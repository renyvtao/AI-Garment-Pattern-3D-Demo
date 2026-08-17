from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from urllib.request import Request, urlopen


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

# body_service also exposes scheme A, whose adapter imports torch.  Scheme-B
# route tests intentionally replace only that import boundary so they can run
# on the project's documented CPU/no-Torch preparation environment.
fake_shapy = types.ModuleType("shapy_adapter")
fake_shapy.ATTRIBUTE_KEYS = {"female": [], "male": []}


class _FakeCheckpointMissing(Exception):
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint


fake_shapy.ShapyCheckpointMissing = _FakeCheckpointMissing
fake_shapy.checkpoint_path = lambda *args, **kwargs: Path("missing.ckpt")
fake_shapy.route_status = lambda *args, **kwargs: []
sys.modules["shapy_adapter"] = fake_shapy

from body_service import BodyHandler, ThreadingHTTPServer
from gbt1335_adapter import (
    BODY_FIELDS,
    GBT1335Error,
    generate_garmentcode_body,
    write_garmentcode_artifacts,
)


def writable_temporary_directory(testcase: unittest.TestCase) -> tempfile.TemporaryDirectory:
    temporary = tempfile.TemporaryDirectory(
        dir=MODULE_ROOT,
        ignore_cleanup_errors=True,
    )
    probe = Path(temporary.name) / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        # A read-only sandbox may let tempfile reserve a directory name but
        # deny every operation inside it.  Detach its cleanup finalizer to
        # avoid a second permission error while reporting the skipped test.
        temporary._finalizer.detach()  # type: ignore[attr-defined]
        testcase.skipTest(f"temporary filesystem is read-only: {exc}")
    testcase.addCleanup(temporary.cleanup)
    return temporary


class AdapterTests(unittest.TestCase):
    def test_female_c_maps_to_b_and_estimates_hips(self) -> None:
        body, audit = generate_garmentcode_body(
            {
                "gender": "female",
                "height_cm": 165,
                "chest_cm": 88,
                "waist_cm": 82,
            }
        )
        self.assertEqual(audit["body_type"]["inferred"], "C")
        self.assertEqual(audit["body_type"]["mapped"], "B")
        self.assertTrue(audit["body_type"]["mapping_applied"])
        self.assertEqual(audit["base_source_id"], "00402.yaml")
        self.assertEqual(audit["target"]["hips_source"], "estimated_from_waist")
        self.assertEqual(body["hips"], 99.2)
        self.assertEqual(len(body), 26)
        self.assertEqual(set(body), set(BODY_FIELDS))
        self.assertEqual(sum(audit["source_counts"].values()), 26)

    def test_user_measurements_win_even_when_proportions_conflict(self) -> None:
        body, audit = generate_garmentcode_body(
            {
                "gender": "male",
                "height_cm": 181.2,
                "chest_cm": 107.3,
                "waist_cm": 101.1,
                "hips_cm": 83.4,
                "boundary_policy": "clamp",
            }
        )
        self.assertEqual(body["height"], 181.2)
        self.assertEqual(body["bust"], 107.3)
        self.assertEqual(body["waist"], 101.1)
        self.assertEqual(body["hips"], 83.4)
        self.assertEqual(audit["target"]["hips_source"], "user_input")

    def test_clamp_only_limits_linked_deltas(self) -> None:
        common = {
            "gender": "female",
            "height_cm": 190,
            "chest_cm": 120,
            "waist_cm": 100,
        }
        extrapolated, extrapolated_audit = generate_garmentcode_body(common)
        clamped, clamped_audit = generate_garmentcode_body(
            {**common, "out_of_range_mode": "clamp"}
        )
        self.assertEqual(extrapolated["height"], 190)
        self.assertEqual(clamped["height"], 190)
        self.assertNotEqual(extrapolated["arm_length"], clamped["arm_length"])
        self.assertEqual(
            clamped_audit["deltas_from_standard_anchor"]["height"]["applied_to_linked_fields_cm"],
            10,
        )
        self.assertEqual(extrapolated_audit["boundary_policy"], "extrapolate")
        self.assertEqual(clamped_audit["boundary_policy"], "clamp")

    def test_outside_body_type_range_is_audited(self) -> None:
        _, audit = generate_garmentcode_body(
            {
                "gender": "male",
                "height_cm": 170,
                "chest_cm": 90,
                "waist_cm": 90,
                "out_of_range_mode": "clamp",
            }
        )
        self.assertEqual(audit["body_type"]["inferred"], "C")
        self.assertEqual(audit["body_type"]["mapped"], "B")
        self.assertEqual(audit["body_type"]["classification_drop_cm"], 2)
        self.assertTrue(any("夹到2 cm" in warning for warning in audit["warnings"]))

    def test_rejects_non_positive_measurement(self) -> None:
        with self.assertRaises(GBT1335Error):
            generate_garmentcode_body(
                {"gender": "female", "height_cm": 160, "chest_cm": 84, "waist_cm": 0}
            )

    def test_writes_yaml_and_json_audit(self) -> None:
        temporary = writable_temporary_directory(self)
        output_dir = Path(temporary.name) / "result"
        body, audit, files = write_garmentcode_artifacts(
            {
                "gender": "female", "height_cm": 160,
                "chest_cm": 84, "waist_cm": 68,
            },
            output_dir,
        )
        yaml_text = (output_dir / files["body_yaml"]).read_text(encoding="utf-8")
        saved_audit = json.loads(
            (output_dir / files["audit_json"]).read_text(encoding="utf-8")
        )
        self.assertTrue(yaml_text.startswith("body:\n"))
        self.assertEqual(sum(1 for line in yaml_text.splitlines() if line.startswith("  ")), 26)
        self.assertEqual(saved_audit["body"], body)
        self.assertEqual(saved_audit["base_source_id"], audit["base_source_id"])


class HttpRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = writable_temporary_directory(self)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), BodyHandler)
        self.server.config = {"output_root": Path(self.temporary.name)}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_schema_and_generate_routes(self) -> None:
        with urlopen(f"{self.base_url}/api/body/gbt1335/schema") as response:
            schema = json.load(response)
        self.assertEqual(schema["output"]["body_field_count"], 26)
        self.assertEqual(schema["body_type_mapping"]["C"], "B")

        data = json.dumps(
            {
                "gender": "female", "height_cm": 165,
                "chest_cm": 88, "waist_cm": 74, "hips_cm": None,
                "out_of_range_mode": "extrapolate",
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/body/gbt1335/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            self.assertEqual(response.status, 201)
            result = json.load(response)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["base_source_id"], "00319.yaml")
        self.assertEqual(len(result["body"]), 26)
        self.assertIn("body_yaml", result["downloads"])
        self.assertIn("audit_json", result["downloads"])
        output_dir = Path(self.temporary.name) / result["request_id"]
        self.assertTrue((output_dir / "garmentcode_body.yaml").is_file())
        self.assertTrue((output_dir / "audit.json").is_file())


if __name__ == "__main__":
    unittest.main()
