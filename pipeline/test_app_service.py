from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from app_service import (
    JobStore,
    Pipeline,
    delete_job,
    load_lower_specifications,
    parse_byte_range,
    read_lower_garment_type,
    resolve_suit_button_count,
    row_payload,
    safe_child,
)


class JobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = JobStore(self.root / "data")
        self.pipeline = Pipeline(self.root / "project", self.store)
        self.config = {
            "body_mode": "preset",
            "gender": "female",
            "action_id": "none",
            "input_files": ["001_test.png"],
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_job(self) -> tuple[str, Path]:
        job_id = self.store.create("test", self.config)
        root = self.store.job_root(job_id)
        (root / "inputs").mkdir(parents=True)
        (root / "inputs" / "001_test.png").write_bytes(b"png")
        (root / "outputs").mkdir()
        (root / "outputs" / "audit.json").write_text("{}", encoding="utf-8")
        return job_id, root

    def test_manifest_and_bundle(self) -> None:
        job_id, root = self.make_job()
        self.pipeline.write_result_manifest(job_id, self.config)
        bundle = self.pipeline.make_bundle(job_id)
        self.assertTrue(bundle.is_file())
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
        self.assertIn("inputs/001_test.png", names)
        self.assertIn("outputs/audit.json", names)
        self.assertIn("result_manifest.json", names)

    def test_cache_trash_and_permanent_delete(self) -> None:
        job_id, root = self.make_job()
        (root / "work").mkdir()
        (root / "work" / "cache.bin").write_bytes(b"cache")
        sequence = root / "outputs" / "contourcraft_sequence.npz"
        sequence.write_bytes(b"sequence")
        body_sequence = root / "outputs" / "body_sequence.pkl"
        body_sequence.write_bytes(b"body-sequence")
        self.pipeline.make_bundle(job_id)
        self.store.update(job_id, state="completed")
        delete_job(self.store, job_id, "cache")
        self.assertFalse((root / "work").exists())
        self.assertFalse(sequence.exists())
        self.assertFalse(body_sequence.exists())
        with zipfile.ZipFile(root / "result_bundle.zip") as archive:
            names = set(archive.namelist())
        self.assertNotIn("outputs/contourcraft_sequence.npz", names)
        self.assertNotIn("outputs/body_sequence.pkl", names)
        delete_job(self.store, job_id, "trash")
        self.assertFalse(root.exists())
        self.assertEqual(self.store.row(job_id)["state"], "trashed")
        delete_job(self.store, job_id, "permanent")
        self.assertIsNone(self.store.row(job_id))

    def test_path_escape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            safe_child(self.store.runs_root, "../outside")

    def test_video_byte_ranges(self) -> None:
        self.assertIsNone(parse_byte_range(None, 1000))
        self.assertEqual(parse_byte_range("bytes=0-99", 1000), (0, 99))
        self.assertEqual(parse_byte_range("bytes=900-", 1000), (900, 999))
        self.assertEqual(parse_byte_range("bytes=-100", 1000), (900, 999))
        self.assertEqual(parse_byte_range("bytes=950-1200", 1000), (950, 999))
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=1000-", 1000)

    def test_runtime_labels_follow_the_active_substep(self) -> None:
        self.pipeline.runtime_profile.update(
            gpu_name="NVIDIA GeForce RTX 4090",
            chatgarment_cuda=True,
            static_cuda=True,
            ccraft_cuda=True,
            nvidia_egl=True,
        )
        simulation = self.pipeline.runtime_for_job(
            state="running",
            step="dynamic_3d",
            message="ContourCraft 正在使用 CUDA 进行动态布料仿真",
            config=self.config,
        )
        rendering = self.pipeline.runtime_for_job(
            state="running",
            step="dynamic_3d",
            message="动态视频逐帧渲染：12/100 帧",
            config=self.config,
        )
        self.assertEqual(simulation["kind"], "gpu")
        self.assertIn("RTX 4090", simulation["label"])
        self.assertIn("ContourCraft", simulation["detail"])
        self.assertIn("NVIDIA EGL", rendering["detail"])
        self.assertTrue(rendering["verified"])

    def test_static_simulation_quality_is_exposed(self) -> None:
        job_id, root = self.make_job()
        summary = root / "outputs" / "mens_suit" / "static_3d_simulation_summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(
            """[
              {
                "status": "completed",
                "simulation_stats": {
                  "body_collisions": {"suit": 0},
                  "self_collisions": {"suit": 14},
                  "fin_frame": {"suit": 320},
                  "fails": {"cloth_self_intersection": ["suit"]}
                }
              }
            ]""",
            encoding="utf-8",
        )
        payload = row_payload(self.store, self.pipeline, self.store.row(job_id))
        quality = payload["simulation_quality"]
        self.assertEqual(quality["completed_count"], 1)
        self.assertEqual(quality["body_collisions"], 0)
        self.assertEqual(quality["self_collisions"], 14)
        self.assertEqual(quality["min_frames"], 320)
        self.assertEqual(quality["warnings"], ["cloth_self_intersection"])

    def test_suit_button_count_is_resolved_and_exposed(self) -> None:
        job_id, root = self.make_job()
        result = root / "result.json"
        result.write_text(
            '{"predicted": {"button_count": 3}}', encoding="utf-8"
        )
        resolved = resolve_suit_button_count(result)
        self.assertEqual(resolved["model_button_count"], 3)
        self.assertEqual(resolved["resolved_button_count"], 2)
        self.assertTrue(resolved["clamped"])

        config = dict(self.config, garment_mode="mens_suit")
        self.store.update(job_id, config_json=json.dumps(config))
        audit = (
            root
            / "outputs"
            / "mens_suit"
            / "case"
            / "static_3d"
            / "case_k62_3d_adapter_audit.json"
        )
        audit.parent.mkdir(parents=True)
        audit.write_text(
            '{"front_closure": {"button_count": 2, "stitches_added": [{}, {}]}}',
            encoding="utf-8",
        )
        payload = row_payload(self.store, self.pipeline, self.store.row(job_id))
        self.assertEqual(payload["suit_closure_summary"]["counts"], {"2": 1})
        self.assertEqual(payload["suit_closure_summary"]["closure_stitch_count"], 2)

    def test_official_lower_specs_and_summary_are_exposed(self) -> None:
        chatgarment = self.root / "ChatGarment"
        vis_root = chatgarment / "runs" / "case" / "vis_new"
        lower = vis_root / "sample" / "sample_lower" / "sample_lower_specification.json"
        upper = vis_root / "sample" / "sample_upper" / "sample_upper_specification.json"
        lower.parent.mkdir(parents=True)
        upper.parent.mkdir(parents=True)
        lower.write_text("{}", encoding="utf-8")
        upper.write_text("{}", encoding="utf-8")
        spec_list = vis_root / "all_json_spec_files.json"
        spec_list.write_text(
            json.dumps([str(upper), str(lower)]), encoding="utf-8"
        )
        self.assertEqual(
            load_lower_specifications(chatgarment, vis_root, spec_list), [lower]
        )
        design = lower.parent / "design.yaml"
        design.write_text(
            "design:\n  meta:\n    bottom:\n      v: Pants\n",
            encoding="utf-8",
        )
        self.assertEqual(read_lower_garment_type(design), "Pants")

        job_id, root = self.make_job()
        config = dict(self.config, garment_mode="mens_suit")
        self.store.update(job_id, config_json=json.dumps(config))
        manifest = root / "outputs" / "official_lower" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "expected_image_count": 1,
                    "detected_lower_count": 1,
                    "static_completed_count": 1,
                    "garment_counts": {"Pants": 1},
                    "dynamic_included": False,
                    "simulation_quality": {
                        "body_collisions": 2,
                        "self_collisions": 5,
                        "min_frames": 300,
                        "max_frames": 300,
                        "warnings": ["cloth_body_intersection"],
                    },
                }
            ),
            encoding="utf-8",
        )
        payload = row_payload(self.store, self.pipeline, self.store.row(job_id))
        self.assertEqual(payload["official_lower_summary"]["garment_counts"], {"Pants": 1})
        self.assertEqual(payload["official_lower_summary"]["static_completed_count"], 1)
        self.assertEqual(
            payload["official_lower_summary"]["simulation_quality"]["body_collisions"],
            2,
        )
        self.assertFalse(payload["official_lower_summary"]["dynamic_included"])


if __name__ == "__main__":
    unittest.main()
