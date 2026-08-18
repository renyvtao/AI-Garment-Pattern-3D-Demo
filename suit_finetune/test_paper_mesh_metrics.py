import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("evaluate_paper_mesh_metrics.py")
SPEC = importlib.util.spec_from_file_location("evaluate_paper_mesh_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
MESH_DEPS_AVAILABLE = (
    importlib.util.find_spec("trimesh") is not None
    and importlib.util.find_spec("scipy") is not None
)
PYTORCH3D_AVAILABLE = importlib.util.find_spec("pytorch3d") is not None


class PaperMeshMetricTests(unittest.TestCase):
    def test_fscore_uses_squared_distance_threshold(self) -> None:
        pred = np.array([[0.0005, 0.0020]])
        gt = np.array([[0.0001, 0.0002]])
        fscore, precision, recall = MODULE.fscore_from_squared_distances(
            pred, gt, threshold=0.001
        )
        self.assertAlmostEqual(precision.item(), 0.5)
        self.assertAlmostEqual(recall.item(), 1.0)
        self.assertAlmostEqual(fscore.item(), 2 / 3)

    @unittest.skipUnless(MESH_DEPS_AVAILABLE, "trimesh and scipy are required")
    def test_mesh_metric_runs_without_pytorch3d(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / "triangle.obj"
            mesh.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="ascii",
            )
            result = MODULE.evaluate_case(mesh, mesh, 128, 0.001, 7)
        self.assertGreaterEqual(result["chamfer_distance_x1000"], 0.0)
        self.assertGreaterEqual(result["fscore"], 0.0)
        self.assertLessEqual(result["fscore"], 1.0)

    @unittest.skipUnless(PYTORCH3D_AVAILABLE, "pytorch3d is optional locally")
    def test_pytorch3d_backend_runs_on_a_triangle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / "triangle.obj"
            mesh.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="ascii",
            )
            result = MODULE.evaluate_case_pytorch3d(mesh, mesh, 128, 0.001, "cpu")
        self.assertGreaterEqual(result["chamfer_distance_x1000"], 0.0)
        self.assertGreaterEqual(result["fscore"], 0.0)
        self.assertLessEqual(result["fscore"], 1.0)

    def test_auto_backend_is_explicit(self) -> None:
        expected = "pytorch3d" if PYTORCH3D_AVAILABLE else "scipy"
        self.assertEqual(MODULE.select_backend("auto"), expected)

    def test_manifest_rejects_missing_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case",
                                "prediction": "prediction.obj",
                                "ground_truth": "missing.obj",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                MODULE.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
