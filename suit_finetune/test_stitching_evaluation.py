import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_suit_stitching_evaluation.py")
SPEC = importlib.util.spec_from_file_location("run_suit_stitching_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StitchingSummaryTests(unittest.TestCase):
    def test_preparation_and_simulation_failures_are_counted(self) -> None:
        prepared = [
            {"case_id": "ok", "prepared": True},
            {"case_id": "sim_fail", "prepared": True},
            {"case_id": "adapter_fail", "prepared": False, "error": "bad spec"},
        ]
        simulations = [
            {"garment_name": "ok_k62", "status": "completed", "elapsed_seconds": 2},
            {"garment_name": "sim_fail_k62", "status": "failed", "error": "stitch"},
        ]
        summary = MODULE.summarize(prepared, simulations)
        self.assertEqual(summary["total_case_count"], 3)
        self.assertEqual(summary["completed_case_count"], 1)
        self.assertEqual(summary["failure_count"], 2)
        self.assertAlmostEqual(summary["stitching_failure_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
