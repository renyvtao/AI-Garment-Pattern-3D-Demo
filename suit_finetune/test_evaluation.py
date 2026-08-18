from __future__ import annotations

import unittest

from evaluate_suit_outputs import summarize_variant


class SuitEvaluationTests(unittest.TestCase):
    def test_tolerances_and_failures_are_counted(self) -> None:
        expected = {
            "garment_length_ratio": 1.75,
            "waist_ease_cm": 15.0,
            "lapel_style": "notched",
            "button_count": 2,
            "small_pocket_enabled": True,
            "large_pockets_enabled": True,
        }
        good = dict(expected, garment_length_ratio=1.759, waist_ease_cm=15.5)
        results = [
            {
                "id": "good",
                "generation_success": True,
                "parse_success": True,
                "schema_complete": True,
                "raw_predicted": good,
                "predicted": good,
                "expected": expected,
                "corrections": [],
                "pattern_attempted": True,
                "pattern_success": True,
            },
            {
                "id": "failed",
                "generation_success": True,
                "parse_success": False,
                "schema_complete": False,
                "expected": expected,
                "pattern_attempted": False,
                "pattern_success": None,
            },
        ]
        summary, rows = summarize_variant("suit_lora", results)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["parse_success_rate"], 0.5)
        self.assertEqual(summary["pattern_success_rate"], 1.0)
        self.assertEqual(summary["field_metrics"]["garment_length_ratio"]["accuracy"], 0.5)
        self.assertEqual(
            summary["field_metrics"]["garment_length_ratio"]["balanced_accuracy"], 0.5
        )
        self.assertEqual(
            summary["field_metrics"]["garment_length_ratio"]["majority_baseline_accuracy"],
            1.0,
        )
        self.assertEqual(summary["field_metrics"]["waist_ease_cm"]["accuracy"], 0.5)
        self.assertEqual(summary["all_fields_pass_rate"], 0.5)
        self.assertEqual(summary["majority_baseline_all_fields_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
