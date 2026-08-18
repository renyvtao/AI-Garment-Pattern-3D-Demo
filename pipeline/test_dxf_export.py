from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dxf_export import export_specification, relative_to_absolute


class DxfExportTests(unittest.TestCase):
    def test_relative_control_point_matches_edge_frame(self) -> None:
        self.assertEqual(relative_to_absolute((2.0, 3.0), (12.0, 3.0), (0.5, 0.2)), (7.0, 5.0))

    def test_metric_dxf_contains_native_lines_and_splines(self) -> None:
        specification = {
            "pattern": {
                "panels": {
                    "front": {
                        "vertices": [[0, 0], [10, 0], [10, 20], [0, 20]],
                        "edges": [
                            {"endpoints": [0, 1], "label": "hem"},
                            {
                                "endpoints": [1, 2],
                                "curvature": {
                                    "type": "cubic",
                                    "params": [[0.25, 0.1], [0.75, 0.1]],
                                },
                            },
                            {"endpoints": [2, 3]},
                            {"endpoints": [3, 0]},
                        ],
                    }
                },
                "panel_order": ["front"],
                "stitches": [],
            },
            "properties": {"curvature_coords": "relative", "units_in_meter": 100},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "pattern_specification.json", root / "pattern.dxf"
            preview = root / "pattern_dxf_preview.svg"
            source.write_text(json.dumps(specification), encoding="utf-8")
            report = export_specification(source, output, preview=preview, edge_labels=True)
            content = output.read_text(encoding="ascii")
            preview_content = preview.read_text(encoding="utf-8")
        self.assertIn("$INSUNITS", content)
        self.assertIn("\nSPLINE\n", content)
        self.assertIn("\nLINE\n", content)
        self.assertIn("front", content)
        self.assertIn("DXF 样片预览", preview_content)
        self.assertIn("<path", preview_content)
        self.assertEqual(report["output_units"], "millimetres")
        self.assertEqual(report["millimetres_per_source_unit"], 10.0)
        self.assertEqual(report["panel_count"], 1)
        self.assertEqual(report["entity_counts"]["cubic"], 1)


if __name__ == "__main__":
    unittest.main()
