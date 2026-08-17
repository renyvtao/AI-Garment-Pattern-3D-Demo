#!/usr/bin/env python3
"""Print compact panel/stitch metadata for GarmentCode specification files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.specs:
        document = json.loads(path.read_text(encoding="utf-8"))
        pattern = document["pattern"]
        panels = pattern["panels"]
        stitches = pattern.get("stitches", [])
        print(json.dumps({
            "path": str(path),
            "panel_count": len(panels),
            "stitch_count": len(stitches),
            "panels": {
                name: {
                    "edge_count": len(panel.get("edges", [])),
                    "translation": panel.get("translation"),
                    "rotation": panel.get("rotation"),
                }
                for name, panel in panels.items()
            },
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
