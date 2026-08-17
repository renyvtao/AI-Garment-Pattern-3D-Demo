#!/usr/bin/env python3
"""Build a first K62-compatible 3D assembly specification from a suit 2D spec.

The adapter deliberately preserves the validated K62 panel topology, stitch
graph, labels, and 3D placement.  Target parameterization is introduced by
scaling each K62 panel around its local origin to the bounding dimensions of
the corresponding generated 2D panel.  This is the first static-3D POC; a
later semantic-boundary transfer can replace the coarse per-panel scaling
without changing the downstream simulation contract.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROLES = (
    "front_center",
    "front_side",
    "sleeve_big",
    "sleeve_small",
    "back",
    "collar",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", type=Path, required=True)
    parser.add_argument("--golden-spec", type=Path, required=True)
    parser.add_argument("--output-spec", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-scale", type=float, default=0.75)
    parser.add_argument("--max-scale", type=float, default=1.25)
    parser.add_argument(
        "--button-count",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Add K62 virtual-button closure stitches; 0 keeps the front open.",
    )
    return parser.parse_args()


def panel_role(name: str) -> str:
    for role in ROLES:
        if role in name:
            return role
    raise ValueError(f"unrecognized suit panel role: {name}")


def dimensions(panel: dict[str, Any]) -> tuple[float, float]:
    vertices = panel.get("vertices", [])
    if not vertices:
        raise ValueError("panel has no vertices")
    xs = [float(point[0]) for point in vertices]
    ys = [float(point[1]) for point in vertices]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate panel dimensions: {width} x {height}")
    return width, height


def load_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_panel_by_side_and_role(
    panels: dict[str, dict[str, Any]],
    side: str,
    role: str,
) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, panel)
        for name, panel in panels.items()
        if side in name and role in name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {side} {role} panel; found {[name for name, _ in matches]}"
        )
    return matches[0]


def edge_index_by_label(panel: dict[str, Any], label: str) -> int:
    matches = [
        index
        for index, edge in enumerate(panel.get("edges", []))
        if edge.get("label") == label
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one edge labeled {label!r}; found {matches}")
    return matches[0]


def add_button_closure(
    output: dict[str, Any],
    button_count: int,
) -> list[dict[str, Any]]:
    if button_count == 0:
        return []
    panels = output["pattern"]["panels"]
    left_name, left_panel = find_panel_by_side_and_role(
        panels, "left_clean2d", "front_center"
    )
    right_name, right_panel = find_panel_by_side_and_role(
        panels, "right_clean2d", "front_center"
    )
    stitches = output["pattern"].setdefault("stitches", [])
    added: list[dict[str, Any]] = []
    for number in range(1, button_count + 1):
        label = f"virtual_button_{number}"
        left_edge = edge_index_by_label(left_panel, label)
        right_edge = edge_index_by_label(right_panel, label)
        left_ref = {"panel": left_name, "edge": left_edge}
        right_ref = {"panel": right_name, "edge": right_edge}
        duplicate = any(
            len(stitch) >= 2
            and {
                (stitch[0].get("panel"), stitch[0].get("edge")),
                (stitch[1].get("panel"), stitch[1].get("edge")),
            }
            == {
                (left_name, left_edge),
                (right_name, right_edge),
            }
            for stitch in stitches
        )
        if duplicate:
            raise ValueError(f"closure stitch already exists for {label}")
        stitches.append([left_ref, right_ref, f"model_closure_{label}"])
        added.append(
            {
                "button_number": number,
                "label": label,
                "left": left_ref,
                "right": right_ref,
            }
        )
    return added


def main() -> None:
    args = parse_args()
    if not 0 < args.min_scale <= args.max_scale:
        raise ValueError("invalid scale bounds")

    source = load_document(args.input_spec)
    golden = load_document(args.golden_spec)
    source_panels = source["pattern"]["panels"]
    golden_panels = golden["pattern"]["panels"]

    source_by_role: dict[str, dict[str, Any]] = {}
    source_names: dict[str, str] = {}
    for name, panel in source_panels.items():
        role = panel_role(name)
        if role in source_by_role:
            raise ValueError(f"duplicate source role: {role}")
        source_by_role[role] = panel
        source_names[role] = name
    missing = sorted(set(ROLES) - set(source_by_role))
    if missing:
        raise ValueError(f"source specification is missing roles: {missing}")

    output = copy.deepcopy(golden)
    output_panels = output["pattern"]["panels"]
    panel_audit: dict[str, Any] = {}
    for name, panel in output_panels.items():
        role = panel_role(name)
        source_panel = source_by_role[role]
        source_width, source_height = dimensions(source_panel)
        golden_width, golden_height = dimensions(golden_panels[name])
        raw_scale_x = source_width / golden_width
        raw_scale_y = source_height / golden_height
        scale_x = min(args.max_scale, max(args.min_scale, raw_scale_x))
        scale_y = min(args.max_scale, max(args.min_scale, raw_scale_y))

        # GarmentCode panel translations locate the local origin in body space.
        # Scaling around that origin keeps the validated K62 shoulder/neck or
        # sleeve-cap anchor in place while allowing width/length variation.
        panel["vertices"] = [
            [float(point[0]) * scale_x, float(point[1]) * scale_y]
            for point in panel["vertices"]
        ]
        panel_audit[name] = {
            "role": role,
            "source_panel": source_names[role],
            "golden_dimensions_cm": [golden_width, golden_height],
            "source_dimensions_cm": [source_width, source_height],
            "raw_scale": [raw_scale_x, raw_scale_y],
            "applied_scale": [scale_x, scale_y],
            "clamped": [raw_scale_x != scale_x, raw_scale_y != scale_y],
            "translation": panel.get("translation"),
            "rotation": panel.get("rotation"),
        }

    closure_stitches = add_button_closure(output, args.button_count)

    audit = {
        "schema": "k62_suit_3d_adapter_v2",
        "method": "preserve_golden_topology_stitches_placement_scale_panels_and_apply_model_button_closure",
        "input_spec": str(args.input_spec.resolve()),
        "golden_spec": str(args.golden_spec.resolve()),
        "output_spec": str(args.output_spec.resolve()),
        "panel_count": len(output_panels),
        "stitch_count": len(output["pattern"].get("stitches", [])),
        "source_panel_count": len(source_panels),
        "source_stitch_count": len(source["pattern"].get("stitches", [])),
        "front_closure": {
            "mode": "open" if args.button_count == 0 else "model_button_count",
            "button_count": args.button_count,
            "stitches_added": closure_stitches,
        },
        "scale_bounds": [args.min_scale, args.max_scale],
        "panels": panel_audit,
        "limitations": [
            "Panel topology, stitch graph, labels, and 3D placement come from the fixed K62 Golden reference.",
            "Generated 2D parameters currently affect per-panel width and height, not every semantic boundary curve.",
            "Button closure uses K62's pre-existing virtual-button edge segments; no decorative button or buttonhole geometry is generated.",
            "This output must pass BoxMesh and Warp validation before it can be presented as a 3D result.",
        ],
    }

    args.output_spec.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_spec.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
