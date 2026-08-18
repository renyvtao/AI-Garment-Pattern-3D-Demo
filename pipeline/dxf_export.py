#!/usr/bin/env python3
"""Export GarmentCode sewing-pattern specifications as metric DXF files.

The exporter reads panel geometry directly from ``*_specification.json``.
Straight edges and Bezier curves are preserved as native DXF LINE/SPLINE
entities. Circular edges are preserved as ARC entities. Panels are packed on
one non-overlapping sheet without rotation, so the source dimensions remain
unchanged.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Point = tuple[float, float]


@dataclass(frozen=True)
class Curve:
    kind: str
    points: tuple[Point, ...]
    radius: float | None = None
    large_arc: bool = False
    clockwise: bool = False
    label: str = ""


@dataclass
class PanelGeometry:
    name: str
    curves: list[Curve]
    bounds: tuple[float, float, float, float]
    offset: Point = (0.0, 0.0)


def relative_to_absolute(start: Point, end: Point, relative: Iterable[float]) -> Point:
    """Match GarmentCode's rel_to_abs_2d edge-coordinate conversion."""
    rx, ry = (float(value) for value in relative)
    dx, dy = end[0] - start[0], end[1] - start[1]
    return (
        start[0] + rx * dx - ry * dy,
        start[1] + rx * dy + ry * dx,
    )


def _bezier_point(points: tuple[Point, ...], t: float) -> Point:
    work = [list(point) for point in points]
    while len(work) > 1:
        work = [
            [
                (1.0 - t) * left[0] + t * right[0],
                (1.0 - t) * left[1] + t * right[1],
            ]
            for left, right in zip(work, work[1:])
        ]
    return work[0][0], work[0][1]


def _circle_geometry(curve: Curve) -> tuple[Point, float, float, bool]:
    """Return center/start/end angles and whether the source path is clockwise."""
    start, end = curve.points[0], curve.points[-1]
    radius = float(curve.radius or 0.0)
    dx, dy = end[0] - start[0], end[1] - start[1]
    chord = math.hypot(dx, dy)
    if chord == 0 or radius < chord / 2.0:
        raise ValueError(f"invalid circular edge radius={radius} chord={chord}")
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    height = math.sqrt(max(0.0, radius * radius - (chord / 2.0) ** 2))
    perpendicular = (-dy / chord, dx / chord)
    candidates = [
        (
            midpoint[0] + sign * height * perpendicular[0],
            midpoint[1] + sign * height * perpendicular[1],
        )
        for sign in (1.0, -1.0)
    ]
    for center in candidates:
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        if curve.clockwise:
            sweep = (start_angle - end_angle) % (2.0 * math.pi)
        else:
            sweep = (end_angle - start_angle) % (2.0 * math.pi)
        if (sweep > math.pi) == curve.large_arc or math.isclose(sweep, math.pi):
            return center, start_angle, end_angle, curve.clockwise
    raise ValueError("unable to resolve circular edge flags")


def _curve_samples(curve: Curve, count: int = 65) -> list[Point]:
    if curve.kind in {"line", "quadratic", "cubic"}:
        if curve.kind == "line":
            return list(curve.points)
        return [_bezier_point(curve.points, index / (count - 1)) for index in range(count)]
    if curve.kind == "circle":
        center, start_angle, end_angle, clockwise = _circle_geometry(curve)
        if clockwise:
            sweep = -((start_angle - end_angle) % (2.0 * math.pi))
        else:
            sweep = (end_angle - start_angle) % (2.0 * math.pi)
        radius = float(curve.radius)
        return [
            (
                center[0] + radius * math.cos(start_angle + sweep * index / (count - 1)),
                center[1] + radius * math.sin(start_angle + sweep * index / (count - 1)),
            )
            for index in range(count)
        ]
    raise ValueError(f"unsupported curve kind: {curve.kind}")


def _curve_from_edge(vertices: list[list[float]], edge: dict[str, Any], scale: float) -> Curve:
    endpoints = edge.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError(f"invalid edge endpoints: {endpoints!r}")
    try:
        start = tuple(float(value) * scale for value in vertices[int(endpoints[0])])
        end = tuple(float(value) * scale for value in vertices[int(endpoints[1])])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid edge vertices: {endpoints!r}") from exc
    label = str(edge.get("label", ""))
    curvature = edge.get("curvature")
    if curvature is None:
        return Curve("line", (start, end), label=label)

    if isinstance(curvature, list):
        params = [curvature]
        kind = "quadratic"
    elif isinstance(curvature, dict):
        kind = str(curvature.get("type", "")).lower()
        params = curvature.get("params")
    else:
        raise ValueError(f"invalid curvature: {curvature!r}")

    unscaled_start = (start[0] / scale, start[1] / scale)
    unscaled_end = (end[0] / scale, end[1] / scale)
    if kind in {"quadratic", "cubic"}:
        expected = 1 if kind == "quadratic" else 2
        if not isinstance(params, list) or len(params) != expected:
            raise ValueError(f"{kind} edge requires {expected} control point(s)")
        controls = [
            tuple(value * scale for value in relative_to_absolute(unscaled_start, unscaled_end, point))
            for point in params
        ]
        return Curve(kind, (start, *controls, end), label=label)
    if kind == "circle":
        if not isinstance(params, list) or len(params) != 3:
            raise ValueError("circle edge requires radius, large-arc and side flags")
        radius, large_arc, right = params
        return Curve(
            "circle",
            (start, end),
            radius=float(radius) * scale,
            large_arc=bool(large_arc),
            clockwise=bool(right),
            label=label,
        )
    raise ValueError(f"unsupported GarmentCode curvature type: {kind!r}")


def _panel_geometry(name: str, panel: dict[str, Any], scale: float) -> PanelGeometry:
    vertices = panel.get("vertices")
    edges = panel.get("edges")
    if not isinstance(vertices, list) or not isinstance(edges, list) or not edges:
        raise ValueError(f"panel {name!r} has no usable vertices/edges")
    curves = [_curve_from_edge(vertices, edge, scale) for edge in edges]
    samples = [point for curve in curves for point in _curve_samples(curve)]
    xs = [point[0] for point in samples]
    ys = [point[1] for point in samples]
    return PanelGeometry(name, curves, (min(xs), min(ys), max(xs), max(ys)))


def _pack_panels(panels: list[PanelGeometry], gap_mm: float) -> tuple[float, float]:
    area = sum(
        max(1.0, panel.bounds[2] - panel.bounds[0])
        * max(1.0, panel.bounds[3] - panel.bounds[1])
        for panel in panels
    )
    widest = max(panel.bounds[2] - panel.bounds[0] for panel in panels)
    target_width = max(widest, math.sqrt(area) * 1.7)
    cursor_x = cursor_y = row_height = 0.0
    used_width = 0.0
    for panel in panels:
        min_x, min_y, max_x, max_y = panel.bounds
        width, height = max_x - min_x, max_y - min_y
        if cursor_x > 0 and cursor_x + width > target_width:
            cursor_x = 0.0
            cursor_y += row_height + gap_mm
            row_height = 0.0
        panel.offset = (cursor_x - min_x, cursor_y - min_y)
        cursor_x += width + gap_mm
        row_height = max(row_height, height)
        used_width = max(used_width, cursor_x - gap_mm)
    return used_width, cursor_y + row_height


class DxfWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, code: int, value: Any) -> None:
        if isinstance(value, float):
            rendered = f"{value:.9f}".rstrip("0").rstrip(".") or "0"
        else:
            rendered = str(value)
        self.lines.extend((str(code), rendered))

    def line(self, start: Point, end: Point, layer: str = "CUT") -> None:
        self.add(0, "LINE")
        self.add(100, "AcDbEntity")
        self.add(8, layer)
        self.add(100, "AcDbLine")
        self.add(10, start[0]); self.add(20, start[1]); self.add(30, 0.0)
        self.add(11, end[0]); self.add(21, end[1]); self.add(31, 0.0)

    def spline(self, points: tuple[Point, ...], layer: str = "CUT") -> None:
        degree = len(points) - 1
        knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        self.add(0, "SPLINE")
        self.add(100, "AcDbEntity")
        self.add(8, layer)
        self.add(100, "AcDbSpline")
        self.add(210, 0.0); self.add(220, 0.0); self.add(230, 1.0)
        self.add(70, 8)
        self.add(71, degree)
        self.add(72, len(knots))
        self.add(73, len(points))
        self.add(74, 0)
        self.add(42, 1e-7); self.add(43, 1e-7); self.add(44, 1e-10)
        for knot in knots:
            self.add(40, knot)
        for x, y in points:
            self.add(10, x); self.add(20, y); self.add(30, 0.0)

    def arc(self, curve: Curve, offset: Point, layer: str = "CUT") -> None:
        center, start_angle, end_angle, clockwise = _circle_geometry(curve)
        if clockwise:
            start_angle, end_angle = end_angle, start_angle
        self.add(0, "ARC")
        self.add(100, "AcDbEntity")
        self.add(8, layer)
        self.add(100, "AcDbCircle")
        self.add(10, center[0] + offset[0]); self.add(20, center[1] + offset[1]); self.add(30, 0.0)
        self.add(40, float(curve.radius))
        self.add(100, "AcDbArc")
        self.add(50, math.degrees(start_angle) % 360.0)
        self.add(51, math.degrees(end_angle) % 360.0)

    def text(self, point: Point, value: str, height: float, layer: str) -> None:
        safe = value.encode("ascii", "replace").decode("ascii")[:250]
        self.add(0, "TEXT")
        self.add(100, "AcDbEntity")
        self.add(8, layer)
        self.add(100, "AcDbText")
        self.add(10, point[0]); self.add(20, point[1]); self.add(30, 0.0)
        self.add(40, height)
        self.add(1, safe)
        self.add(50, 0.0)


def _translated(points: tuple[Point, ...], offset: Point) -> tuple[Point, ...]:
    return tuple((x + offset[0], y + offset[1]) for x, y in points)


def _write_header(writer: DxfWriter, layer_names: list[str]) -> None:
    writer.add(0, "SECTION"); writer.add(2, "HEADER")
    writer.add(9, "$ACADVER"); writer.add(1, "AC1015")
    writer.add(9, "$INSUNITS"); writer.add(70, 4)  # millimetres
    writer.add(9, "$MEASUREMENT"); writer.add(70, 1)
    writer.add(0, "ENDSEC")
    writer.add(0, "SECTION"); writer.add(2, "TABLES")
    writer.add(0, "TABLE"); writer.add(2, "LAYER"); writer.add(70, len(layer_names))
    colors = {"CUT": 7, "PANEL_LABEL": 3, "EDGE_LABEL": 5}
    for layer in layer_names:
        writer.add(0, "LAYER")
        writer.add(100, "AcDbSymbolTableRecord")
        writer.add(100, "AcDbLayerTableRecord")
        writer.add(2, layer); writer.add(70, 0); writer.add(62, colors.get(layer, 7)); writer.add(6, "CONTINUOUS")
    writer.add(0, "ENDTAB"); writer.add(0, "ENDSEC")


def _svg_point(point: Point) -> str:
    return f"{point[0]:.6f},{point[1]:.6f}"


def write_svg_preview(
    panels: list[PanelGeometry],
    output: Path,
    sheet_width: float,
    sheet_height: float,
) -> None:
    """Write a browser preview from the exact same packed geometry as the DXF."""
    margin = 12.0
    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{-margin:.3f} {-margin:.3f} {sheet_width + 2 * margin:.3f} '
            f'{sheet_height + 2 * margin:.3f}" role="img" aria-label="DXF 样片预览">'
        ),
        '<rect x="-12" y="-12" width="100%" height="100%" fill="#f7f4ef"/>',
        f'<g transform="translate(0 {sheet_height:.6f}) scale(1 -1)" fill="none" stroke="#171717" stroke-width="0.8" vector-effect="non-scaling-stroke">',
    ]
    for panel in panels:
        for curve in panel.curves:
            points = _translated(curve.points, panel.offset)
            if curve.kind == "line":
                command = f"M {_svg_point(points[0])} L {_svg_point(points[-1])}"
            elif curve.kind == "quadratic":
                command = (
                    f"M {_svg_point(points[0])} Q {_svg_point(points[1])} {_svg_point(points[2])}"
                )
            elif curve.kind == "cubic":
                command = (
                    f"M {_svg_point(points[0])} C {_svg_point(points[1])} "
                    f"{_svg_point(points[2])} {_svg_point(points[3])}"
                )
            else:
                sampled = [
                    (point[0] + panel.offset[0], point[1] + panel.offset[1])
                    for point in _curve_samples(curve, 97)
                ]
                command = f"M {_svg_point(sampled[0])} " + " ".join(
                    f"L {_svg_point(point)}" for point in sampled[1:]
                )
            svg.append(f'<path d="{command}"/>')
    svg.append("</g>")
    for panel in panels:
        min_x, min_y, max_x, max_y = panel.bounds
        x = (min_x + max_x) / 2.0 + panel.offset[0]
        y = sheet_height - ((min_y + max_y) / 2.0 + panel.offset[1])
        svg.append(
            f'<text x="{x:.6f}" y="{y:.6f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="5" fill="#c95c2d">'
            f'{html.escape(panel.name)}</text>'
        )
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def export_specification(
    specification: Path,
    output: Path,
    *,
    preview: Path | None = None,
    edge_labels: bool = False,
    gap_mm: float = 30.0,
) -> dict[str, Any]:
    payload = json.loads(specification.read_text(encoding="utf-8"))
    pattern = payload.get("pattern")
    if not isinstance(pattern, dict) or not isinstance(pattern.get("panels"), dict):
        raise ValueError(f"not a GarmentCode pattern specification: {specification}")
    properties = payload.get("properties") if isinstance(payload.get("properties"), dict) else {}
    units_per_meter = float(properties.get("units_in_meter", 100.0))
    if units_per_meter <= 0:
        raise ValueError("properties.units_in_meter must be positive")
    millimetres_per_unit = 1000.0 / units_per_meter

    panel_map = pattern["panels"]
    declared_order = pattern.get("panel_order")
    names = [name for name in declared_order or [] if name in panel_map]
    names.extend(name for name in panel_map if name not in names)
    panels = [_panel_geometry(name, panel_map[name], millimetres_per_unit) for name in names]
    if not panels:
        raise ValueError("pattern contains no panels")
    sheet_width, sheet_height = _pack_panels(panels, gap_mm)

    writer = DxfWriter()
    layers = ["CUT", "PANEL_LABEL"] + (["EDGE_LABEL"] if edge_labels else [])
    _write_header(writer, layers)
    writer.add(0, "SECTION"); writer.add(2, "ENTITIES")
    counts = {"line": 0, "quadratic": 0, "cubic": 0, "circle": 0}
    for panel in panels:
        min_x, min_y, max_x, max_y = panel.bounds
        label_point = (
            (min_x + max_x) / 2.0 + panel.offset[0],
            (min_y + max_y) / 2.0 + panel.offset[1],
        )
        writer.text(label_point, panel.name, 5.0, "PANEL_LABEL")
        for curve in panel.curves:
            counts[curve.kind] += 1
            points = _translated(curve.points, panel.offset)
            if curve.kind == "line":
                writer.line(points[0], points[-1])
            elif curve.kind in {"quadratic", "cubic"}:
                writer.spline(points)
            elif curve.kind == "circle":
                writer.arc(curve, panel.offset)
            if edge_labels and curve.label:
                midpoint = _curve_samples(curve, 3)[1]
                writer.text(
                    (midpoint[0] + panel.offset[0], midpoint[1] + panel.offset[1]),
                    curve.label,
                    2.5,
                    "EDGE_LABEL",
                )
    writer.add(0, "ENDSEC"); writer.add(0, "EOF")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(writer.lines) + "\n", encoding="ascii")
    if preview is not None:
        write_svg_preview(panels, preview, sheet_width, sheet_height)
    report = {
        "source_specification": str(specification),
        "output_dxf": str(output),
        "format": "AutoCAD DXF 2000 (AC1015)",
        "output_units": "millimetres",
        "source_units_per_meter": units_per_meter,
        "millimetres_per_source_unit": millimetres_per_unit,
        "panel_count": len(panels),
        "stitch_count": len(pattern.get("stitches", [])),
        "entity_counts": counts,
        "sheet_bounds_mm": {"width": sheet_width, "height": sheet_height},
        "geometry_policy": {
            "panel_rotation": "unchanged",
            "line": "native LINE",
            "quadratic_and_cubic_bezier": "exact clamped DXF SPLINE",
            "circle": "native ARC",
        },
    }
    if preview is not None:
        report["output_preview"] = str(preview)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preview", type=Path, help="Optional SVG browser preview")
    parser.add_argument("--edge-labels", action="store_true")
    parser.add_argument("--gap-mm", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_specification(
        args.specification.resolve(),
        args.output.resolve(),
        preview=args.preview.resolve() if args.preview else None,
        edge_labels=args.edge_labels,
        gap_mm=args.gap_mm,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
