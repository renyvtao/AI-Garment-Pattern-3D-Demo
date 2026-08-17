#!/usr/bin/env python3
"""Convert a suit-mode ChatGarment response into canonical GarmentCodeRC YAML."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import yaml


GARMENT_TYPE = "MensSuitJacketCleanFinal"
DEFAULT_TEMPLATE = "assets/design_params/mens-suit-jacket-clean-final.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-output", type=Path)
    source.add_argument("--model-output-text")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def extract_mapping(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```python", "").replace("```json", "").replace("```", "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a dictionary")
    candidate = cleaned[start : end + 1]
    errors: list[str] = []
    for parser in (ast.literal_eval, json.loads):
        try:
            value = parser(candidate)
            if not isinstance(value, dict):
                raise TypeError("parsed value is not a dictionary")
            return value
        except Exception as exc:  # keep both parse errors for diagnosis
            errors.append(f"{type(exc).__name__}: {exc}")
    raise ValueError("unable to parse model output: " + " | ".join(errors))


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return min(high, max(low, numeric))


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def normalize(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    raw = parsed.get("suit", parsed)
    if not isinstance(raw, dict):
        raise ValueError("parsed model output does not contain a suit dictionary")
    corrections: list[str] = []

    button_raw = raw.get("button_count", 2)
    try:
        button_count = min(2, max(1, int(round(float(button_raw)))))
    except (TypeError, ValueError):
        button_count = 2
    if button_raw != button_count:
        corrections.append(f"button_count: {button_raw!r} -> {button_count}")

    lapel = str(raw.get("lapel_style", "notched")).strip().lower()
    if lapel not in {"notched", "peak"}:
        corrections.append(f"lapel_style: {lapel!r} -> 'notched'")
        lapel = "notched"

    normalized = {
        "garment_length_ratio": clamp(
            raw.get("garment_length_ratio"), 1.7, 1.8, 1.75
        ),
        "overlap_M_cm": 2.0,
        "break_point_height_cm": 0.0,
        "button_spacing_cm": 9.0,
        "button_count": button_count,
        "front_lower_edge_style": "curved",
        "body_panel_layout": "six_panel",
        "waist_ease_cm": clamp(raw.get("waist_ease_cm"), 11.0, 19.0, 15.0),
        "back_collar_width_BCW_cm": 2.5,
        "point_width_delta_cm": 0.0,
        "collar_point_angle_s3_s4_s5_deg": 90.0,
        "sleeve_length_reduction_X_cm": 0.5,
        "lapel_style": lapel,
        "small_pocket_enabled": as_bool(
            raw.get("small_pocket_enabled"), True
        ),
        "large_pockets_enabled": as_bool(
            raw.get("large_pockets_enabled"), True
        ),
    }
    return normalized, corrections


def apply_to_template(template: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    design = template.get("design")
    if not isinstance(design, dict):
        raise ValueError("template does not contain a design mapping")
    design.setdefault("meta", {}).setdefault("upper", {})["v"] = GARMENT_TYPE
    suit = design.get("suit")
    if not isinstance(suit, dict):
        raise ValueError("template does not contain design.suit")
    missing = sorted(set(values) - set(suit))
    if missing:
        raise ValueError(f"template is missing suit fields: {missing}")
    for field, value in values.items():
        entry = suit[field]
        if not isinstance(entry, dict) or "v" not in entry:
            raise ValueError(f"template field design.suit.{field} has no v entry")
        entry["v"] = value
    return template


def main() -> None:
    args = parse_args()
    text = (
        args.model_output.read_text(encoding="utf-8-sig")
        if args.model_output
        else args.model_output_text
    )
    parsed = extract_mapping(text)
    normalized, corrections = normalize(parsed)
    template = yaml.safe_load(args.template.read_text(encoding="utf-8-sig"))
    output_document = apply_to_template(template, normalized)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(output_document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report_path = args.report or args.output.with_suffix(".adapter.json")
    report_path.write_text(
        json.dumps(
            {
                "garment_type": GARMENT_TYPE,
                "raw_model_output": parsed,
                "applied_suit_parameters": normalized,
                "corrections": corrections,
                "design_yaml": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(normalized, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
