#!/usr/bin/env python3
"""Coarse GB/T 1335 adapter backed by six fixed GarmentCode bodies.

This is intentionally a rule based scheme-B adapter.  It does not attempt to
recover a unique human shape from four measurements.  The user's principal
measurements always win; the selected base body supplies the remaining fields.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


BODY_FIELDS = (
    "arm_length", "arm_pose_angle", "armscye_depth", "back_width",
    "bum_points", "bust", "bust_line", "bust_points", "crotch_hip_diff",
    "head_l", "height", "hip_back_width", "hip_inclination", "hips",
    "hips_line", "leg_circ", "neck_w", "shoulder_incl", "shoulder_w",
    "underbust", "vert_bust_line", "waist", "waist_back_width",
    "waist_line", "waist_over_bust_line", "wrist",
)

BODY_TYPE_RANGES = {
    "male": {"Y": (17, 22), "A": (12, 16), "B": (7, 11), "C": (2, 6)},
    "female": {"Y": (19, 24), "A": (14, 18), "B": (9, 13), "C": (4, 8)},
}

STANDARDS = {
    "male": "GB/T 1335.1—2008",
    "female": "GB/T 1335.2—2008",
}

# Clamp is deliberately limited to rule-driving deltas.  Direct user fields
# remain untouched.  This keeps "user input wins" compatible with a safe mode.
LINKED_DELTA_LIMITS = {"height": 10.0, "bust": 8.0, "waist": 8.0}

COMMON_COEFFICIENTS = {
    "male": {
        "cervical_height": 0.8,
        "waist_height": 0.6,
        "arm_length": 0.3,
        "shoulder_w": 0.3,
        "neck_girth_reference": 0.25,
        "sitting_cervical_height_reference": 0.4,
    },
    "female": {
        "cervical_height": 0.8,
        "waist_height": 0.6,
        "arm_length": 0.3,
        "shoulder_w": 0.25,
        "neck_girth_reference": 0.2,
        "sitting_cervical_height_reference": 0.4,
    },
}

BASES: dict[tuple[str, str], dict[str, Any]] = {
    ("female", "Y"): {
        "source_id": "03066.yaml", "anchor": (160.0, 84.0, 64.0, 90.0),
        "hip_coefficient": 0.9,
        "body": [51.0078,49.4802,10.9948,40.7631,14.9579,86.0759,23.1435,13.6978,8.2703,24.87,160.319,48.2105,12.7271,91.0295,19.1656,54.6013,16.2259,19.5945,33.8048,70.2702,19.2364,65.2553,31.3579,34.2789,37.0016,14.5693],
    },
    ("female", "A"): {
        "source_id": "00319.yaml", "anchor": (160.0, 84.0, 68.0, 90.0),
        "hip_coefficient": 0.9,
        "body": [50.7459,47.1321,11.0516,41.5014,16.3882,87.5291,23.6158,14.3752,7.25577,25.2727,160.931,49.4999,12.0366,91.9101,21.5667,52.8688,16.6709,22.3355,33.6333,71.7427,19.7094,69.0244,32.5946,35.0605,38.0216,14.8069],
    },
    ("female", "B"): {
        "source_id": "00402.yaml", "anchor": (160.0, 88.0, 78.0, 96.0),
        "hip_coefficient": 0.8,
        "body": [50.0952,38.8001,12.3665,44.5709,17.2366,94.341,24.3032,15.7438,7.44789,24.547,160.503,52.9711,10.4831,97.7216,22.0535,55.5874,17.3135,16.2832,34.3248,79.0359,19.5024,78.0654,36.6982,34.4049,38.3275,15.077],
    },
    ("male", "Y"): {
        "source_id": "02589.yaml", "anchor": (170.0, 88.0, 70.0, 90.0),
        "hip_coefficient": 0.8,
        "body": [56.4933,43.7596,11.2108,43.3558,15.1996,90.0089,24.2429,14.8736,8.66516,26.9327,172.205,48.6517,9.11877,90.5393,20.9841,52.808,17.8513,22.9631,36.8156,76.8006,20.1056,69.6982,32.8194,36.268,39.5221,15.5796],
    },
    ("male", "A"): {
        "source_id": "03181.yaml", "anchor": (170.0, 88.0, 74.0, 90.0),
        "hip_coefficient": 0.8,
        "body": [55.032,56.826,11.573,43.7647,16.5188,91.7499,24.479,15.1987,8.57673,27.0477,171.142,49.4902,9.03193,92.1342,22.3296,52.7245,17.6328,20.7765,35.9926,79.8563,19.0709,73.9004,34.0347,35.4888,40.2121,15.4832],
    },
    ("male", "B"): {
        "source_id": "04673.yaml", "anchor": (170.0, 92.0, 84.0, 95.0),
        "hip_coefficient": 0.7,
        "body": [54.3112,39.5326,12.469,47.7603,16.0747,98.5547,24.9555,17.1491,9.32193,25.9551,168.995,51.8437,6.52537,96.9912,22.5092,56.1427,19.3801,22.1504,36.9721,86.6789,20.6711,84.287,38.6903,35.8293,39.1003,16.4978],
    },
}


class GBT1335Error(ValueError):
    """Invalid scheme-B request."""


def _number(value: Any, name: str, *, optional: bool = False) -> float | None:
    if optional and value in (None, ""):
        return None
    if isinstance(value, bool):
        raise GBT1335Error(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GBT1335Error(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise GBT1335Error(f"{name} must be a positive finite number")
    return result


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def validate_gbt_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GBT1335Error("request body must be a JSON object")
    gender = str(payload.get("gender", payload.get("sex", ""))).strip().lower()
    if gender not in {"male", "female"}:
        raise GBT1335Error("gender must be male or female")
    policy = str(
        payload.get(
            "out_of_range_mode",
            payload.get("boundary_policy", "extrapolate"),
        )
    ).strip().lower()
    if policy not in {"extrapolate", "clamp"}:
        raise GBT1335Error(
            "out_of_range_mode/boundary_policy must be extrapolate or clamp"
        )

    # chest_cm is the public/UI name; bust is the GarmentCode field name.
    chest_value = payload.get("chest_cm", payload.get("bust_cm"))
    request = {
        "gender": gender,
        "height_cm": _number(payload.get("height_cm"), "height_cm"),
        "chest_cm": _number(chest_value, "chest_cm"),
        "waist_cm": _number(payload.get("waist_cm"), "waist_cm"),
        "hips_cm": _number(payload.get("hips_cm"), "hips_cm", optional=True),
        "boundary_policy": policy,
        "out_of_range_mode": policy,
        "name": str(payload.get("name", "")).strip()[:80],
    }
    return request


def _infer_type(gender: str, rounded_drop: int, policy: str) -> tuple[str, int, bool]:
    ranges = BODY_TYPE_RANGES[gender]
    minimum = min(low for low, _ in ranges.values())
    maximum = max(high for _, high in ranges.values())
    outside = rounded_drop < minimum or rounded_drop > maximum
    effective_drop = rounded_drop
    if outside and policy == "clamp":
        effective_drop = min(max(rounded_drop, minimum), maximum)

    for body_type, (low, high) in ranges.items():
        if low <= effective_drop <= high:
            return body_type, effective_drop, outside

    # Extrapolation continues the closest outer category beyond the table.
    return ("C" if effective_drop < minimum else "Y"), effective_drop, outside


def _linked_delta(delta: float, field: str, policy: str) -> float:
    if policy == "extrapolate":
        return delta
    limit = LINKED_DELTA_LIMITS[field]
    return min(max(delta, -limit), limit)


def generate_garmentcode_body(payload: Any) -> tuple[dict[str, float], dict[str, Any]]:
    request = validate_gbt_request(payload)
    gender = request["gender"]
    height = request["height_cm"]
    bust = request["chest_cm"]
    waist = request["waist_cm"]
    hips = request["hips_cm"]
    policy = request["boundary_policy"]

    raw_drop = bust - waist
    rounded_drop = round_half_up(raw_drop)
    inferred, effective_drop, classification_outside = _infer_type(
        gender, rounded_drop, policy
    )
    mapped = "B" if inferred == "C" else inferred
    base = BASES[(gender, mapped)]
    anchor_height, anchor_bust, anchor_waist, anchor_hips = base["anchor"]
    base_body = dict(zip(BODY_FIELDS, base["body"]))

    raw_deltas = {
        "height": height - anchor_height,
        "bust": bust - anchor_bust,
        "waist": waist - anchor_waist,
    }
    applied_deltas = {
        key: _linked_delta(value, key, policy)
        for key, value in raw_deltas.items()
    }
    coeff = COMMON_COEFFICIENTS[gender]

    base_cervical_height = base_body["height"] - base_body["head_l"]
    base_waist_height = (
        base_body["height"] - base_body["head_l"] - base_body["waist_line"]
    )
    target_cervical_height = (
        base_cervical_height + coeff["cervical_height"] * applied_deltas["height"]
    )
    target_waist_height = (
        base_waist_height + coeff["waist_height"] * applied_deltas["height"]
    )

    body = dict(base_body)
    body.update({"height": height, "bust": bust, "waist": waist})
    if hips is None:
        body["hips"] = anchor_hips + base["hip_coefficient"] * applied_deltas["waist"]
        hips_source = "estimated_from_waist"
    else:
        body["hips"] = hips
        hips_source = "user_input"
    body["arm_length"] = base_body["arm_length"] + coeff["arm_length"] * applied_deltas["height"]
    body["head_l"] = height - target_cervical_height
    body["waist_line"] = target_cervical_height - target_waist_height
    body["shoulder_w"] = base_body["shoulder_w"] + coeff["shoulder_w"] * applied_deltas["bust"]
    body = {key: round(float(body[key]), 4) for key in BODY_FIELDS}

    for key, value in body.items():
        if not math.isfinite(value) or value <= 0:
            raise GBT1335Error(
                f"generated body.{key} is non-positive; measurements are too far from the fixed base"
            )

    field_sources = {key: "base_inherited" for key in BODY_FIELDS}
    field_sources.update({
        "height": "user_input", "bust": "user_input", "waist": "user_input",
        "hips": hips_source,
        "arm_length": "derived", "head_l": "derived", "waist_line": "derived",
        "shoulder_w": "derived",
    })
    source_counts = dict(Counter(field_sources.values()))

    warnings: list[str] = []
    if inferred == "C":
        warnings.append("C体型当前没有独立基础人体，已按约定映射到同一性别的B体型基础人体。")
    if classification_outside:
        if policy == "clamp":
            warnings.append(
                f"胸腰差{rounded_drop} cm超出国标体型范围；分类用值已夹到{effective_drop} cm，用户胸围和腰围未改写。"
            )
        else:
            warnings.append(
                f"胸腰差{rounded_drop} cm超出国标体型范围；已按最近外侧体型继续外推。"
            )
    for field, raw_delta in raw_deltas.items():
        limit = LINKED_DELTA_LIMITS[field]
        if abs(raw_delta) > limit:
            if policy == "clamp":
                warnings.append(
                    f"{field}相对锚点变化{raw_delta:+.2f} cm；联动计算已夹到{applied_deltas[field]:+.2f} cm，直接输入字段保持原值。"
                )
            else:
                warnings.append(
                    f"{field}相对锚点变化{raw_delta:+.2f} cm，超过粗粒度建议范围±{limit:g} cm，当前按外推计算。"
                )

    audit = {
        "schema_version": 2,
        "method": "gbt1335_fixed_base",
        "coarse_estimate": True,
        "gender": gender,
        "standard": STANDARDS[gender],
        "body_type": {
            "inferred": inferred,
            "mapped": mapped,
            "mapping_applied": inferred != mapped,
            "bust_waist_drop_raw_cm": round(raw_drop, 4),
            "bust_waist_drop_rounded_cm": rounded_drop,
            "classification_drop_cm": effective_drop,
        },
        "base_source_id": base["source_id"],
        "base_anchor": {
            "height_cm": anchor_height, "chest_cm": anchor_bust,
            "waist_cm": anchor_waist, "hips_cm": anchor_hips,
        },
        "boundary_policy": policy,
        "out_of_range_mode": policy,
        "boundary_policy_semantics": (
            "Direct user measurements are never clamped. Classification drop is clamped to the "
            "GB/T outer boundary and only linked/estimated-field deltas are clamped to ±10 cm "
            "for height and ±8 cm for chest/waist."
            if policy == "clamp" else
            "Direct user measurements and all linked/estimated-field deltas are extrapolated; "
            "an out-of-range chest-waist drop uses the nearest outer body type."
        ),
        "target": {
            "height_cm": height, "chest_cm": bust, "waist_cm": waist,
            "hips_cm": body["hips"], "hips_source": hips_source,
        },
        "deltas_from_standard_anchor": {
            key: {"raw_cm": round(raw_deltas[key], 4), "applied_to_linked_fields_cm": round(applied_deltas[key], 4)}
            for key in raw_deltas
        },
        "body": body,
        "field_sources": field_sources,
        "source_counts": source_counts,
        "warnings": warnings,
        "request": request,
    }
    return body, audit


def dump_body_yaml(body: dict[str, float]) -> str:
    """Serialize the simple GarmentCode body schema without a YAML dependency."""
    lines = ["body:"]
    for key in BODY_FIELDS:
        lines.append(f"  {key}: {body[key]:g}")
    return "\n".join(lines) + "\n"


def write_garmentcode_artifacts(
    payload: Any,
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, Any], dict[str, str]]:
    body, audit = generate_garmentcode_body(payload)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {"body_yaml": "garmentcode_body.yaml", "audit_json": "audit.json"}
    (output_dir / files["body_yaml"]).write_text(dump_body_yaml(body), encoding="utf-8")
    (output_dir / files["audit_json"]).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return body, audit, files


def schema_payload() -> dict[str, Any]:
    return {
        "status": "ready",
        "method": "gbt1335_fixed_base",
        "coarse_estimate": True,
        "genders": ["female", "male"],
        "measurements": {
            "required": ["height_cm", "chest_cm", "waist_cm"],
            "optional": ["hips_cm"],
            "unit": "cm",
            "user_values_take_precedence": True,
        },
        "body_type_ranges": BODY_TYPE_RANGES,
        "body_type_mapping": {"Y": "Y", "A": "A", "B": "B", "C": "B"},
        "base_source_ids": {
            gender: {body_type: BASES[(gender, body_type)]["source_id"] for body_type in ("Y", "A", "B")}
            for gender in ("female", "male")
        },
        "boundary_policies": {
            "default": "extrapolate",
            "extrapolate": "Keep direct inputs and extrapolate linked/estimated fields; use the nearest outer type when chest-waist drop is outside the table.",
            "clamp": "Keep direct inputs unchanged; clamp classification drop to the table boundary and linked deltas to height ±10 cm, chest/waist ±8 cm.",
        },
        "out_of_range_modes": ["extrapolate", "clamp"],
        "output": {"body_fields": list(BODY_FIELDS), "body_field_count": len(BODY_FIELDS)},
    }
