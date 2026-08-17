#!/usr/bin/env python3
"""Lightweight inference adapter for SHAPY's official polynomial A2S checkpoints.

This module deliberately avoids importing the 2022 training environment.  It
reproduces the polynomial forward pass stored in ``last.ckpt`` with the
project's existing PyTorch runtime, which is compatible with RTX 4090.
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
import torch


ATTRIBUTE_KEYS = {
    "female": (
        "big",
        "broad_shoulders",
        "feminine",
        "large_breasts",
        "long_legs",
        "long_neck",
        "long_torso",
        "muscular",
        "pear_shaped",
        "petite",
        "short",
        "short_arms",
        "skinny_legs",
        "slim_waist",
        "tall",
    ),
    "male": (
        "average",
        "big",
        "broad_shoulders",
        "delicate_build",
        "long_legs",
        "long_neck",
        "long_torso",
        "masculine",
        "muscular",
        "rectangular",
        "short",
        "short_arms",
        "skinny_arms",
        "soft_body",
        "tall",
    ),
}

VARIANTS = {
    "04b_ahcwh2s": ("height_gt", "chest", "waist", "hips"),
    "05b_ahwcwh2s": ("height_gt", "weight_gt", "chest", "waist", "hips"),
}


class ShapyCheckpointMissing(RuntimeError):
    def __init__(self, checkpoint: Path):
        super().__init__(f"SHAPY official checkpoint is missing: {checkpoint}")
        self.checkpoint = checkpoint


def _install_lightning_checkpoint_stub() -> None:
    """Provide the sole legacy Lightning type referenced by SHAPY checkpoints.

    The callback object is training metadata and is not used for inference.
    Avoiding a full pytorch-lightning 1.3 install keeps the official 2022
    checkpoint usable in the project's PyTorch 2.1 / CUDA 11.8 runtime.
    """

    try:
        __import__("pytorch_lightning.callbacks.model_checkpoint")
        return
    except ModuleNotFoundError:
        pass

    root = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    model_checkpoint = types.ModuleType(
        "pytorch_lightning.callbacks.model_checkpoint"
    )

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = model_checkpoint.__name__
    model_checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = model_checkpoint
    root.callbacks = callbacks
    sys.modules[root.__name__] = root
    sys.modules[callbacks.__name__] = callbacks
    sys.modules[model_checkpoint.__name__] = model_checkpoint


def checkpoint_path(
    data_root: Path,
    *,
    model_gender: str,
    semantic_profile: str,
    variant: str,
) -> Path:
    base = (
        data_root
        / "trained_models"
        / "a2b"
        / f"caesar-{semantic_profile}_smplx-{model_gender}-10betas"
    )
    canonical = base / "poynomial" / f"{variant}.yaml" / "last.ckpt"
    corrected = base / "polynomial" / f"{variant}.yaml" / "last.ckpt"
    return canonical if canonical.exists() or not corrected.exists() else corrected


def route_status(data_root: Path) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for model_gender in ("female", "male", "neutral"):
        profiles = (
            ("female", "male") if model_gender == "neutral" else (model_gender,)
        )
        for semantic_profile in profiles:
            for variant in VARIANTS:
                path = checkpoint_path(
                    data_root,
                    model_gender=model_gender,
                    semantic_profile=semantic_profile,
                    variant=variant,
                )
                routes.append(
                    {
                        "model_gender": model_gender,
                        "semantic_profile": semantic_profile,
                        "variant": variant,
                        "checkpoint": str(path),
                        "available": path.is_file(),
                    }
                )
    return routes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_feature_dim(polynomial_width: int) -> int:
    # SHAPY degree-2, no-bias polynomial width: d + d*(d+1)/2.
    dim = int((-3 + math.sqrt(9 + 8 * polynomial_width)) / 2)
    if dim + dim * (dim + 1) // 2 != polynomial_width:
        raise ValueError(
            f"Unsupported SHAPY polynomial width: {polynomial_width}"
        )
    return dim


def _extract_linear_state(checkpoint: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Unsupported SHAPY checkpoint: state dictionary not found")

    candidates: list[tuple[str, torch.Tensor]] = []
    for key, value in state.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.shape[0] == 10
            and key.endswith("linear.weight")
        ):
            candidates.append((key, value))
    candidates.sort(key=lambda item: ("a2b.linear.weight" not in item[0], item[0]))
    if not candidates:
        raise ValueError("SHAPY polynomial linear weight was not found")

    weight_key, weight = candidates[0]
    bias_key = weight_key[: -len("weight")] + "bias"
    bias = state.get(bias_key)
    if not isinstance(bias, torch.Tensor) or bias.shape != (10,):
        raise ValueError(f"SHAPY polynomial bias was not found at {bias_key}")
    return weight.float().cpu(), bias.float().cpu()


def _official_feature_vector(
    request: dict[str, Any],
    *,
    semantic_profile: str,
    variant: str,
) -> tuple[torch.Tensor, list[str]]:
    attribute_keys = ATTRIBUTE_KEYS[semantic_profile]
    attributes = request.get("attributes", {})
    feature_values = [float(attributes.get(key, 3.0)) for key in attribute_keys]
    for key, value in zip(attribute_keys, feature_values):
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"attribute {key!r} must be between 1 and 5")

    measurements = {
        "height_gt": float(request["height_cm"]) / 100.0,
        "chest": float(request["chest_cm"]) / 100.0,
        "waist": float(request["waist_cm"]) / 100.0,
        "hips": float(request["hips_cm"]) / 100.0,
    }
    if variant == "05b_ahwcwh2s":
        measurements["weight_gt"] = float(request["weight_kg"])

    measurement_keys = VARIANTS[variant]
    measurement_values = [measurements[key] for key in measurement_keys]

    # Reproduce SHAPY's create_input_feature_vec + to_whw2s preprocessing.
    # This looks unusual (height is scaled in both functions), but matching the
    # training/inference code is necessary for faithful checkpoint inference.
    processed = dict(zip(measurement_keys, measurement_values))
    processed["height_gt"] *= 100.0
    if "weight_gt" in processed:
        processed["weight_gt"] = processed["weight_gt"] ** (1.0 / 3.0)
    processed["height_gt"] *= 100.0
    if "weight_gt" in processed:
        processed["weight_gt"] = math.sqrt(processed["weight_gt"])

    names = [*attribute_keys, *measurement_keys]
    values = [*feature_values, *(processed[key] for key in measurement_keys)]
    return torch.tensor([values], dtype=torch.float32), names


def _polynomial_features(features: torch.Tensor) -> torch.Tensor:
    columns = [features]
    pairs = list(combinations_with_replacement(range(features.shape[1]), 2))
    pair_index = torch.tensor(pairs, dtype=torch.long)
    columns.append(torch.prod(features[:, pair_index], dim=-1))
    return torch.cat(columns, dim=-1)


def predict_betas(
    request: dict[str, Any],
    *,
    data_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_gender = str(request["gender"]).lower()
    semantic_profile = str(
        request.get(
            "semantic_profile",
            model_gender if model_gender != "neutral" else "female",
        )
    ).lower()
    if model_gender not in {"female", "male", "neutral"}:
        raise ValueError("gender must be female, male, or neutral")
    if semantic_profile not in ATTRIBUTE_KEYS:
        raise ValueError("semantic_profile must be female or male")
    if model_gender != "neutral" and semantic_profile != model_gender:
        raise ValueError("gender-specific models require the matching semantic profile")

    variant = (
        "05b_ahwcwh2s"
        if request.get("weight_kg") not in (None, "")
        else "04b_ahcwh2s"
    )
    path = checkpoint_path(
        data_root,
        model_gender=model_gender,
        semantic_profile=semantic_profile,
        variant=variant,
    )
    if not path.is_file():
        raise ShapyCheckpointMissing(path)

    _install_lightning_checkpoint_stub()
    checkpoint = torch.load(path, map_location="cpu")
    weight, bias = _extract_linear_state(checkpoint)
    feature_vector, feature_names = _official_feature_vector(
        request,
        semantic_profile=semantic_profile,
        variant=variant,
    )
    feature_dim = _infer_feature_dim(weight.shape[1])
    if feature_vector.shape[1] != feature_dim:
        raise ValueError(
            "SHAPY checkpoint/input mismatch: "
            f"checkpoint expects {feature_dim}, request produced "
            f"{feature_vector.shape[1]}"
        )

    polynomial = _polynomial_features(feature_vector)
    betas = torch.nn.functional.linear(polynomial, weight, bias)
    if betas.shape != (1, 10) or not torch.isfinite(betas).all():
        raise ValueError("SHAPY produced invalid body shape parameters")

    metadata = {
        "method": "shapy_official_polynomial_a2s",
        "variant": variant,
        "model_gender": model_gender,
        "semantic_profile": semantic_profile,
        "feature_names": feature_names,
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "output_betas": 10,
        "trained_here": False,
    }
    return betas.numpy().astype(np.float32)[0], metadata
