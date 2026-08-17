#!/usr/bin/env python3
"""Fit selected CMU ASF/AMC joint trajectories to SMPL-X pose parameters.

The conversion is deterministic optimization, not model training.  Raw CMU
files remain the source of truth.  The generated files are internal runtime
assets and must not be included in user download bundles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fairmotion.data import asfamc


CMU_TO_SMPLX = [
    ("root", 0),
    ("lfemur", 1),
    ("rfemur", 2),
    ("lowerback", 3),
    ("ltibia", 4),
    ("rtibia", 5),
    ("upperback", 6),
    ("lfoot", 7),
    ("rfoot", 8),
    ("thorax", 9),
    ("ltoes", 10),
    ("rtoes", 11),
    ("lowerneck", 12),
    ("lclavicle", 13),
    ("rclavicle", 14),
    ("head", 15),
    ("lhumerus", 16),
    ("rhumerus", 17),
    ("lradius", 18),
    ("rradius", 19),
    ("lwrist", 20),
    ("rwrist", 21),
]

SMPLX_EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13),
    (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20), (19, 21),
]

# FAQ conversion for CMU root positions and bone lengths:
# (1 / 0.45) inches, converted to metres.
CMU_UNIT_TO_METER = (1.0 / 0.45) * 2.54 / 100.0

ACTION_CONFIG: dict[str, dict[str, Any]] = {
    "standing_turn": {
        "asf": "69.asf",
        "amc": "69_16.amc",
        "seconds": 5.0,
        "selector": "turn",
        "output": "cmu_69_16_standing_turn_smplx.npz",
    },
    "wave": {
        "asf": "141.asf",
        "amc": "141_16.amc",
        "seconds": 5.0,
        "selector": "wave",
        "output": "cmu_141_16_wave_smplx.npz",
    },
    "walk_in_place": {
        "asf": "69.asf",
        "amc": "69_01.amc",
        "seconds": 5.0,
        "selector": "walk",
        "output": "cmu_69_01_walk_in_place_smplx.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=sorted(ACTION_CONFIG), required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--official-motion", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--max-action-frames", type=int)
    return parser.parse_args()


def load_cmu_positions(
    asf_path: Path,
    amc_path: Path,
) -> tuple[np.ndarray, list[str]]:
    motion = asfamc.load(str(asf_path), motion=str(amc_path))
    names = [joint.name for joint in motion.skel.joints]
    missing = [name for name, _ in CMU_TO_SMPLX if name not in names]
    if missing:
        raise KeyError(f"CMU skeleton is missing mapped joints: {missing}")
    indices = [names.index(name) for name, _ in CMU_TO_SMPLX]
    positions = np.asarray(motion.positions(local=False), dtype=np.float32)
    return positions[:, indices] * CMU_UNIT_TO_METER, names


def unwrap_angle(values: np.ndarray) -> np.ndarray:
    return np.unwrap(values.astype(np.float64)).astype(np.float32)


def shoulder_heading(positions: np.ndarray) -> np.ndarray:
    shoulder_axis = positions[:, 17] - positions[:, 16]
    return unwrap_angle(np.arctan2(shoulder_axis[:, 2], shoulder_axis[:, 0]))


def select_window(
    positions: np.ndarray,
    *,
    selector: str,
    frames: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    frames = min(frames, len(positions))
    if frames == len(positions):
        return positions, (0, len(positions))
    best_start = 0
    best_score = -float("inf")
    if selector == "wave":
        wrist_height = np.maximum(
            positions[:, 20, 1] - positions[:, 16, 1],
            positions[:, 21, 1] - positions[:, 17, 1],
        )
        center = int(np.argmax(wrist_height))
        best_start = max(0, min(len(positions) - frames, center - frames // 2))
    elif selector == "turn":
        heading = shoulder_heading(positions)
        for start in range(0, len(positions) - frames + 1):
            window = heading[start : start + frames]
            score = float(window.max() - window.min())
            if score > best_score:
                best_start, best_score = start, score
    else:
        # Avoid capture lead-in and prefer the window with the most ankle motion.
        for start in range(0, len(positions) - frames + 1):
            window = positions[start : start + frames]
            ankle = window[:, [7, 8], :]
            score = float(np.var(ankle[:, :, 0]) + np.var(ankle[:, :, 2]))
            if score > best_score:
                best_start, best_score = start, score
    end = best_start + frames
    return positions[best_start:end], (best_start, end)


def source_scale(source: np.ndarray, target_rest: np.ndarray) -> float:
    ratios = []
    for parent, child in SMPLX_EDGES:
        source_length = np.linalg.norm(source[:, child] - source[:, parent], axis=1)
        target_length = np.linalg.norm(target_rest[child] - target_rest[parent])
        median_source = float(np.median(source_length))
        if median_source > 1e-5 and target_length > 1e-5:
            ratios.append(float(target_length / median_source))
    return float(np.median(ratios))


def fit_pose_chunks(
    layer: Any,
    target: np.ndarray,
    *,
    device: str,
    iterations: int,
    chunk_size: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    fitted = []
    reports = []
    previous: torch.Tensor | None = None
    joint_weights = torch.ones(22, dtype=torch.float32, device=device)
    joint_weights[[0, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]] = 2.0
    for start in range(0, len(target), chunk_size):
        end = min(len(target), start + chunk_size)
        count = end - start
        initial = torch.zeros(count, 55, 3, dtype=torch.float32, device=device)
        if previous is not None:
            initial[:] = previous
        pose = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam([pose], lr=0.035)
        betas = torch.zeros(count, 300, dtype=torch.float32, device=device)
        transl = torch.zeros(count, 3, dtype=torch.float32, device=device)
        chunk_target = target_tensor[start:end]
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            output = layer.forward_simple(
                betas=betas,
                full_pose=pose,
                transl=transl,
                pose2rot=True,
            )
            predicted = output.joints[:, :22]
            predicted = predicted - predicted[:, :1]
            difference = (predicted - chunk_target) * joint_weights[None, :, None]
            position_loss = difference.square().mean()
            temporal = (
                (pose[1:] - pose[:-1]).square().mean()
                if count > 1
                else torch.zeros((), device=device)
            )
            boundary = (
                (pose[0] - previous).square().mean()
                if previous is not None
                else torch.zeros((), device=device)
            )
            regularizer = pose[:, 1:22].square().mean()
            loss = position_loss + 0.002 * temporal + 0.004 * boundary + 0.00005 * regularizer
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            output = layer.forward_simple(
                betas=betas,
                full_pose=pose,
                transl=transl,
                pose2rot=True,
            )
            predicted = output.joints[:, :22]
            predicted = predicted - predicted[:, :1]
            error_cm = torch.linalg.norm(predicted - chunk_target, dim=-1).mean() * 100
        fitted.append(pose.detach().cpu().numpy())
        previous = pose[-1].detach().clone()
        reports.append(
            {
                "start": float(start),
                "end": float(end),
                "mean_joint_error_cm": float(error_cm.cpu()),
            }
        )
        print(
            json.dumps(reports[-1]),
            flush=True,
        )
    return np.concatenate(fitted, axis=0), reports


def prepend_transition(
    poses: np.ndarray,
    official_first_pose: np.ndarray,
    frames: int = 30,
) -> np.ndarray:
    official = official_first_pose.reshape(55, 3).astype(np.float32)
    alpha = np.linspace(0.0, 1.0, frames, endpoint=False, dtype=np.float32)
    transition = official[None] * (1 - alpha[:, None, None]) + poses[0:1] * alpha[:, None, None]
    return np.concatenate([transition, poses], axis=0)


def main() -> None:
    args = parse_args()
    config = ACTION_CONFIG[args.action]
    source_120, joint_names = load_cmu_positions(
        args.raw_dir / config["asf"],
        args.raw_dir / config["amc"],
    )
    source_30 = source_120[::4]
    window_frames = int(round(float(config["seconds"]) * 30))
    selected, selected_range_30 = select_window(
        source_30,
        selector=config["selector"],
        frames=window_frames,
    )
    if args.max_action_frames:
        selected = selected[: args.max_action_frames]

    contourcraft_root = args.project_root / "dynamic3d" / "src" / "ContourCraft-CG"
    sys.path.insert(0, str(contourcraft_root))
    from runners.smplx.body_models import SMPLXLayer

    layer = SMPLXLayer(str(args.smplx_model), ext="pkl", num_betas=300).to(args.device)
    with torch.inference_mode():
        rest = layer.forward_simple(
            betas=torch.zeros(1, 300, device=args.device),
            full_pose=torch.zeros(1, 55, 3, device=args.device),
            transl=torch.zeros(1, 3, device=args.device),
            pose2rot=True,
        )
    target_rest = rest.joints[0, :22].detach().cpu().numpy()
    scale = source_scale(selected, target_rest)
    scaled = selected * scale
    root_motion = scaled[:, 0].copy()
    target = scaled - scaled[:, :1]

    poses, chunk_reports = fit_pose_chunks(
        layer,
        target,
        device=args.device,
        iterations=args.iterations,
        chunk_size=args.chunk_size,
    )
    official = np.load(args.official_motion, allow_pickle=True)
    poses = prepend_transition(poses, np.asarray(official["poses"])[0])

    # All three presentation actions are intentionally in-place.  Preserve only
    # the vertical root trajectory after conversion and ground the first frame.
    vertical = root_motion[:, 1] - root_motion[0, 1]
    vertical = np.concatenate([np.zeros(30, dtype=np.float32), vertical.astype(np.float32)])
    with torch.inference_mode():
        preview = layer.forward_simple(
            betas=torch.zeros(len(poses), 300, device=args.device),
            full_pose=torch.as_tensor(poses, dtype=torch.float32, device=args.device),
            transl=torch.zeros(len(poses), 3, device=args.device),
            pose2rot=True,
        )
    first_floor = float(preview.vertices[0, :, 1].min().detach().cpu())
    trans = np.zeros((len(poses), 3), dtype=np.float32)
    trans[:, 1] = -first_floor + vertical

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / config["output"]
    np.savez_compressed(
        output_path,
        poses=poses.reshape(len(poses), 165).astype(np.float32),
        trans=trans,
        betas=np.zeros(300, dtype=np.float32),
        gender=np.asarray("neutral"),
        fps=np.asarray(30, dtype=np.int64),
        action_id=np.asarray(args.action),
        source=np.asarray(f"CMU {config['amc']}"),
    )
    report = {
        "status": "converted",
        "action": args.action,
        "source_asf": config["asf"],
        "source_amc": config["amc"],
        "source_frames_120fps": int(len(source_120)),
        "selected_range_30fps": list(selected_range_30),
        "action_frames_30fps": int(len(selected)),
        "transition_frames": 30,
        "output_frames": int(len(poses)),
        "scale_to_target_smplx": scale,
        "cmu_joint_names": joint_names,
        "chunk_reports": chunk_reports,
        "mean_joint_error_cm": float(
            np.mean([item["mean_joint_error_cm"] for item in chunk_reports])
        ),
        "output": str(output_path),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
