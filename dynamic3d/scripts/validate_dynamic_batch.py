#!/usr/bin/env python3
"""Validate ContourCraft sequence archives and rendered videos in one batch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def probe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,avg_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    container = payload["format"]
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]),
        "frame_rate": stream["avg_frame_rate"],
        "duration_seconds": float(container["duration"]),
        "size_bytes": int(container["size"]),
    }


def validate_case(case_dir: Path) -> dict[str, object]:
    sequence_path = case_dir / "contourcraft_sequence.npz"
    video_path = case_dir / "contourcraft.mp4"
    with np.load(sequence_path, allow_pickle=True) as payload:
        cloth = payload["pred"]
        body = payload["body_vertices"]
        metrics = payload["metrics"].item()

    steps = np.linalg.norm(np.diff(cloth, axis=0), axis=2)
    collisions = [int(value) for value in metrics.get("ncoll", [])]
    simulation_collisions = collisions[1:] if len(collisions) > body.shape[0] else collisions
    finite = bool(np.isfinite(cloth).all() and np.isfinite(body).all())
    frame_layout_ok = cloth.shape[0] == body.shape[0] + 2
    success = metrics.get("fail_reason") == "SUCCESS"

    result = {
        "case_id": case_dir.name,
        "sequence": {
            "cloth_shape": list(cloth.shape),
            "body_shape": list(body.shape),
            "finite": finite,
            "frame_layout_ok": frame_layout_ok,
            "mean_step_m": float(steps.mean()),
            "p99_step_m": float(np.percentile(steps, 99)),
            "max_step_m": float(steps.max()),
        },
        "collision": {
            "initial": collisions[0] if collisions else None,
            "simulation_max": max(simulation_collisions) if simulation_collisions else None,
            "final": simulation_collisions[-1] if simulation_collisions else None,
            "zero_frames": sum(value == 0 for value in simulation_collisions),
            "sample_count": len(simulation_collisions),
        },
        "model_status": metrics.get("fail_reason"),
        "model_time_seconds": float(metrics.get("time", 0.0)),
        "video": probe_video(video_path),
    }
    result["passed"] = bool(
        finite
        and frame_layout_ok
        and success
        and result["video"]["frames"] == body.shape[0]
    )
    return result


def main() -> None:
    args = parse_args()
    cases = []
    for case_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir()):
        sequence = case_dir / "contourcraft_sequence.npz"
        video = case_dir / "contourcraft.mp4"
        if sequence.is_file() and video.is_file():
            cases.append(validate_case(case_dir))

    report = {
        "output_root": str(args.output_root.resolve()),
        "case_count": len(cases),
        "passed_count": sum(case["passed"] for case in cases),
        "all_passed": bool(cases) and all(case["passed"] for case in cases),
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
