#!/usr/bin/env python3
"""Run all prepared garments and optionally render/copy their dynamic videos."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--hood-data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--smplx-model", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--body-mesh-sequence", type=Path)
    parser.add_argument("--rest-body-params", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--gallery-public", type=Path)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.body_mesh_sequence is None and (
        args.smplx_model is None or args.motion is None
    ):
        parser.error(
            "provide --body-mesh-sequence, or provide both --smplx-model and --motion"
        )
    return args


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    available = [case["case_id"] for case in payload["cases"]]
    case_ids = args.case_ids or available
    unknown = sorted(set(case_ids).difference(available))
    if unknown:
        raise SystemExit(f"Unknown case ids: {unknown}")
    if not args.dry_run and args.blender is None:
        raise SystemExit("--blender is required unless --dry-run is set")

    completed = 0
    skipped = 0
    for case_id in case_ids:
        case_output = args.output_root / case_id
        sequence = case_output / "contourcraft_sequence.npz"
        video = case_output / "contourcraft.mp4"
        if args.skip_existing and video.is_file() and video.stat().st_size > 0:
            print(f"[SKIP] {case_id}: {video}")
            skipped += 1
            continue

        inference = [
            sys.executable,
            str(script_root / "run_dynamic_case.py"),
            "--project-root",
            str(args.project_root),
            "--hood-data",
            str(args.hood_data),
            "--manifest",
            str(args.manifest),
            "--case-id",
            case_id,
            "--checkpoint",
            str(args.checkpoint),
            "--rest-body-params",
            str(args.rest_body_params),
            "--output-dir",
            str(case_output),
            "--device",
            args.device,
        ]
        if args.body_mesh_sequence is not None:
            inference.extend(
                ["--body-mesh-sequence", str(args.body_mesh_sequence)]
            )
        else:
            inference.extend(
                [
                    "--smplx-model",
                    str(args.smplx_model),
                    "--motion",
                    str(args.motion),
                ]
            )
        if args.max_frames is not None:
            inference.extend(["--max-frames", str(args.max_frames)])
        if args.dry_run:
            inference.append("--dry-run")
        run(inference)

        if not args.dry_run:
            render = [
                str(args.blender),
                "-b",
                "--python",
                str(script_root / "render_dynamic_sequence.py"),
                "--",
                "--sequence",
                str(sequence),
                "--output",
                str(video),
                "--fps",
                str(args.fps),
                "--stride",
                str(args.stride),
            ]
            run(render)

            if args.gallery_public:
                destination = (
                    args.gallery_public
                    / "cases"
                    / f"valid_garment_{case_id}"
                    / "dynamic"
                    / "contourcraft.mp4"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video, destination)
                print(f"[GALLERY] {destination}")
        completed += 1

    print(
        json.dumps(
            {
                "requested": len(case_ids),
                "completed": completed,
                "skipped": skipped,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
