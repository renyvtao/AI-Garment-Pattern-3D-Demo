#!/usr/bin/env python3
"""Extract selected garment/body OBJ frames from a ContourCraft NPZ sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--include-body", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    payload = np.load(args.sequence.resolve(), allow_pickle=True)
    required = {"pred", "cloth_faces"}
    missing = required.difference(payload.files)
    if missing:
        raise KeyError(f"Sequence is missing keys: {sorted(missing)}")

    cloth_vertices = np.asarray(payload["pred"])
    cloth_faces = np.asarray(payload["cloth_faces"], dtype=np.int64)
    body_vertices = (
        np.asarray(payload["body_vertices"])
        if "body_vertices" in payload.files
        else None
    )
    body_faces = (
        np.asarray(payload["body_faces"], dtype=np.int64)
        if "body_faces" in payload.files
        else None
    )

    if cloth_vertices.ndim != 3 or cloth_vertices.shape[-1] != 3:
        raise ValueError(f"Unexpected pred shape: {cloth_vertices.shape}")
    if cloth_faces.ndim != 2 or cloth_faces.shape[-1] != 3:
        raise ValueError(f"Unexpected cloth_faces shape: {cloth_faces.shape}")
    if not np.isfinite(cloth_vertices).all():
        raise ValueError("Dynamic sequence contains NaN or Inf")
    if args.include_body and (body_vertices is None or body_faces is None):
        raise ValueError("--include-body requested but body mesh is absent")

    stop = min(args.stop or len(cloth_vertices), len(cloth_vertices))
    frame_indices = list(range(args.start, stop, args.stride))
    if not frame_indices:
        raise ValueError("No frames selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for frame_index in frame_indices:
        cloth = trimesh.Trimesh(
            vertices=cloth_vertices[frame_index],
            faces=cloth_faces,
            process=False,
        )
        cloth_path = args.output_dir / f"garment_{frame_index:06d}.obj"
        cloth.export(cloth_path, file_type="obj", include_texture=False)

        combined_path = None
        if args.include_body:
            body_frame = min(frame_index, len(body_vertices) - 1)
            body = trimesh.Trimesh(
                vertices=body_vertices[body_frame],
                faces=body_faces,
                process=False,
            )
            combined = trimesh.util.concatenate((body, cloth))
            combined_path = args.output_dir / f"body_garment_{frame_index:06d}.obj"
            combined.export(combined_path, file_type="obj", include_texture=False)

        records.append(
            {
                "frame": frame_index,
                "garment_obj": str(cloth_path.resolve()),
                "combined_obj": (
                    str(combined_path.resolve()) if combined_path else None
                ),
                "cloth_vertices": int(len(cloth.vertices)),
                "cloth_faces": int(len(cloth.faces)),
            }
        )
        print(f"[FRAME] {frame_index}")

    manifest_path = args.output_dir / "frame_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sequence": str(args.sequence.resolve()),
                "frame_count": len(records),
                "frames": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
