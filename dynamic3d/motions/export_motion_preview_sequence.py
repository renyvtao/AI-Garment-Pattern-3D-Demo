#!/usr/bin/env python3
"""Expand one SMPL-X motion into a body-only sequence for Blender QA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contourcraft_root = args.project_root / "dynamic3d" / "src" / "ContourCraft-CG"
    sys.path.insert(0, str(contourcraft_root))
    from runners.smplx.body_models import SMPLXLayer

    motion = np.load(args.motion, allow_pickle=True)
    poses = np.asarray(motion["poses"], dtype=np.float32).reshape(-1, 55, 3)
    trans = np.asarray(motion["trans"], dtype=np.float32).reshape(-1, 3)
    betas = np.asarray(motion["betas"], dtype=np.float32).reshape(1, -1)
    layer = SMPLXLayer(str(args.smplx_model), ext="pkl", num_betas=300).to(args.device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(poses), 60):
            end = min(len(poses), start + 60)
            output = layer.forward_simple(
                betas=torch.as_tensor(betas, device=args.device).expand(end - start, -1),
                full_pose=torch.as_tensor(poses[start:end], device=args.device),
                transl=torch.as_tensor(trans[start:end], device=args.device),
                pose2rot=True,
            )
            chunks.append(output.vertices.detach().cpu().numpy().astype(np.float32))
    body = np.concatenate(chunks, axis=0)
    faces = layer.faces_tensor.detach().cpu().numpy().astype(np.int32)
    # The existing renderer expects a cloth sequence. Put one invisible-scale
    # triangle below the floor and render the body fields for visual QA.
    anchor = body[0, 0]
    triangle = np.stack(
        [anchor, anchor + [1e-5, 0, 0], anchor + [0, 0, 1e-5]]
    ).astype(np.float32)
    dummy = np.repeat(triangle[None], len(body) + 2, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pred=dummy,
        cloth_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        body_vertices=body,
        body_faces=faces,
    )
    print(f"wrote {args.output} with {len(body)} frames")


if __name__ == "__main__":
    main()
