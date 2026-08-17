#!/usr/bin/env python3
"""Check ContourCraft-CG code, dependencies, licensed assets, and GPU readiness."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_IMPORTS = {
    "torch": "PyTorch",
    "torch_geometric": "PyTorch Geometric",
    "pytorch3d": "PyTorch3D",
    "smplx": "SMPL-X Python package",
    "trimesh": "Trimesh",
    "omegaconf": "OmegaConf",
    "warp": "NVIDIA Warp",
    "cccollisions": "ContourCraft CCCollision extension",
    "cudf": "RAPIDS cuDF",
    "cugraph": "RAPIDS cuGraph",
}

OPTIONAL_IMPORTS: dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--body-mesh-sequence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def import_status(module_name: str, label: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "label": label,
            "module": module_name,
            "available": True,
            "version": getattr(module, "__version__", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "label": label,
            "module": module_name,
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def command_status(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "command": command,
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def first_existing(root: Path, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return str(matches[0].resolve())
    return None


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()

    imports = {
        name: import_status(name, label)
        for name, label in {**REQUIRED_IMPORTS, **OPTIONAL_IMPORTS}.items()
    }

    torch_status: dict[str, Any] = {
        "imported": imports["torch"]["available"],
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "device_names": [],
    }
    if imports["torch"]["available"]:
        import torch

        torch_status.update(
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=torch.version.cuda,
            device_count=int(torch.cuda.device_count()),
            device_names=[
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        )

    assets = {
        "contourcraft_checkpoint": first_existing(
            data_root,
            (
                "trained_models/contourcraft.pth",
                "**/trained_models/contourcraft.pth",
            ),
        ),
        "smplx_neutral": first_existing(
            data_root,
            (
                "aux_data/body_models/smplx/SMPLX_NEUTRAL.pkl",
                "**/SMPLX_NEUTRAL.pkl",
            ),
        ),
        "motion": first_existing(
            data_root,
            (
                "examples/**/*.npz",
                "motions/**/*.npz",
                "**/*motion*.npz",
            ),
        ),
        "registered_rest_body": first_existing(
            data_root,
            (
                "rest_pose/registered_params.pkl",
                "**/registered_params.pkl",
                "**/*rest*body*.pkl",
            ),
        ),
        "body_mesh_sequence": (
            str(args.body_mesh_sequence.resolve())
            if args.body_mesh_sequence and args.body_mesh_sequence.is_file()
            else first_existing(
                data_root,
                (
                    "examples/fromanypose/mesh_sequence.pkl",
                    "**/mesh_sequence.pkl",
                ),
            )
        ),
    }

    manifest_status: dict[str, Any] | None = None
    if args.manifest:
        manifest_path = args.manifest.resolve()
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_status = {
                "path": str(manifest_path),
                "exists": True,
                "case_count": payload.get("case_count"),
                "failure_count": payload.get("failure_count"),
            }
        else:
            manifest_status = {
                "path": str(manifest_path),
                "exists": False,
            }

    required_imports_ready = all(
        imports[name]["available"] for name in REQUIRED_IMPORTS
    )
    common_assets_ready = bool(
        assets["contourcraft_checkpoint"] and assets["registered_rest_body"]
    )
    parametric_motion_ready = bool(
        common_assets_ready and assets["smplx_neutral"] and assets["motion"]
    )
    mesh_sequence_ready = bool(
        common_assets_ready and assets["body_mesh_sequence"]
    )
    assets_ready = parametric_motion_ready or mesh_sequence_ready
    gpu_ready = torch_status["cuda_available"]

    report = {
        "schema_version": 1,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "environment": {
            "HOOD_PROJECT": os.environ.get("HOOD_PROJECT"),
            "HOOD_DATA": os.environ.get("HOOD_DATA"),
        },
        "project_root": str(project_root),
        "data_root": str(data_root),
        "project_exists": project_root.is_dir(),
        "imports": imports,
        "torch": torch_status,
        "commands": {
            "nvcc": command_status(
                ["/usr/local/cuda-11.8/bin/nvcc", "--version"]
            ),
            "ffmpeg": command_status(["ffmpeg", "-version"]),
            "blender": command_status(["blender", "--version"]),
        },
        "assets": assets,
        "manifest": manifest_status,
        "readiness": {
            "cpu_preparation_ready": project_root.is_dir(),
            "required_python_imports_ready": required_imports_ready,
            "licensed_and_checkpoint_assets_ready": assets_ready,
            "parametric_smplx_motion_ready": parametric_motion_ready,
            "precomputed_mesh_sequence_ready": mesh_sequence_ready,
            "gpu_runtime_ready": gpu_ready,
            "contourcraft_inference_ready": (
                project_root.is_dir()
                and required_imports_ready
                and assets_ready
                and gpu_ready
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["readiness"], ensure_ascii=False, indent=2))
    print(f"[REPORT] {args.output.resolve()}")


if __name__ == "__main__":
    main()
