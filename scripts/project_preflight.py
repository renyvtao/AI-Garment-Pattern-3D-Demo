#!/usr/bin/env python3
"""Validate public data, pinned sources, restricted assets and local runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--hash-large-assets", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_check(path: Path, *, required: bool = True) -> dict:
    return {
        "path": str(path),
        "required": required,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def command_version(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command, check=True, text=True, capture_output=True, timeout=15
        )
        line = (result.stdout or result.stderr).splitlines()[0]
        return {"available": True, "version": line}
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return {"available": False, "error": str(exc)}


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest = json.loads(
        (root / "PROJECT_MANIFEST.json").read_text(encoding="utf-8")
    )
    public_files = [
        root / "suit_finetune/prepared_data/train.json",
        root / "suit_finetune/prepared_data/validation.json",
        root / "suit_finetune/prepared_data/test.json",
        root / "suit_finetune/prepared_data/manifest.json",
        root / "incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816/01_GOLDEN_BASE/K62_specification.json",
        root / "incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816/02_BODY/mean_all.obj",
        root / "gallery_site/pnpm-lock.yaml",
    ]
    package_names = [
        "torch",
        "torchvision",
        "pytorch3d",
        "fvcore",
        "iopath",
        "transformers",
        "peft",
        "deepspeed",
        "numpy",
        "scipy",
        "PyYAML",
        "Pillow",
        "trimesh",
        "paramiko",
    ]
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    report = {
        "project_root": str(root),
        "python": sys.version.split()[0],
        "node": command_version(["node", "--version"]),
        "git": command_version(["git", "--version"]),
        "public_files": [file_check(path) for path in public_files],
        "packages": packages,
    }

    if not args.source_only:
        official = (
            root
            / "ChatGarment/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin"
        )
        restricted_files = [
            official,
            root / "models/llava-v1.5-7b-4481d270/config.json",
            root
            / "cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/config.json",
            root / "dynamic3d/assets/ccraft_data/trained_models/contourcraft.pth",
            root
            / "dynamic3d/assets/ccraft_data/aux_data/body_models/smplx/SMPLX_MALE.pkl",
            root
            / "dynamic3d/assets/ccraft_data/aux_data/body_models/smplx/SMPLX_FEMALE.pkl",
            root
            / "dynamic3d/assets/ccraft_data/aux_data/body_models/smplx/SMPLX_NEUTRAL.pkl",
            root / "dynamic3d/blender-3.6.14-linux-x64/blender",
            root / "GarmentCodeRC_K62_3D/pygarment/meshgen/garment.py",
        ]
        checks = [file_check(path) for path in restricted_files]
        if official.is_file():
            checks[0]["expected_bytes"] = manifest["sources"]["chatgarment_checkpoint"][
                "expected_bytes"
            ]
            checks[0]["size_matches"] = (
                official.stat().st_size == checks[0]["expected_bytes"]
            )
            if args.hash_large_assets:
                checks[0]["sha256"] = sha256(official)
                checks[0]["hash_matches"] = checks[0]["sha256"] == manifest[
                    "sources"
                ]["chatgarment_checkpoint"]["sha256"]
        report["restricted_assets"] = checks

    required_checks = report["public_files"] + report.get("restricted_assets", [])
    report["ready"] = all(item["exists"] for item in required_checks)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
