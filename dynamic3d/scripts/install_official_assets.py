#!/usr/bin/env python3
"""Safely unpack and register official ContourCraft/ChatGarment-CG assets.

The script never downloads weights. It only handles archives and the licensed
SMPL-X file that the user has placed in the inbox.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz")
MOTION_HINTS = ("male_", "female_", "motion", "sequence", "animation")


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return target


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                safe_target(destination, member.filename)
            handle.extractall(destination)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                safe_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(
                        f"Archive links are not accepted: {member.name}"
                    )
            handle.extractall(destination)
        return

    raise RuntimeError(f"Unsupported archive format: {archive}")


def first_match(files: list[Path], predicate) -> Path | None:
    candidates = sorted((path for path in files if predicate(path)), key=str)
    return candidates[0] if candidates else None


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def looks_like_motion(path: Path) -> bool:
    if path.suffix.lower() != ".npz":
        return False
    lowered = path.name.lower()
    return any(hint in lowered for hint in MOTION_HINTS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chatgarment-cg-root", type=Path, required=True)
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep temporary extracted trees below inbox/extracted.",
    )
    args = parser.parse_args()

    inbox = args.inbox.resolve()
    data_root = args.data_root.resolve()
    cg_root = args.chatgarment_cg_root.resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    archives = sorted(
        path for path in inbox.iterdir() if path.is_file() and is_archive(path)
    )
    direct_files = [
        path for path in inbox.iterdir() if path.is_file() and not is_archive(path)
    ]
    extracted_parent = inbox / "extracted" if args.keep_extracted else None
    temp_context = (
        tempfile.TemporaryDirectory(prefix="official-assets-")
        if extracted_parent is None
        else None
    )
    workspace = (
        Path(temp_context.name)
        if temp_context is not None
        else extracted_parent.resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        extracted_roots: list[Path] = []
        for index, archive in enumerate(archives):
            target = workspace / f"{index:02d}_{archive.stem}"
            extract_archive(archive, target)
            extracted_roots.append(target)

        discovered = list(direct_files)
        for root in extracted_roots:
            discovered.extend(path for path in root.rglob("*") if path.is_file())

        checkpoint = first_match(
            discovered,
            lambda path: path.name.lower() == "contourcraft.pth",
        )
        smplx = first_match(
            discovered,
            lambda path: path.name.upper() == "SMPLX_NEUTRAL.PKL",
        )
        registered = first_match(
            discovered,
            lambda path: path.name.lower() == "registered_params.pkl",
        )
        motions = sorted(
            (path for path in discovered if looks_like_motion(path)),
            key=str,
        )

        installed: dict[str, object] = {
            "checkpoint": None,
            "smplx_neutral": None,
            "registered_rest_body": None,
            "motions": [],
            "contourcraft_data_tree": None,
            "chatgarment_cg_assets": None,
        }

        data_dirs = [
            path
            for root in extracted_roots
            for path in root.rglob("ccraft_data")
            if path.is_dir()
        ]
        official_data_root = first_match(
            data_dirs,
            lambda path: path.name == "ccraft_data",
        )
        if official_data_root:
            shutil.copytree(official_data_root, data_root, dirs_exist_ok=True)
            installed["contourcraft_data_tree"] = str(data_root)

        checkpoint_target = data_root / "trained_models" / "contourcraft.pth"
        if checkpoint:
            copy_file(checkpoint, checkpoint_target)
        if checkpoint_target.is_file():
            installed["checkpoint"] = str(checkpoint_target)

        smplx_target = (
            data_root
            / "aux_data"
            / "body_models"
            / "smplx"
            / "SMPLX_NEUTRAL.pkl"
        )
        if smplx:
            copy_file(smplx, smplx_target)
        if smplx_target.is_file():
            installed["smplx_neutral"] = str(smplx_target)

        registered_target = data_root / "rest_pose" / "registered_params.pkl"
        if registered:
            copy_file(registered, registered_target)
        if registered_target.is_file():
            installed["registered_rest_body"] = str(registered_target)

        motion_targets: list[str] = []
        for motion in motions:
            target = data_root / "motions" / motion.name
            copy_file(motion, target)
            motion_targets.append(str(target))
        if not motion_targets:
            existing_motions = sorted(
                {
                    *data_root.glob("motions/**/*.npz"),
                    *data_root.glob("examples/**/*.npz"),
                },
                key=str,
            )
            motion_targets.extend(str(path) for path in existing_motions)
        installed["motions"] = motion_targets

        assets_dirs = [
            path
            for root in extracted_roots
            for path in root.rglob("assets")
            if path.is_dir()
        ]
        assets_root = first_match(assets_dirs, lambda path: path.name == "assets")
        if assets_root:
            target = cg_root / "assets"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(assets_root, target, dirs_exist_ok=True)
            installed["chatgarment_cg_assets"] = str(target)

        missing = [
            key
            for key in ("checkpoint", "smplx_neutral", "registered_rest_body")
            if installed[key] is None
        ]
        if not motion_targets:
            missing.append("motions")

        result = {
            "inbox": str(inbox),
            "archives_seen": [str(path) for path in archives],
            "installed": installed,
            "missing": missing,
            "ready": not missing,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
