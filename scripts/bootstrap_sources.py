#!/usr/bin/env python3
"""Clone and pin every upstream source required by the project."""

from __future__ import annotations

import argparse
import json
import shutil
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
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--apply-overrides", action="store_true")
    parser.add_argument("--install-k62-overlay", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, capture: bool = False) -> str:
    print("[RUN]", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def source_records(root: Path) -> list[dict[str, str]]:
    project = json.loads(
        (root / "PROJECT_MANIFEST.json").read_text(encoding="utf-8")
    )
    dynamic = json.loads(
        (root / "dynamic3d" / "SOURCE_LOCK.json").read_text(encoding="utf-8")
    )
    records = [
        {
            "name": "ChatGarment",
            "url": project["sources"]["ChatGarment"]["repository"],
            "commit": project["sources"]["ChatGarment"]["commit"],
            "destination": "ChatGarment",
        },
        {
            "name": "GarmentCodeRC",
            "url": project["sources"]["GarmentCodeRC"]["repository"],
            "commit": project["sources"]["GarmentCodeRC"]["commit"],
            "destination": "GarmentCodeRC",
        },
    ]
    destinations = {
        "ContourCraft-CG": "dynamic3d/src/ContourCraft-CG",
        "ContourCraft": "dynamic3d/src/ContourCraft",
        "CCCollisions": "dynamic3d/src/CCCollisions",
        "cuda-samples": "dynamic3d/src/cuda-samples",
        "PyTorch3D": "dynamic3d/src/pytorch3d",
    }
    for name, value in dynamic["sources"].items():
        records.append(
            {
                "name": name,
                "url": value["url"],
                "commit": value["commit"],
                "destination": destinations[name],
            }
        )
    return records


def ensure_source(root: Path, record: dict[str, str], check_only: bool) -> dict:
    destination = root / record["destination"]
    git_dir = destination / ".git"
    if check_only:
        if not git_dir.is_dir():
            return {**record, "status": "missing"}
        head = run(["git", "-C", str(destination), "rev-parse", "HEAD"], capture=True)
        return {
            **record,
            "status": "ok" if head == record["commit"] else "revision_mismatch",
            "actual_commit": head,
        }

    if destination.exists() and not git_dir.is_dir():
        raise RuntimeError(f"Refusing to replace non-git directory: {destination}")
    if not git_dir.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", record["url"], str(destination)])
    dirty = run(
        ["git", "-C", str(destination), "status", "--porcelain"], capture=True
    )
    if dirty:
        raise RuntimeError(f"Refusing to change dirty checkout: {destination}")
    run(["git", "-C", str(destination), "fetch", "origin", record["commit"]])
    run(["git", "-C", str(destination), "checkout", "--detach", record["commit"]])
    return {**record, "status": "installed", "actual_commit": record["commit"]}


def apply_overrides(root: Path) -> None:
    source = root / "upstream_overrides" / "ChatGarment"
    destination = root / "ChatGarment"
    if not (destination / ".git").is_dir():
        raise FileNotFoundError("Install ChatGarment before applying overrides")
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        print(f"[OVERRIDE] {target}")

    suit_trainer = root / "suit_finetune" / "train_garmentcode_outfit_suit_poc.py"
    trainer_target = (
        destination / "llava" / "train" / "train_garmentcode_outfit_suit_poc.py"
    )
    if not suit_trainer.is_file():
        raise FileNotFoundError(f"Missing public suit trainer: {suit_trainer}")
    trainer_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(suit_trainer, trainer_target)
    print(f"[SUIT TRAINER] {trainer_target}")


def install_k62_overlay(root: Path) -> None:
    source_repo = root / "GarmentCodeRC"
    target_repo = root / "GarmentCodeRC_K62_3D"
    package = (
        root
        / "incoming"
        / "K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
    )
    installer = package / "04_RUNNER" / "install_3d_overlay_to_2d_copy.py"
    if not source_repo.is_dir() or not installer.is_file():
        raise FileNotFoundError("GarmentCodeRC or the public K62 handoff is missing")
    if not target_repo.exists():
        shutil.copytree(source_repo, target_repo, ignore=shutil.ignore_patterns(".git"))
    run([sys.executable, str(installer), "--repo", str(target_repo)])


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    results = [
        ensure_source(root, record, args.check_only)
        for record in source_records(root)
    ]
    if not args.check_only and args.apply_overrides:
        apply_overrides(root)
    if not args.check_only and args.install_k62_overlay:
        install_k62_overlay(root)
    print(json.dumps({"sources": results}, ensure_ascii=False, indent=2))
    if args.check_only and any(item["status"] != "ok" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
