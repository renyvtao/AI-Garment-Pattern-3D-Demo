#!/usr/bin/env python3
"""Install the user-downloaded official SHAPY data archive safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "shapy" / "data",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_relative_path(name: str) -> Path | None:
    parts = PurePosixPath(name).parts
    if "data" in parts:
        index = parts.index("data")
        relative = parts[index + 1 :]
    elif parts and parts[0] in {
        "trained_models",
        "utility_files",
        "expose_release",
    }:
        # The official shapy_data.zip release stores these directories at the
        # archive root.  They belong directly below SHAPY_REPO/data.
        relative = parts
    else:
        return None
    if not relative or any(part in {"", ".", ".."} for part in relative):
        return None
    return Path(*relative)


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    target = args.target.resolve()
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise ValueError(f"not a valid ZIP archive: {archive}")
    target.mkdir(parents=True, exist_ok=True)

    installed = 0
    skipped = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            relative = data_relative_path(member.filename)
            if relative is None:
                continue
            destination = (target / relative).resolve()
            destination.relative_to(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size == member.file_size:
                    skipped += 1
                    continue
                raise FileExistsError(
                    f"refusing to overwrite different existing file: {destination}"
                )
            temporary = destination.with_name(destination.name + ".installing")
            with bundle.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
            installed += 1

    checkpoints = sorted(target.glob("trained_models/a2b/**/last.ckpt"))
    expected_variants = {"04b_ahcwh2s.yaml", "05b_ahwcwh2s.yaml"}
    ready = [
        str(path)
        for path in checkpoints
        if path.parent.name in expected_variants
        and "smplx-" in str(path)
        and "10betas" in str(path)
    ]
    result = {
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "target": str(target),
        "installed_files": installed,
        "skipped_files": skipped,
        "a2s_checkpoints": len(checkpoints),
        "required_route_checkpoints_found": ready,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if len(ready) < 8:
        raise RuntimeError(
            "archive installed, but fewer than 8 required 04b/05b "
            "gender/profile checkpoints were found"
        )


if __name__ == "__main__":
    main()
