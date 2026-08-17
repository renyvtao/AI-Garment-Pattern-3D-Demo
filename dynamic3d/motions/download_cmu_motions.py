#!/usr/bin/env python3
"""Download the selected original CMU ASF/AMC files and record checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-cmu-legacy-certificate",
        action="store_true",
        help="CMU's legacy server currently presents an incomplete TLS chain.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_motion_file(path: Path) -> None:
    head = path.read_bytes()[:4096].lower()
    if b"<html" in head or b"<!doctype" in head:
        raise ValueError(f"server returned HTML instead of a motion file: {path}")
    if path.suffix == ".amc" and b":degrees" not in head:
        raise ValueError(f"AMC header is invalid: {path}")
    if path.suffix == ".asf" and b":bonedata" not in path.read_bytes().lower():
        raise ValueError(f"ASF skeleton is invalid: {path}")


def main() -> None:
    args = parse_args()
    source_list = json.loads(args.source_list.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = (
        ssl._create_unverified_context()
        if args.allow_cmu_legacy_certificate
        else ssl.create_default_context()
    )
    records = []
    for item in source_list["files"]:
        target = args.output_dir / item["filename"]
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(
            item["url"],
            headers={"User-Agent": "AI-Garment-Pattern-3D asset installer/1.0"},
        )
        with urllib.request.urlopen(request, context=context, timeout=180) as response:
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        validate_motion_file(partial)
        partial.replace(target)
        records.append(
            {
                **item,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
        print(f"downloaded {item['id']}: {target}", flush=True)

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": source_list["homepage"],
        "license_summary": source_list["license_summary"],
        "files": records,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
