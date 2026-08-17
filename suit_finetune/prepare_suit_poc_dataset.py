#!/usr/bin/env python3
"""Prepare the collaborator-provided suit images for a ChatGarment POC.

The script keeps all augmented variants of one base garment in the same split,
removes byte-identical images, applies the agreed first-pass parameter rules,
and writes a deterministic manifest beside the converted LLaVA-style JSON.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


BASE_ID_RE = re.compile(r"^(?P<base>.+)_([1-5])$")
EXPECTED_FIELDS = {
    "body_panel_layout",
    "button_count",
    "front_lower_edge_style",
    "garment_length_ratio",
    "lapel_style",
    "large_pockets_enabled",
    "small_pocket_enabled",
    "waist_ease_cm",
}
PROMPT = (
    "<image>\nCan you estimate the men's suit jacket sewing pattern "
    "parameters based on the image?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--button-spacing-cm", type=float, default=9.0)
    return parser.parse_args()


def base_id(record_id: str) -> str:
    match = BASE_ID_RE.match(record_id)
    return match.group("base") if match else record_id


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe zip member: {name}")
    return path


def python_style(value: object) -> str:
    """Use the Python-dict text style emitted by the original ChatGarment."""
    return repr(value)


def convert_target(raw_text: str, button_spacing_cm: float) -> tuple[dict, int]:
    parsed = ast.literal_eval(raw_text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("suit"), dict):
        raise ValueError("assistant target must contain a suit dictionary")
    raw = parsed["suit"]
    if set(raw) != EXPECTED_FIELDS:
        missing = sorted(EXPECTED_FIELDS - set(raw))
        extra = sorted(set(raw) - EXPECTED_FIELDS)
        raise ValueError(f"unexpected suit fields; missing={missing}, extra={extra}")

    original_button_count = int(raw["button_count"])
    mapped_button_count = min(2, max(1, original_button_count))
    suit = {
        "garment_length_ratio": float(raw["garment_length_ratio"]),
        "button_spacing_cm": float(button_spacing_cm),
        "waist_ease_cm": float(raw["waist_ease_cm"]),
        "body_panel_layout": "six_panel",
        "lapel_style": str(raw["lapel_style"]),
        "front_lower_edge_style": "curved",
        "button_count": mapped_button_count,
        "small_pocket_enabled": bool(raw["small_pocket_enabled"]),
        "large_pockets_enabled": bool(raw["large_pockets_enabled"]),
    }
    return {
        "garment_type": "MensSuitJacketCleanFinal",
        "suit": suit,
    }, original_button_count


def main() -> None:
    args = parse_args()
    source_zip = args.source_zip.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True)

    with zipfile.ZipFile(source_zip) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        image_members = {
            PurePosixPath(name).name: name
            for name in names
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        json_names = [name for name in names if name.lower().endswith(".json")]
        if len(json_names) != 1:
            raise ValueError(f"expected exactly one JSON file, found {json_names}")
        records = json.loads(archive.read(json_names[0]).decode("utf-8-sig"))
        if not isinstance(records, list):
            raise ValueError("dataset JSON root must be a list")

        converted: list[dict] = []
        seen_hashes: dict[str, str] = {}
        duplicate_images: list[dict] = []
        button_counts_before: Counter[int] = Counter()
        button_counts_after: Counter[int] = Counter()

        for record in records:
            image_name = safe_member(str(record["image"]))
            archive_image_name = image_members.get(image_name.name)
            if archive_image_name is None:
                raise FileNotFoundError(f"image not found in archive: {image_name}")
            image_bytes = archive.read(archive_image_name)
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest in seen_hashes:
                duplicate_images.append(
                    {"dropped": image_name.name, "kept": seen_hashes[digest], "sha256": digest}
                )
                continue
            seen_hashes[digest] = image_name.name

            conversations = record.get("conversations", [])
            if len(conversations) != 2 or conversations[1].get("from") != "gpt":
                raise ValueError(f"invalid conversations for {record.get('id')}")
            target, original_button_count = convert_target(
                conversations[1]["value"], args.button_spacing_cm
            )
            mapped_button_count = target["suit"]["button_count"]
            button_counts_before[original_button_count] += 1
            button_counts_after[mapped_button_count] += 1

            destination = images_dir / image_name.name
            destination.write_bytes(image_bytes)
            converted.append(
                {
                    "id": str(record["id"]),
                    "base_id": base_id(str(record["id"])),
                    "image": f"images/{image_name.name}",
                    "conversations": [
                        {"from": "human", "value": PROMPT},
                        {"from": "gpt", "value": python_style(target)},
                    ],
                }
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in converted:
        grouped[record["base_id"]].append(record)
    base_ids = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(base_ids)

    validation_count = round(len(base_ids) * 0.10)
    test_count = round(len(base_ids) * 0.10)
    train_count = len(base_ids) - validation_count - test_count
    split_ids = {
        "train": set(base_ids[:train_count]),
        "validation": set(base_ids[train_count : train_count + validation_count]),
        "test": set(base_ids[train_count + validation_count :]),
    }

    split_stats: dict[str, dict] = {}
    for split_name, selected in split_ids.items():
        split_records = [
            {key: value for key, value in record.items() if key != "base_id"}
            for record in converted
            if record["base_id"] in selected
        ]
        split_records.sort(key=lambda item: item["id"])
        target = output_dir / f"{split_name}.json"
        target.write_text(
            json.dumps(split_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        split_stats[split_name] = {
            "base_garments": len(selected),
            "images": len(split_records),
            "json": target.name,
        }

    manifest = {
        "source_zip": source_zip.name,
        "source_sha256": hashlib.sha256(source_zip.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rules": {
            "mode": "user_selected_mens_suit",
            "garment_type": "MensSuitJacketCleanFinal",
            "button_count": "clamp to [1, 2]",
            "body_panel_layout": "six_panel",
            "front_lower_edge_style": "curved",
            "button_spacing_cm": args.button_spacing_cm,
        },
        "source_records": len(records),
        "kept_records": len(converted),
        "base_garments": len(base_ids),
        "duplicates_removed": duplicate_images,
        "button_count_before": dict(sorted(button_counts_before.items())),
        "button_count_after": dict(sorted(button_counts_after.items())),
        "splits": split_stats,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
