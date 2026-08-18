#!/usr/bin/env python3
"""Run the suit LoRA on held-out images and generate GarmentCodeRC 2D outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image

from suit_output_adapter import apply_to_template, extract_mapping, normalize


PROMPT = (
    "Can you estimate the men's suit jacket sewing pattern parameters "
    "based on the image?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-json", type=Path)
    source.add_argument("--image-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suit-lora",
        type=Path,
        help="Suit LoRA state. Omit it to evaluate the untouched official task checkpoint.",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--body", type=Path)
    parser.add_argument(
        "--skip-pattern",
        action="store_true",
        help="Only run model/parameter evaluation; skip GarmentCode pattern construction.",
    )
    return parser.parse_args()


def find_lora_targets(model: torch.nn.Module) -> list[str]:
    targets: set[str] = set()
    excluded = (
        "mm_projector",
        "vision_tower",
        "vision_resampler",
        "float_layer",
    )
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and not any(item in name for item in excluded)
            and any(item in name for item in ("q_proj", "v_proj"))
        ):
            # Keep the full module path.  Using only q_proj/v_proj would also
            # match CLIP attention layers when PEFT resolves target suffixes.
            targets.add(name)
    return sorted(targets)


def expand_square(image: Image.Image, mean: list[float]) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    result = Image.new(image.mode, (size, size), tuple(int(item * 255) for item in mean))
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


def load_model(args: argparse.Namespace):
    project_root = args.project_root.resolve()
    chatgarment_root = project_root / "ChatGarment"
    sys.path.insert(0, str(chatgarment_root))

    from llava import conversation as conversation_lib
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
    from llava.model import GarmentGPTFloat50ForCausalLM

    base_model = project_root / "models/llava-v1.5-7b-4481d270"
    clip_model = (
        project_root
        / "cache/huggingface/hub/models--openai--clip-vit-large-patch14-336"
        / "snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    )
    official_checkpoint = (
        chatgarment_root
        / "checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final"
        / "pytorch_model.bin"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(base_model), model_max_length=1024, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    model = GarmentGPTFloat50ForCausalLM.from_pretrained(
        str(base_model),
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        seg_token_idx=seg_token_idx,
    )
    model_args = SimpleNamespace(
        vision_tower=str(clip_model),
        mm_vision_select_layer=-2,
        mm_projector_type="mlp2x_gelu",
        pretrain_mm_mlp_adapter=None,
        mm_use_im_patch_token=False,
        mm_use_im_start_end=False,
        mm_patch_merge_type="flat",
        mm_vision_select_feature="patch",
        tune_mm_mlp_adapter=False,
    )
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device="cuda")

    model = get_peft_model(
        model,
        LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_lora_targets(model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.resize_token_embeddings(len(tokenizer))
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    official_state = torch.load(official_checkpoint, map_location="cpu")
    if args.suit_lora:
        suit_state = torch.load(args.suit_lora, map_location="cpu")
        unexpected = sorted(set(suit_state) - set(official_state))
        if unexpected:
            raise RuntimeError(f"suit LoRA contains unknown state keys: {unexpected[:5]}")
        official_state.update(suit_state)
        del suit_state
    model.load_state_dict(official_state, strict=True)
    del official_state
    model = model.bfloat16().cuda().eval()

    helpers = {
        "conversation_lib": conversation_lib,
        "default_image_token": DEFAULT_IMAGE_TOKEN,
        "tokenizer_image_token": tokenizer_image_token,
    }
    return model, tokenizer, vision_tower, helpers


def generate_text(
    model,
    tokenizer,
    vision_tower,
    helpers: dict[str, Any],
    image_path: Path,
    max_new_tokens: int,
) -> str:
    processor = vision_tower.image_processor
    image = Image.open(image_path).convert("RGB")
    image = expand_square(image, processor.image_mean)
    image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(device="cuda", dtype=torch.bfloat16)

    conversation = helpers["conversation_lib"].conv_templates["v1"].copy()
    conversation.messages = []
    conversation.append_message(
        conversation.roles[0], helpers["default_image_token"] + "\n" + PROMPT
    )
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = helpers["tokenizer_image_token"](
        prompt, tokenizer, return_tensors="pt"
    ).unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids, _, _ = model.evaluate(
            image_tensor,
            image_tensor,
            input_ids,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
        )
    return (
        tokenizer.decode(output_ids[0, 1:], skip_special_tokens=False)
        .replace("</s>", "")
        .strip()
    )


def generate_pattern(
    garmentcode_root: Path,
    body_path: Path,
    design_path: Path,
    output_root: Path,
    case_id: str,
) -> Path:
    sys.path.insert(0, str(garmentcode_root))
    old_cwd = Path.cwd()
    try:
        os.chdir(garmentcode_root)
        from assets.bodies.body_params import BodyParameters
        from assets.garment_programs.meta_garment import MetaGarment

        design = yaml.safe_load(design_path.read_text(encoding="utf-8"))["design"]
        body = BodyParameters(str(body_path))
        pattern = MetaGarment(case_id, body, design).assembly()
        return Path(
            pattern.serialize(
                output_root,
                tag="",
                to_subfolder=True,
                with_3d=False,
                with_text=False,
                view_ids=False,
                with_printable=True,
            )
        )
    finally:
        os.chdir(old_cwd)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    garmentcode_root = project_root / "GarmentCodeRC"
    template_path = (
        garmentcode_root / "assets/design_params/mens-suit-jacket-clean-final.yaml"
    )
    body_path = args.body or garmentcode_root / "assets/bodies/mean_all.yaml"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_json:
        if args.data_root is None:
            raise ValueError("--data-root is required with --dataset-json")
        records = json.loads(args.dataset_json.read_text(encoding="utf-8-sig"))
    else:
        image_dir = args.image_dir.resolve()
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not image_paths:
            raise FileNotFoundError(f"no supported images found in {image_dir}")
        records = [
            {"id": path.stem, "_image_path": str(path)}
            for path in image_paths
        ]
    if args.limit > 0:
        records = records[: args.limit]
    model, tokenizer, vision_tower, helpers = load_model(args)

    learned_fields = (
        "garment_length_ratio",
        "waist_ease_cm",
        "lapel_style",
        "button_count",
        "small_pocket_enabled",
        "large_pockets_enabled",
    )
    model_variant = "official_base_plus_suit_lora" if args.suit_lora else "official_base"
    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        case_id = str(record["id"])
        case_dir = output_dir / f"{index:02d}_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = (
            Path(record["_image_path"])
            if "_image_path" in record
            else args.data_root / record["image"]
        )
        expected_text = None
        conversations = record.get("conversations")
        if isinstance(conversations, list) and len(conversations) > 1:
            expected_text = conversations[1].get("value")
        if expected_text:
            shutil.copy2(image_path, case_dir / f"input{image_path.suffix.lower()}")
            (case_dir / "expected_output.txt").write_text(expected_text, encoding="utf-8")
            expected_values, _ = normalize(extract_mapping(expected_text))
        else:
            expected_values = None

        result: dict[str, Any] = {
            "id": case_id,
            "image": str(image_path),
            "model_variant": model_variant,
            "generation_success": False,
            "parse_success": False,
            "schema_complete": False,
            "pattern_attempted": False,
            "pattern_success": None,
        }
        if expected_values is not None:
            result["expected"] = expected_values
        try:
            generated = generate_text(
                model,
                tokenizer,
                vision_tower,
                helpers,
                image_path,
                args.max_new_tokens,
            )
            (case_dir / "model_output.txt").write_text(generated, encoding="utf-8")
            result["generation_success"] = True
        except Exception as exc:
            result.update(
                {
                    "error_stage": "generation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if result["generation_success"]:
            try:
                parsed = extract_mapping(generated)
                raw_values = parsed.get("suit", parsed)
                if not isinstance(raw_values, dict):
                    raise ValueError("model output does not contain a suit parameter mapping")
                values, corrections = normalize(parsed)
                result.update(
                    {
                        "parse_success": True,
                        "schema_complete": all(field in raw_values for field in learned_fields),
                        "raw_predicted": raw_values,
                        "predicted": values,
                        "corrections": corrections,
                    }
                )
                if expected_text:
                    result.update(
                        {
                            "field_matches": {
                                field: values[field] == expected_values[field]
                                for field in learned_fields
                            },
                        }
                    )
                template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
                design_document = apply_to_template(template, values)
                design_path = case_dir / "design_params.yaml"
                design_path.write_text(
                    yaml.safe_dump(design_document, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                if not args.skip_pattern:
                    result["pattern_attempted"] = True
                    try:
                        pattern_dir = generate_pattern(
                            garmentcode_root,
                            body_path,
                            design_path,
                            case_dir / "pattern",
                            case_id,
                        )
                        result.update(
                            {"pattern_success": True, "pattern_dir": str(pattern_dir)}
                        )
                    except Exception as exc:
                        result.update(
                            {
                                "pattern_success": False,
                                "pattern_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
            except Exception as exc:
                result.update(
                    {
                        "error_stage": "parse_or_adapter",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        results.append(result)
        (case_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))

    summary = {
        "model_variant": model_variant,
        "cases": len(results),
        "generation_success": sum(bool(item.get("generation_success")) for item in results),
        "parse_success": sum(bool(item.get("parse_success")) for item in results),
        "schema_complete": sum(bool(item.get("schema_complete")) for item in results),
        "pattern_attempted": sum(bool(item.get("pattern_attempted")) for item in results),
        "pattern_success": sum(bool(item.get("pattern_success")) for item in results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
