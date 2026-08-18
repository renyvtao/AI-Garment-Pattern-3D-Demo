#!/usr/bin/env python3
"""Summarize held-out mens-suit parameter predictions and compare model variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LEARNED_FIELDS = (
    "garment_length_ratio",
    "waist_ease_cm",
    "lapel_style",
    "button_count",
    "small_pocket_enabled",
    "large_pockets_enabled",
)
NUMERIC_TOLERANCES = {
    "garment_length_ratio": 0.01,
    "waist_ease_cm": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Inference output to evaluate. Repeat for official-base / suit-LoRA A/B.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--run must be LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ValueError(f"--run must be LABEL=PATH, got {value!r}")
    return label.strip(), Path(raw_path).resolve()


def load_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if results:
        return results
    summary_path = root / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload.get("results"), list):
            return payload["results"]
    raise FileNotFoundError(f"no result.json or usable summary.json under {root}")


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def field_pass(field: str, predicted: Any, expected: Any) -> bool:
    if field in NUMERIC_TOLERANCES:
        left, right = as_number(predicted), as_number(expected)
        return (
            left is not None
            and right is not None
            and abs(left - right) <= NUMERIC_TOLERANCES[field]
        )
    return predicted == expected


def summarize_variant(label: str, results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = len(results)
    parse_success = sum(bool(item.get("parse_success")) for item in results)
    schema_complete = sum(bool(item.get("schema_complete")) for item in results)
    correction_cases = sum(bool(item.get("corrections")) for item in results)
    pattern_attempted = sum(bool(item.get("pattern_attempted")) for item in results)
    pattern_success = sum(item.get("pattern_success") is True for item in results)
    raw_covered = 0
    ground_truth_cases = 0
    all_fields_passed = 0
    ground_truth_records: list[dict[str, Any]] = []
    field_rows: dict[str, list[tuple[Any, Any]]] = {field: [] for field in LEARNED_FIELDS}
    case_rows: list[dict[str, Any]] = []

    for item in results:
        predicted = item.get("predicted") if isinstance(item.get("predicted"), dict) else {}
        expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
        raw = item.get("raw_predicted") if isinstance(item.get("raw_predicted"), dict) else predicted
        raw_covered += sum(field in raw for field in LEARNED_FIELDS)
        has_ground_truth = all(field in expected for field in LEARNED_FIELDS)
        passes: dict[str, bool] = {}
        if has_ground_truth:
            ground_truth_cases += 1
            ground_truth_records.append(expected)
            for field in LEARNED_FIELDS:
                field_rows[field].append((predicted.get(field), expected[field]))
                passes[field] = field_pass(field, predicted.get(field), expected[field])
            all_fields_passed += all(passes.values())
        row: dict[str, Any] = {
            "variant": label,
            "id": item.get("id"),
            "generation_success": bool(item.get("generation_success", item.get("parse_success"))),
            "parse_success": bool(item.get("parse_success")),
            "schema_complete": bool(item.get("schema_complete")),
            "correction_count": len(item.get("corrections", [])),
            "pattern_attempted": bool(item.get("pattern_attempted")),
            "pattern_success": item.get("pattern_success"),
            "error": item.get("error", ""),
        }
        for field in LEARNED_FIELDS:
            row[f"predicted_{field}"] = predicted.get(field)
            row[f"expected_{field}"] = expected.get(field)
            row[f"pass_{field}"] = passes.get(field)
        row["all_fields_pass"] = all(passes.values()) if passes else False
        case_rows.append(row)

    field_metrics: dict[str, dict[str, Any]] = {}
    primary_accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    majority_accuracies: list[float] = []
    majority_values: dict[str, Any] = {}
    for field, pairs in field_rows.items():
        denominator = len(pairs)
        passed = sum(field_pass(field, predicted, expected) for predicted, expected in pairs)
        accuracy = passed / denominator if denominator else None
        distribution = Counter(distribution_key(expected) for _, expected in pairs)
        class_recalls = {
            value: (
                sum(
                    field_pass(field, predicted, expected)
                    for predicted, expected in pairs
                    if distribution_key(expected) == value
                )
                / count
            )
            for value, count in sorted(distribution.items())
        }
        balanced_accuracy = (
            sum(class_recalls.values()) / len(class_recalls) if class_recalls else None
        )
        majority_label = (
            max(distribution, key=lambda value: (distribution[value], value))
            if distribution
            else None
        )
        majority_accuracy = (
            distribution[majority_label] / denominator
            if majority_label is not None and denominator
            else None
        )
        if majority_label is not None:
            majority_values[field] = next(
                expected
                for _, expected in pairs
                if distribution_key(expected) == majority_label
            )
        metric: dict[str, Any] = {
            "ground_truth_count": denominator,
            "pass_count": passed,
            "accuracy": accuracy,
            "ground_truth_distribution": dict(sorted(distribution.items())),
            "class_recalls": class_recalls,
            "balanced_accuracy": balanced_accuracy,
            "majority_baseline_label": majority_label,
            "majority_baseline_accuracy": majority_accuracy,
        }
        if field in NUMERIC_TOLERANCES:
            errors = [
                abs(predicted_number - expected_number)
                for predicted, expected in pairs
                if (predicted_number := as_number(predicted)) is not None
                and (expected_number := as_number(expected)) is not None
            ]
            metric.update(
                {
                    "tolerance": NUMERIC_TOLERANCES[field],
                    "valid_prediction_count": len(errors),
                    "mae": sum(errors) / len(errors) if errors else None,
                    "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None,
                    "exact_match_accuracy": (
                        sum(predicted == expected for predicted, expected in pairs) / denominator
                        if denominator
                        else None
                    ),
                }
            )
        field_metrics[field] = metric
        if accuracy is not None:
            primary_accuracies.append(accuracy)
        if balanced_accuracy is not None:
            balanced_accuracies.append(balanced_accuracy)
        if majority_accuracy is not None:
            majority_accuracies.append(majority_accuracy)

    majority_all_fields_passed = sum(
        all(
            field_pass(field, majority_values.get(field), expected[field])
            for field in LEARNED_FIELDS
        )
        for expected in ground_truth_records
    )

    summary = {
        "label": label,
        "case_count": total,
        "ground_truth_case_count": ground_truth_cases,
        "generation_success_rate": (
            sum(bool(item.get("generation_success", item.get("parse_success"))) for item in results) / total
            if total
            else None
        ),
        "parse_success_rate": parse_success / total if total else None,
        "schema_complete_rate": schema_complete / total if total else None,
        "raw_field_coverage": raw_covered / (total * len(LEARNED_FIELDS)) if total else None,
        "adapter_correction_case_rate": correction_cases / total if total else None,
        "pattern_attempted_count": pattern_attempted,
        "pattern_success_rate": pattern_success / pattern_attempted if pattern_attempted else None,
        "field_metrics": field_metrics,
        "macro_field_accuracy": sum(primary_accuracies) / len(primary_accuracies) if primary_accuracies else None,
        "macro_balanced_accuracy": (
            sum(balanced_accuracies) / len(balanced_accuracies) if balanced_accuracies else None
        ),
        "majority_baseline_macro_field_accuracy": (
            sum(majority_accuracies) / len(majority_accuracies) if majority_accuracies else None
        ),
        "all_fields_pass_rate": all_fields_passed / ground_truth_cases if ground_truth_cases else None,
        "majority_baseline_all_fields_pass_rate": (
            majority_all_fields_passed / ground_truth_cases if ground_truth_cases else None
        ),
        "numeric_tolerances": NUMERIC_TOLERANCES,
    }
    return summary, case_rows


def ratio(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def distribution_key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 男西装 LoRA 留出测试集评测",
        "",
        "本报告评估西装 LoRA 对六个监督制版字段的学习效果。连续字段按容差判定："
        "`garment_length_ratio ≤ 0.01`、`waist_ease_cm ≤ 0.5 cm`；其余字段采用严格相等。",
        "",
        "## 汇总",
        "",
        "| 模型 | 样本 | 生成成功 | 解析成功 | 六字段完整 | 原始字段覆盖 | 常规宏平均 | 平衡宏平均 | 全字段同时通过 | 板片生成 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        pattern = (
            ratio(variant["pattern_success_rate"])
            if variant["pattern_attempted_count"]
            else "未执行"
        )
        lines.append(
            f"| {variant['label']} | {variant['case_count']} | {ratio(variant['generation_success_rate'])} | "
            f"{ratio(variant['parse_success_rate'])} | {ratio(variant['schema_complete_rate'])} | "
            f"{ratio(variant['raw_field_coverage'])} | {ratio(variant['macro_field_accuracy'])} | "
            f"{ratio(variant['macro_balanced_accuracy'])} | "
            f"{ratio(variant['all_fields_pass_rate'])} | {pattern} |"
        )
    if payload["variants"]:
        lines.extend(
            [
                "",
                "## 测试集标签覆盖",
                "",
                "| 字段 | 真值分布（取值: 样本数） |",
                "|---|---|",
            ]
        )
        for field in LEARNED_FIELDS:
            distribution = payload["variants"][0]["field_metrics"][field][
                "ground_truth_distribution"
            ]
            rendered = "；".join(f"{value}: {count}" for value, count in distribution.items())
            lines.append(f"| `{field}` | {rendered} |")
    for variant in payload["variants"]:
        lines.extend(
            [
                "",
                f"## {variant['label']} 字段明细",
                "",
                "| 字段 | 通过率 | 平衡通过率 | 多数类基线 | 各类别召回 | MAE | RMSE | 判定方式 |",
                "|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for field in LEARNED_FIELDS:
            metric = variant["field_metrics"][field]
            if field in NUMERIC_TOLERANCES:
                rule = f"绝对误差 ≤ {NUMERIC_TOLERANCES[field]}"
                mae, rmse = number(metric["mae"]), number(metric["rmse"])
            else:
                rule, mae, rmse = "严格相等", "—", "—"
            recalls = "；".join(
                f"{value}: {ratio(recall)}"
                for value, recall in metric["class_recalls"].items()
            )
            lines.append(
                f"| `{field}` | {ratio(metric['accuracy'])} | "
                f"{ratio(metric['balanced_accuracy'])} | "
                f"{ratio(metric['majority_baseline_accuracy'])} | "
                f"{recalls} | {mae} | {rmse} | {rule} |"
            )
    comparison = payload.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## LoRA 相对官方基础权重",
                "",
                f"- 解析成功率提升：{comparison['parse_success_rate_pp']:+.2f} 个百分点。",
                f"- 六字段完整率提升：{comparison['schema_complete_rate_pp']:+.2f} 个百分点。",
                f"- 六字段宏平均提升：{comparison['macro_field_accuracy_pp']:+.2f} 个百分点。",
                f"- 全字段同时通过率提升：{comparison['all_fields_pass_rate_pp']:+.2f} 个百分点。",
                f"- 原始字段覆盖率提升：{comparison['raw_field_coverage_pp']:+.2f} 个百分点。",
                "",
                "## LoRA 相对测试集多数类常量基线",
                "",
                f"- 常规六字段宏平均：LoRA {ratio(comparison['lora_macro_field_accuracy'])}，"
                f"多数类基线 {ratio(comparison['majority_macro_field_accuracy'])}，"
                f"差值 {comparison['lora_vs_majority_macro_pp']:+.2f} 个百分点。",
                f"- 全字段同时通过率：LoRA {ratio(comparison['lora_all_fields_pass_rate'])}，"
                f"多数类组合基线 {ratio(comparison['majority_all_fields_pass_rate'])}，"
                f"差值 {comparison['lora_vs_majority_all_fields_pp']:+.2f} 个百分点。",
                f"- LoRA 六字段平衡宏平均：{ratio(comparison['lora_macro_balanced_accuracy'])}。",
                "",
                "多数类基线不读取图片，只固定输出每个字段在测试集里出现最多的取值。"
                "它用于判断常规准确率中有多少来自类别不平衡。",
            ]
        )
    lines.extend(
        [
            "",
            "## 与 ChatGarment 论文指标的关系",
            "",
            "这里的参数指标用于判断 LoRA 是否学会当前西装监督信号。缝合失败率由 "
            "run_suit_stitching_evaluation.py 单独统计；CD 与 F-Score 需要逐样本三维真值网格，"
            "不能用预测网格或渲染图替代。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for raw_run in args.run:
        label, root = parse_run(raw_run)
        summary, rows = summarize_variant(label, load_results(root))
        summary["source"] = str(root)
        variants.append(summary)
        case_rows.extend(rows)

    payload: dict[str, Any] = {
        "evaluation": "mens_suit_supervised_held_out",
        "learned_fields": LEARNED_FIELDS,
        "variants": variants,
    }
    base = next((item for item in variants if "base" in item["label"].lower()), None)
    lora = next((item for item in variants if "lora" in item["label"].lower()), None)
    if base and lora:
        payload["comparison"] = {
            "base": base["label"],
            "lora": lora["label"],
            "parse_success_rate_pp": 100.0 * (lora["parse_success_rate"] - base["parse_success_rate"]),
            "schema_complete_rate_pp": 100.0 * (lora["schema_complete_rate"] - base["schema_complete_rate"]),
            "macro_field_accuracy_pp": 100.0 * (lora["macro_field_accuracy"] - base["macro_field_accuracy"]),
            "all_fields_pass_rate_pp": 100.0 * (lora["all_fields_pass_rate"] - base["all_fields_pass_rate"]),
            "raw_field_coverage_pp": 100.0 * (lora["raw_field_coverage"] - base["raw_field_coverage"]),
            "lora_macro_field_accuracy": lora["macro_field_accuracy"],
            "lora_macro_balanced_accuracy": lora["macro_balanced_accuracy"],
            "majority_macro_field_accuracy": lora["majority_baseline_macro_field_accuracy"],
            "lora_vs_majority_macro_pp": 100.0 * (
                lora["macro_field_accuracy"] - lora["majority_baseline_macro_field_accuracy"]
            ),
            "lora_all_fields_pass_rate": lora["all_fields_pass_rate"],
            "majority_all_fields_pass_rate": lora["majority_baseline_all_fields_pass_rate"],
            "lora_vs_majority_all_fields_pp": 100.0 * (
                lora["all_fields_pass_rate"] - lora["majority_baseline_all_fields_pass_rate"]
            ),
        }

    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = list(case_rows[0]) if case_rows else ["variant", "id"]
    with (output_dir / "evaluation_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)
    (output_dir / "evaluation_report.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
