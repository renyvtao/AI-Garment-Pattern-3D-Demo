#!/usr/bin/env python3
"""Measure suit-pattern stitching failure rate with the K62 Warp pipeline.

This is the paper-aligned validity metric that can be computed without a
ground-truth garment mesh.  A case fails if its prediction cannot be adapted to
the fixed K62 topology, meshed, simulated, and written as a completed garment.
The script is resumable because the underlying simulator can reuse completed
outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--inference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Run Warp after preparing K62 specifications.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse cases whose simulator outputs are already complete.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def find_cases(inference_root: Path, limit: int) -> list[Path]:
    cases = sorted(
        path.parent
        for path in inference_root.glob("*/result.json")
        if path.parent.is_dir()
    )
    return cases[:limit] if limit > 0 else cases


def prepare_specs(
    project_root: Path,
    case_dirs: list[Path],
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    specs_dir = output_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        project_root / "GarmentCodeRC_K62_3D/assets/bodies/mean_all.yaml",
        specs_dir / "body_measurements.yaml",
    )

    adapter = project_root / "suit_finetune/build_suit_3d_spec.py"
    golden_spec = (
        project_root
        / "incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
        / "01_GOLDEN_BASE/K62_specification.json"
    )
    prepared: list[Path] = []
    records: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        case_id = case_dir.name.split("_", 1)[-1]
        record: dict[str, Any] = {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "prepared": False,
        }
        try:
            result = read_json(case_dir / "result.json")
            if not result.get("pattern_success"):
                raise RuntimeError("prediction did not generate a 2D pattern")
            source_specs = sorted(case_dir.rglob("*_specification.json"))
            if len(source_specs) != 1:
                raise RuntimeError(
                    f"expected one source specification, found {len(source_specs)}"
                )
            button_count = int(result["predicted"]["button_count"])
            output_spec = specs_dir / f"{case_id}_k62_specification.json"
            audit_path = specs_dir / f"{case_id}_k62_adapter_audit.json"
            command = [
                sys.executable,
                str(adapter),
                "--input-spec",
                str(source_specs[0]),
                "--golden-spec",
                str(golden_spec),
                "--output-spec",
                str(output_spec),
                "--audit",
                str(audit_path),
                "--button-count",
                str(button_count),
            ]
            completed = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "K62 adapter failed")
            prepared.append(output_spec)
            record.update(
                prepared=True,
                button_count=button_count,
                source_spec=str(source_specs[0]),
                adapted_spec=str(output_spec),
                audit=str(audit_path),
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        print(
            f"[PREPARE] {len(records)}/{len(case_dirs)} {case_id}: "
            f"{'ok' if record['prepared'] else record['error']}",
            flush=True,
        )

    return prepared, records


def run_simulation(
    project_root: Path,
    spec_list: Path,
    simulation_summary: Path,
    skip_completed: bool,
) -> int:
    command = [
        sys.executable,
        str(project_root / "scripts/garment_sim_runner.py"),
        "--garmentcode-root",
        str(project_root / "GarmentCodeRC_K62_3D"),
        "--spec-list",
        str(spec_list),
        "--config",
        str(
            project_root
            / "incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
            / "01_GOLDEN_BASE/sim_props.yaml"
        ),
        "--system",
        str(
            project_root
            / "incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816"
            / "REPLAY_OUTPUT/system.generated.json"
        ),
        "--body",
        "mean_all",
        "--summary",
        str(simulation_summary),
    ]
    if skip_completed:
        command.append("--skip-completed")

    environment = os.environ.copy()
    environment.update(
        {
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
            "PYTHONPATH": str(project_root / "GarmentCodeRC_K62_3D"),
            "PYOPENGL_PLATFORM": "egl",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    completed = subprocess.run(
        command,
        cwd=project_root / "GarmentCodeRC_K62_3D",
        env=environment,
        check=False,
    )
    return completed.returncode


def summarize(
    preparation_records: list[dict[str, Any]],
    simulation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    simulation_by_case = {
        str(item.get("garment_name", "")).removesuffix("_k62"): item
        for item in simulation_records
    }
    cases: list[dict[str, Any]] = []
    for prepared in preparation_records:
        case_id = str(prepared["case_id"])
        item = {
            "case_id": case_id,
            "prepared": bool(prepared.get("prepared")),
            "status": "failed",
        }
        if not prepared.get("prepared"):
            item["failure_stage"] = "pattern_or_adapter"
            item["error"] = prepared.get("error")
        else:
            simulation = simulation_by_case.get(case_id)
            if simulation is None:
                item["failure_stage"] = "simulation_missing"
                item["error"] = "simulation result is missing"
            elif simulation.get("status") != "completed":
                item["failure_stage"] = "stitching_or_simulation"
                item["error"] = simulation.get("error")
            else:
                item.update(
                    status="completed",
                    elapsed_seconds=simulation.get("elapsed_seconds"),
                    reused_existing=bool(simulation.get("reused_existing")),
                    output_files=simulation.get("output_files", {}),
                )
        cases.append(item)

    failed = sum(item["status"] != "completed" for item in cases)
    total = len(cases)
    return {
        "schema": "suit_stitching_evaluation_v1",
        "metric": "stitching_failure_rate",
        "definition": (
            "Cases that fail parameter adaptation, K62 meshing, Warp stitching, "
            "simulation, or completed garment export divided by all evaluated cases."
        ),
        "total_case_count": total,
        "completed_case_count": total - failed,
        "failure_count": failed,
        "stitching_failure_rate": failed / total if total else None,
        "cases": cases,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    rate = summary["stitching_failure_rate"]
    rate_text = "N/A" if rate is None else f"{rate * 100:.2f}%"
    failures = [item for item in summary["cases"] if item["status"] != "completed"]
    lines = [
        "# 男西装 LoRA 缝合失败率",
        "",
        "该指标尽量对齐 ChatGarment 原论文的 stitching failure rate。",
        "不使用渲染图计算分数；渲染文件仅是现有 Warp 流程完成性的输出检查。",
        "",
        "| 测试样本 | 完成缝合 | 失败 | 缝合失败率 |",
        "|---:|---:|---:|---:|",
        (
            f"| {summary['total_case_count']} | {summary['completed_case_count']} | "
            f"{summary['failure_count']} | {rate_text} |"
        ),
        "",
        "失败被定义为预测板片无法完成 K62 适配、网格构建、Warp 缝合/垂坠或最终服装网格导出。",
        "该测试使用本项目西装留出集，不是论文的 Dress4D/CLoSE 测试集，因此只能比较指标定义，",
        "不能把数值解释为同一数据集上的模型排名。",
    ]
    if failures:
        lines.extend(["", "## 失败样本", ""])
        for item in failures:
            lines.append(
                f"- `{item['case_id']}`：{item.get('failure_stage')}；{item.get('error')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    inference_root = args.inference_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = find_cases(inference_root, args.limit)
    if not cases:
        raise FileNotFoundError(f"no evaluation cases found in {inference_root}")
    specs, preparation = prepare_specs(project_root, cases, output_dir)
    write_json(output_dir / "preparation_summary.json", preparation)
    spec_list = output_dir / "adapted_specs.json"
    write_json(spec_list, [str(path) for path in specs])

    if not args.run_simulation:
        print(f"Prepared {len(specs)}/{len(cases)} specifications: {spec_list}")
        return

    simulation_summary = output_dir / "simulation_summary.json"
    return_code = run_simulation(
        project_root, spec_list, simulation_summary, args.skip_completed
    )
    simulation_records = (
        read_json(simulation_summary) if simulation_summary.is_file() else []
    )
    summary = summarize(preparation, simulation_records)
    write_json(output_dir / "stitching_evaluation_summary.json", summary)
    write_report(output_dir / "stitching_evaluation_report.md", summary)
    print(json.dumps({key: summary[key] for key in (
        "total_case_count", "completed_case_count", "failure_count",
        "stitching_failure_rate")}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if return_code or summary["failure_count"] else 0)


if __name__ == "__main__":
    main()
