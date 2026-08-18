#!/usr/bin/env python3
"""Compute ChatGarment-style CD and F-Score for aligned garment meshes.

The input meshes must already share the same body shape, pose, coordinate
system, and length unit.  This tool intentionally does not guess registration
or scale because doing so would make the result incomparable with the paper.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "JSON with a cases list. Each case needs id, prediction, and "
            "ground_truth paths; relative paths are resolved from the manifest."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=10_000)
    parser.add_argument(
        "--fscore-threshold",
        type=float,
        default=0.001,
        help="Threshold applied to squared point distances, matching paper code.",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--backend",
        choices=("auto", "pytorch3d", "scipy"),
        default="auto",
        help="Prefer PyTorch3D for paper alignment; SciPy is a portable fallback.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def fscore_from_squared_distances(dist_pred, dist_gt, threshold: float):
    pred = np.asarray(dist_pred, dtype=np.float64)
    gt = np.asarray(dist_gt, dtype=np.float64)
    precision = np.mean(pred < threshold, axis=-1)
    recall = np.mean(gt < threshold, axis=-1)
    denominator = precision + recall
    fscore = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    return fscore, precision, recall


def _load_triangles(path: Path) -> np.ndarray:
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    geometries = list(loaded.geometry.values())
    if not geometries:
        raise ValueError(f"mesh contains no triangle geometry: {path}")
    mesh = trimesh.util.concatenate(geometries)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError(f"mesh contains no usable triangles: {path}")
    return triangles


def _sample_surface(triangles: np.ndarray, count: int, seed: int) -> np.ndarray:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("mesh surface area must be positive")
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(triangles), size=count, p=areas / total_area)
    chosen = triangles[face_indices]
    root_u = np.sqrt(rng.random(count))
    v = rng.random(count)
    return (
        (1.0 - root_u)[:, None] * chosen[:, 0]
        + (root_u * (1.0 - v))[:, None] * chosen[:, 1]
        + (root_u * v)[:, None] * chosen[:, 2]
    )


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain a non-empty cases list")
    base = path.resolve().parent
    normalized = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            raise TypeError(f"case {index} is not an object")
        for key in ("id", "prediction", "ground_truth"):
            if key not in item:
                raise ValueError(f"case {index} is missing {key}")
        normalized.append(
            {
                "id": str(item["id"]),
                "prediction": resolve_path(base, str(item["prediction"])),
                "ground_truth": resolve_path(base, str(item["ground_truth"])),
            }
        )
    missing_gt = [str(item["ground_truth"]) for item in normalized if not item["ground_truth"].is_file()]
    if missing_gt:
        raise FileNotFoundError(
            "ground-truth meshes are required and must not be synthesized by the evaluator: "
            + ", ".join(missing_gt[:5])
        )
    return normalized


def evaluate_case(
    prediction: Path,
    ground_truth: Path,
    sample_count: int,
    threshold: float,
    seed: int,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    pred_points = _sample_surface(_load_triangles(prediction), sample_count, seed)
    gt_points = _sample_surface(_load_triangles(ground_truth), sample_count, seed + 1)
    dist_pred = np.square(cKDTree(gt_points).query(pred_points, k=1, workers=-1)[0])
    dist_gt = np.square(cKDTree(pred_points).query(gt_points, k=1, workers=-1)[0])
    fscore, precision, recall = fscore_from_squared_distances(
        dist_pred, dist_gt, threshold
    )
    chamfer_x1000 = float((dist_pred.mean() + dist_gt.mean()) * 1_000)
    return {
        "chamfer_distance_x1000": chamfer_x1000,
        "fscore": float(np.asarray(fscore).item()),
        "precision": float(np.asarray(precision).item()),
        "recall": float(np.asarray(recall).item()),
    }


def evaluate_case_pytorch3d(
    prediction: Path,
    ground_truth: Path,
    sample_count: int,
    threshold: float,
    device: str,
) -> dict[str, float]:
    from pytorch3d.io import IO
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops import sample_points_from_meshes

    pred_mesh = IO().load_mesh(str(prediction), device=device)
    gt_mesh = IO().load_mesh(str(ground_truth), device=device)
    pred_points = sample_points_from_meshes(pred_mesh, sample_count)
    gt_points = sample_points_from_meshes(gt_mesh, sample_count)
    directional = chamfer_distance(
        pred_points,
        gt_points,
        batch_reduction=None,
        point_reduction=None,
    )[0]
    dist_pred, dist_gt = directional
    fscore, precision, recall = fscore_from_squared_distances(
        dist_pred.detach().cpu().numpy(),
        dist_gt.detach().cpu().numpy(),
        threshold,
    )
    chamfer_x1000 = (dist_pred.mean() + dist_gt.mean()) * 1_000
    return {
        "chamfer_distance_x1000": float(chamfer_x1000.item()),
        "fscore": float(np.asarray(fscore).item()),
        "precision": float(np.asarray(precision).item()),
        "recall": float(np.asarray(recall).item()),
    }


def select_backend(requested: str) -> str:
    available = importlib.util.find_spec("pytorch3d") is not None
    if requested == "pytorch3d" and not available:
        raise ModuleNotFoundError(
            "PyTorch3D backend requested but pytorch3d is not installed"
        )
    if requested == "auto":
        return "pytorch3d" if available else "scipy"
    return requested


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mesh_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "mesh_metrics_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "status",
                "chamfer_distance_x1000",
                "fscore",
                "precision",
                "recall",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["cases"])

    lines = [
        "# 论文口径三维网格指标",
        "",
        "| 有效样本 | CD × 1000（越低越好） | F-Score（越高越好） |",
        "|---:|---:|---:|",
        (
            f"| {summary['valid_case_count']} | "
            f"{summary['mean_chamfer_distance_x1000']:.6f} | "
            f"{summary['mean_fscore']:.6f} |"
        ),
        "",
        f"- 每个网格采样点数：{summary['sample_count']}",
        f"- F-Score 平方距离阈值：{summary['fscore_threshold']}",
        f"- 计算后端：{summary['backend']}（{summary['nearest_neighbor_backend']}）",
        f"- 计算设备：{summary['device']}",
        "- 输入要求：预测与真值网格必须已完成同人体、同姿态、同坐标系和同单位对齐。",
        "- 本脚本不执行渲染，也不会自动缩放或配准网格。",
    ]
    (output_dir / "mesh_metrics_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.fscore_threshold <= 0:
        raise ValueError("--fscore-threshold must be positive")

    backend = select_backend(args.backend)
    device = "cpu"
    if backend == "pytorch3d":
        import torch

        device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else args.device
        )
        if device == "auto":
            device = "cpu"
        torch.manual_seed(args.seed)
    cases = load_manifest(args.manifest.resolve())
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(cases, start=1):
        row: dict[str, Any] = {
            "id": item["id"],
            "status": "failed",
            "chamfer_distance_x1000": None,
            "fscore": None,
            "precision": None,
            "recall": None,
            "error": None,
        }
        if not item["prediction"].is_file():
            row["error"] = f"prediction mesh is missing: {item['prediction']}"
        else:
            try:
                metrics = (
                    evaluate_case_pytorch3d(
                        item["prediction"],
                        item["ground_truth"],
                        args.sample_count,
                        args.fscore_threshold,
                        device,
                    )
                    if backend == "pytorch3d"
                    else evaluate_case(
                        item["prediction"],
                        item["ground_truth"],
                        args.sample_count,
                        args.fscore_threshold,
                        args.seed + index * 2,
                    )
                )
                row.update(status="completed", **metrics)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"[METRIC] {index}/{len(cases)} {item['id']}: {row['status']}")

    valid = [row for row in rows if row["status"] == "completed"]
    if not valid:
        raise RuntimeError("no valid prediction/ground-truth mesh pairs were evaluated")
    summary = {
        "schema": "chatgarment_mesh_metrics_v1",
        "sample_count": args.sample_count,
        "fscore_threshold": args.fscore_threshold,
        "threshold_applies_to": "squared_point_distance",
        "backend": backend,
        "device": device,
        "surface_sampling": (
            "pytorch3d.sample_points_from_meshes"
            if backend == "pytorch3d"
            else "area_weighted_triangle_sampling"
        ),
        "nearest_neighbor_backend": (
            "pytorch3d.chamfer_distance"
            if backend == "pytorch3d"
            else "scipy.spatial.cKDTree"
        ),
        "total_case_count": len(rows),
        "valid_case_count": len(valid),
        "failure_count": len(rows) - len(valid),
        "mean_chamfer_distance_x1000": sum(
            row["chamfer_distance_x1000"] for row in valid
        )
        / len(valid),
        "mean_fscore": sum(row["fscore"] for row in valid) / len(valid),
        "cases": rows,
    }
    write_outputs(args.output_dir.resolve(), summary)
    print(json.dumps({key: summary[key] for key in (
        "total_case_count", "valid_case_count", "failure_count",
        "mean_chamfer_distance_x1000", "mean_fscore")}, indent=2))


if __name__ == "__main__":
    main()
