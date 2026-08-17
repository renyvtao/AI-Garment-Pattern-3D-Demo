#!/usr/bin/env python3
"""Prepare ChatGarment Warp meshes for ContourCraft-CG inference.

The script groups static garment meshes by image case, combines upper/lower
garments when necessary, converts centimeters to meters, and writes a
machine-readable manifest. It is intentionally CPU-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


GARMENT_KINDS = ("upper", "lower", "wholebody")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vis-root",
        required=True,
        type=Path,
        help="ChatGarment vis_new directory containing valid_garment_<case> folders.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New directory for combined ContourCraft input meshes.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Output manifest path. Defaults to <output-root>/dynamic_manifest.json.",
    )
    parser.add_argument(
        "--scale-to-meters",
        type=float,
        default=0.01,
        help="Scale applied to GarmentCodeRC OBJ vertices. Default converts cm to m.",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=10,
        help="Expected number of image cases. Defaults to the original 10-case batch.",
    )
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"OBJ scene has no geometry: {path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type {type(loaded).__name__}: {path}")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Invalid vertex array {vertices.shape}: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Invalid triangle array {faces.shape}: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh contains NaN or Inf vertices: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh contains out-of-range face indices: {path}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def find_garment_mesh(case_dir: Path, kind: str) -> Path | None:
    candidates = sorted(
        case_dir.glob(
            f"valid_garment_{kind}/valid_garment_{kind}/"
            f"valid_garment_{kind}_sim.obj"
        )
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Multiple {kind} meshes found in {case_dir}")
    return candidates[0]


def mesh_record(kind: str, source: Path, mesh: trimesh.Trimesh) -> dict[str, Any]:
    bounds = np.asarray(mesh.bounds)
    return {
        "kind": kind,
        "source_obj": str(source),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_source_units": bounds.round(8).tolist(),
    }


def finite_json_number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value cannot be written to JSON: {value}")
    return round(float(value), 8)


def prepare_case(
    case_dir: Path,
    output_root: Path,
    scale_to_meters: float,
) -> dict[str, Any]:
    case_id = case_dir.name.removeprefix("valid_garment_")
    sources = {kind: find_garment_mesh(case_dir, kind) for kind in GARMENT_KINDS}
    present = [kind for kind, path in sources.items() if path is not None]

    if present == ["wholebody"]:
        ordered_kinds = ["wholebody"]
    elif set(present) == {"upper", "lower"}:
        ordered_kinds = ["upper", "lower"]
    else:
        raise ValueError(
            f"Expected wholebody or upper+lower meshes for {case_dir.name}; got {present}"
        )

    meshes: list[trimesh.Trimesh] = []
    garments: list[dict[str, Any]] = []
    vertex_cursor = 0
    face_cursor = 0

    for kind in ordered_kinds:
        source = sources[kind]
        assert source is not None
        mesh = load_mesh(source)
        record = mesh_record(kind, source, mesh)
        record["start_vertex_index"] = vertex_cursor
        record["start_face_index"] = face_cursor
        vertex_cursor += len(mesh.vertices)
        face_cursor += len(mesh.faces)

        scaled = mesh.copy()
        scaled.vertices = np.asarray(scaled.vertices) * scale_to_meters
        meshes.append(scaled)
        garments.append(record)

    combined = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    case_output = output_root / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    combined_path = case_output / "combined_garment_meters.obj"
    combined.export(combined_path, file_type="obj", include_texture=False)

    reloaded = load_mesh(combined_path)
    if len(reloaded.vertices) != vertex_cursor or len(reloaded.faces) != face_cursor:
        raise ValueError(
            f"Round-trip count mismatch for {case_id}: "
            f"expected {vertex_cursor}/{face_cursor}, "
            f"got {len(reloaded.vertices)}/{len(reloaded.faces)}"
        )

    bounds_m = np.asarray(reloaded.bounds)
    extent_m = bounds_m[1] - bounds_m[0]
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "mode": "wholebody" if ordered_kinds == ["wholebody"] else "outfit",
        "garments": garments,
        "combined_obj": str(combined_path),
        "scale_to_meters": finite_json_number(scale_to_meters),
        "total_vertices": int(len(reloaded.vertices)),
        "total_faces": int(len(reloaded.faces)),
        "bounds_meters": bounds_m.round(8).tolist(),
        "extent_meters": [finite_json_number(value) for value in extent_m],
        "contourcraft": {
            "template_pkl": None,
            "pinned_vertices": None,
            "body_correspondences": None,
            "motion_npz": None,
            "output_npz": None,
            "status": "mesh_prepared_waiting_for_smplx_and_gpu",
        },
    }


def main() -> None:
    args = parse_args()
    vis_root = args.vis_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else output_root / "dynamic_manifest.json"
    )

    if not vis_root.is_dir():
        raise FileNotFoundError(f"vis_new directory does not exist: {vis_root}")
    if args.scale_to_meters <= 0:
        raise ValueError("--scale-to-meters must be positive")

    case_dirs = sorted(
        path
        for path in vis_root.glob("valid_garment_*")
        if path.is_dir()
    )
    if not case_dirs:
        raise FileNotFoundError(f"No valid_garment_* cases found under {vis_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for case_dir in case_dirs:
        try:
            case = prepare_case(case_dir, output_root, args.scale_to_meters)
            cases.append(case)
            print(
                f"[OK] {case['case_id']} mode={case['mode']} "
                f"vertices={case['total_vertices']} faces={case['total_faces']}"
            )
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_dir.name.removeprefix("valid_garment_"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[FAILED] {case_dir.name}: {failures[-1]['error']}")

    manifest = {
        "schema_version": 1,
        "source_vis_root": str(vis_root),
        "output_root": str(output_root),
        "scale_to_meters": finite_json_number(args.scale_to_meters),
        "case_count": len(cases),
        "outfit_count": sum(case["mode"] == "outfit" for case in cases),
        "wholebody_count": sum(case["mode"] == "wholebody" for case in cases),
        "failure_count": len(failures),
        "cases": cases,
        "failures": failures,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[MANIFEST] {manifest_path}")

    expected_cases = args.expected_cases
    if expected_cases <= 0:
        raise ValueError("--expected-cases must be positive")
    if len(cases) != expected_cases or failures:
        raise SystemExit(
            f"Prepared {len(cases)}/{expected_cases} cases with {len(failures)} failures"
        )


if __name__ == "__main__":
    main()
