#!/usr/bin/env python3
"""Prepare one or more K62 suit outfits for ContourCraft-CG inference.

Each ``--case`` argument uses ``CASE_ID=PATH``.  GarmentCodeRC meshes are in
centimetres. An optional matching ``--lower CASE_ID=PATH`` adds the lower
garment generated for the same input image. The script preserves both source
meshes for garment-specific pin calculation and writes one combined OBJ in
metres for ContourCraft template construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument(
        "--lower",
        action="append",
        default=[],
        help="Optional lower garment mesh using the same CASE_ID=PATH as --case.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scale-to-meters", type=float, default=0.01)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"OBJ scene has no geometry: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"invalid vertices {vertices.shape}: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"invalid faces {faces.shape}: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"mesh contains NaN/Inf: {path}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def parse_case(value: str) -> tuple[str, Path]:
    case_id, separator, path_text = value.partition("=")
    if not separator or not case_id.strip() or not path_text.strip():
        raise ValueError(f"--case must use CASE_ID=PATH, got: {value!r}")
    return case_id.strip(), Path(path_text).resolve()


def prepare_case(
    case_id: str,
    source: Path,
    lower_source: Path | None,
    output_root: Path,
    scale: float,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    sources = [("upper", source)] if lower_source is not None else [("wholebody", source)]
    if lower_source is not None:
        if not lower_source.is_file():
            raise FileNotFoundError(lower_source)
        sources.append(("lower", lower_source))

    scaled_meshes: list[trimesh.Trimesh] = []
    garments: list[dict[str, Any]] = []
    vertex_cursor = 0
    face_cursor = 0
    for kind, mesh_path in sources:
        mesh = load_mesh(mesh_path)
        garments.append(
            {
                "kind": kind,
                "source_obj": str(mesh_path),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "bounds_source_units": np.asarray(mesh.bounds).round(8).tolist(),
                "start_vertex_index": vertex_cursor,
                "start_face_index": face_cursor,
            }
        )
        vertex_cursor += len(mesh.vertices)
        face_cursor += len(mesh.faces)
        scaled = mesh.copy()
        scaled.vertices = np.asarray(scaled.vertices) * scale
        scaled_meshes.append(scaled)

    case_root = output_root / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    combined = (
        scaled_meshes[0]
        if len(scaled_meshes) == 1
        else trimesh.util.concatenate(scaled_meshes)
    )
    combined_path = case_root / "combined_garment_meters.obj"
    combined.export(combined_path, file_type="obj", include_texture=False)
    reloaded = load_mesh(combined_path)
    if len(reloaded.vertices) != vertex_cursor or len(reloaded.faces) != face_cursor:
        raise ValueError(f"mesh round-trip count mismatch for {case_id}")
    bounds = np.asarray(reloaded.bounds)
    return {
        "case_id": case_id,
        "case_dir": str(source.parent),
        "mode": "outfit" if lower_source is not None else "wholebody",
        "garments": garments,
        "combined_obj": str(combined_path),
        "scale_to_meters": scale,
        "total_vertices": int(len(reloaded.vertices)),
        "total_faces": int(len(reloaded.faces)),
        "bounds_meters": bounds.round(8).tolist(),
        "extent_meters": (bounds[1] - bounds[0]).round(8).tolist(),
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
    if args.scale_to_meters <= 0:
        raise ValueError("--scale-to-meters must be positive")
    parsed = [parse_case(value) for value in args.case]
    case_ids = [case_id for case_id, _ in parsed]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case id")
    parsed_lower = [parse_case(value) for value in args.lower]
    lower_ids = [case_id for case_id, _ in parsed_lower]
    if len(lower_ids) != len(set(lower_ids)):
        raise ValueError("duplicate lower case id")
    unknown_lower_ids = sorted(set(lower_ids).difference(case_ids))
    if unknown_lower_ids:
        raise ValueError(
            "--lower case ids must also be present in --case: "
            + ", ".join(unknown_lower_ids)
        )
    lower_by_case = dict(parsed_lower)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        prepare_case(
            case_id,
            source,
            lower_by_case.get(case_id),
            output_root,
            args.scale_to_meters,
        )
        for case_id, source in parsed
    ]
    payload = {
        "schema_version": 1,
        "source_kind": "k62_suit_static_drape",
        "output_root": str(output_root),
        "scale_to_meters": args.scale_to_meters,
        "case_count": len(cases),
        "outfit_count": sum(case["mode"] == "outfit" for case in cases),
        "wholebody_count": sum(case["mode"] == "wholebody" for case in cases),
        "failure_count": 0,
        "cases": cases,
        "failures": [],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
