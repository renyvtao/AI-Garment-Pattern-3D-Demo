#!/usr/bin/env python3
"""Run one prepared ChatGarment case through ContourCraft-CG.

The implementation follows the official ContourCraft-CG simulation_example.py
while replacing author-machine paths with command-line arguments. Use
--dry-run in CPU-only mode to validate inputs without importing CUDA modules.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--hood-data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--smplx-model", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument(
        "--body-mesh-sequence",
        type=Path,
        help=(
            "Official precomputed body mesh pickle with 'verts' and 'faces'. "
            "When set, --smplx-model and --motion are not required."
        ),
    )
    parser.add_argument(
        "--rest-body-params",
        required=True,
        type=Path,
        help="Pickle with joints and vertices/smplx_vertices in the static garment pose.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="aux/from_any_pose")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit the body sequence for a smoke test; omit for the full sequence.",
    )
    parser.add_argument(
        "--no-align-body-sequence",
        action="store_true",
        help=(
            "Keep the original global translation of a precomputed body mesh "
            "sequence. By default its first frame is aligned to the registered "
            "rest body."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.body_mesh_sequence is None and (
        args.smplx_model is None or args.motion is None
    ):
        parser.error(
            "provide --body-mesh-sequence, or provide both --smplx-model and --motion"
        )
    if args.max_frames is not None and args.max_frames < 3:
        parser.error("--max-frames must be at least 3")
    return args


def load_case(manifest_path: Path, case_id: str) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [case for case in payload.get("cases", []) if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one case {case_id!r}; found {len(matches)}")
    return matches[0]


def validate_inputs(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "project_root": args.project_root,
        "hood_data": args.hood_data,
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "rest_body_params": args.rest_body_params,
        "combined_obj": Path(case["combined_obj"]),
    }
    if args.body_mesh_sequence is not None:
        paths["body_mesh_sequence"] = args.body_mesh_sequence
    else:
        paths["smplx_model"] = args.smplx_model
        paths["motion"] = args.motion
    status = {
        name: {"path": str(path.resolve()), "exists": path.exists()}
        for name, path in paths.items()
    }
    status["case"] = {
        "case_id": case["case_id"],
        "mode": case["mode"],
        "garment_count": len(case["garments"]),
        "total_vertices": case["total_vertices"],
        "total_faces": case["total_faces"],
    }
    return status


def setup_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_material_params() -> dict[str, Any]:
    from utils.common import random_between_log

    return {
        "density": random_between_log(4.34e-2, 7e-1, [1]).item(),
        "lame_mu": random_between_log(15909, 63636, [1]).item(),
        "lame_lambda": random_between_log(
            3535.414406069427,
            93333.73508005822,
            [1],
        ).item(),
        "bending_coeff": random_between_log(
            6.370782056371576e-08,
            0.0013139737991266374,
            [1],
        ).item(),
        "smpl_model": None,
    }


def load_source_garments(case: dict[str, Any]) -> list[dict[str, Any]]:
    import trimesh

    loaded: list[dict[str, Any]] = []
    for garment in case["garments"]:
        path = Path(garment["source_obj"])
        mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
        vertices_m = np.asarray(mesh.vertices, dtype=np.float32) * float(
            case["scale_to_meters"]
        )
        loaded.append(
            {
                "kind": garment["kind"],
                "path": path,
                "vertices": vertices_m,
                "faces": np.asarray(mesh.faces, dtype=np.int64),
            }
        )
    return loaded


def calculate_pins(
    garments: list[dict[str, Any]],
    rest_body_params_path: Path,
    device: str,
) -> tuple[list[int], list[int]]:
    import torch
    from utils.anypose_utils import calculate_pinned_v_dense

    with rest_body_params_path.open("rb") as handle:
        rest_params = pickle.load(handle)

    if "joints" not in rest_params:
        raise KeyError("rest-body-params must contain 'joints'")
    body_key = "smplx_vertices" if "smplx_vertices" in rest_params else "vertices"
    if body_key not in rest_params:
        raise KeyError("rest-body-params must contain 'vertices' or 'smplx_vertices'")

    joints = torch.as_tensor(rest_params["joints"], dtype=torch.float32, device=device)
    joints = joints.reshape(-1, 3)
    body_vertices = torch.as_tensor(
        rest_params[body_key],
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 3)

    pinned_final: list[int] = []
    body_index_final: list[int] = []
    vertex_offset = 0
    for garment in garments:
        vertices = torch.as_tensor(
            garment["vertices"],
            dtype=torch.float32,
            device=device,
        )
        lower = garment["kind"] == "lower"
        pinned, closest = calculate_pinned_v_dense(
            vertices,
            None,
            joints,
            body_vertices,
            lower_garment=lower,
        )
        pinned_final.extend(int(index) + vertex_offset for index in pinned)
        body_index_final.extend(int(index) for index in closest)
        vertex_offset += len(vertices)
    return pinned_final, body_index_final


def build_template_and_map_indices(
    combined_obj: Path,
    garments: list[dict[str, Any]],
    pinned_vertices: list[int],
    output_dir: Path,
    device: str,
) -> tuple[Path, np.ndarray, int, list[int]]:
    import torch
    import trimesh
    from pytorch3d.ops import knn_points
    from utils.common import pickle_dump
    from utils.mesh_creation import add_pinned_verts_single_template, obj2template

    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "garment_template.pkl"
    template = obj2template(
        str(combined_obj),
        verbose=True,
        approximate_center=True,
    )
    pickle_dump(template, str(template_path))

    template_vertices = np.asarray(template["rest_pos"]).reshape(1, -1, 3)
    combined = trimesh.load(
        combined_obj,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    source_vertices = np.asarray(combined.vertices).reshape(1, -1, 3)

    _, source_to_template, _ = knn_points(
        torch.as_tensor(source_vertices, dtype=torch.float32, device=device),
        torch.as_tensor(template_vertices, dtype=torch.float32, device=device),
        K=1,
    )
    source_to_template = source_to_template.reshape(-1).cpu().numpy()

    mapped_pins = [int(source_to_template[index]) for index in pinned_vertices]
    add_pinned_verts_single_template(str(template_path), mapped_pins)

    if len(garments) == 2:
        upper_count = len(garments[0]["vertices"])
        upper_mapped = source_to_template[:upper_count]
        upper_vertex_end = int(upper_mapped.max()) + 1
    else:
        upper_vertex_end = -1

    return template_path, template_vertices, upper_vertex_end, mapped_pins


def run_contourcraft(args: argparse.Namespace, case: dict[str, Any]) -> Path:
    project_root = args.project_root.resolve()
    hood_data = args.hood_data.resolve()
    os.environ["HOOD_PROJECT"] = str(project_root)
    os.environ["HOOD_DATA"] = str(hood_data)
    sys.path.insert(0, str(project_root))

    import torch
    from utils.arguments import create_modules, create_runner, load_params
    from utils.datasets import make_fromanypose_dataloader
    from utils.validation import apply_material2_params, apply_material_params

    if not torch.cuda.is_available():
        raise RuntimeError("ContourCraft dynamic inference requires a CUDA GPU")

    setup_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    garments = load_source_garments(case)
    pinned_vertices, closest_body_indices = calculate_pins(
        garments,
        args.rest_body_params.resolve(),
        device,
    )

    combined_obj = Path(case["combined_obj"]).resolve()
    template_path, template_vertices, upper_vertex_end, mapped_pins = (
        build_template_and_map_indices(
            combined_obj,
            garments,
            pinned_vertices,
            output_dir,
            device,
        )
    )

    modules, config = load_params(args.config)
    config = apply_material_params(config, random_material_params())
    runner_name = list(config.runner.keys())[0]
    runner_config = config.runner[runner_name]

    if len(garments) == 2:
        config = apply_material2_params(config, random_material_params())
        runner_config.material2.use_meterial2 = True
        runner_config.material2.start_face_indices = len(garments[0]["faces"])
        runner_config.material2.start_vertex_indices = upper_vertex_end
    else:
        runner_config.material2.use_meterial2 = False
        runner_config.material2.start_face_indices = -1
        runner_config.material2.start_vertex_indices = -1

    runner_module, runner, _ = create_runner(modules, config)
    state_dict = torch.load(args.checkpoint.resolve(), map_location="cpu")
    runner.load_state_dict(state_dict["training_module"])
    runner = runner.to(device)
    runner.eval()

    if args.body_mesh_sequence is not None:
        with args.body_mesh_sequence.resolve().open("rb") as handle:
            body_sequence = pickle.load(handle)
        if not isinstance(body_sequence, dict):
            raise TypeError("--body-mesh-sequence must contain a dictionary")
        missing_body_keys = {"verts", "faces"}.difference(body_sequence)
        if missing_body_keys:
            raise KeyError(
                "Body mesh sequence is missing keys: "
                f"{sorted(missing_body_keys)}"
            )
        body_vertices = np.asarray(body_sequence["verts"], dtype=np.float32)
        body_faces = np.asarray(body_sequence["faces"], dtype=np.int64)
        body_source = str(args.body_mesh_sequence.resolve())
        body_alignment_offset = np.zeros(3, dtype=np.float32)
        alignment_applied = not args.no_align_body_sequence
        if alignment_applied:
            with args.rest_body_params.resolve().open("rb") as handle:
                rest_params = pickle.load(handle)
            rest_key = (
                "smplx_vertices"
                if "smplx_vertices" in rest_params
                else "vertices"
            )
            if rest_key not in rest_params:
                raise KeyError(
                    "rest-body-params must contain 'vertices' or "
                    "'smplx_vertices' for body-sequence alignment"
                )
            rest_vertices = np.asarray(
                rest_params[rest_key],
                dtype=np.float32,
            ).reshape(-1, 3)
            first_body = body_vertices[0]
            source_anchor = np.asarray(
                [
                    first_body[:, 0].mean(),
                    first_body[:, 1].min(),
                    first_body[:, 2].mean(),
                ],
                dtype=np.float32,
            )
            target_anchor = np.asarray(
                [
                    rest_vertices[:, 0].mean(),
                    rest_vertices[:, 1].min(),
                    rest_vertices[:, 2].mean(),
                ],
                dtype=np.float32,
            )
            body_alignment_offset = target_anchor - source_anchor
            body_vertices = body_vertices + body_alignment_offset[None, None, :]
    else:
        from runners.smplx.body_models import SMPLXLayer

        motion = np.load(args.motion.resolve())
        required_motion_keys = {"betas", "poses", "trans"}
        missing_motion_keys = required_motion_keys.difference(motion.files)
        if missing_motion_keys:
            raise KeyError(
                f"Motion NPZ is missing keys: {sorted(missing_motion_keys)}"
            )

        smplx_layer = SMPLXLayer(
            str(args.smplx_model.resolve()),
            ext="pkl",
            num_betas=300,
        ).to(device)
        frame_count = len(motion["poses"])
        smplx_output = smplx_layer.forward_simple(
            betas=torch.as_tensor(
                motion["betas"],
                dtype=torch.float32,
                device=device,
            ).expand(frame_count, -1),
            full_pose=torch.as_tensor(
                motion["poses"],
                dtype=torch.float32,
                device=device,
            ),
            transl=torch.as_tensor(
                motion["trans"],
                dtype=torch.float32,
                device=device,
            ),
            pose2rot=True,
        )

        body_vertices = smplx_output.vertices.detach().cpu().numpy()
        body_faces = smplx_layer.faces_tensor.detach().cpu().numpy()
        body_source = str(args.motion.resolve())
        body_alignment_offset = np.zeros(3, dtype=np.float32)
        alignment_applied = False

    if body_vertices.ndim != 3 or body_vertices.shape[-1] != 3:
        raise ValueError(
            f"Expected body verts shaped [frames, vertices, 3], got "
            f"{body_vertices.shape}"
        )
    if body_faces.ndim != 2 or body_faces.shape[-1] != 3:
        raise ValueError(f"Expected body faces shaped [faces, 3], got {body_faces.shape}")

    frame_limit = args.max_frames or len(body_vertices)
    body_vertices = body_vertices[:frame_limit]
    if len(body_vertices) < 3:
        raise ValueError("Body mesh sequence must contain at least 3 frames")

    if (
        args.body_mesh_sequence is not None
        and args.max_frames is None
        and not alignment_applied
        and len(body_vertices) == len(body_sequence["verts"])
    ):
        body_sequence_path = args.body_mesh_sequence.resolve()
    else:
        body_sequence_path = output_dir / "body_sequence.pkl"
        with body_sequence_path.open("wb") as handle:
            pickle.dump({"verts": body_vertices, "faces": body_faces}, handle)

    dataloader = make_fromanypose_dataloader(
        pose_sequence_type="mesh",
        pose_sequence_path=str(body_sequence_path),
        garment_template_path=str(template_path),
        garment_dict2={"vertices": template_vertices.reshape(-1, 3)},
    )
    sample = next(iter(dataloader))
    trajectories = runner.valid_rollout(sample)

    output_path = output_dir / "contourcraft_sequence.npz"
    np.savez_compressed(
        output_path,
        pred=trajectories["pred"],
        cloth_faces=trajectories["cloth_faces"],
        body_vertices=body_vertices,
        body_faces=body_faces,
        pinned_vertices=np.asarray(mapped_pins, dtype=np.int64),
        closest_body_indices=np.asarray(closest_body_indices, dtype=np.int64),
        start_face_index=np.asarray(
            runner_config.material2.start_face_indices,
            dtype=np.int64,
        ),
        start_vertex_index=np.asarray(
            runner_config.material2.start_vertex_indices,
            dtype=np.int64,
        ),
        metrics=np.asarray(trajectories.get("metrics", {}), dtype=object),
        case_id=np.asarray(case["case_id"]),
        body_source=np.asarray(body_source),
        body_alignment_offset=body_alignment_offset,
    )
    return output_path


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.hood_data = args.hood_data.resolve()
    args.manifest = args.manifest.resolve()
    case = load_case(args.manifest, args.case_id)
    validation = validate_inputs(args, case)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "input_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missing = [
        name
        for name, item in validation.items()
        if name != "case" and not item["exists"]
    ]
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(f"Missing required inputs: {', '.join(missing)}")
    if args.dry_run:
        print(f"[DRY-RUN PASSED] {args.case_id}")
        return

    output_path = run_contourcraft(args, case)
    print(f"[COMPLETED] {output_path}")


if __name__ == "__main__":
    main()
