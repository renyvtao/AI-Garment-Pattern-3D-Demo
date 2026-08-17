#!/usr/bin/env python3
"""Refine SHAPY betas against SHAPY-compatible virtual measurements.

The official SHAPY checkpoints regress ten SMPL-X shape coefficients.  This
module keeps that prediction as a prior and performs a small, per-request
optimization against the supplied anthropometric measurements.  It uses the
official SMPL-X landmark YAML and the same measurement conventions:

* height: HeelLeft to HeadTop along the vertical axis;
* chest/waist/hips: convex perimeter of a horizontal mesh cross-section;
* weight: watertight mesh volume multiplied by 985 kg/m^3.

The official first ten coefficients are frozen; a small set of later SMPL-X
shape coefficients absorbs the measurement residual. Pose, translation,
topology, gender and motion remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


MEASUREMENT_LANDMARKS = {
    "chest_m": "NippleRight",
    "waist_m": "BellyButton",
    "hips_m": "Crotch",
}
BODY_DENSITY_KG_M3 = 985.0


@dataclass
class CrossSection:
    """Ordered mesh edges forming one local horizontal cross-section."""

    edge_pairs: torch.Tensor


def _json_floats(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


class VirtualMeasurements:
    """Differentiable measurements on an unposed SMPL-X shaped mesh."""

    def __init__(
        self,
        *,
        v_template: torch.Tensor,
        shapedirs: torch.Tensor,
        faces: torch.Tensor,
        landmarks_path: Path,
        active_beta_count: int,
    ) -> None:
        self.v_template = torch.as_tensor(v_template).detach().cpu().float()
        self.shapedirs = torch.as_tensor(shapedirs).detach().cpu().float()[
            ..., :active_beta_count
        ]
        self.faces = torch.as_tensor(faces).detach().cpu().long()
        with landmarks_path.open("r", encoding="utf-8") as stream:
            self.landmarks: dict[str, dict[str, Any]] = yaml.safe_load(stream)

        triangle_edges = torch.cat(
            [
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            ],
            dim=0,
        )
        self.edges = torch.unique(
            torch.sort(triangle_edges, dim=1).values,
            dim=0,
        )

    def vertices(self, betas10: torch.Tensor) -> torch.Tensor:
        return self.v_template + torch.einsum(
            "vci,i->vc",
            self.shapedirs,
            betas10,
        )

    def landmark(self, vertices: torch.Tensor, name: str) -> torch.Tensor:
        definition = self.landmarks[name]
        face = self.faces[int(definition["face_idx"])]
        barycentric = torch.as_tensor(
            definition["bc"],
            dtype=vertices.dtype,
            device=vertices.device,
        )
        return (vertices[face] * barycentric[:, None]).sum(dim=0)

    @staticmethod
    def _edge_intersections(
        vertices: torch.Tensor,
        edge_pairs: torch.Tensor,
        plane_height: torch.Tensor,
    ) -> torch.Tensor:
        starts = vertices[edge_pairs[:, 0]]
        ends = vertices[edge_pairs[:, 1]]
        denominator = ends[:, 1] - starts[:, 1]
        epsilon = torch.full_like(denominator, 1e-8)
        safe_denominator = torch.where(
            denominator.abs() < 1e-8,
            torch.where(denominator < 0, -epsilon, epsilon),
            denominator,
        )
        fraction = (plane_height - starts[:, 1]) / safe_denominator
        return starts + fraction[:, None] * (ends - starts)

    def build_section(
        self,
        vertices: torch.Tensor,
        plane_height: torch.Tensor,
    ) -> CrossSection:
        detached_vertices = vertices.detach()
        detached_height = plane_height.detach()
        starts = detached_vertices[self.edges[:, 0], 1]
        ends = detached_vertices[self.edges[:, 1], 1]
        crossing = ((starts <= detached_height) & (ends > detached_height)) | (
            (ends <= detached_height) & (starts > detached_height)
        )
        pairs = self.edges[crossing]
        if len(pairs) < 3:
            raise ValueError("body cross-section contains fewer than three edges")
        points = self._edge_intersections(
            detached_vertices,
            pairs,
            detached_height,
        )[:, [0, 2]]
        hull = ConvexHull(points.numpy(), qhull_options="QJ")
        ordered_pairs = pairs[torch.as_tensor(hull.vertices, dtype=torch.long)]
        return CrossSection(edge_pairs=ordered_pairs)

    def sections(self, vertices: torch.Tensor) -> dict[str, CrossSection]:
        return {
            key: self.build_section(
                vertices,
                self.landmark(vertices, landmark_name)[1],
            )
            for key, landmark_name in MEASUREMENT_LANDMARKS.items()
        }

    def measure(
        self,
        vertices: torch.Tensor,
        sections: dict[str, CrossSection],
    ) -> dict[str, torch.Tensor]:
        heel = self.landmark(vertices, "HeelLeft")
        head = self.landmark(vertices, "HeadTop")
        values: dict[str, torch.Tensor] = {
            "height_m": (head[1] - heel[1]).abs(),
        }
        for key, landmark_name in MEASUREMENT_LANDMARKS.items():
            height = self.landmark(vertices, landmark_name)[1]
            points = self._edge_intersections(
                vertices,
                sections[key].edge_pairs,
                height,
            )[:, [0, 2]]
            values[key] = torch.linalg.vector_norm(
                torch.roll(points, shifts=-1, dims=0) - points,
                dim=1,
            ).sum()

        triangles = vertices[self.faces]
        signed_six_volume = torch.sum(
            triangles[:, 0]
            * torch.linalg.cross(
                triangles[:, 1],
                triangles[:, 2],
                dim=1,
            )
        )
        volume_m3 = signed_six_volume.abs() / 6.0
        values["weight_kg"] = volume_m3 * BODY_DENSITY_KG_M3
        return values

    def exact_report(self, betas10: torch.Tensor) -> dict[str, float]:
        vertices = self.vertices(betas10)
        sections = self.sections(vertices)
        return _json_floats(
            {
                key: value.detach().cpu().item()
                for key, value in self.measure(vertices, sections).items()
            }
        )

    def raw_report(self, betas10: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            vertices = self.vertices(betas10)
            sections = self.sections(vertices)
            return {
                key: float(value.detach().cpu())
                for key, value in self.measure(vertices, sections).items()
            }


def _target_measurements(request: dict[str, Any]) -> dict[str, float]:
    targets = {
        "height_m": float(request["height_cm"]) / 100.0,
        "chest_m": float(request["chest_cm"]) / 100.0,
        "waist_m": float(request["waist_cm"]) / 100.0,
        "hips_m": float(request["hips_cm"]) / 100.0,
    }
    if request.get("weight_kg") not in (None, ""):
        targets["weight_kg"] = float(request["weight_kg"])
    return targets


def _errors(
    measured: dict[str, float],
    targets: dict[str, float],
) -> dict[str, float]:
    return _json_floats(
        {key: measured[key] - target for key, target in targets.items()}
    )


def refine_betas(
    initial_betas: np.ndarray,
    request: dict[str, Any],
    *,
    layer: Any,
    measurement_root: Path,
    max_evaluations: int = 300,
    refinement_beta_count: int = 20,
    prior_weight: float = 0.000001,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the ten SHAPY betas and optimize later SMPL-X coefficients."""

    landmarks_path = measurement_root / "smplx_measurements.yaml"
    if not landmarks_path.is_file():
        raise FileNotFoundError(landmarks_path)
    initial_np = np.asarray(initial_betas, dtype=np.float32).reshape(-1)
    if initial_np.shape != (10,):
        raise ValueError(f"expected ten SHAPY betas, got {initial_np.shape}")

    active_beta_count = 10 + refinement_beta_count
    virtual = VirtualMeasurements(
        v_template=layer.v_template,
        shapedirs=layer.shapedirs,
        faces=layer.faces_tensor,
        landmarks_path=landmarks_path,
        active_beta_count=active_beta_count,
    )
    targets = _target_measurements(request)
    initial_active = np.zeros(active_beta_count, dtype=np.float32)
    initial_active[:10] = initial_np
    initial_report = virtual.exact_report(torch.from_numpy(initial_active))
    target_items = list(targets.items())

    def active_betas(candidate: np.ndarray) -> np.ndarray:
        result = initial_active.copy()
        result[10:] = np.asarray(candidate, dtype=np.float32)
        return result

    def residuals(candidate: np.ndarray) -> np.ndarray:
        report = virtual.raw_report(
            torch.from_numpy(active_betas(candidate))
        )
        measurement_residuals = np.asarray(
            [
                (report[key] - target) / target
                for key, target in target_items
            ],
            dtype=np.float64,
        )
        prior_residuals = (
            np.sqrt(prior_weight)
            * np.asarray(candidate, dtype=np.float64)
            / 2.0
        )
        return np.concatenate([measurement_residuals, prior_residuals])

    def jacobian(candidate: np.ndarray) -> np.ndarray:
        # Hull membership is discrete, so it is rebuilt at the current point
        # and frozen only while autograd differentiates the local perimeter.
        # This takes five reverse passes instead of 2*N full mesh evaluations.
        fixed_shapy = torch.from_numpy(initial_active[:10])
        tail = torch.tensor(
            np.asarray(candidate, dtype=np.float32),
            requires_grad=True,
        )
        active = torch.cat([fixed_shapy, tail])
        sections = virtual.sections(virtual.vertices(active))

        def measurement_residuals(
            candidate_tail: torch.Tensor,
        ) -> torch.Tensor:
            candidate_active = torch.cat([fixed_shapy, candidate_tail])
            vertices = virtual.vertices(candidate_active)
            measured = virtual.measure(vertices, sections)
            return torch.stack(
                [
                    (measured[key] - target) / target
                    for key, target in target_items
                ]
            )

        measurement_jacobian = torch.autograd.functional.jacobian(
            measurement_residuals,
            tail,
        ).detach().cpu().numpy().astype(np.float64)
        prior_jacobian = (
            np.sqrt(prior_weight)
            / 2.0
            * np.eye(refinement_beta_count, dtype=np.float64)
        )
        return np.vstack([measurement_jacobian, prior_jacobian])

    # The official ten-dimensional SHAPY result is frozen. Measurement
    # correction uses the next SMPL-X basis vectors, preventing the optimizer
    # from erasing the semantic attributes encoded by SHAPY.
    lower = np.full(refinement_beta_count, -5.0, dtype=np.float64)
    upper = np.full(refinement_beta_count, 5.0, dtype=np.float64)
    solution = least_squares(
        residuals,
        np.zeros(refinement_beta_count, dtype=np.float64),
        bounds=(lower, upper),
        method="trf",
        jac=jacobian,
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=max_evaluations,
    )
    best_betas = active_betas(solution.x)
    final_report = virtual.exact_report(torch.from_numpy(best_betas))
    relative_errors = np.asarray(
        [
            (final_report[key] - target) / target
            for key, target in target_items
        ],
        dtype=np.float64,
    )
    metadata = {
        "enabled": True,
        "implementation": "official_landmarks_local_beta_optimization",
        "solver": "scipy_trust_region_reflective",
        "frozen_shapy_beta_count": 10,
        "refinement_beta_start": 10,
        "refinement_beta_count": refinement_beta_count,
        "active_beta_count": active_beta_count,
        "function_evaluations": int(solution.nfev),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "prior_weight": prior_weight,
        "targets": _json_floats(targets),
        "initial": initial_report,
        "initial_error": _errors(initial_report, targets),
        "final": final_report,
        "final_error": _errors(final_report, targets),
        "mean_squared_relative_error": round(
            float(np.mean(relative_errors**2)),
            10,
        ),
    }
    return best_betas, metadata
