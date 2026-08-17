# -*- coding: utf-8 -*-
"""
Generic local-topology point-triangle self-collision exclusion for GarmentCodeRC.

This module does NOT hard-code any garment component name. The caller supplies
panel name patterns (fnmatch syntax) and a graph-ring distance.

The validated formal-suit configuration uses:
    rings = 2
    panel_patterns = ["*sleeve_small*"]

Semantics:
- point and triangle must belong to the SAME selected panel;
- H0 / directly-containing triangles are left to Warp's existing filter;
- H1..Hr pairs are additionally filtered;
- the filter is applied after wp.sim.collide() and before XPBD substeps;
- the Warp kernel is imported from a normal module and is safe to capture in
  the same CUDA graph as the official Cloth frame.
"""
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import warp as wp

from pygarment.meshgen.boxmeshgen import BoxMesh


@wp.kernel
def filter_forbidden_point_tri_pairs(
    point_tri_contact_count: wp.array(dtype=int),
    point_tri_contact_pairs: wp.array(dtype=wp.vec2i),
    point_tri_contact_filter: wp.array(dtype=bool),
    forbidden_pairs: wp.array(dtype=wp.vec2i),
    forbidden_count: int,
    newly_filtered_count: wp.array(dtype=int),
):
    """Binary-search the sorted (particle, face) forbidden-pair table."""
    tid = wp.tid()

    if tid >= point_tri_contact_count[0]:
        return

    if point_tri_contact_filter[tid]:
        return

    pair = point_tri_contact_pairs[tid]
    particle = pair[0]
    face = pair[1]

    lo = int(0)
    hi = forbidden_count

    while lo < hi:
        mid = (lo + hi) // 2
        q = forbidden_pairs[mid]

        if q[0] < particle or (q[0] == particle and q[1] < face):
            lo = mid + 1
        else:
            hi = mid

    if lo < forbidden_count:
        q = forbidden_pairs[lo]
        if q[0] == particle and q[1] == face:
            point_tri_contact_filter[tid] = True
            wp.atomic_add(newly_filtered_count, 0, 1)


def _panel_global_vertex_ids(bm: BoxMesh, panel_name: str) -> np.ndarray:
    panel = bm.panels[panel_name]
    ids = []

    for local_id in range(len(panel.panel_vertices)):
        if local_id < panel.n_stitches:
            gid = int(bm.verts_loc_glob[(panel_name, local_id)])
        else:
            gid = int(panel.glob_offset + (local_id - panel.n_stitches))
        ids.append(gid)

    return np.asarray(sorted(set(ids)), dtype=np.int64)


def _panel_global_faces(bm: BoxMesh, panel_name: str):
    panel = bm.panels[panel_name]
    faces = []

    for face in panel.panel_faces:
        glob = bm._get_glob_ids(panel, face)
        glob = tuple(map(int, glob))
        if len(set(glob)) == 3:
            faces.append(glob)

    return faces


def _runtime_face_index(runtime_faces: np.ndarray):
    out = {}
    for face_id, tri in enumerate(np.asarray(runtime_faces, dtype=np.int64)):
        key = tuple(sorted(map(int, tri)))
        if key in out:
            raise RuntimeError(
                "LocalTopologySelfCollisionFilter: duplicated runtime triangle "
                f"key {key}; cannot build unambiguous face authority"
            )
        out[key] = int(face_id)
    return out


def _select_panels(panel_names, panel_patterns):
    patterns = list(panel_patterns or [])
    if not patterns:
        raise ValueError(
            "enable_local_topology_self_collision_filter=true requires "
            "local_topology_self_collision_panel_patterns"
        )

    selected = [
        name
        for name in panel_names
        if any(fnmatch(name, pattern) for pattern in patterns)
    ]

    if not selected:
        raise RuntimeError(
            "LocalTopologySelfCollisionFilter: no panels matched patterns "
            f"{patterns}; available panels={list(panel_names)}"
        )

    return selected


def build_forbidden_pairs(
    spec_path,
    mesh_resolution,
    runtime_faces,
    rings,
    panel_patterns,
):
    """
    Reconstruct panel topology from the exact specification and convert selected
    same-panel H1..Hr point/triangle relations to runtime (particle, face) ids.
    """
    rings = int(rings)
    if rings < 1:
        raise ValueError("local_topology_self_collision_rings must be >= 1")

    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    runtime_faces = np.asarray(runtime_faces, dtype=np.int64).reshape(-1, 3)
    face_index = _runtime_face_index(runtime_faces)

    bm = BoxMesh(str(spec_path), res=float(mesh_resolution))
    bm.load()

    selected_panels = _select_panels(bm.panelNames, panel_patterns)
    forbidden = set()
    missing_faces = []

    for panel_name in selected_panels:
        runtime_panel_faces = []

        for tri in _panel_global_faces(bm, panel_name):
            key = tuple(sorted(tri))
            face_id = face_index.get(key)

            if face_id is None:
                missing_faces.append((panel_name, tri))
                continue

            runtime_panel_faces.append((int(face_id), tri))

        panel_vertices = set(
            map(int, _panel_global_vertex_ids(bm, panel_name))
        )
        adjacency = {gid: set() for gid in panel_vertices}
        incident_faces = defaultdict(set)
        containing_faces = defaultdict(set)

        for face_id, tri in runtime_panel_faces:
            a, b, c = tri

            for gid in tri:
                if gid in panel_vertices:
                    incident_faces[gid].add(face_id)
                    containing_faces[gid].add(face_id)

            if a in adjacency:
                adjacency[a].update((b, c))
            if b in adjacency:
                adjacency[b].update((a, c))
            if c in adjacency:
                adjacency[c].update((a, b))

        for particle_id in panel_vertices:
            visited = {particle_id}
            frontier = {particle_id}
            candidate_faces = set()

            for _ in range(rings):
                next_frontier = set()

                for vertex_id in frontier:
                    next_frontier |= adjacency.get(vertex_id, set())

                next_frontier -= visited
                visited |= next_frontier
                frontier = next_frontier

                for vertex_id in frontier:
                    candidate_faces |= incident_faces.get(vertex_id, set())

            # H0 is already excluded by Warp's direct triangle-vertex filter.
            candidate_faces -= containing_faces.get(particle_id, set())

            for face_id in candidate_faces:
                forbidden.add((int(particle_id), int(face_id)))

    if missing_faces:
        preview = missing_faces[:10]
        raise RuntimeError(
            "LocalTopologySelfCollisionFilter: panel/runtime face authority "
            f"mismatch; missing={len(missing_faces)}, examples={preview}"
        )

    pairs = np.asarray(sorted(forbidden), dtype=np.int32)
    if pairs.size == 0:
        raise RuntimeError(
            "LocalTopologySelfCollisionFilter produced zero forbidden pairs"
        )

    pairs = pairs.reshape(-1, 2)
    return pairs, selected_panels


class LocalTopologySelfCollisionFilter:
    """GPU-resident, CUDA-graph-capturable local topology filter."""

    def __init__(
        self,
        *,
        spec_path,
        mesh_resolution,
        model,
        device,
        rings,
        panel_patterns,
    ):
        runtime_faces = wp.array.numpy(
            model.particle_shape.indices
        ).reshape(-1, 3).astype(np.int64)

        pairs, selected_panels = build_forbidden_pairs(
            spec_path=spec_path,
            mesh_resolution=mesh_resolution,
            runtime_faces=runtime_faces,
            rings=rings,
            panel_patterns=panel_patterns,
        )

        self.rings = int(rings)
        self.panel_patterns = list(panel_patterns)
        self.selected_panels = list(selected_panels)
        self.forbidden_pairs_host = pairs

        self.forbidden_pairs = wp.array(
            pairs,
            dtype=wp.vec2i,
            device=device,
        )
        self.newly_filtered_count = wp.zeros(
            1,
            dtype=int,
            device=device,
        )

    @property
    def forbidden_pair_count(self):
        return int(len(self.forbidden_pairs_host))

    def apply(self, model, device):
        self.newly_filtered_count.zero_()

        wp.launch(
            kernel=filter_forbidden_point_tri_pairs,
            dim=model.point_tri_contact_max,
            inputs=[
                model.point_tri_contact_count,
                model.point_tri_contact_pairs,
                model.point_tri_contact_filter,
                self.forbidden_pairs,
                self.forbidden_pair_count,
            ],
            outputs=[self.newly_filtered_count],
            device=device,
        )

    def newly_filtered_count_host(self):
        return int(
            wp.array.numpy(self.newly_filtered_count).reshape(-1)[0]
        )
