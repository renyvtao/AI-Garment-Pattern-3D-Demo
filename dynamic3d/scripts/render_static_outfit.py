#!/usr/bin/env python3
"""Render a static upper-and-lower outfit on one body with Blender.

Run through Blender, for example::

    blender -b --python render_static_outfit.py -- \
      --upper upper_sim.obj --lower lower_sim.obj --body body.obj \
      --output-dir combined_outfit

GarmentCode simulation meshes use centimetres while the generated SMPL-X
body uses metres, so the default garment scale is 0.01.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper", required=True, type=Path)
    parser.add_argument("--lower", required=True, type=Path)
    parser.add_argument("--body", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--garment-scale", type=float, default=0.01)
    parser.add_argument("--body-scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args(argv)


def load_obj(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                values = line.split()[1:]
                indices: list[int] = []
                for value in values:
                    index = int(value.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[offset], indices[offset + 1]])
    vertex_array = np.asarray(vertices, dtype=np.float32) * scale
    face_array = np.asarray(faces, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not len(vertex_array):
        raise ValueError(f"invalid vertices in {path}: {vertex_array.shape}")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or not len(face_array):
        raise ValueError(f"invalid faces in {path}: {face_array.shape}")
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"mesh contains NaN/Inf: {path}")
    return to_blender_coordinates(vertex_array), face_array


def to_blender_coordinates(vertices: np.ndarray) -> np.ndarray:
    converted = vertices[..., [0, 2, 1]].copy()
    converted[..., 1] *= -1
    return converted


def material(name: str, color: tuple[float, float, float, float], roughness: float):
    value = bpy.data.materials.new(name=name)
    value.diffuse_color = color
    value.use_nodes = True
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return value


def mesh_object(name: str, vertices: np.ndarray, faces: np.ndarray, mesh_material):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mesh_material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def point_camera(camera, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_scene(
    output_dir: Path,
    resolution: int,
    samples: int,
    bounds: np.ndarray,
):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.035, 0.04)
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples

    center = Vector(((bounds[0] + bounds[1]) / 2).tolist())
    extent = float(np.max(bounds[1] - bounds[0]))

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 58

    key_data = bpy.data.lights.new(name="Key", type="AREA")
    key_data.energy = 1000
    key_data.size = max(extent, 1.0)
    key = bpy.data.objects.new(name="Key", object_data=key_data)
    bpy.context.collection.objects.link(key)
    key.location = center + Vector((-extent, -extent, 1.8 * extent))
    point_camera(key, center)

    fill_data = bpy.data.lights.new(name="Fill", type="AREA")
    fill_data.energy = 650
    fill_data.size = max(extent * 1.5, 1.0)
    fill = bpy.data.objects.new(name="Fill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = center + Vector((extent, 0.6 * extent, 0.9 * extent))
    point_camera(fill, center)

    floor_z = float(bounds[0, 2] - 0.015 * max(extent, 1.0))
    bpy.ops.mesh.primitive_plane_add(
        size=max(5 * extent, 4.0),
        location=(center.x, center.y, floor_z),
    )
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(
        material("FloorMaterial", (0.08, 0.1, 0.105, 1.0), 0.82)
    )

    views = {
        "front": center + Vector((0.0, -2.25 * extent, 0.12 * extent)),
        "back": center + Vector((0.0, 2.25 * extent, 0.12 * extent)),
    }
    for name, location in views.items():
        camera.location = location
        point_camera(camera, center)
        scene.render.filepath = str(output_dir / f"combined_outfit_render_{name}.png")
        bpy.ops.render.render(write_still=True)
    return scene


def main() -> None:
    args = parse_args()
    if args.garment_scale <= 0 or args.body_scale <= 0:
        raise ValueError("mesh scales must be positive")
    if args.resolution < 128 or args.samples < 1:
        raise ValueError("resolution/samples are too small")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    upper_vertices, upper_faces = load_obj(args.upper.resolve(), args.garment_scale)
    lower_vertices, lower_faces = load_obj(args.lower.resolve(), args.garment_scale)
    body_vertices, body_faces = load_obj(args.body.resolve(), args.body_scale)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    upper_material = material("UpperMaterial", (0.72, 0.17, 0.075, 1.0), 0.5)
    lower_material = material("LowerMaterial", (0.56, 0.10, 0.045, 1.0), 0.52)
    body_material = material("BodyMaterial", (0.42, 0.5, 0.52, 1.0), 0.63)
    mesh_object("SuitUpper", upper_vertices, upper_faces, upper_material)
    mesh_object("SuitLower", lower_vertices, lower_faces, lower_material)
    mesh_object("Body", body_vertices, body_faces, body_material)

    points = np.concatenate([upper_vertices, lower_vertices, body_vertices], axis=0)
    bounds = np.stack([points.min(axis=0), points.max(axis=0)])
    configure_scene(args.output_dir, args.resolution, args.samples, bounds)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(args.output_dir / "combined_outfit.blend")
    )
    print(
        "[COMPLETED] static outfit front/back "
        f"upper={len(upper_vertices)} lower={len(lower_vertices)} body={len(body_vertices)}"
    )


if __name__ == "__main__":
    main()
