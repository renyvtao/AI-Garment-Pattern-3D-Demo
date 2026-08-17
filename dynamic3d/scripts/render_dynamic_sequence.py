#!/usr/bin/env python3
"""Render a ContourCraft sequence to MP4 with Blender.

Run this script through Blender, for example:

    blender -b --python render_dynamic_sequence.py -- \
      --sequence contourcraft_sequence.npz --output contourcraft.mp4

The mesh is updated by a frame-change handler, so the script does not create
one Blender object or shape key per frame. This keeps long sequences usable.
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
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=2,
        help="ContourCraft commonly stores two warm-up frames before frame zero.",
    )
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--no-body", action="store_true")
    return parser.parse_args(argv)


def normalized_faces(values: np.ndarray) -> np.ndarray:
    faces = np.asarray(values)
    while faces.ndim > 2 and faces.shape[0] == 1:
        faces = faces[0]
    faces = faces.reshape(-1, 3).astype(np.int32, copy=False)
    return faces


def normalized_sequence(values: np.ndarray) -> np.ndarray:
    sequence = np.asarray(values)
    while sequence.ndim > 3 and sequence.shape[0] == 1:
        sequence = sequence[0]
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(f"Expected [frames, vertices, 3], got {sequence.shape}")
    return sequence.astype(np.float32, copy=False)


def to_blender_coordinates(sequence: np.ndarray) -> np.ndarray:
    """Map source (X lateral, Y up, Z depth) to Blender (X, Y depth, Z up)."""
    converted = sequence[..., [0, 2, 1]].copy()
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


def mesh_object(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_material,
):
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
    output: Path,
    fps: int,
    resolution: int,
    samples: int,
    bounds: np.ndarray,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.fps = fps
    scene.render.filepath = str(output)
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
    camera.location = center + Vector((1.25 * extent, -1.9 * extent, 0.7 * extent))
    camera.data.lens = 55
    point_camera(camera, center)

    key_data = bpy.data.lights.new(name="Key", type="AREA")
    key_data.energy = 1000
    key_data.size = max(extent, 1.0)
    key = bpy.data.objects.new(name="Key", object_data=key_data)
    bpy.context.collection.objects.link(key)
    key.location = center + Vector((-extent, -extent, 1.8 * extent))
    point_camera(key, center)

    fill_data = bpy.data.lights.new(name="Fill", type="AREA")
    fill_data.energy = 600
    fill_data.size = max(extent * 1.5, 1.0)
    fill = bpy.data.objects.new(name="Fill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = center + Vector((extent, 0.5 * extent, 0.8 * extent))
    point_camera(fill, center)

    floor_z = float(bounds[0, 2] - 0.015 * max(extent, 1.0))
    bpy.ops.mesh.primitive_plane_add(size=max(5 * extent, 4.0), location=(center.x, center.y, floor_z))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(material("FloorMaterial", (0.08, 0.1, 0.105, 1.0), 0.82))


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.fps < 1:
        raise ValueError("--stride and --fps must be positive")
    args.sequence = args.sequence.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.sequence, allow_pickle=True)
    cloth_all = to_blender_coordinates(normalized_sequence(payload["pred"]))
    cloth = cloth_all[args.skip_first :: args.stride]
    cloth_faces = normalized_faces(payload["cloth_faces"])
    if len(cloth) == 0:
        raise ValueError("No cloth frames remain after --skip-first/--stride")

    body = None
    body_faces = None
    if not args.no_body and {"body_vertices", "body_faces"}.issubset(payload.files):
        body_all = to_blender_coordinates(
            normalized_sequence(payload["body_vertices"])
        )
        body = body_all[:: args.stride][: len(cloth)]
        body_faces = normalized_faces(payload["body_faces"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    cloth_material = material("ClothMaterial", (0.78, 0.16, 0.07, 1.0), 0.48)
    body_material = material("BodyMaterial", (0.42, 0.5, 0.52, 1.0), 0.63)
    cloth_obj = mesh_object("Garment", cloth[0], cloth_faces, cloth_material)
    body_obj = (
        mesh_object("Body", body[0], body_faces, body_material)
        if body is not None and body_faces is not None
        else None
    )

    bounds_points = [cloth.reshape(-1, 3)]
    if body is not None:
        bounds_points.append(body.reshape(-1, 3))
    points = np.concatenate(bounds_points, axis=0)
    bounds = np.stack([points.min(axis=0), points.max(axis=0)])
    configure_scene(args.output, args.fps, args.resolution, args.samples, bounds)

    def update_meshes(scene) -> None:
        index = max(0, min(scene.frame_current - 1, len(cloth) - 1))
        cloth_obj.data.vertices.foreach_set("co", cloth[index].reshape(-1))
        cloth_obj.data.update()
        if body_obj is not None and body is not None:
            body_index = min(index, len(body) - 1)
            body_obj.data.vertices.foreach_set("co", body[body_index].reshape(-1))
            body_obj.data.update()

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(update_meshes)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(cloth)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix(".blend")))
    bpy.ops.render.render(animation=True)
    print(f"[COMPLETED] {args.output} ({len(cloth)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
