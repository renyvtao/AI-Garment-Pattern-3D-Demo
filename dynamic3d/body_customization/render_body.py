#!/usr/bin/env python3
"""Blender background renderer for a generated SMPL-X OBJ."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    separator = sys.argv.index("--")
    obj_path = Path(sys.argv[separator + 1]).resolve()
    output_path = Path(sys.argv[separator + 2]).resolve()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    try:
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {obj_path}")
    bpy.context.view_layer.objects.active = meshes[0]
    for mesh in meshes:
        mesh.select_set(True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    body = bpy.context.view_layer.objects.active
    body.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.shade_smooth()

    material = bpy.data.materials.new("BodyMaterial")
    material.diffuse_color = (0.43, 0.60, 0.57, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.31, 0.52, 0.49, 1.0)
    principled.inputs["Roughness"].default_value = 0.68
    body.data.materials.append(material)

    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground_material = bpy.data.materials.new("GroundMaterial")
    ground_material.diffuse_color = (0.92, 0.91, 0.87, 1.0)
    ground.data.materials.append(ground_material)

    bpy.ops.object.light_add(type="AREA", location=(3.0, -3.5, 4.2))
    key = bpy.context.active_object
    key.data.energy = 900
    key.data.shape = "DISK"
    key.data.size = 4.0
    look_at(key, Vector((0, 0, 1.0)))

    bpy.ops.object.light_add(type="AREA", location=(-2.5, -1.0, 2.2))
    fill = bpy.context.active_object
    fill.data.energy = 550
    fill.data.size = 3.0
    look_at(fill, Vector((0, 0, 1.0)))

    bpy.ops.object.camera_add(location=(2.7, -4.7, 1.35))
    camera = bpy.context.active_object
    look_at(camera, Vector((0, 0, 0.92)))
    camera.data.lens = 62
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 1.3
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.94, 0.93, 0.89)
    scene.render.filepath = str(output_path)
    scene.view_settings.look = "Medium High Contrast"
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
