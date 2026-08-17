"""EXPERIMENTAL / NOT PART OF THE FROZEN K62 REPLAY.

A minimal body-collider update hook for future moving-body experiments.
It uses the same Warp replace_mesh_points mechanism already used by GarmentCodeRC's
body-smoothing path. It is intentionally separate from 03_RUNTIME_PATCH so the
validated static K62 replay remains unchanged.

Important: K62 sim_props has cloth_reference_drag enabled. Those reference submeshes
are built from the initial body. A production motion runner must either update those
reference meshes as well or disable/revalidate that feature. Therefore this helper
refuses to operate when cloth_reference_drag is enabled unless explicitly allowed.
"""
import numpy as np
import warp as wp
from warp.sim.integrator_xpbd import replace_mesh_points

def update_body_collider(cloth, body_vertices_m, allow_stale_reference_drag=False):
    if getattr(cloth, "enable_cloth_reference_drag", False) and not allow_stale_reference_drag:
        raise RuntimeError("cloth_reference_drag is enabled; update its reference submeshes or disable/revalidate it before moving-body simulation")
    v=np.asarray(body_vertices_m,float)
    if v.shape != np.asarray(cloth.v_body).shape:
        raise ValueError(f"body topology/vertex count mismatch: {v.shape} vs {np.asarray(cloth.v_body).shape}")
    v=v*float(cloth.b_scale)
    if getattr(cloth,"shift_y",0): v[:,1]+=float(cloth.shift_y)
    if not hasattr(cloth,"_motion_body_buffer"):
        cloth._motion_body_buffer=wp.array(v,dtype=wp.vec3,device=cloth.device)
    else:
        wp.copy(cloth._motion_body_buffer,wp.array(v,dtype=wp.vec3,device='cpu',copy=False))
    wp.launch(kernel=replace_mesh_points,dim=len(v),inputs=[cloth.body_mesh.mesh.id,cloth._motion_body_buffer],device=cloth.device)
    cloth.body_mesh.mesh.refit()
    cloth.v_body=v
    if getattr(cloth,"sim_use_graph",False): cloth.create_graph()
