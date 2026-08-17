from pathlib import Path
import argparse, hashlib, json, shutil, sys
import numpy as np

PKG = Path(__file__).resolve().parents[1]
BASE = PKG / "01_GOLDEN_BASE"
BODY = PKG / "02_BODY"

EXPECTED_GOLDEN_SHA = "707c21b2f1e1893df665ce1f683e1db18f5436842b2ae960c5c1c97ea4cf4778"
EXPECTED_SPEC_SHA = "8eced1a3d64e832ebd8fd668b0e657a83813367c7abc58d83fa65328a29606e8"
EXPECTED_VERTICES = 13498
EXPECTED_FACES = 26584

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def obj_counts(path):
    nv = nf = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "): nv += 1
            elif line.startswith("f "): nf += 1
    return nv, nf

def copy_required(src, dst):
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def save_checkpoint(cloth, paths, completed_frame):
    old = paths.g_sim
    target = paths.out_el / f"K62_GOLDEN_DIRECT_FRAME{completed_frame:04d}.obj"
    paths.g_sim = target
    cloth.save_frame(save_v_norms=False)
    paths.g_sim = old
    return target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Separate original+2D GarmentCodeRC copy with 03_RUNTIME_PATCH installed")
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import warp as wp
    import pygarment.data_config as data_config
    from pygarment.meshgen.sim_config import PathCofig, SimConfig
    from pygarment.meshgen.garment import Cloth

    golden = BASE / "K62_GOLDEN_boxmesh.obj"
    spec = BASE / "K62_specification.json"
    sim_props = BASE / "sim_props.yaml"

    if sha256(golden) != EXPECTED_GOLDEN_SHA:
        raise RuntimeError("Golden BoxMesh SHA mismatch")
    if sha256(spec) != EXPECTED_SPEC_SHA:
        raise RuntimeError("Specification SHA mismatch")
    nv, nf = obj_counts(golden)
    if (nv, nf) != (EXPECTED_VERTICES, EXPECTED_FACES):
        raise RuntimeError(f"Golden topology mismatch: {nv}/{nf}")

    print("GOLDEN_INPUT_IDENTITY = PASS")
    print("GOLDEN_SHA256 =", sha256(golden))
    print("GOLDEN_TOPOLOGY =", nv, nf)

    props = data_config.Properties(str(sim_props))
    config = SimConfig(props["sim"]["config"])
    print("WARP_VERSION =", getattr(wp, "__version__", "UNKNOWN"))
    wp.init()
    print("WARP_INIT = PASS")
    print("DEVICE =", wp.get_device())
    print("SIM_SUBSTEPS =", config.sim_substeps)

    # Exact limitation of the portable direct-load smoke:
    # generic local-topology reconstruction is not identity-safe for this deferred Golden mesh.
    config.enable_local_topology_self_collision_filter = False
    print("DIRECT_REPLAY_LOCAL_TOPOLOGY_FILTER = OFF_IN_MEMORY")

    out = PKG / "REPLAY_OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    generated_system = out / "system.generated.json"
    generated_system.write_text(json.dumps({
        "output": str(out),
        "datasets_path": "",
        "datasets_sim": "",
        "sim_configs_path": str(BASE),
        "bodies_default_path": str(BODY),
        "body_samples_path": ""
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    paths = PathCofig(
        in_element_path=str(BASE), out_path=str(out),
        in_name="K62_GOLDEN_DIRECT", out_name="K62_GOLDEN_DIRECT",
        body_name="mean_all", smpl_body=False, add_timestamp=True,
        system_path=str(generated_system), file_name=spec.name,
    )

    copy_required(golden, paths.g_box_mesh)
    copy_required(spec, paths.g_specs)
    copy_required(BASE / "K62_sim_segmentation.txt", paths.g_mesh_segmentation)
    copy_required(BASE / "K62_orig_lens.pickle", paths.g_orig_edge_len)
    copy_required(BASE / "K62_vertex_labels.yaml", paths.g_vert_labels)
    copy_required(sim_props, paths.element_sim_props)

    cloth = Cloth("K62_GOLDEN_DIRECT", config, paths, caching=False)
    particle_count = len(wp.array.numpy(cloth.state_0.particle_q))
    face_count = len(cloth.f_cloth)
    print("MODEL_PARTICLE_COUNT =", particle_count)
    print("MODEL_FACE_COUNT =", face_count)
    if particle_count != EXPECTED_VERTICES or face_count != EXPECTED_FACES:
        raise RuntimeError("ACCIDENTAL_REMESH_OR_RUNTIME_MISMATCH")

    save_checkpoint(cloth, paths, 0)
    checkpoints = {5, 10, 15, 20, args.frames}
    for completed in range(1, args.frames + 1):
        cloth.frame = completed - 1
        cloth.run_frame()
        q = np.asarray(cloth.current_verts, float)
        if not np.isfinite(q).all():
            raise RuntimeError(f"NaN/Inf at frame {completed}")
        if completed in checkpoints:
            p = save_checkpoint(cloth, paths, completed)
            print("SAVED", completed, p)

    last = {
        "status": "PASS",
        "output": str(paths.out_el),
        "frames": args.frames,
        "frame20": str(paths.out_el / "K62_GOLDEN_DIRECT_FRAME0020.obj") if args.frames >= 20 else None,
        "golden_sha256": sha256(golden),
        "topology": [EXPECTED_VERTICES, EXPECTED_FACES],
        "historical_protected20_byte_identical_claim": False,
    }
    (out / "LAST_RUN.json").write_text(json.dumps(last, indent=2, ensure_ascii=True), encoding="utf-8")

    print("DIRECT_GOLDEN_LOAD = PASS")
    print("ACCIDENTAL_REMESH = false")
    print("WARP_FRAMES_RUN =", args.frames)
    print("OUTPUT =", paths.out_el)
    print("IMPORTANT: ordinary Golden direct Warp replay; not a byte-identical claim of historical protected20 dynamics.")

if __name__ == "__main__":
    main()
