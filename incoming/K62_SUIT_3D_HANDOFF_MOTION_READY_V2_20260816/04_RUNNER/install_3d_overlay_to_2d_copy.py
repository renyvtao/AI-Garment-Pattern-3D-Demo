from pathlib import Path
import argparse, shutil, json, hashlib

PKG = Path(__file__).resolve().parents[1]
PATCH = PKG / "03_RUNTIME_PATCH"
BODY = PKG / "02_BODY"
EXPECTED_PATCH_FILES = [
    Path("pygarment/meshgen/garment.py"),
    Path("pygarment/meshgen/sim_config.py"),
    Path("pygarment/meshgen/topology_self_collision.py"),
    Path("pygarment/meshgen/render/pythonrender.py"),
]
EXPECTED_BODY_FILES = ["ggg_body_segmentation.json", "mean_all.obj", "mean_all.yaml"]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def tree_digest(root: Path):
    if not root.exists(): return None
    h=hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        rel=p.relative_to(root).as_posix().encode("utf-8")
        h.update(rel); h.update(b"\0")
        with open(p,"rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def backup_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    existed=dst.exists()
    bak=None
    if existed:
        bak = dst.with_suffix(dst.suffix + ".pre_k62_3d.bak")
        if not bak.exists():
            shutil.copy2(dst, bak)
            print("BACKUP:", bak)
    shutil.copy2(src, dst)
    print("COPY:", src, "->", dst)
    return existed, bak

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="A COPY of the student's original+2D GarmentCodeRC repo")
    args=ap.parse_args()
    repo=Path(args.repo).resolve()
    required_repo=[repo/"pygarment/meshgen/garment.py", repo/"pygarment/meshgen/sim_config.py", repo/"pygarment/meshgen/render/pythonrender.py", repo/"assets"]
    missing=[str(p) for p in required_repo if not p.exists()]
    if missing:
        raise SystemExit("Not a compatible GarmentCodeRC repo; missing: " + "; ".join(missing))

    two_d_roots=[repo/"assets/garment_programs", repo/"assets/design_params"]
    before={str(p.relative_to(repo)):tree_digest(p) for p in two_d_roots}
    touched=[]; created_new=[]; backups=[]
    for rel in EXPECTED_PATCH_FILES:
        src=PATCH/rel
        if not src.is_file(): raise FileNotFoundError(src)
        dst=repo/rel
        existed,bak=backup_copy(src,dst)
        touched.append(str(dst))
        if not existed: created_new.append(str(dst))
        if bak is not None: backups.append(str(bak))
    for name in EXPECTED_BODY_FILES:
        src=BODY/name
        if not src.is_file(): raise FileNotFoundError(src)
        dst=repo/"assets/bodies"/name
        existed,bak=backup_copy(src,dst)
        touched.append(str(dst))
        if not existed: created_new.append(str(dst))
        if bak is not None: backups.append(str(bak))

    after={str(p.relative_to(repo)):tree_digest(p) for p in two_d_roots}
    preserved=before==after
    report={
        "status":"PASS" if preserved else "FAIL",
        "repo":str(repo),
        "touched":touched,
        "created_new":created_new,
        "backups":backups,
        "two_d_tree_before":before,
        "two_d_tree_after":after,
        "two_d_programs_preserved":preserved,
        "note":"3D runtime/body/render overlay only; 2D garment programs and design params are hash-checked before/after."
    }
    rep=repo/"K62_3D_OVERLAY_INSTALL_REPORT.json"
    rep.write_text(json.dumps(report,indent=2,ensure_ascii=True),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if preserved else 3)

if __name__=="__main__": main()
