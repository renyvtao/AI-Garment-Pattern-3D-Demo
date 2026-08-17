from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import yaml

PKG=Path(__file__).resolve().parents[1]
BODY_OBJ=PKG/"02_BODY/mean_all.obj"
BODY_DIR=PKG/"02_BODY"
PROPS=PKG/"01_GOLDEN_BASE/sim_props.yaml"

def read_obj(path):
    v=[]; f=[]
    for line in path.read_text(encoding="utf-8",errors="ignore").splitlines():
        if line.startswith("v "): v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): f.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    return np.asarray(v,float),np.asarray(f,int)

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True,type=Path)
    ap.add_argument("--result",required=True,type=Path,help="Folder containing one *_sim.obj plus K62BDIR2 material companions")
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    repo=args.repo.resolve(); result=args.result.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo))
    from pygarment.meshgen.sim_config import PathCofig
    from pygarment.meshgen.render.pythonrender import render_images

    sim=next(iter(sorted(result.glob("*_sim.obj"))),None)
    if sim is None: raise FileNotFoundError(f"No *_sim.obj in {result}")
    for n in ["K62BDIR2_material.mtl","K62BDIR2_texture_fabric.png"]:
        if not (result/n).is_file(): raise FileNotFoundError(result/n)
    name=sim.stem[:-4] if sim.stem.endswith("_sim") else sim.stem

    generated_system=out/"system.render.generated.json"
    generated_system.write_text(json.dumps({
        "output":str(out),"datasets_path":"","datasets_sim":"",
        "sim_configs_path":str(PKG/"01_GOLDEN_BASE"),
        "bodies_default_path":str(BODY_DIR),"body_samples_path":""
    },indent=2,ensure_ascii=True),encoding="utf-8")

    paths=PathCofig(str(result),str(out),name,out_name=name,body_name="mean_all",add_timestamp=False,system_path=str(generated_system))
    paths.g_sim=sim
    bv,bf=read_obj(BODY_OBJ); bv=bv*100.0
    props=yaml.safe_load(PROPS.read_text(encoding="utf-8"))
    cfg=dict(props["render"]["config"]); cfg["sides"]=["front","back","left45","right45"]; cfg.setdefault("resolution",[800,800])
    render_images(paths,bv,bf,cfg)
    outputs={}; hashes={}
    for side in cfg["sides"]:
        p=Path(paths.render_path(side))
        if not p.is_file(): raise RuntimeError(f"Renderer did not produce {side}: {p}")
        outputs[side]=str(p); hashes[side]=sha(p)
    if len(set(hashes.values())) < 3:
        raise RuntimeError("Rendered views are unexpectedly identical; check that the handed-off pythonrender.py camera patch was installed")
    report={"status":"PASS","source_sim_obj":str(sim),"render_authority":"pygarment.meshgen.render.pythonrender.render_images","outputs":outputs,"output_sha256":hashes,"WARP_FRAMES_RUN":0,"CLOTH_INSTANTIATION":False,"official_material_expected":True}
    (out/"render_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=True),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
