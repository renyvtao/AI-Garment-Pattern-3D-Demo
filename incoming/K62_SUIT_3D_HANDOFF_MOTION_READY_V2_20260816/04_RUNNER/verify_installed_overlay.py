from pathlib import Path
import argparse, hashlib, json
PKG=Path(__file__).resolve().parents[1]
PATCH=PKG/"03_RUNTIME_PATCH"
BODY=PKG/"02_BODY"
FILES=[
    (PATCH/"pygarment/meshgen/garment.py", Path("pygarment/meshgen/garment.py")),
    (PATCH/"pygarment/meshgen/sim_config.py", Path("pygarment/meshgen/sim_config.py")),
    (PATCH/"pygarment/meshgen/topology_self_collision.py", Path("pygarment/meshgen/topology_self_collision.py")),
    (PATCH/"pygarment/meshgen/render/pythonrender.py", Path("pygarment/meshgen/render/pythonrender.py")),
    (BODY/"ggg_body_segmentation.json", Path("assets/bodies/ggg_body_segmentation.json")),
    (BODY/"mean_all.obj", Path("assets/bodies/mean_all.obj")),
    (BODY/"mean_all.yaml", Path("assets/bodies/mean_all.yaml")),
]
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",required=True,type=Path); args=ap.parse_args()
    repo=args.repo.resolve(); rows=[]; ok=True
    for src,rel in FILES:
        dst=repo/rel; exists=dst.is_file(); same=exists and sha(src)==sha(dst)
        rows.append({"path":str(rel),"exists":exists,"sha_match":same})
        ok &= same
    report={"status":"PASS" if ok else "FAIL","repo":str(repo),"files":rows}
    print(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if ok else 2)
if __name__=="__main__": main()
