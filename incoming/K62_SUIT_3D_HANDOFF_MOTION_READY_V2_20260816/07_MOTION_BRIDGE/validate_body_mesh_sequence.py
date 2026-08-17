from pathlib import Path
import argparse, json, numpy as np
PKG=Path(__file__).resolve().parents[1]
REF=PKG/"02_BODY/mean_all.obj"

def read_obj(p):
    v=[]; f=[]
    for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
        if line.startswith("v "): v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): f.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
    return np.asarray(v,float),np.asarray(f,int)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sequence",required=True,type=Path); ap.add_argument("--fps",type=float,default=30.0); args=ap.parse_args()
    seq=args.sequence.resolve(); frames=sorted(seq.glob("body_*.obj"))
    if not frames: raise FileNotFoundError(f"No body_*.obj in {seq}")
    rv,rf=read_obj(REF); records=[]; ok=True; prev=None
    for i,p in enumerate(frames):
        v,f=read_obj(p)
        topo=(v.shape==rv.shape and f.shape==rf.shape and np.array_equal(f,rf))
        finite=bool(np.isfinite(v).all())
        step=None if prev is None else float(np.linalg.norm(v-prev,axis=1).max())
        records.append({"index":i,"file":str(p),"vertices":int(len(v)),"faces":int(len(f)),"topology_identity":bool(topo),"finite":finite,"max_vertex_step_m":step})
        ok &= topo and finite; prev=v
    out={"status":"PASS" if ok else "FAIL","fps":args.fps,"reference":str(REF),"reference_vertices":int(len(rv)),"reference_faces":int(len(rf)),"frames":records}
    (seq/"motion_sequence_manifest.json").write_text(json.dumps(out,indent=2,ensure_ascii=True),encoding="utf-8")
    print(json.dumps({"status":out["status"],"frames":len(records),"reference_vertices":len(rv),"reference_faces":len(rf)},indent=2))
    raise SystemExit(0 if ok else 2)
if __name__=="__main__": main()
