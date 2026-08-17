from pathlib import Path
import argparse, shutil

PKG=Path(__file__).resolve().parents[1]
GOLDEN=PKG/"01_GOLDEN_BASE/K62_GOLDEN_boxmesh.obj"
MAT=PKG/"06_OFFICIAL_MATERIAL"

def vertices(path):
    return [line for line in path.read_text(encoding="utf-8",errors="ignore").splitlines() if line.startswith("v ")]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--name", default="K62_FRAME0020")
    args=ap.parse_args()
    frame=args.frame.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    newv=vertices(frame)
    glines=GOLDEN.read_text(encoding="utf-8",errors="ignore").splitlines()
    oldn=sum(1 for x in glines if x.startswith("v "))
    if len(newv)!=oldn: raise RuntimeError(f"vertex count mismatch frame={len(newv)} golden={oldn}")
    it=iter(newv); restored=[]
    for line in glines:
        restored.append(next(it) if line.startswith("v ") else line)
    sim=out/f"{args.name}_sim.obj"
    sim.write_text("\n".join(restored)+"\n",encoding="utf-8")
    for n in ["K62BDIR2_material.mtl","K62BDIR2_texture_fabric.png","K62BDIR2_texture.png"]:
        shutil.copy2(MAT/n,out/n)
    print("FRAME_VERTICES =",len(newv))
    print("OFFICIAL_UV_MATERIAL_RESTORED = PASS")
    print("OUTPUT =",sim)
if __name__=="__main__": main()
