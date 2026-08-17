from pathlib import Path
import argparse, json, subprocess, sys
PKG=Path(__file__).resolve().parents[1]
RUN=PKG/"04_RUNNER"

def call(cmd):
    print("\n>>", " ".join(map(str,cmd)))
    subprocess.run([str(x) for x in cmd],check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",required=True,type=Path); args=ap.parse_args(); repo=args.repo.resolve()
    call([sys.executable,RUN/"verify_delivery.py"])
    call([sys.executable,RUN/"run_golden_base_direct20.py","--repo",repo,"--frames","20"])
    last=json.loads((PKG/"REPLAY_OUTPUT/LAST_RUN.json").read_text(encoding="utf-8"))
    frame=Path(last["frame20"])
    material_dir=frame.parent/"render_input_frame20_official_color"
    call([sys.executable,RUN/"restore_official_material.py","--frame",frame,"--output-dir",material_dir,"--name","K62_FRAME0020"])
    render_out=frame.parent/"render_frame20_official_color"
    call([sys.executable,RUN/"render_existing_k62_result.py","--repo",repo,"--result",material_dir,"--output",render_out])
    print("\nFINAL_REPLAY = PASS")
    print("FRAME20 =",frame)
    print("OFFICIAL_COLOR_RENDER =",render_out)
if __name__=="__main__": main()
