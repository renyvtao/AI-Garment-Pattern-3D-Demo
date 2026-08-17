from pathlib import Path
import argparse, json, shutil

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",required=True,type=Path); args=ap.parse_args(); repo=args.repo.resolve()
    report_path=repo/"K62_3D_OVERLAY_INSTALL_REPORT.json"
    if not report_path.is_file(): raise FileNotFoundError(report_path)
    report=json.loads(report_path.read_text(encoding="utf-8"))
    created=set(report.get("created_new",[])); restored=[]; removed=[]; missing=[]
    for s in report.get("touched",[]):
        dst=Path(s)
        bak=dst.with_suffix(dst.suffix+".pre_k62_3d.bak")
        if bak.is_file():
            shutil.copy2(bak,dst); restored.append(str(dst))
        elif str(dst) in created and dst.exists():
            dst.unlink(); removed.append(str(dst))
        else:
            missing.append(str(dst))
    out={"status":"PASS" if not missing else "PARTIAL","restored":restored,"removed_created_files":removed,"unresolved":missing}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    raise SystemExit(0 if not missing else 2)
if __name__=="__main__": main()
