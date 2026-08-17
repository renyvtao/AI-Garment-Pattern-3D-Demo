from __future__ import annotations
import importlib, json, sys

REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("trimesh", "trimesh"),
    ("pyrender", "pyrender"),
    ("PIL", "Pillow"),
    ("igl", "libigl"),
]
EXPECTED_WARP = "1.0.0-beta.6"

def main():
    report = {
        "python": sys.version,
        "modules": {},
        "warp": {},
        "status": "PASS",
        "notes": [],
    }
    ok = True
    for mod, package in REQUIRED_MODULES:
        try:
            m = importlib.import_module(mod)
            report["modules"][mod] = {"status": "PASS", "version": getattr(m, "__version__", "UNKNOWN")}
        except Exception as e:
            report["modules"][mod] = {"status": "FAIL", "pip_name": package, "error": repr(e)}
            ok = False
    try:
        import warp as wp
        report["warp"]["version"] = getattr(wp, "__version__", "UNKNOWN")
        report["warp"]["expected_version"] = EXPECTED_WARP
        report["warp"]["version_match"] = report["warp"]["version"] == EXPECTED_WARP
        if not report["warp"]["version_match"]:
            ok = False
        try:
            import warp.collision.panel_assignment  # noqa: F401
            report["warp"]["garmentcode_panel_assignment"] = True
        except Exception as e:
            report["warp"]["garmentcode_panel_assignment"] = False
            report["warp"]["panel_assignment_error"] = repr(e)
            ok = False
        try:
            from warp.sim.integrator_xpbd import replace_mesh_points  # noqa: F401
            report["warp"]["replace_mesh_points"] = True
        except Exception as e:
            report["warp"]["replace_mesh_points"] = False
            report["warp"]["replace_mesh_points_error"] = repr(e)
            ok = False
        wp.init()
        dev = wp.get_device()
        report["warp"]["device"] = str(dev)
        report["warp"]["cuda"] = bool(getattr(dev, "is_cuda", False))
        if not report["warp"]["cuda"]:
            report["notes"].append("No CUDA device selected. The handed-off K62 replay was validated on CUDA; CPU-only execution is not the validated reference environment.")
    except Exception as e:
        report["warp"]["status"] = "FAIL"
        report["warp"]["error"] = repr(e)
        ok = False
    report["status"] = "PASS" if ok else "FAIL"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 2)

if __name__ == "__main__":
    main()
