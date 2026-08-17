import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Drape and render ChatGarment pattern specifications.")
    parser.add_argument("--garmentcode-root", required=True)
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--spec-list")
    parser.add_argument("--config", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--body", default="mean_all")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def load_specs(args):
    specs = list(args.spec)
    if args.spec_list:
        with open(args.spec_list, "r", encoding="utf-8") as handle:
            specs.extend(json.load(handle))

    normalized = []
    for spec in specs:
        path = Path(spec).resolve()
        if not path.is_file() and "/GarmentCodeRC/runs/" in str(path):
            chatgarment_path = Path(str(path).replace("/GarmentCodeRC/runs/", "/ChatGarment/runs/"))
            if chatgarment_path.is_file():
                path = chatgarment_path
        normalized.append(str(path))
    return normalized


def main():
    args = parse_args()
    garmentcode_root = str(Path(args.garmentcode_root).resolve())
    sys.path.insert(0, garmentcode_root)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    import pygarment.data_config as data_config
    from pygarment.meshgen.sim_config import PathCofig

    results = []
    for raw_spec in load_specs(args):
        spec_path = Path(raw_spec)
        garment_name, _, _ = spec_path.stem.rpartition("_specification")
        if not garment_name:
            garment_name, _, _ = spec_path.stem.rpartition("_")

        started = time.time()
        result = {"spec": str(spec_path), "garment_name": garment_name, "status": "running"}
        result_dir = spec_path.parent / garment_name
        expected_outputs = {
            "mesh": result_dir / f"{garment_name}_sim.obj",
            "render_front": result_dir / f"{garment_name}_render_front.png",
            "render_back": result_dir / f"{garment_name}_render_back.png",
        }

        if args.skip_completed and all(path.is_file() for path in expected_outputs.values()):
            result.update(
                status="completed",
                elapsed_seconds=0,
                output_dir=str(result_dir),
                output_files={name: str(path) for name, path in expected_outputs.items()},
                reused_existing=True,
            )
            results.append(result)
            write_summary(args.summary, results)
            print(f"[SKIP] completed outputs already exist for {garment_name}", flush=True)
            continue

        try:
            props = data_config.Properties(args.config)
            props.set_section_stats(
                "sim",
                fails={},
                sim_time={},
                spf={},
                fin_frame={},
                body_collisions={},
                self_collisions={},
            )
            props.set_section_stats("render", render_time={})
            paths = PathCofig(
                in_element_path=spec_path.parent,
                out_path=spec_path.parent,
                in_name=garment_name,
                body_name=args.body,
                # A non-empty samples name tells GarmentCodeRC to read the
                # task-local body_measurements.yaml next to the specification.
                # default_body=True deliberately keeps the validated standard
                # collision OBJ while preserving the task measurements.
                samples_name="task_measurements",
                default_body=True,
                smpl_body=False,
                add_timestamp=False,
                system_path=args.system,
                easy_texture_path="",
            )

            print(f"[MESH] {spec_path}", flush=True)
            garment_box_mesh = BoxMesh(paths.in_g_spec, props["sim"]["config"]["resolution_scale"])
            garment_box_mesh.load()
            garment_box_mesh.serialize(
                paths,
                store_panels=False,
                uv_config=props["render"]["config"]["uv_texture"],
            )
            props.serialize(paths.element_sim_props)

            print(f"[SIM] {garment_name}", flush=True)
            run_sim(
                garment_box_mesh.name,
                props,
                paths,
                save_v_norms=False,
                store_usd=False,
                optimize_storage=False,
                verbose=False,
            )
            props.serialize(paths.element_sim_props)

            missing_outputs = [str(path) for path in expected_outputs.values() if not path.is_file()]
            if missing_outputs:
                raise RuntimeError(f"Expected outputs were not generated: {missing_outputs}")

            result.update(
                status="completed",
                elapsed_seconds=round(time.time() - started, 3),
                output_dir=str(result_dir),
                simulation_stats=props["sim"]["stats"],
                render_stats=props["render"]["stats"],
                output_files={name: str(path) for name, path in expected_outputs.items()},
                reused_existing=False,
            )
        except BaseException as exc:
            result.update(
                status="failed",
                elapsed_seconds=round(time.time() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"[FAILED] {spec_path}: {result['error']}", flush=True)

        results.append(result)
        write_summary(args.summary, results)

    failed = sum(item["status"] != "completed" for item in results)
    print(f"[SUMMARY] total={len(results)} failed={failed}", flush=True)
    raise SystemExit(1 if failed else 0)


def write_summary(summary_path, results):
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
