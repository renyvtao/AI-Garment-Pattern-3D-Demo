#!/usr/bin/env python3
"""Rebuild one completed job manifest and ZIP after an authorized repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "pipeline"))
    from app_service import JobStore, Pipeline, tree_size

    store = JobStore(args.data_root)
    row = store.row(args.job_id)
    if row is None:
        raise SystemExit(f"job not found: {args.job_id}")
    if row["state"] not in {"completed", "failed", "cancelled"}:
        raise SystemExit(f"job is not safe to rebundle: {row['state']}")

    pipeline = Pipeline(project_root, store)
    config = json.loads(row["config_json"])
    dxf_manifest = pipeline.export_job_dxfs(store.job_root(args.job_id))
    pipeline.write_result_manifest(args.job_id, config)
    bundle = pipeline.make_bundle(args.job_id)
    size = tree_size(store.job_root(args.job_id))
    store.update(args.job_id, size_bytes=size)
    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "dxf_export_count": dxf_manifest["export_count"],
                "bundle": str(bundle),
                "size_bytes": size,
            }
        )
    )


if __name__ == "__main__":
    main()
