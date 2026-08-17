#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AI_GARMENT_PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
RUN="$ROOT/ChatGarment/runs/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/example_imgs_img_recon"

cd "$ROOT/GarmentCodeRC"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="$ROOT/GarmentCodeRC"
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$ROOT/outputs"

"$ROOT/venv/bin/python" "$ROOT/garment_sim_runner.py" \
  --garmentcode-root "$ROOT/GarmentCodeRC" \
  --spec-list "$RUN/vis_new/all_json_spec_files.json" \
  --config "$ROOT/GarmentCodeRC/assets/Sim_props/default_sim_props.yaml" \
  --system "$ROOT/GarmentCodeRC/system.json" \
  --summary "$ROOT/outputs/batch_sim_summary.json" \
  --skip-completed
