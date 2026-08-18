#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AI_GARMENT_PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
PYTHON="${SUIT_EVAL_PYTHON:-$PROJECT_ROOT/venv/bin/python}"
DATA_ROOT="${SUIT_DATA_ROOT:-$PROJECT_ROOT/suit_finetune/prepared_data}"
if [[ -z "${SUIT_LORA:-}" ]]; then
  SUIT_LORA="$PROJECT_ROOT/ChatGarment/checkpoints/suit_poc_lora_v1/suit_lora_state.bin"
  if [[ ! -f "$SUIT_LORA" && -f "$PROJECT_ROOT/ChatGarment/runs/suit_poc_lora_v1/suit_lora_state.bin" ]]; then
    SUIT_LORA="$PROJECT_ROOT/ChatGarment/runs/suit_poc_lora_v1/suit_lora_state.bin"
  fi
fi
EVAL_ROOT="${SUIT_EVAL_ROOT:-$PROJECT_ROOT/evaluation/suit_lora}"
LIMIT="${SUIT_EVAL_LIMIT:-0}"
MAX_NEW_TOKENS="${SUIT_EVAL_MAX_NEW_TOKENS:-1024}"
PATTERN_FLAG=()
if [[ "${SUIT_EVAL_SKIP_PATTERN:-0}" == "1" ]]; then
  PATTERN_FLAG+=(--skip-pattern)
fi

for required in "$PYTHON" "$DATA_ROOT/test.json" "$SUIT_LORA"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required evaluation input: $required" >&2
    exit 1
  fi
done

mkdir -p "$EVAL_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"$PYTHON" "$PROJECT_ROOT/suit_finetune/run_suit_poc_inference.py" \
  --project-root "$PROJECT_ROOT" \
  --dataset-json "$DATA_ROOT/test.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$EVAL_ROOT/official_base" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${PATTERN_FLAG[@]}"

"$PYTHON" "$PROJECT_ROOT/suit_finetune/run_suit_poc_inference.py" \
  --project-root "$PROJECT_ROOT" \
  --dataset-json "$DATA_ROOT/test.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$EVAL_ROOT/suit_lora" \
  --suit-lora "$SUIT_LORA" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${PATTERN_FLAG[@]}"

"$PYTHON" "$PROJECT_ROOT/suit_finetune/evaluate_suit_outputs.py" \
  --run "official_base=$EVAL_ROOT/official_base" \
  --run "suit_lora=$EVAL_ROOT/suit_lora" \
  --output-dir "$EVAL_ROOT/report"

echo "Evaluation report: $EVAL_ROOT/report/evaluation_report.md"
