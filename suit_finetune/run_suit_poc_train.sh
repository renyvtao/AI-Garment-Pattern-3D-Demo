#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AI_GARMENT_PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
CHATGARMENT_ROOT="$PROJECT_ROOT/ChatGarment"
SUIT_DATA_ROOT="${SUIT_DATA_ROOT:-$PROJECT_ROOT/suit_finetune/prepared_data}"
LLAVA_MODEL="${LLAVA_MODEL:-$PROJECT_ROOT/models/llava-v1.5-7b-4481d270}"
CLIP_MODEL="${CLIP_MODEL:-$PROJECT_ROOT/cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1}"
TASK_CHECKPOINT="${TASK_CHECKPOINT:-$CHATGARMENT_ROOT/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin}"

source "$PROJECT_ROOT/venv/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$CHATGARMENT_ROOT"
deepspeed llava/train/train_garmentcode_outfit_suit_poc.py \
  --suit_poc_mode True \
  --suit_init_checkpoint "$TASK_CHECKPOINT" \
  --suit_exp_name suit_poc_lora_v1 \
  --suit_epochs "${SUIT_EPOCHS:-1}" \
  --suit_steps_per_epoch "${SUIT_STEPS_PER_EPOCH:-220}" \
  --suit_batch_size 1 \
  --suit_grad_accumulation_steps "${SUIT_GRAD_ACCUMULATION_STEPS:-8}" \
  --suit_learning_rate "${SUIT_LEARNING_RATE:-2e-5}" \
  --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "$LLAVA_MODEL" \
  --version v1 \
  --data_path "$SUIT_DATA_ROOT/train.json" \
  --data_path_eval "$SUIT_DATA_ROOT/validation.json" \
  --image_folder "$SUIT_DATA_ROOT" \
  --vision_tower "$CLIP_MODEL" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/suit_poc_lora_v1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 2e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length "${SUIT_MODEL_MAX_LENGTH:-1024}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none
