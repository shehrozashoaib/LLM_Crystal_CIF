#!/bin/bash
# ============================================================================
# training/run_crystext_grpo.sh — GRPO on Qwen2.5 with the CrysText reward.
# Same crash-resume pattern as training/grpo_run.sh: the trainer auto-resumes from the
# newest checkpoint, so each retry continues where the last left off.
#
#   GRPO_START_MODEL=experiments/rank_r32_s3407/checkpoints/checkpoint-3000 \
#     bash training/run_crystext_grpo.sh
# ============================================================================
set -u
cd "$(dirname "$0")"

PY="${PY:-/venv/py312/bin/python}"
export GRPO_START_MODEL="${GRPO_START_MODEL:-experiments/rank_r32_s3407/checkpoints/checkpoint-3000}"
export GRPO_START_TOKENIZER="${GRPO_START_TOKENIZER:-$GRPO_START_MODEL}"
export GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-experiments/grpo_crystext_reward}"

# --- CrysText's exact batch geometry (grpo_training.py) --------------------
# per_device_train_batch_size=12, gradient_accumulation_steps=1,
# num_generations=6  ->  12 completions scored per step = 2 crystals x 6 gens.
export GRPO_PDBS="${GRPO_PDBS:-12}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
export GRPO_NUM_GEN="${GRPO_NUM_GEN:-6}"
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-3000}"
export GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
LOG="logs/grpo_crystext_reward.log"
WLOG="logs/grpo_crystext_watchdog.log"
mkdir -p logs

echo "[crystext $(date -u +%FT%TZ)] start model=$GRPO_START_MODEL out=$GRPO_OUTPUT_DIR" >> "$WLOG"
for i in $(seq 1 "${RETRY_MAX:-10}"); do
  echo "[crystext $(date -u +%FT%TZ)] attempt $i" >> "$WLOG"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PY" training/grpo_crystext_reward.py >> "$LOG" 2>&1
  rc=$?
  echo "[crystext $(date -u +%FT%TZ)] exited rc=$rc" >> "$WLOG"
  [ "$rc" -eq 0 ] && { echo "[crystext $(date -u +%FT%TZ)] completed cleanly" >> "$WLOG"; break; }
  sleep 30
done
