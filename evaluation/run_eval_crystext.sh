#!/bin/bash
# ============================================================================
# evaluation/run_eval_crystext.sh — evaluate a GRPO-with-CrysText-reward checkpoint on the
# full MPTS-52 test set: vLLM generation -> pymatgen/StructureMatcher validation.
# Mirrors the eval_model() stage of training/grpo_continuous_all.sh.
#
#   bash evaluation/run_eval_crystext.sh best        # experiments/grpo_crystext_reward/best_model
#   bash evaluation/run_eval_crystext.sh final
#   SYSTEM_PROMPT=sft bash evaluation/run_eval_crystext.sh best
#
# PROMPT: defaults to `grpo` — the system message the grpo_*.py trainers
# actually optimise under. The repo's other evals in results/ used `sft`
# (training/code_FineTune.py's message), which is a DIFFERENT prompt; pass
# SYSTEM_PROMPT=sft for a number comparable with those.
#
# FLASHINFER on Blackwell (sm120): flashinfer resolves the CUDA version from
# the system nvcc (12.8 here) rather than torch's bundled cu130, and refuses
# sm12.x under CUDA < 12.9. vLLM then reports the misleading error
# "FlashInfer requires GPUs with sm75 or higher". Routing attention to
# FLASH_ATTN and the sampler off flashinfer avoids it entirely.
# ============================================================================
set -u
cd "$(dirname "$0")"

LABEL="${1:-best}"
EXP="${EXP:-experiments/grpo_crystext_reward}"
PY="${PY:-/venv/py312/bin/python}"
VLLM_PY="${VLLM_PY:-/venv/vllm/bin/python}"
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"
TEST_N="${TEST_N:-8096}"
RET_SEQS="${RET_SEQS:-10}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-grpo}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MD="$EXP/${LABEL}_model/model"
TOK="$EXP/${LABEL}_model/tokenizer"
[ -d "$TOK" ] || TOK="$MD"
GDIR="generated/grpo_crystext_${LABEL}"
LOG="logs/eval_crystext_${LABEL}.log"
mkdir -p "$GDIR" logs

[ -n "$(ls -A "$MD" 2>/dev/null)" ] || { echo "no model at $MD"; exit 1; }

echo "[eval] model=$MD prompt=$SYSTEM_PROMPT n=$TEST_N x $RET_SEQS seqs" | tee -a "$LOG"

# ---- 1. generate with vLLM ----
GCSV=$(ls "$GDIR"/generated_cifs_*_0_$((TEST_N-1)).csv 2>/dev/null | head -1)
if [ -z "$GCSV" ]; then
  "$VLLM_PY" evaluation/generate_cifs_vllm.py \
    --model_dir "$MD" --tokenizer_dir "$TOK" --csv_path "$TEST_CSV" \
    --output_dir "$GDIR" --start_ix 0 --stop_ix "$TEST_N" \
    --ret_seqs "$RET_SEQS" --max_new_tokens 3072 \
    --gpu_mem_util 0.90 --chunk 1000 --system_prompt "$SYSTEM_PROMPT" >> "$LOG" 2>&1
  rc=$?; echo "[eval] generation rc=$rc" | tee -a "$LOG"
  [ "$rc" -eq 0 ] || exit $rc
  GCSV=$(ls "$GDIR"/generated_cifs_*_0_$((TEST_N-1)).csv 2>/dev/null | head -1)
fi
[ -n "$GCSV" ] || { echo "[eval] no generated csv"; exit 1; }

# ---- 2. validate with pymatgen ----
echo "[eval] validating $GCSV" | tee -a "$LOG"
"$PY" evaluation/cif_structure_validator_mp52.py \
  --input_csv "$GCSV" --output_dir "$EXP/eval_${LABEL}" \
  --dataset mp52 --model "grpo_crystext_${LABEL}" --precision 16bit \
  --seq_num "$RET_SEQS" >> "$LOG" 2>&1
echo "[eval] done -> $EXP/eval_${LABEL}" | tee -a "$LOG"
