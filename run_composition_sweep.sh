#!/bin/bash
# ============================================================================
# run_composition_sweep.sh — COMPOSITION (Data) lever, §4.1
# ============================================================================
# For each MP-20:MPTS-52 ratio, run the full pipeline ONE AT A TIME:
#     1. SFT train   (pinned to MAX_STEPS, no early stopping)
#     2. Generate    (RET_SEQS CIFs per prompt on the MPTS-52 test set)
#     3. Validate    (best-of-N match + RMSE panel)
#
# Every run is graded on the SAME MPTS-52 test set (full 8096 by default; set
# TEST_CSV/TEST_N for the frozen 1000 subset), at identical steps and identical
# training volume (24,000 crystals) — only the MP-20:MPTS-52 ratio moves.
#
# Usage:
#   ./run_composition_sweep.sh                 # all 5 ratios: 00 25 50 75 100
#   ./run_composition_sweep.sh 50              # just the 50:50 run
#   ./run_composition_sweep.sh 25 50 75        # a subset
#   MAX_STEPS=4500 RET_SEQS=10 ./run_composition_sweep.sh
#
# Resumable: a ratio whose validation summary already exists is SKIPPED.
# Re-run a single stage by deleting its output and re-invoking.
# ============================================================================
set -euo pipefail

# ---- Config (override via env) ---------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MAX_STEPS="${MAX_STEPS:-4500}"
RET_SEQS="${RET_SEQS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
# Default: grade on the FULL MPTS-52 test set (8096), not the 1000 subset.
# For a fast sanity run use the frozen subset instead:
#   TEST_CSV=Data/composition_sweep/test_frozen_mp52_1000.csv TEST_N=1000 ./run_composition_sweep.sh
# NOTE: train sets are already leakage-filtered against the ENTIRE mp52_test,
# so using all 8096 is leakage-safe (no train set contains any test material).
TEST_CSV="${TEST_CSV:-mp_52_test_cifs_description_reduced_withmpids.csv}"
TEST_N="${TEST_N:-8096}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
# vLLM generation knobs (continuous batching — no fixed batch size needed)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
GEN_CHUNK="${GEN_CHUNK:-1000}"          # materials per incremental save
# Two envs: py312 (unsloth+trl) for TRAIN/VALIDATE; vllm for GENERATE.
PY="${PYTHON:-/venv/py312/bin/python}"
VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"

DATA_DIR="Data/composition_sweep"
GEN_SCRIPT="generate_cifs_vllm.py"      # vLLM inference (fast); HF version kept as fallback
SFT_SCRIPT="code_FineTune.py"
VAL_SCRIPT="cif_structure_validator_mp52.py"
LOG_DIR="logs"
GEN_ROOT="generated"

# ratios: positional args if given, else the full sweep
if [ "$#" -gt 0 ]; then RATIOS=("$@"); else RATIOS=(00 25 50 75 100); fi

mkdir -p "$LOG_DIR" "$GEN_ROOT"

# ---- Preflight -------------------------------------------------------------
[ -x "$PY" ] || { echo "[FATAL] py312 python not found/executable: $PY"; exit 1; }
"$PY" -c "import unsloth, trl, pymatgen, seaborn" 2>/dev/null \
  || { echo "[FATAL] $PY is missing train/validate libs (need: unsloth, trl, pymatgen, seaborn)"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "[FATAL] vllm python not found/executable: $VLLM_PY"; exit 1; }
"$VLLM_PY" -c "import vllm, pandas" 2>/dev/null \
  || { echo "[FATAL] $VLLM_PY is missing inference libs (need: vllm, pandas)"; exit 1; }
for f in "$SFT_SCRIPT" "$GEN_SCRIPT" "$VAL_SCRIPT" "$TEST_CSV"; do
  [ -e "$f" ] || { echo "[FATAL] missing required file: $f  (did you run build_composition_datasets.py?)"; exit 1; }
done
echo "=== Composition sweep ==="
echo "python=$PY"
echo "ratios=${RATIOS[*]}  max_steps=$MAX_STEPS  ret_seqs=$RET_SEQS  gpu_mem_util=$GPU_MEM_UTIL  chunk=$GEN_CHUNK"
echo "test=$TEST_CSV (N=$TEST_N)   lora_r=$LORA_R alpha=$LORA_ALPHA"
echo

run_one () {
  local tag="$1"
  local run_name="comp_mp20_${tag}"
  local train_csv="${DATA_DIR}/train_mp20_${tag}.csv"
  local val_csv="${DATA_DIR}/val_mp20_${tag}.csv"
  local model_dir="experiments/${run_name}/final_model/model"
  local tok_dir="experiments/${run_name}/final_model/tokenizer"
  local gen_out_dir="${GEN_ROOT}/${run_name}"
  local val_out_dir="experiments/${run_name}"
  # generate_cifs auto-names: generated_cifs_<run_name>_<ret>seq_maxtok<mt>_<start>_<stop-1>.csv
  local gen_csv="${gen_out_dir}/generated_cifs_${run_name}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"

  [ -f "$train_csv" ] || { echo "[skip $tag] no train file $train_csv"; return; }

  echo "############################################################"
  echo "# RATIO ${tag}  (run_name=${run_name})"
  echo "############################################################"

  # ---- 1. TRAIN ----
  if [ -d "$model_dir" ] && [ -n "$(ls -A "$model_dir" 2>/dev/null)" ]; then
    echo "[$tag][train] final model already exists -> skip"
  else
    echo "[$tag][train] SFT (max_steps=$MAX_STEPS) ..."
    "$PY" "$SFT_SCRIPT" \
      --train_csv "$train_csv" \
      --val_csv "$val_csv" \
      --run_name "$run_name" \
      --max_steps "$MAX_STEPS" \
      --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
      2>&1 | tee "${LOG_DIR}/${run_name}_train.log"
  fi

  # ---- 2. GENERATE ----
  if [ -f "$gen_csv" ]; then
    echo "[$tag][gen] $gen_csv exists -> skip"
  else
    echo "[$tag][gen] vLLM generating ${RET_SEQS} CIFs/prompt on test set (N=$TEST_N) ..."
    mkdir -p "$gen_out_dir"
    "$VLLM_PY" "$GEN_SCRIPT" \
      --model_dir "$model_dir" \
      --tokenizer_dir "$tok_dir" \
      --csv_path "$TEST_CSV" \
      --output_dir "$gen_out_dir" \
      --start_ix 0 --stop_ix "$TEST_N" \
      --ret_seqs "$RET_SEQS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
      2>&1 | tee "${LOG_DIR}/${run_name}_gen.log"
  fi

  # ---- 3. VALIDATE ----
  echo "[$tag][val] validating (best-of-N + RMSE panel) ..."
  "$PY" "$VAL_SCRIPT" \
    --input_csv "$gen_csv" \
    --output_dir "$val_out_dir" \
    --dataset mp52 \
    --model "$run_name" \
    --precision 16bit \
    --seq_num "$RET_SEQS" \
    2>&1 | tee "${LOG_DIR}/${run_name}_val.log"

  echo "[$tag] DONE. results under ${val_out_dir}/validation_*_${run_name}/"
  echo
}

for tag in "${RATIOS[@]}"; do
  run_one "$tag"
done

echo "=== Sweep complete for ratios: ${RATIOS[*]} ==="
echo "Compare best-of-N success + RMSE across runs in experiments/comp_mp20_*/validation_*/"
