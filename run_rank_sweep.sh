#!/bin/bash
# ============================================================================
# run_rank_sweep.sh — LoRA RANK (Params) lever, §4.2
# ============================================================================
# Sweep LoRA rank r ∈ {16, 32, 64, 128} (alpha = 2r) on PURE MPTS-52, identical
# budget, identical data — ONLY rank/alpha (and seed) change. Brackets the
# paper's "doubling rank gives only ~+1.2%" claim with a curve + a seed band.
#
# Per (rank, seed): train -> generate (RET_SEQS CIFs/prompt on the MPTS-52 test
# set) -> validate (best-of-N match + RMSE panel). Graded on the SAME 8096
# MPTS-52 test crystals as the composition + curriculum runs, at the SAME
# matched budget (MAX_STEPS=4500). dropout / LR / seq-len are held at the
# code_FineTune.py defaults (0.05 / 1e-4 / 4096) — see §4.2 "Gotchas".
#
# Seeds are looped OUTER so the FIRST pass gives a complete 4-point rank curve;
# the second seed adds the error band. --seed feeds both LoRA init and data
# shuffle order in code_FineTune.py, so the two seeds differ meaningfully.
#
# Usage:
#   ./run_rank_sweep.sh                       # ranks 16 32 64 128, seeds 3407 1234
#   ./run_rank_sweep.sh 32 64                 # just two ranks (both seeds)
#   SEEDS="3407 1234 999" ./run_rank_sweep.sh # 3 seeds
#   TEST_CSV=Data/composition_sweep/test_frozen_mp52_1000.csv.gz TEST_N=1000 \
#       ./run_rank_sweep.sh                   # fast frozen-1000 grading
#
# Resumable: a (rank,seed) stage whose output already exists is SKIPPED.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- Config (override via env) ---------------------------------------------
MAX_STEPS="${MAX_STEPS:-4500}"            # matched budget (composition/curriculum)
RET_SEQS="${RET_SEQS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
RANKS=("$@"); [ "${#RANKS[@]}" -gt 0 ] || RANKS=(16 32 64 128)
read -r -a SEEDS <<<"${SEEDS:-3407 1234}"  # 2 seeds by default

# Pure-MPTS-52 training pool (= composition ratio 00 = 0% MP-20), 24k crystals.
TRAIN_CSV="${TRAIN_CSV:-Data/composition_sweep/train_mp20_00.csv.gz}"
VAL_CSV="${VAL_CSV:-Data/composition_sweep/val_mp20_00.csv.gz}"
# Main metric: full MPTS-52 test set — the SAME 8096 crystals (verified id-for-id)
# the composition + curriculum runs are graded on.
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"
TEST_N="${TEST_N:-8096}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
GEN_CHUNK="${GEN_CHUNK:-1000}"

PY="${PYTHON:-/venv/py312/bin/python}"
VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"
SFT_SCRIPT="code_FineTune.py"
GEN_SCRIPT="generate_cifs_vllm.py"
VAL_SCRIPT="cif_structure_validator_mp52.py"
LOG_DIR="logs"; GEN_ROOT="generated"
mkdir -p "$LOG_DIR" "$GEN_ROOT"

# ---- Preflight -------------------------------------------------------------
[ -x "$PY" ] || { echo "[FATAL] py312 python not found: $PY"; exit 1; }
"$PY" -c "import unsloth, trl, pymatgen, seaborn" 2>/dev/null \
  || { echo "[FATAL] $PY missing train/validate libs (unsloth, trl, pymatgen, seaborn)"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "[FATAL] vllm python not found: $VLLM_PY"; exit 1; }
"$VLLM_PY" -c "import vllm, pandas" 2>/dev/null \
  || { echo "[FATAL] $VLLM_PY missing inference libs (vllm, pandas)"; exit 1; }
for f in "$SFT_SCRIPT" "$GEN_SCRIPT" "$VAL_SCRIPT" "$TRAIN_CSV" "$VAL_CSV" "$TEST_CSV"; do
  [ -e "$f" ] || { echo "[FATAL] missing required file: $f"; exit 1; }
done

echo "=== Rank sweep ==="
echo "ranks=${RANKS[*]}  seeds=${SEEDS[*]}  max_steps=$MAX_STEPS  ret_seqs=$RET_SEQS"
echo "train=$TRAIN_CSV  test=$TEST_CSV (N=$TEST_N)"
echo

run_one () {  # rank seed
  local r="$1" seed="$2"
  local alpha=$(( r * 2 ))                 # alpha = 2r (§4.2)
  local run="rank_r${r}_s${seed}"
  local md="experiments/${run}/final_model/model"
  local tok="experiments/${run}/final_model/tokenizer"
  local gen_dir="${GEN_ROOT}/${run}"
  local gen_csv="${gen_dir}/generated_cifs_${run}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"

  echo "############################################################"
  echo "# RANK r=${r} (alpha=${alpha})  SEED=${seed}   run=${run}"
  echo "############################################################"

  # ---- 1. TRAIN ----
  if [ -d "$md" ] && [ -n "$(ls -A "$md" 2>/dev/null)" ]; then
    echo "[$run][train] final model exists -> skip"
  else
    echo "[$run][train] SFT r=$r alpha=$alpha seed=$seed max_steps=$MAX_STEPS"
    "$PY" "$SFT_SCRIPT" \
      --train_csv "$TRAIN_CSV" --val_csv "$VAL_CSV" --run_name "$run" \
      --max_steps "$MAX_STEPS" --lora_r "$r" --lora_alpha "$alpha" --seed "$seed" \
      2>&1 | tee "${LOG_DIR}/${run}_train.log"
  fi

  # ---- 2. GENERATE ----
  if [ -f "$gen_csv" ]; then
    echo "[$run][gen] $gen_csv exists -> skip"
  else
    echo "[$run][gen] vLLM ${RET_SEQS} CIFs/prompt on $TEST_CSV (N=$TEST_N)"
    mkdir -p "$gen_dir"
    "$VLLM_PY" "$GEN_SCRIPT" \
      --model_dir "$md" --tokenizer_dir "$tok" \
      --csv_path "$TEST_CSV" --output_dir "$gen_dir" \
      --start_ix 0 --stop_ix "$TEST_N" \
      --ret_seqs "$RET_SEQS" --max_new_tokens "$MAX_NEW_TOKENS" \
      --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
      2>&1 | tee "${LOG_DIR}/${run}_gen.log"
  fi

  # ---- 3. VALIDATE ----
  # NB: trailing `|| true` is required — under `set -o pipefail`, grep -v exits 1
  # when find returns nothing (a run not yet validated), which would otherwise
  # trip `set -e` and kill the whole sweep at the val stage of every fresh run.
  local val_done
  val_done="$(find "experiments/${run}" -path "*validation_*${run}*" -name "*summary*.txt" 2>/dev/null \
              | grep -v with_failures | head -1 || true)"
  if [ -n "$val_done" ]; then
    echo "[$run][val] summary exists -> skip ($val_done)"
  else
    echo "[$run][val] validating (best-of-N + RMSE panel)"
    "$PY" "$VAL_SCRIPT" \
      --input_csv "$gen_csv" --output_dir "experiments/${run}" \
      --dataset mp52 --model "$run" --precision 16bit --seq_num "$RET_SEQS" \
      2>&1 | tee "${LOG_DIR}/${run}_val.log"
  fi
  echo "[$run] DONE."
  echo
}

# Seeds OUTER -> first pass = complete rank curve; second seed = error band.
for seed in "${SEEDS[@]}"; do
  for r in "${RANKS[@]}"; do
    run_one "$r" "$seed"
  done
done

echo "=== Rank sweep complete: ranks=${RANKS[*]} seeds=${SEEDS[*]} ==="
echo "Compare best-of-N success + RMSE across experiments/rank_r*_s*/validation_*/"
