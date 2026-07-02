#!/bin/bash
# ============================================================================
# run_phase_split_sweep.sh — §4.3.1 phase-split / target-emphasis curriculum
# ============================================================================
# Fixed total budget S=4500 steps. Train ONE MP-20 base run keeping ALL
# checkpoints, then for each fork point k: continue from checkpoint-k on MPTS-52
# for (4500-k) steps -> generate -> validate. Only the MP-20->MPTS-52 SWITCH
# POINT moves; pools held fixed. Endpoints already exist (no retrain):
#   k=0    pure MPTS-52  = comp_mp20_00   = 30.1%
#   k≈2109 forward       = curr_fwd       = 30.7%
#   k=4500 pure MP-20    = comp_mp20_100  = 26.6%
#
# Usage:  ./run_phase_split_sweep.sh            # forks 1000 3000 (default)
#         ./run_phase_split_sweep.sh 1500 2500  # other fork points
# Resumable: a stage whose output exists is skipped.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

S_TOTAL="${S_TOTAL:-4500}"
BASE_STEPS="${BASE_STEPS:-3000}"          # MP-20 base; must be >= max fork k
SEED="${SEED:-3407}"
LORA_R="${LORA_R:-32}"; LORA_ALPHA="${LORA_ALPHA:-64}"
RET_SEQS="${RET_SEQS:-10}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"; TEST_N="${TEST_N:-8096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"; GEN_CHUNK="${GEN_CHUNK:-1000}"
MP20_TRAIN="Data/curriculum/train_phase_mp20.csv"; MP20_VAL="Data/curriculum/val_mp20.csv"
MP52_TRAIN="Data/curriculum/train_phase_mp52.csv"; MP52_VAL="Data/curriculum/val_mp52.csv"
PY="${PYTHON:-/venv/py312/bin/python}"; VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"
SFT=code_FineTune.py; GEN=generate_cifs_vllm.py; VAL=cif_structure_validator_mp52.py
LOG_DIR=logs; GEN_ROOT=generated; mkdir -p "$LOG_DIR" "$GEN_ROOT"
BASE_RUN=psplit_mp20base

FORKS=("$@"); [ "${#FORKS[@]}" -gt 0 ] || FORKS=(1000 3000)

# ---- Preflight -------------------------------------------------------------
for f in "$SFT" "$GEN" "$VAL" "$MP20_TRAIN" "$MP20_VAL" "$MP52_TRAIN" "$MP52_VAL" "$TEST_CSV"; do
  [ -e "$f" ] || { echo "[FATAL] missing: $f"; exit 1; }
done
"$PY" -c "import unsloth,trl,pymatgen,seaborn" 2>/dev/null || { echo "[FATAL] py312 libs"; exit 1; }
"$VLLM_PY" -c "import vllm,pandas" 2>/dev/null || { echo "[FATAL] vllm libs"; exit 1; }
echo "=== Phase-split sweep ===  forks=${FORKS[*]}  S=$S_TOTAL  base=$BASE_STEPS  seed=$SEED"

# ---- 1. MP-20 base run (keep ALL checkpoints) ------------------------------
base_md="experiments/$BASE_RUN/final_model/model"
if [ -d "$base_md" ] && [ -n "$(ls -A "$base_md" 2>/dev/null)" ]; then
  echo "[$BASE_RUN] base exists -> skip"
else
  echo "[$BASE_RUN] MP-20 base: $BASE_STEPS steps, keep all checkpoints"
  "$PY" "$SFT" --train_csv "$MP20_TRAIN" --val_csv "$MP20_VAL" --run_name "$BASE_RUN" \
    --max_steps "$BASE_STEPS" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
    --save_total_limit 0 2>&1 | tee "${LOG_DIR}/${BASE_RUN}_train.log"
fi

# ---- 2. forks: continue on MPTS-52 for (S - k) steps -----------------------
fork_one () {  # k
  local k="$1" run="psplit_k$1"
  local ckpt="experiments/$BASE_RUN/checkpoints/checkpoint-$k"
  [ -d "$ckpt" ] || ckpt="experiments/$BASE_RUN/final_model/model"   # k==BASE_STEPS
  local steps=$(( S_TOTAL - k ))
  local md="experiments/$run/final_model/model" tok="experiments/$run/final_model/tokenizer"
  local gdir="${GEN_ROOT}/${run}"
  local gcsv="${gdir}/generated_cifs_${run}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"
  echo "######## FORK k=$k  (MP-20 $k -> MPTS-52 $steps)  run=$run ########"

  if [ -d "$md" ] && [ -n "$(ls -A "$md" 2>/dev/null)" ]; then
    echo "[$run][train] exists -> skip"
  else
    echo "[$run][train] continue $ckpt on MPTS-52 for $steps steps"
    "$PY" "$SFT" --train_csv "$MP52_TRAIN" --val_csv "$MP52_VAL" --run_name "$run" \
      --max_steps "$steps" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
      --init_adapter "$ckpt" 2>&1 | tee "${LOG_DIR}/${run}_train.log"
  fi

  if [ -f "$gcsv" ]; then echo "[$run][gen] exists -> skip"; else
    echo "[$run][gen] vLLM ${RET_SEQS} CIFs/prompt on $TEST_CSV (N=$TEST_N)"; mkdir -p "$gdir"
    "$VLLM_PY" "$GEN" --model_dir "$md" --tokenizer_dir "$tok" --csv_path "$TEST_CSV" \
      --output_dir "$gdir" --start_ix 0 --stop_ix "$TEST_N" --ret_seqs "$RET_SEQS" \
      --max_new_tokens "$MAX_NEW_TOKENS" --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
      2>&1 | tee "${LOG_DIR}/${run}_gen.log"
  fi

  local vdone; vdone="$(find "experiments/$run" -path "*validation_*${run}*" -name "*summary*.txt" 2>/dev/null | grep -v with_failures | head -1 || true)"
  if [ -n "$vdone" ]; then echo "[$run][val] exists -> skip"; else
    echo "[$run][val] validating"
    "$PY" "$VAL" --input_csv "$gcsv" --output_dir "experiments/$run" --dataset mp52 \
      --model "$run" --precision 16bit --seq_num "$RET_SEQS" 2>&1 | tee "${LOG_DIR}/${run}_val.log"
  fi
  echo "[$run] DONE."
}

for k in "${FORKS[@]}"; do fork_one "$k"; done
echo "=== Phase-split sweep complete: forks=${FORKS[*]} ==="
