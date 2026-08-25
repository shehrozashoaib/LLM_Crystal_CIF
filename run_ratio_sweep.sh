#!/bin/bash
# ============================================================================
# run_ratio_sweep.sh — DIAGONAL (target-emphasis) curriculum sweep
# ----------------------------------------------------------------------------
# For each ratio X:Y the MP-20 : MPTS-52 split is applied identically to the
# DATA budget (24,000 crystals) and the STEP budget (4,500 steps), so every
# phase runs 6.00 epochs — the same per-crystal exposure as the composition
# sweep. Datasets + step splits come from build_ratio_datasets.py's manifest.
#
#   Phase 1: MP-20    (train_phase_mp20_<tag>.csv.gz)  steps.mp20
#   Phase 2: MPTS-52  (train_phase_mp52_<tag>.csv.gz)  steps.mp52, forked from P1
#   then generate 10 CIFs/prompt on the full 8,096 MPTS-52 test set, then validate.
#
# Usage:  ./run_ratio_sweep.sh 3:4 4:3
# Resumable: any stage whose output already exists is skipped.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

SEED="${SEED:-3407}"; LORA_R="${LORA_R:-32}"; LORA_ALPHA="${LORA_ALPHA:-64}"
# Microbatch shape is a pure throughput knob: MICRO_BATCH * GRAD_ACCUM is held at
# the matched effective batch of 32, so the budget control is untouched.
MICRO_BATCH="${MICRO_BATCH:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-8}"
# Attention backend. gh200 pins cuDNN SDPA ahead of xformers — the backend every
# committed GH200 result used. On sm_120 the stock xformers path segfaults.
ARCH="${ARCH:-auto}"
# Checkpoint cadence: must be tighter than the mean time-to-crash, otherwise a
# retry resumes from the same checkpoint forever (livelock).
SAVE_STEPS="${SAVE_STEPS:-100}"
# Eval is LOGGING ONLY here (no early stopping, no load_best_model_at_end), so
# skipping it changes no weights — it only removes a crash surface and its cost.
NO_EVAL="${NO_EVAL:-1}"
# Unsloth's async H2D double-buffered backward is the prime suspect for the
# segfaults seen on this sm_120 card; off by default until proven innocent.
export UNSLOTH_DISABLE_DOUBLE_BUFFER="${UNSLOTH_DISABLE_DOUBLE_BUFFER:-1}"
# vLLM's FlashInfer sampler JIT-compiles kernels and needs CUDA >= 12.9 for
# sm_120; this box has toolkit 12.8, so it dies with a misleading "requires sm75"
# error. Fall back to vLLM's native PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
[ $((MICRO_BATCH * GRAD_ACCUM)) -eq 32 ] || { echo "[FATAL] MICRO_BATCH*GRAD_ACCUM must be 32 (got $((MICRO_BATCH*GRAD_ACCUM)))"; exit 1; }
RET_SEQS="${RET_SEQS:-10}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"; TEST_N="${TEST_N:-8096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"; GEN_CHUNK="${GEN_CHUNK:-1000}"
DATA_DIR="${DATA_DIR:-Data/ratio_sweep}"
MANIFEST="${MANIFEST:-$DATA_DIR/manifest_ratio_sweep.json}"
MP20_VAL="${MP20_VAL:-Data/curriculum/val_mp20.csv.gz}"
MP52_VAL="${MP52_VAL:-Data/curriculum/val_mp52.csv.gz}"
PY="${PYTHON:-/venv/py312/bin/python}"; VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"
SFT=code_FineTune.py; GEN=generate_cifs_vllm.py; VAL=cif_structure_validator_mp52.py
LOG_DIR=logs; GEN_ROOT=generated; mkdir -p "$LOG_DIR" "$GEN_ROOT"

RATIOS=("$@"); [ "${#RATIOS[@]}" -gt 0 ] || { echo "usage: $0 X:Y [X:Y ...]"; exit 1; }

# ---- Preflight -------------------------------------------------------------
for f in "$SFT" "$GEN" "$VAL" "$MANIFEST" "$MP20_VAL" "$MP52_VAL" "$TEST_CSV"; do
  [ -e "$f" ] || { echo "[FATAL] missing: $f"; exit 1; }
done
"$PY" -c "import unsloth,trl,pymatgen,seaborn" 2>/dev/null || { echo "[FATAL] py312 libs"; exit 1; }
"$VLLM_PY" -c "import vllm,pandas" 2>/dev/null || { echo "[FATAL] vllm libs"; exit 1; }

# ---- Read one field out of the manifest ------------------------------------
mfield () {  # ratio, jq-ish path e.g. steps.mp20
  "$PY" -c "
import json,sys
m=json.load(open('$MANIFEST'))['runs'].get('$1')
if m is None: sys.exit('[FATAL] ratio $1 not in $MANIFEST — run build_ratio_datasets.py $1')
d=m
for k in '$2'.split('.'): d=d[k]
print(d)"
}

run_one () {
  local ratio="$1"
  local v20=(--val_csv "$MP20_VAL") v52=(--val_csv "$MP52_VAL")
  [ "$NO_EVAL" = "1" ] && { v20=(); v52=(); }
  local tag s20 s52 f20 f52 p1 run
  tag="$(mfield "$ratio" tag)"
  s20="$(mfield "$ratio" steps.mp20)"; s52="$(mfield "$ratio" steps.mp52)"
  f20="$(mfield "$ratio" files.mp20)"; f52="$(mfield "$ratio" files.mp52)"
  p1="ratio_${tag}_p1"; run="ratio_${tag}"

  echo "######## RATIO $ratio  (MP-20 ${s20} steps -> MPTS-52 ${s52} steps)  run=$run ########"

  # -- Phase 1: MP-20 --------------------------------------------------------
  if [ -n "$(ls -A "experiments/$p1/final_model/model" 2>/dev/null)" ]; then
    echo "[$p1] exists -> skip"
  else
    echo "[$p1] MP-20 phase: $s20 steps on $f20"
    "$PY" "$SFT" --train_csv "$f20" "${v20[@]}" --run_name "$p1" \
      --max_steps "$s20" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
      --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
      --save_steps "$SAVE_STEPS" --arch "$ARCH" \
      2>&1 | tee "${LOG_DIR}/${p1}_train.log"
  fi

  # -- Phase 2: MPTS-52, forked from phase 1 ---------------------------------
  local md="experiments/$run/final_model/model" tok="experiments/$run/final_model/tokenizer"
  if [ -n "$(ls -A "$md" 2>/dev/null)" ]; then
    echo "[$run] exists -> skip"
  else
    echo "[$run] MPTS-52 phase: $s52 steps on $f52, forked from $p1"
    "$PY" "$SFT" --train_csv "$f52" "${v52[@]}" --run_name "$run" \
      --max_steps "$s52" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
      --init_adapter "experiments/$p1/final_model/model" \
      --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
      --save_steps "$SAVE_STEPS" --arch "$ARCH" \
      2>&1 | tee "${LOG_DIR}/${run}_train.log"
  fi

  # -- Generate --------------------------------------------------------------
  local gdir="${GEN_ROOT}/${run}"
  local gcsv="${gdir}/generated_cifs_${run}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"
  # A file existing is NOT proof of completion — a crashed run can leave a partial
  # CSV, and validating that would silently grade an incomplete test set.
  local grows=0
  [ -f "$gcsv" ] && grows=$("$PY" -c "
import csv,sys; csv.field_size_limit(10**9)
try:
    fh=open('$gcsv'); r=csv.reader(fh); next(r,None); print(sum(1 for _ in r))
except Exception: print(0)")
  if [ "$grows" -eq "$TEST_N" ]; then echo "[$run][gen] complete ($grows rows) -> skip"; else
    [ "$grows" -gt 0 ] && echo "[$run][gen] PARTIAL ($grows/$TEST_N rows) -> regenerating"
    echo "[$run][gen] vLLM ${RET_SEQS} CIFs/prompt on $TEST_CSV (N=$TEST_N)"; mkdir -p "$gdir"
    "$VLLM_PY" "$GEN" --model_dir "$md" --tokenizer_dir "$tok" --csv_path "$TEST_CSV" \
      --output_dir "$gdir" --start_ix 0 --stop_ix "$TEST_N" --ret_seqs "$RET_SEQS" \
      --max_new_tokens "$MAX_NEW_TOKENS" --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
      2>&1 | tee "${LOG_DIR}/${run}_gen.log"
  fi

  # -- Validate --------------------------------------------------------------
  local vdone; vdone="$(find "experiments/$run" -path "*validation_*${run}*" -name "*summary*.txt" 2>/dev/null | grep -v with_failures | head -1 || true)"
  if [ -n "$vdone" ]; then echo "[$run][val] exists -> skip"; else
    echo "[$run][val] validating"
    "$PY" "$VAL" --input_csv "$gcsv" --output_dir "experiments/$run" --dataset mp52 \
      --model "$run" --precision 16bit --seq_num "$RET_SEQS" 2>&1 | tee "${LOG_DIR}/${run}_val.log"
  fi
  echo "[$run] DONE."
}

for r in "${RATIOS[@]}"; do run_one "$r"; done
echo "=== Ratio sweep complete: ${RATIOS[*]} ==="
