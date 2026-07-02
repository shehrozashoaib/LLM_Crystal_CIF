#!/bin/bash
# ============================================================================
# run_datamatched.sh — §4.3.1 variant: data composition matched to the winning
# step split. Winning point was k=1000 (MP-20 1000 steps -> MPTS-52 3500 steps,
# = 2:7). Here the DATA ratio is also 2:7: MP-20 subsampled to 7,823 (= 2/7 of
# the full 27,380 MPTS-52), MPTS-52 kept full. Net: both phases run ~4.1 epochs.
# Only change vs k=1000 is the MP-20 pool size (24,154 -> 7,823).
#   Phase 1: MP-20 (subsampled) 1000 steps ; Phase 2: MPTS-52 (full) 3500 steps.
# Resumable.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

SEED=3407; LORA_R=32; LORA_ALPHA=64
RET_SEQS=10; MAX_NEW_TOKENS=3072
TEST_CSV=Data/source/mp_52_test.csv.gz; TEST_N=8096
GPU_MEM_UTIL=0.90; GEN_CHUNK=1000
MP20_SUB=Data/curriculum/train_phase_mp20_sub2of7.csv.gz
MP20_VAL=Data/curriculum/val_mp20.csv
MP52_TRAIN=Data/curriculum/train_phase_mp52.csv; MP52_VAL=Data/curriculum/val_mp52.csv
PY=/venv/py312/bin/python; VLLM_PY=/venv/vllm/bin/python
SFT=code_FineTune.py; GEN=generate_cifs_vllm.py; VAL=cif_structure_validator_mp52.py
LOG_DIR=logs; GEN_ROOT=generated; mkdir -p "$LOG_DIR" "$GEN_ROOT"
P1=psplit_datamatch_p1; RUN=psplit_datamatch

echo "=== Data-matched phase split (2:7 data + 2:7 steps) ==="

# Phase 1: MP-20 subsampled, 1000 steps
if [ -n "$(ls -A experiments/$P1/final_model/model 2>/dev/null)" ]; then echo "[$P1] exists -> skip"; else
  echo "[$P1] MP-20 (subsampled 7823) 1000 steps"
  "$PY" "$SFT" --train_csv "$MP20_SUB" --val_csv "$MP20_VAL" --run_name "$P1" \
    --max_steps 1000 --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
    2>&1 | tee "${LOG_DIR}/${P1}_train.log"
fi

# Phase 2: MPTS-52 full, 3500 steps, fork phase-1
md="experiments/$RUN/final_model/model"; tok="experiments/$RUN/final_model/tokenizer"
if [ -n "$(ls -A "$md" 2>/dev/null)" ]; then echo "[$RUN] exists -> skip"; else
  echo "[$RUN] MPTS-52 (full) 3500 steps, continue $P1"
  "$PY" "$SFT" --train_csv "$MP52_TRAIN" --val_csv "$MP52_VAL" --run_name "$RUN" \
    --max_steps 3500 --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
    --init_adapter "experiments/$P1/final_model/model" 2>&1 | tee "${LOG_DIR}/${RUN}_train.log"
fi

# Generate + validate on 8096 test set
gdir="${GEN_ROOT}/${RUN}"
gcsv="${gdir}/generated_cifs_${RUN}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"
if [ -f "$gcsv" ]; then echo "[$RUN][gen] exists -> skip"; else
  echo "[$RUN][gen] vLLM ${RET_SEQS} CIFs/prompt on $TEST_CSV (N=$TEST_N)"; mkdir -p "$gdir"
  "$VLLM_PY" "$GEN" --model_dir "$md" --tokenizer_dir "$tok" --csv_path "$TEST_CSV" \
    --output_dir "$gdir" --start_ix 0 --stop_ix "$TEST_N" --ret_seqs "$RET_SEQS" \
    --max_new_tokens "$MAX_NEW_TOKENS" --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
    2>&1 | tee "${LOG_DIR}/${RUN}_gen.log"
fi
vdone="$(find "experiments/$RUN" -path "*validation_*${RUN}*" -name "*summary*.txt" 2>/dev/null | grep -v with_failures | head -1 || true)"
if [ -n "$vdone" ]; then echo "[$RUN][val] exists -> skip"; else
  echo "[$RUN][val] validating"
  "$PY" "$VAL" --input_csv "$gcsv" --output_dir "experiments/$RUN" --dataset mp52 \
    --model "$RUN" --precision 16bit --seq_num "$RET_SEQS" 2>&1 | tee "${LOG_DIR}/${RUN}_val.log"
fi
echo "=== Data-matched run complete ==="
