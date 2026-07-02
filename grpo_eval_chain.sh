#!/bin/bash
# ============================================================================
# grpo_eval_chain.sh — runs the FULL test-set eval automatically once GRPO
# training finishes. Generation uses vLLM (generate_cifs_vllm.py, vllm venv);
# validation uses cif_structure_validator_mp52.py (py312). Grades on the same
# 8,096 MPTS-52 test set, 10 CIFs/prompt — identical pipeline/params to the
# composition/rank/curriculum runs, so the GRPO number drops into the same tables.
#
# Evaluates TWO models:
#   final  — step-1500 final model (matched-budget; the apples-to-apples number)
#   best   — best-val checkpoint (the GRPO ceiling; gap vs final = late-train drift)
# ============================================================================
set -u
cd /home/ubuntu/LLM_Crystal_CIF
PY=/venv/py312/bin/python
VLLM_PY=/venv/vllm/bin/python
EXP=experiments/grpo_r32_from3000_discrete
LOG=logs/grpo_eval_chain.log
TEST_CSV=Data/source/mp_52_test.csv.gz
TEST_N=8096
mkdir -p logs
log(){ echo "[chain $(date -u +%FT%TZ)] $*" >> "$LOG"; }

log "waiting for GRPO training process to finish..."
while pgrep -f "grpo_r32_from3000_discrete.py" >/dev/null 2>&1; do sleep 300; done
# wait for the watchdog too, so we don't start mid-retry
if [ -f logs/grpo_watchdog.pid ]; then
  while kill -0 "$(cat logs/grpo_watchdog.pid)" 2>/dev/null; do sleep 120; done
fi
log "GRPO finished. Pausing 60s for final-model save to flush."
sleep 60

eval_model(){  # label  parent_dir
  local label="$1" parent="$2"
  local md="$parent/model" tok="$parent/tokenizer"
  if [ ! -d "$md" ] || [ -z "$(ls -A "$md" 2>/dev/null)" ]; then
    log "[$label] no model at $md -> skip"; return; fi
  [ -d "$tok" ] || tok="$md"
  local gdir="generated/grpo_${label}"; mkdir -p "$gdir"
  local gcsv; gcsv=$(ls "$gdir"/generated_cifs_*_10seq_maxtok3072_0_$((TEST_N-1)).csv 2>/dev/null | head -1)
  if [ -n "$gcsv" ] && [ -f "$gcsv" ]; then
    log "[$label] generation exists ($gcsv) -> skip gen"
  else
    log "[$label] vLLM: 10 CIFs/prompt on $TEST_CSV (N=$TEST_N) from $md"
    $VLLM_PY generate_cifs_vllm.py --model_dir "$md" --tokenizer_dir "$tok" \
      --csv_path "$TEST_CSV" --output_dir "$gdir" --start_ix 0 --stop_ix "$TEST_N" \
      --ret_seqs 10 --max_new_tokens 3072 --gpu_mem_util 0.90 --chunk 1000 >> "$LOG" 2>&1
    gcsv=$(ls "$gdir"/generated_cifs_*_10seq_maxtok3072_0_$((TEST_N-1)).csv 2>/dev/null | head -1)
  fi
  if [ -z "$gcsv" ] || [ ! -f "$gcsv" ]; then log "[$label] gen produced no csv -> skip val"; return; fi
  log "[$label] validating (best-of-10 + RMSE panel) -> $EXP/eval_${label}"
  $PY cif_structure_validator_mp52.py --input_csv "$gcsv" --output_dir "$EXP/eval_${label}" \
    --dataset mp52 --model "grpo_r32_${label}" --precision 16bit --seq_num 10 >> "$LOG" 2>&1
  log "[$label] DONE."
}

eval_model final "$EXP/final_model"   # matched-budget, comparable number
eval_model best  "$EXP/best_model"    # best-val ceiling
log "ALL GRPO EVALS COMPLETE."
