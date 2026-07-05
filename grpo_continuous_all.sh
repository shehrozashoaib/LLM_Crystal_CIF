#!/bin/bash
# grpo_continuous_all.sh — GRPO's last shot: continuous reward + group 8.
# Trains under a crash-resume retry loop, then auto-evals (final + best) on the
# full 8,096 MPTS-52 test set with vLLM — same pipeline as everything else.
set -u
cd /home/ubuntu/LLM_Crystal_CIF
PY=/venv/py312/bin/python; VLLM_PY=/venv/vllm/bin/python
EXP=experiments/grpo_r32_from3000_continuous
RUNLOG=logs/grpo_r32_from3000_continuous.log; WLOG=logs/grpo_continuous_watchdog.log
TEST_CSV=Data/source/mp_52_test.csv.gz; TEST_N=8096
mkdir -p logs
log(){ echo "[cont $(date -u +%FT%TZ)] $*" >> "$WLOG"; }

# ---- 1. train (auto-resume from newest checkpoint on any crash) ----
for i in $(seq 1 10); do
  log "train attempt $i"
  PYTORCH_ALLOC_CONF=expandable_segments:True $PY grpo_r32_from3000_continuous.py >> "$RUNLOG" 2>&1
  rc=$?; log "train exited rc=$rc"
  [ "$rc" -eq 0 ] && { log "training complete"; break; }
  sleep 30
done

# ---- 2. eval final + best on the full test set ----
eval_model(){  # label parent_dir
  local label="$1" parent="$2" md="$2/model" tok="$2/tokenizer"
  [ -n "$(ls -A "$md" 2>/dev/null)" ] || { log "[$label] no model -> skip"; return; }
  [ -d "$tok" ] || tok="$md"
  local gdir="generated/grpo_cont_${label}"; mkdir -p "$gdir"
  local gcsv; gcsv=$(ls "$gdir"/generated_cifs_*_10seq_maxtok3072_0_$((TEST_N-1)).csv 2>/dev/null|head -1)
  if [ -z "$gcsv" ]; then
    log "[$label] vLLM gen on $TEST_CSV"
    $VLLM_PY generate_cifs_vllm.py --model_dir "$md" --tokenizer_dir "$tok" --csv_path "$TEST_CSV" \
      --output_dir "$gdir" --start_ix 0 --stop_ix "$TEST_N" --ret_seqs 10 --max_new_tokens 3072 \
      --gpu_mem_util 0.90 --chunk 1000 >> "$RUNLOG" 2>&1
    gcsv=$(ls "$gdir"/generated_cifs_*_0_$((TEST_N-1)).csv 2>/dev/null|head -1)
  fi
  [ -n "$gcsv" ] || { log "[$label] no gen csv"; return; }
  log "[$label] validating -> $EXP/eval_${label}"
  $PY cif_structure_validator_mp52.py --input_csv "$gcsv" --output_dir "$EXP/eval_${label}" \
    --dataset mp52 --model "grpo_cont_${label}" --precision 16bit --seq_num 10 >> "$RUNLOG" 2>&1
  log "[$label] DONE."
}
eval_model final "$EXP/final_model"
eval_model best  "$EXP/best_model"
log "ALL DONE (continuous GRPO + eval)."
