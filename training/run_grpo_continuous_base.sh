#!/bin/bash
# ============================================================================
# training/run_grpo_continuous_base.sh — GRPO (continuous reward) from the BASE model
# ----------------------------------------------------------------------------
# 1. Train training/grpo_continuous_base.py to exactly MAX_STEPS (default 4500),
#    re-launching after a crash; the trainer resumes from its newest checkpoint.
# 2. Generate 10 CIFs/prompt on the full 8,096 MPTS-52 test set with vLLM
#    (system prompt "sft" = the training prompt; seed = SEED), then validate —
#    the same generate/validate path and settings as the SFT sweeps.
# Every stage is skipped only when its output is verified complete.
#
#   bash training/run_grpo_continuous_base.sh                    # seed 123456
#   SEED=25478 bash training/run_grpo_continuous_base.sh         # another seed
#   GRPO_SMOKE=1 MAX_STEPS=2 TEST_N=8 bash training/run_grpo_continuous_base.sh   # wiring test
#   GRPO_REWARD=discrete bash training/run_grpo_continuous_base.sh   # discrete reward, same everything else
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."

SEED="${SEED:-123456}"; MAX_STEPS="${MAX_STEPS:-4500}"
GRPO_REWARD="${GRPO_REWARD:-continuous}"
case "$GRPO_REWARD" in continuous) RTAG=cont;; discrete) RTAG=disc;; *) echo "[FATAL] GRPO_REWARD=$GRPO_REWARD"; exit 1;; esac
RUN_ID="${RUN_ID:-grpo_${RTAG}_base_s${SEED}}"
[ "${GRPO_SMOKE:-0}" = "1" ] && RUN_ID="smoke_${RUN_ID}"
EXP="experiments/$RUN_ID"
TRAIN_RETRIES="${TRAIN_RETRIES:-30}"
RET_SEQS="${RET_SEQS:-10}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.6}"; TOP_P="${TOP_P:-0.9}"; SYSTEM_PROMPT="${SYSTEM_PROMPT:-sft}"
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"; TEST_N="${TEST_N:-8096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"; GEN_CHUNK="${GEN_CHUNK:-1000}"
RESULTS_DIR="${RESULTS_DIR:-results/grpo_${GRPO_REWARD}_base_s${SEED}}"
PY="${PYTHON:-/venv/py312/bin/python}"; VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"
export UNSLOTH_DISABLE_DOUBLE_BUFFER="${UNSLOTH_DISABLE_DOUBLE_BUFFER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export GRPO_SEED="$SEED" GRPO_MAX_STEPS="$MAX_STEPS" GRPO_RUN_ID="$RUN_ID" GRPO_REWARD
mkdir -p logs

trained () {  # training_stats.json written only after global_step == MAX_STEPS
  [ -f "$EXP/final_model/model/adapter_model.safetensors" ] && [ -f "$EXP/training_stats.json" ] &&
  "$PY" -c "import json,sys; d=json.load(open('$EXP/training_stats.json')); sys.exit(0 if d['global_step']==$MAX_STEPS==d['max_steps'] and d['seed']==$SEED and d.get('reward','continuous')=='$GRPO_REWARD' else 1)"
}

# ---- 1. Train (retry on crash; resumes from newest checkpoint) --------------
for i in $(seq 1 "$TRAIN_RETRIES"); do
  if trained; then echo "[$RUN_ID] trained ($MAX_STEPS steps, verified)"; break; fi
  echo "[$RUN_ID] train attempt $i/$TRAIN_RETRIES ($(date -u +%FT%TZ))"
  "$PY" training/grpo_continuous_base.py 2>&1 | tee -a "logs/${RUN_ID}_train.log" || true
  trained || sleep 30
done
trained || { echo "[FATAL] $RUN_ID not trained after $TRAIN_RETRIES attempts"; exit 1; }
"$PY" - "$EXP" "$SEED" "$GRPO_REWARD" <<'PY'
import json, sys, hashlib, pathlib, datetime
d, seed, reward = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
st = json.load(open(d / "training_stats.json"))
sha = hashlib.sha256(open(d / "final_model/model/adapter_model.safetensors", "rb").read()).hexdigest()[:16]
json.dump({"steps": st["global_step"], "seed": seed, "adapter_sha256": sha, "init_adapter": None,
           "start_model": "unsloth/Qwen2.5-7B-Instruct", "method": f"grpo_{reward}_from_base",
           "finished": datetime.datetime.now().isoformat(timespec="seconds")},
          open(d / "DONE.json", "w"), indent=2)
PY
ASHA="$("$PY" -c "import json; print(json.load(open('$EXP/DONE.json'))['adapter_sha256'])")"

# ---- 2. Generate -------------------------------------------------------------
GDIR="generated/$RUN_ID"; mkdir -p "$GDIR"
GCSV="$GDIR/generated_cifs_${RUN_ID}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"
META="{\"adapter_sha256\": \"$ASHA\", \"seed\": $SEED, \"temperature\": $TEMPERATURE, \"top_p\": $TOP_P, \"ret_seqs\": $RET_SEQS, \"max_new_tokens\": $MAX_NEW_TOKENS, \"system_prompt\": \"$SYSTEM_PROMPT\", \"test_csv\": \"$TEST_CSV\", \"test_n\": $TEST_N}"
if [ -f "$GDIR/gen_meta.json" ]; then
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$GDIR/gen_meta.json'))==json.loads('''$META''') else 1)" \
    || { echo "[FATAL] $GDIR holds generations from a different model/settings — move it aside"; exit 1; }
else
  if [ -n "$(ls -A "$GDIR")" ]; then echo "[FATAL] $GDIR non-empty without gen_meta.json"; exit 1; fi
  echo "$META" > "$GDIR/gen_meta.json"
fi
rows () { "$PY" -c "import pandas as pd
try: print(len(pd.read_csv('$GCSV')))
except Exception: print(0)"; }
if [ -f "$GCSV" ] && [ "$(rows)" -eq "$TEST_N" ]; then echo "[$RUN_ID][gen] complete -> skip"; else
  "$VLLM_PY" evaluation/generate_cifs_vllm.py --model_dir "$EXP/final_model/model" \
    --tokenizer_dir "$EXP/final_model/tokenizer" --csv_path "$TEST_CSV" --output_dir "$GDIR" \
    --start_ix 0 --stop_ix "$TEST_N" --ret_seqs "$RET_SEQS" --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" --top_p "$TOP_P" --seed "$SEED" --system_prompt "$SYSTEM_PROMPT" \
    --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" 2>&1 | tee -a "logs/${RUN_ID}_gen.log"
  [ "$(rows)" -eq "$TEST_N" ] || { echo "[FATAL] $GCSV incomplete"; exit 1; }
fi
"$PY" -c "
import pandas as pd, sys
g=pd.read_csv('$GCSV'); t=pd.read_csv('$TEST_CSV').iloc[:$TEST_N]
sys.exit(0 if (g.material_id.values==t.material_id.values).all() else '[FATAL] generated rows misaligned with test set')"

# ---- 3. Validate + collect ---------------------------------------------------
VDIR="$EXP/validation_seed${SEED}"
if [ -n "$(find "$VDIR" -name '*_summary.txt' 2>/dev/null | head -1)" ] && [ "$(cat "$VDIR/VALIDATED_FROM" 2>/dev/null)" = "$ASHA" ]; then
  echo "[$RUN_ID][val] exists -> skip"
else
  rm -rf "$VDIR"; mkdir -p "$VDIR"
  "$PY" evaluation/cif_structure_validator_mp52.py --input_csv "$GCSV" --output_dir "$VDIR" --dataset mp52 \
    --model "$RUN_ID" --precision 16bit --seq_num "$RET_SEQS" 2>&1 | tee -a "logs/${RUN_ID}_val.log"
  [ -n "$(find "$VDIR" -name '*_summary.txt' | head -1)" ] || { echo "[FATAL] no validation summary"; exit 1; }
  echo "$ASHA" > "$VDIR/VALIDATED_FROM"
fi
R="$RESULTS_DIR/$RUN_ID"; mkdir -p "$R/validation"
gzip -c "$GCSV" > "$R/predicted_cifs.csv.gz"
cp -r "$(dirname "$(find "$VDIR" -name '*_summary.txt' | head -1)")"/. "$R/validation/"
cp "$GDIR/gen_meta.json" "$R/"; cp "$EXP/DONE.json" "$R/train_DONE.json"
for f in reward_trace.jsonl reward_group_trace.jsonl val_trace.jsonl; do [ -f "$EXP/$f" ] && gzip -c "$EXP/$f" > "$R/$f.gz"; done
grep -h "Success rate" "$R"/validation/*_summary.txt | sed "s/^/[$RUN_ID] /" || true
echo "[$RUN_ID] DONE."
