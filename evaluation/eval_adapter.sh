#!/bin/bash
# ============================================================================
# evaluation/eval_adapter.sh — evaluate one trained LoRA run: vLLM generate (10 CIFs per
# prompt on the full 8,096 MPTS-52 test set, system prompt "sft", seed SEED) -> pymatgen
# validation -> collect into RESULTS_DIR/<run>. Same settings and verification as the
# sweep / GRPO launchers (gen_meta guard, row count + alignment with the test set,
# validation tied to the adapter hash). Every stage is skipped only when verified complete.
#
#   bash evaluation/eval_adapter.sh experiments/sft_mp52_s123456_to2000
#   SEED=25478 RESULTS_DIR=results/x bash evaluation/eval_adapter.sh experiments/<run>
#   # another test set for the same adapter (outputs keyed by EVAL_TAG so nothing collides):
#   EVAL_TAG=mp20probe DATASET_LABEL=mp20 TEST_CSV=Data/curriculum/test_frozen_mp20_1000.csv.gz \
#     TEST_N=1000 bash evaluation/eval_adapter.sh experiments/<run>
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."

EXP="${1:?usage: eval_adapter.sh experiments/<run>}"; EXP="${EXP%/}"
EVAL_TAG="${EVAL_TAG:-}"; DATASET_LABEL="${DATASET_LABEL:-mp52}"
RUN_ID="${RUN_ID:-$(basename "$EXP")}${EVAL_TAG:+_$EVAL_TAG}"
SEED="${SEED:-123456}"
RET_SEQS="${RET_SEQS:-10}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.6}"; TOP_P="${TOP_P:-0.9}"; SYSTEM_PROMPT="${SYSTEM_PROMPT:-sft}"
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"; TEST_N="${TEST_N:-8096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"; GEN_CHUNK="${GEN_CHUNK:-1000}"
RESULTS_DIR="${RESULTS_DIR:-results/evals_s${SEED}}"
PY="${PYTHON:-/venv/py312/bin/python}"; VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
mkdir -p logs

MD="$EXP/final_model/model"; TOK="$EXP/final_model/tokenizer"; [ -d "$TOK" ] || TOK="$MD"
[ -f "$MD/adapter_model.safetensors" ] || { echo "[FATAL] no adapter at $MD"; exit 1; }
ASHA="$(sha256sum "$MD/adapter_model.safetensors" | cut -c1-16)"
echo "[$RUN_ID] adapter $MD ($ASHA), seed $SEED, $TEST_N x $RET_SEQS"

# ---- Generate ------------------------------------------------------------------
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
  "$VLLM_PY" evaluation/generate_cifs_vllm.py --model_dir "$MD" --tokenizer_dir "$TOK" \
    --csv_path "$TEST_CSV" --output_dir "$GDIR" --start_ix 0 --stop_ix "$TEST_N" \
    --ret_seqs "$RET_SEQS" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" --seed "$SEED" --system_prompt "$SYSTEM_PROMPT" \
    --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" 2>&1 | tee -a "logs/${RUN_ID}_gen.log"
  [ "$(rows)" -eq "$TEST_N" ] || { echo "[FATAL] $GCSV incomplete"; exit 1; }
fi
"$PY" -c "
import pandas as pd, sys
g=pd.read_csv('$GCSV'); t=pd.read_csv('$TEST_CSV').iloc[:$TEST_N]
sys.exit(0 if (g.material_id.values==t.material_id.values).all() and (g.output.values==t.output.values).all() else '[FATAL] generated rows misaligned with test set')"

# ---- Validate + collect ----------------------------------------------------------
VDIR="$EXP/validation_seed${SEED}${EVAL_TAG:+_$EVAL_TAG}"
if [ -n "$(find "$VDIR" -name '*_summary.txt' 2>/dev/null | head -1)" ] && [ "$(cat "$VDIR/VALIDATED_FROM" 2>/dev/null)" = "$ASHA" ]; then
  echo "[$RUN_ID][val] exists -> skip"
else
  rm -rf "$VDIR"; mkdir -p "$VDIR"
  "$PY" evaluation/cif_structure_validator_mp52.py --input_csv "$GCSV" --output_dir "$VDIR" --dataset "$DATASET_LABEL" \
    --model "$RUN_ID" --precision 16bit --seq_num "$RET_SEQS" 2>&1 | tee -a "logs/${RUN_ID}_val.log"
  [ -n "$(find "$VDIR" -name '*_summary.txt' | head -1)" ] || { echo "[FATAL] no validation summary"; exit 1; }
  echo "$ASHA" > "$VDIR/VALIDATED_FROM"
fi
R="$RESULTS_DIR/$RUN_ID"; mkdir -p "$R/validation"
gzip -c "$GCSV" > "$R/predicted_cifs.csv.gz"
cp -r "$(dirname "$(find "$VDIR" -name '*_summary.txt' | head -1)")"/. "$R/validation/"
cp "$GDIR/gen_meta.json" "$R/"
if [ -f "$EXP/DONE.json" ]; then cp "$EXP/DONE.json" "$R/train_DONE.json"
else echo "{\"adapter_sha256\": \"$ASHA\", \"seed\": $SEED}" > "$R/train_DONE.json"; fi
grep -h "Success rate" "$R"/validation/*_summary.txt | sed "s/^/[$RUN_ID] /" || true
echo "[$RUN_ID] DONE."
