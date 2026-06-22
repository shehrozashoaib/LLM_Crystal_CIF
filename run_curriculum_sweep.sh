#!/bin/bash
# ============================================================================
# run_curriculum_sweep.sh — CURRICULUM (Schedule) lever, §4.3
# ============================================================================
# Compares the SAME leakage-safe crystal pool trained in two ORDERS at a
# MATCHED total step budget S (only the schedule moves):
#
#   forward  : MP-20 phase (S1) -> MPTS-52 phase (S2),  S1+S2 = S
#   reverse  : MPTS-52 phase (S1) -> MP-20 phase (S2),  same budget
#   mixed    : shuffled MP-20+MPTS-52, S steps           (reference; OPT-IN only,
#              since the composition sweep already trains the shuffled union)
#
# Per condition: train -> generate on the MPTS-52 test set -> validate
# (best-of-N + RMSE panel), graded on the SAME held-out MPTS-52 crystals as the
# composition runs. Plus a FORGETTING PROBE: MP-20 accuracy on a frozen MP-20
# test set, before vs after the 2nd phase.
#
# Matched-budget controls (framework §3): pinned max_steps, early stopping OFF,
# final model graded (not best-eval). The budget S defaults to MAX_STEPS=4500 to
# MATCH the composition sweep (run_composition_sweep.sh / README), NOT the 5170
# stored in the manifest. forward/reverse split S proportional to phase crystal
# count (pool sizes read from Data/curriculum/manifest_curriculum.json), so each
# pool gets the same #steps in both orders and each order sums to exactly S.
#
# Usage:
#   ./run_curriculum_sweep.sh                      # forward reverse  (default)
#   ./run_curriculum_sweep.sh mixed                # just the reference
#   ./run_curriculum_sweep.sh forward reverse      # the two curricula
#   MAX_STEPS=5170 ./run_curriculum_sweep.sh       # override the matched budget
#   TEST_CSV=Data/composition_sweep/test_frozen_mp52_1000.csv TEST_N=1000 \
#       ./run_curriculum_sweep.sh                  # fast frozen-1000 grading
#
# Resumable: a stage whose output already exists is SKIPPED.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- Config (override via env) ---------------------------------------------
DATA_DIR="Data/curriculum"
MANIFEST="${DATA_DIR}/manifest_curriculum.json"

RET_SEQS="${RET_SEQS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
# Microbatch shape is a pure throughput knob: MICRO_BATCH * GRAD_ACCUM is held at
# the matched effective batch of 32, so the budget control is untouched.
MICRO_BATCH="${MICRO_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# Main metric: full MPTS-52 test set (matches the committed composition results).
# pandas reads the .gz directly. Use the frozen 1000 subset for a fast pass.
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"
TEST_N="${TEST_N:-8096}"
# Forgetting probe: frozen MP-20 test set.
MP20_TEST_CSV="${MP20_TEST_CSV:-Data/curriculum/test_frozen_mp20_1000.csv.gz}"
MP20_TEST_N="${MP20_TEST_N:-1000}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
GEN_CHUNK="${GEN_CHUNK:-1000}"

PY="${PYTHON:-/venv/py312/bin/python}"
VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"

SFT_SCRIPT="code_FineTune.py"
GEN_SCRIPT="generate_cifs_vllm.py"
VAL_SCRIPT="cif_structure_validator_mp52.py"
LOG_DIR="logs"
GEN_ROOT="generated"

CONDS=("$@"); [ "${#CONDS[@]}" -gt 0 ] || CONDS=(forward reverse)
mkdir -p "$LOG_DIR" "$GEN_ROOT"

# ---- Step budget split -----------------------------------------------------
# S = MAX_STEPS (default 4500, MATCHED to the composition sweep — NOT the 5170
# baked into the manifest). Phase steps are recomputed proportionally to the
# pool crystal counts so each pool gets the same #steps in both orders and each
# order sums to exactly S:
#   MP20_STEPS = round(S * n_mp20 / (n_mp20 + n_mp52)),  MP52_STEPS = S - MP20_STEPS
#   forward = MP20 -> MP52 ;  reverse = MP52 -> MP20
S_TOTAL="${MAX_STEPS:-4500}"
read -r MP20_STEPS MP52_STEPS <<<"$("$PY" -c "
import json
m=json.load(open('$MANIFEST'))['union']
n20,n52=m['n_mp20_phase'],m['n_mp52_phase']
S=$S_TOTAL
s20=round(S*n20/(n20+n52)); s52=S-s20
print(s20,s52)")"
S1_FWD=$MP20_STEPS; S2_FWD=$MP52_STEPS   # forward: MP-20 then MPTS-52
S1_REV=$MP52_STEPS; S2_REV=$MP20_STEPS   # reverse: MPTS-52 then MP-20

# ---- Preflight -------------------------------------------------------------
[ -x "$PY" ] || { echo "[FATAL] py312 python not found: $PY"; exit 1; }
"$PY" -c "import unsloth, trl, pymatgen, seaborn" 2>/dev/null \
  || { echo "[FATAL] $PY missing train/validate libs (unsloth, trl, pymatgen, seaborn)"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "[FATAL] vllm python not found: $VLLM_PY"; exit 1; }
"$VLLM_PY" -c "import vllm, pandas" 2>/dev/null \
  || { echo "[FATAL] $VLLM_PY missing inference libs (vllm, pandas)"; exit 1; }
for f in "$SFT_SCRIPT" "$GEN_SCRIPT" "$VAL_SCRIPT" "$MANIFEST" "$TEST_CSV" "$MP20_TEST_CSV"; do
  [ -e "$f" ] || { echo "[FATAL] missing required file: $f (run build_curriculum_datasets.py?)"; exit 1; }
done

echo "=== Curriculum sweep ==="
echo "conditions=${CONDS[*]}"
echo "S=$S_TOTAL  forward(mp20 $S1_FWD + mp52 $S2_FWD)  reverse(mp52 $S1_REV + mp20 $S2_REV)"
echo "main_test=$TEST_CSV (N=$TEST_N)   forgetting_test=$MP20_TEST_CSV (N=$MP20_TEST_N)"
echo "ret_seqs=$RET_SEQS  lora_r=$LORA_R alpha=$LORA_ALPHA"
echo

# ---- Stage helpers ---------------------------------------------------------
model_dir_of () { echo "experiments/$1/final_model/model"; }
tok_dir_of   () { echo "experiments/$1/final_model/tokenizer"; }

train_single () {  # run_name train_csv val_csv max_steps
  local run="$1" tr="$2" va="$3" steps="$4"
  local md; md="$(model_dir_of "$run")"
  if [ -d "$md" ] && [ -n "$(ls -A "$md" 2>/dev/null)" ]; then
    echo "[$run][train] final model exists -> skip"; return
  fi
  echo "[$run][train] SFT max_steps=$steps  train=$tr"
  "$PY" "$SFT_SCRIPT" \
    --train_csv "$tr" --val_csv "$va" --run_name "$run" \
    --max_steps "$steps" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    2>&1 | tee "${LOG_DIR}/${run}_train.log"
}

train_phase2 () {  # run_name train_csv val_csv max_steps init_adapter
  local run="$1" tr="$2" va="$3" steps="$4" init="$5"
  local md; md="$(model_dir_of "$run")"
  if [ -d "$md" ] && [ -n "$(ls -A "$md" 2>/dev/null)" ]; then
    echo "[$run][train] final model exists -> skip"; return
  fi
  echo "[$run][train] SFT (continue $init) max_steps=$steps  train=$tr"
  "$PY" "$SFT_SCRIPT" \
    --train_csv "$tr" --val_csv "$va" --run_name "$run" \
    --max_steps "$steps" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --init_adapter "$init" \
    2>&1 | tee "${LOG_DIR}/${run}_train.log"
}

# generate + validate on a given test set into a given output_dir/label
eval_on () {  # run_name test_csv test_n out_dir gen_subdir dataset_label
  local run="$1" test_csv="$2" test_n="$3" out_dir="$4" gen_sub="$5" dslabel="$6"
  local md tok gen_csv
  md="$(model_dir_of "$run")"; tok="$(tok_dir_of "$run")"
  gen_csv="${GEN_ROOT}/${gen_sub}/generated_cifs_${run}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((test_n-1)).csv"

  if [ -f "$gen_csv" ]; then
    echo "[$run][gen:$dslabel] $gen_csv exists -> skip"
  else
    echo "[$run][gen:$dslabel] vLLM ${RET_SEQS} CIFs/prompt on $test_csv (N=$test_n)"
    mkdir -p "${GEN_ROOT}/${gen_sub}"
    "$VLLM_PY" "$GEN_SCRIPT" \
      --model_dir "$md" --tokenizer_dir "$tok" \
      --csv_path "$test_csv" --output_dir "${GEN_ROOT}/${gen_sub}" \
      --start_ix 0 --stop_ix "$test_n" \
      --ret_seqs "$RET_SEQS" --max_new_tokens "$MAX_NEW_TOKENS" \
      --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
      2>&1 | tee "${LOG_DIR}/${gen_sub}_gen.log"
  fi

  echo "[$run][val:$dslabel] validating (best-of-N + RMSE panel) -> $out_dir"
  mkdir -p "$out_dir"
  "$PY" "$VAL_SCRIPT" \
    --input_csv "$gen_csv" --output_dir "$out_dir" \
    --dataset "$dslabel" --model "$run" --precision 16bit --seq_num "$RET_SEQS" \
    2>&1 | tee "${LOG_DIR}/${run}_val_${dslabel}.log"
}

# main MPTS-52 metric for a final model
eval_mpts52 () { eval_on "$1" "$TEST_CSV" "$TEST_N" "experiments/$1" "$1" "mp52"; }
# MP-20 forgetting probe for any model (final or intermediate phase-1)
eval_forget () { eval_on "$1" "$MP20_TEST_CSV" "$MP20_TEST_N" "experiments/$1/forgetting" "${1}_mp20" "mp20"; }

# ---- Conditions ------------------------------------------------------------
run_mixed () {
  echo "######## MIXED (reference) ########"
  train_single curr_mixed "${DATA_DIR}/train_mixed.csv.gz" "${DATA_DIR}/val_mixed.csv.gz" "$S_TOTAL"
  eval_mpts52  curr_mixed
  eval_forget  curr_mixed          # MP-20 no-forgetting reference (simultaneous)
}

run_forward () {
  echo "######## FORWARD: MP-20 -> MPTS-52 ########"
  train_single curr_fwd_p1 "${DATA_DIR}/train_phase_mp20.csv.gz" "${DATA_DIR}/val_mp20.csv.gz" "$S1_FWD"
  eval_forget  curr_fwd_p1         # MP-20 BEFORE phase 2 (peak MP-20)
  train_phase2 curr_fwd "${DATA_DIR}/train_phase_mp52.csv.gz" "${DATA_DIR}/val_mp52.csv.gz" "$S2_FWD" \
               "$(model_dir_of curr_fwd_p1)"
  eval_mpts52  curr_fwd            # main MPTS-52 metric (forward)
  eval_forget  curr_fwd            # MP-20 AFTER phase 2 (forgetting)
}

run_reverse () {
  echo "######## REVERSE: MPTS-52 -> MP-20 ########"
  train_single curr_rev_p1 "${DATA_DIR}/train_phase_mp52.csv.gz" "${DATA_DIR}/val_mp52.csv.gz" "$S1_REV"
  eval_forget  curr_rev_p1         # MP-20 BEFORE the MP-20 phase (MPTS-52-only)
  train_phase2 curr_rev "${DATA_DIR}/train_phase_mp20.csv.gz" "${DATA_DIR}/val_mp20.csv.gz" "$S2_REV" \
               "$(model_dir_of curr_rev_p1)"
  eval_mpts52  curr_rev            # main MPTS-52 metric (reverse; expect drop vs p1)
  eval_forget  curr_rev            # MP-20 AFTER MP-20 phase (recency, should be high)
}

for c in "${CONDS[@]}"; do
  case "$c" in
    mixed)   run_mixed ;;
    forward) run_forward ;;
    reverse) run_reverse ;;
    *) echo "[skip] unknown condition: $c (use mixed|forward|reverse)";;
  esac
done

echo "=== Curriculum sweep complete: ${CONDS[*]} ==="
echo "MPTS-52 metric : experiments/curr_*/validation_*/"
echo "Forgetting     : experiments/curr_*/forgetting/validation_*/"
