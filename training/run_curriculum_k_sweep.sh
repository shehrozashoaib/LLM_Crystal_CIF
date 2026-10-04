#!/bin/bash
# ============================================================================
# training/run_curriculum_k_sweep.sh — forward-curriculum switch-point sweep, one seed
# ----------------------------------------------------------------------------
# For each k in {0, 500, 1000, 1500, 2000, 2500, 4500} (MP-20 steps of 4,500):
#   Phase 1: MP-20   k steps on train_mp20_k<k>  (fresh LoRA on unsloth/Qwen2.5-7B-Instruct)
#   Phase 2: MPTS-52 4500-k steps on train_mp52_k<k>, continuing phase 1's adapter
#   k=0 / k=4500 are single-phase runs (fresh LoRA, 4,500 steps on one pool).
#   Then: vLLM, 10 CIFs/prompt on the full 8,096 MPTS-52 test set -> pymatgen validation.
# Datasets + step splits come from datasets/build_curriculum_k_sweep.py's manifest
# (data ratio == step ratio, 24,000 unique crystals per run, 6.00 epochs per phase).
#
# Phases use separate trainers exactly as training/run_ratio_sweep.sh did (per-phase 10%
# warmup + cosine, optimizer reset at the switch) — a deliberate choice for parity.
#
# SEED is used everywhere randomness exists: dataset sampling (builder), LoRA init +
# trainer/data order (code_FineTune --seed), and vLLM sampling (generate --seed).
#
# Every stage is skipped only when its output is VERIFIED complete:
#   train : DONE.json written after exit 0, adapter present, trainer global_step == steps
#   gen   : gen_meta.json matches (adapter sha256 + sampling settings) AND rows == TEST_N
#   val   : summary present for this exact generation
#
# Usage:  bash training/run_curriculum_k_sweep.sh                 # all k
#         bash training/run_curriculum_k_sweep.sh 500 1000        # subset
#         SMOKE=1 bash training/run_curriculum_k_sweep.sh 0 500 4500   # plumbing test
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."

SEED="${SEED:-123456}"
LORA_R="${LORA_R:-32}"; LORA_ALPHA="${LORA_ALPHA:-64}"
MICRO_BATCH="${MICRO_BATCH:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-8}"
ARCH="${ARCH:-auto}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RET_SEQS="${RET_SEQS:-10}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.6}"; TOP_P="${TOP_P:-0.9}"; SYSTEM_PROMPT="${SYSTEM_PROMPT:-sft}"
TEST_CSV="${TEST_CSV:-Data/source/mp_52_test.csv.gz}"; TEST_N="${TEST_N:-8096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"; GEN_CHUNK="${GEN_CHUNK:-1000}"
DATA_DIR="${DATA_DIR:-Data/curriculum_k_sweep_s${SEED}}"
MANIFEST="$DATA_DIR/manifest.json"
PREFIX="${PREFIX:-cks${SEED}}"
EXP_ROOT="${EXP_ROOT:-experiments}"; GEN_ROOT="${GEN_ROOT:-generated}"; LOG_DIR="${LOG_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results/curriculum_k_sweep_s${SEED}}"
PY="${PYTHON:-/venv/py312/bin/python}"; VLLM_PY="${VLLM_PYTHON:-/venv/vllm/bin/python}"
SFT=training/code_FineTune.py; GEN=evaluation/generate_cifs_vllm.py; VAL=evaluation/cif_structure_validator_mp52.py

# SMOKE=1: same code path, tiny budgets, isolated outputs — proves the plumbing.
STEP_DIV=1
if [ "${SMOKE:-0}" = "1" ]; then
  STEP_DIV=500; TEST_N=8; GEN_CHUNK=4; SAVE_STEPS=2; PREFIX="smoke_${PREFIX}"
  SCR="${SMOKE_ROOT:?set SMOKE_ROOT to a scratch dir}"
  EXP_ROOT="$SCR/experiments"; GEN_ROOT="$SCR/generated"; LOG_DIR="$SCR/logs"; RESULTS_DIR="$SCR/results"
  echo "[SMOKE] steps / $STEP_DIV, TEST_N=$TEST_N, outputs under $SCR"
fi

export UNSLOTH_DISABLE_DOUBLE_BUFFER="${UNSLOTH_DISABLE_DOUBLE_BUFFER:-1}"   # sm_120 segfault suspect
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"     # FlashInfer needs CUDA>=12.9 on sm_120
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

[ $((MICRO_BATCH * GRAD_ACCUM)) -eq 32 ] || { echo "[FATAL] MICRO_BATCH*GRAD_ACCUM must be 32"; exit 1; }
mkdir -p "$LOG_DIR" "$GEN_ROOT" "$EXP_ROOT"

KS=("$@"); [ "${#KS[@]}" -gt 0 ] || KS=(0 500 1000 1500 2000 2500 4500)

# ---- Preflight -------------------------------------------------------------
for f in "$SFT" "$GEN" "$VAL" "$MANIFEST" "$TEST_CSV"; do
  [ -e "$f" ] || { echo "[FATAL] missing: $f  (build: $PY datasets/build_curriculum_k_sweep.py --seed $SEED)"; exit 1; }
done
"$PY" -c "import json,sys; s=json.load(open('$MANIFEST'))['params']['seed']; sys.exit(0 if s==$SEED else f'[FATAL] manifest seed {s} != SEED $SEED')"
"$PY" -c "import unsloth,trl,pymatgen,seaborn" 2>/dev/null || { echo "[FATAL] py312 libs"; exit 1; }
"$VLLM_PY" -c "import vllm,pandas" 2>/dev/null || { echo "[FATAL] vllm libs"; exit 1; }

mfield () {  # k, dotted path into manifest.runs[k]
  "$PY" -c "
import json,sys
m=json.load(open('$MANIFEST'))['runs'].get('$1')
if m is None: sys.exit('[FATAL] k=$1 not in $MANIFEST')
d=m
for p in '$2'.split('.'): d=d[p]
print('' if d is None else d)"
}

sha () { sha256sum "$1/adapter_model.safetensors" | cut -c1-16; }

# A training stage is complete iff DONE.json exists, the adapter exists, and the
# last checkpoint's trainer_state says global_step == expected steps.
train_done () {  # run_dir, steps
  local d="$1" steps="$2"
  [ -f "$d/DONE.json" ] && [ -f "$d/final_model/model/adapter_model.safetensors" ] || return 1
  "$PY" - "$d" "$steps" <<'PY' || return 1
import json, sys, pathlib
d, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
ck = sorted(d.glob("checkpoints/checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
gs = json.load(open(ck[-1]))["global_step"] if ck else -1
done = json.load(open(d / "DONE.json"))
sys.exit(0 if gs == steps and done.get("steps") == steps else 1)
PY
}

train_phase () {  # run, csv, steps, init_adapter(optional)
  local run="$1" csv="$2" steps="$3" init="${4:-}"
  local d="$EXP_ROOT/$run"
  if train_done "$d" "$steps"; then echo "[$run] trained ($steps steps, verified) -> skip"; return; fi
  if [ -f "$d/DONE.json" ]; then echo "[FATAL] $d has DONE.json but fails verification — inspect manually"; exit 1; fi
  local initargs=()
  if [ -n "$init" ]; then
    [ -f "$init/adapter_model.safetensors" ] || { echo "[FATAL] init adapter missing: $init"; exit 1; }
    initargs=(--init_adapter "$init")
  fi
  echo "[$run] training $steps steps on $csv ${init:+(init from $init)}"
  "$PY" "$SFT" --train_csv "$csv" --run_name "$run" --output_root "$EXP_ROOT" \
    --max_steps "$steps" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
    --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_steps "$SAVE_STEPS" --arch "$ARCH" "${initargs[@]}" \
    2>&1 | tee -a "$LOG_DIR/${run}_train.log"
  "$PY" - "$d" "$steps" "$csv" "$SEED" "$init" <<'PY'
import json, sys, hashlib, pathlib, datetime
d, steps, csv, seed, init = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
rec = {"steps": steps, "train_csv": csv, "seed": seed,
       "adapter_sha256": h(d / "final_model/model/adapter_model.safetensors"),
       "init_adapter": init or None,
       "init_adapter_sha256": h(pathlib.Path(init) / "adapter_model.safetensors") if init else None,
       "finished": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(rec, open(d / "DONE.json", "w"), indent=2)
PY
  train_done "$d" "$steps" || { echo "[FATAL] $run finished but global_step != $steps"; exit 1; }
}

run_one () {
  local k="$1" tag s20 s52 f20 f52 run
  tag="$(mfield "$k" tag)"
  s20="$(mfield "$k" steps.mp20)"; s52="$(mfield "$k" steps.mp52)"
  f20="$(mfield "$k" files.mp20)"; f52="$(mfield "$k" files.mp52)"
  s20=$(( s20 / STEP_DIV )); s52=$(( s52 / STEP_DIV ))
  if [ "$k" -gt 0 ] && [ "$s20" -lt 1 ]; then s20=1; fi
  if [ "$k" -lt 4500 ] && [ "$s52" -lt 1 ]; then s52=1; fi
  run="${PREFIX}_${tag}"
  echo "######## k=$k  MP-20 ${s20} steps -> MPTS-52 ${s52} steps   run=$run ########"

  # ---- Train -------------------------------------------------------------
  if [ "$s52" -eq 0 ]; then
    train_phase "$run" "$f20" "$s20"                                  # k=4500: pure MP-20
  elif [ "$s20" -eq 0 ]; then
    train_phase "$run" "$f52" "$s52"                                  # k=0: pure MPTS-52
  else
    train_phase "${run}_p1" "$f20" "$s20"
    train_phase "$run" "$f52" "$s52" "$EXP_ROOT/${run}_p1/final_model/model"
  fi

  local md="$EXP_ROOT/$run/final_model/model" tok="$EXP_ROOT/$run/final_model/tokenizer"
  local asha; asha="$(sha "$md")"

  # ---- Generate ------------------------------------------------------------
  local gdir="$GEN_ROOT/$run"
  local gcsv="$gdir/generated_cifs_${run}_${RET_SEQS}seq_maxtok${MAX_NEW_TOKENS}_0_$((TEST_N-1)).csv"
  local meta
  meta="{\"adapter_sha256\": \"$asha\", \"seed\": $SEED, \"temperature\": $TEMPERATURE, \"top_p\": $TOP_P, \"ret_seqs\": $RET_SEQS, \"max_new_tokens\": $MAX_NEW_TOKENS, \"system_prompt\": \"$SYSTEM_PROMPT\", \"test_csv\": \"$TEST_CSV\", \"test_n\": $TEST_N}"
  mkdir -p "$gdir"
  if [ -f "$gdir/gen_meta.json" ]; then
    "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$gdir/gen_meta.json'))==json.loads('''$meta''') else 1)" \
      || { echo "[FATAL] $gdir holds generations from a different model/settings — move it aside"; exit 1; }
  else
    if [ -n "$(ls -A "$gdir")" ]; then echo "[FATAL] $gdir is non-empty but has no gen_meta.json"; exit 1; fi
    echo "$meta" > "$gdir/gen_meta.json"
  fi
  local grows=0
  if [ -f "$gcsv" ]; then grows=$("$PY" -c "
import pandas as pd
try: print(len(pd.read_csv('$gcsv')))
except Exception: print(0)"); fi
  if [ "$grows" -eq "$TEST_N" ]; then echo "[$run][gen] complete ($grows rows) -> skip"; else
    echo "[$run][gen] vLLM ${RET_SEQS} CIFs/prompt, seed $SEED, on $TEST_CSV (N=$TEST_N)"
    "$VLLM_PY" "$GEN" --model_dir "$md" --tokenizer_dir "$tok" --csv_path "$TEST_CSV" \
      --output_dir "$gdir" --start_ix 0 --stop_ix "$TEST_N" --ret_seqs "$RET_SEQS" \
      --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
      --seed "$SEED" --system_prompt "$SYSTEM_PROMPT" \
      --gpu_mem_util "$GPU_MEM_UTIL" --chunk "$GEN_CHUNK" \
      2>&1 | tee -a "$LOG_DIR/${run}_gen.log"
    grows=$("$PY" -c "import pandas as pd; print(len(pd.read_csv('$gcsv')))")
    [ "$grows" -eq "$TEST_N" ] || { echo "[FATAL] $gcsv has $grows rows, expected $TEST_N"; exit 1; }
  fi
  # rows must be the test set, in order
  "$PY" -c "
import pandas as pd, sys
g=pd.read_csv('$gcsv'); t=pd.read_csv('$TEST_CSV').iloc[:$TEST_N]
sys.exit(0 if (g.material_id.values==t.material_id.values).all() and (g.output.values==t.output.values).all() else '[FATAL] generated rows misaligned with test set')"

  # ---- Validate --------------------------------------------------------------
  local vdir="$EXP_ROOT/$run/validation_seed${SEED}"
  local vdone; vdone="$(find "$vdir" -name "*_summary.txt" 2>/dev/null | head -1 || true)"
  if [ -n "$vdone" ] && [ -f "$vdir/VALIDATED_FROM" ] && [ "$(cat "$vdir/VALIDATED_FROM")" = "$asha" ]; then
    echo "[$run][val] exists -> skip"
  else
    rm -rf "$vdir"; mkdir -p "$vdir"
    echo "[$run][val] validating"
    "$PY" "$VAL" --input_csv "$gcsv" --output_dir "$vdir" --dataset mp52 \
      --model "$run" --precision 16bit --seq_num "$RET_SEQS" 2>&1 | tee -a "$LOG_DIR/${run}_val.log"
    [ -n "$(find "$vdir" -name "*_summary.txt" | head -1)" ] || { echo "[FATAL] validation wrote no summary"; exit 1; }
    echo "$asha" > "$vdir/VALIDATED_FROM"
  fi

  # ---- Collect into results/ (same layout as the committed sweeps) ---------
  local rdir="$RESULTS_DIR/$run"
  mkdir -p "$rdir/validation"
  gzip -c "$gcsv" > "$rdir/predicted_cifs.csv.gz"
  cp -r "$(dirname "$(find "$vdir" -name "*_summary.txt" | head -1)")"/. "$rdir/validation/"
  cp "$gdir/gen_meta.json" "$rdir/"
  cp "$EXP_ROOT/$run/DONE.json" "$rdir/train_DONE.json"
  if [ -f "$EXP_ROOT/${run}_p1/DONE.json" ]; then cp "$EXP_ROOT/${run}_p1/DONE.json" "$rdir/train_p1_DONE.json"; fi
  grep -h -m3 -i "best-of\|any generation\|overall" "$rdir"/validation/*_summary.txt 2>/dev/null | sed "s/^/[$run] /" || true
  echo "[$run] DONE."
}

for k in "${KS[@]}"; do run_one "$k"; done
echo "=== curriculum k-sweep (seed $SEED) complete: ${KS[*]} ==="
