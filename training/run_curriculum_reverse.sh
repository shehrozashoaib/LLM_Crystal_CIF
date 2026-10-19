#!/bin/bash
# ============================================================================
# training/run_curriculum_reverse.sh — REVERSE curriculum (MPTS-52 first, then MP-20)
# on the crystal sets of one k-sweep point (datasets/build_curriculum_k_sweep.py).
#
# For k-sweep point K (forward: MP-20 K steps on train_mp20_kK, then MPTS-52 4500-K steps
# on train_mp52_kK) the reverse run trains the SAME crystals in the opposite order:
#   phase 1: MPTS-52, 4500-K steps on train_mp52_kK   (fresh LoRA)
#   phase 2: MP-20,   K steps      on train_mp20_kK   (continues phase 1's adapter)
# Each phase keeps its 6.0 epochs, so forward kK vs reverse kK differ ONLY in order.
# e.g. K=2000 -> MPTS-52 2500 steps (13,333 crystals) then MP-20 2000 steps (10,667).
#
#   bash training/run_curriculum_reverse.sh 2000            # seed 123456
#   SEED=25478 bash training/run_curriculum_reverse.sh 2000
# Training only; evaluate with evaluation/eval_adapter.sh.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
K="${1:?usage: run_curriculum_reverse.sh K}"
SEED="${SEED:-123456}"; LORA_R="${LORA_R:-32}"; LORA_ALPHA="${LORA_ALPHA:-64}"
MICRO_BATCH="${MICRO_BATCH:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-8}"; ARCH="${ARCH:-auto}"; SAVE_STEPS="${SAVE_STEPS:-100}"
MANIFEST="${MANIFEST:-Data/curriculum_k_sweep_s${SEED}/manifest.json}"
PY="${PYTHON:-/venv/py312/bin/python}"
export UNSLOTH_DISABLE_DOUBLE_BUFFER="${UNSLOTH_DISABLE_DOUBLE_BUFFER:-1}"
[ $((MICRO_BATCH * GRAD_ACCUM)) -eq 32 ] || { echo "[FATAL] MICRO_BATCH*GRAD_ACCUM must be 32"; exit 1; }
mkdir -p logs
mf () { "$PY" -c "import json; m=json.load(open('$MANIFEST')); assert m['params']['seed']==$SEED; print(m['runs']['$K']$1)"; }
S52="$(mf "['steps']['mp52']")"; S20="$(mf "['steps']['mp20']")"
F52="$(mf "['files']['mp52']")"; F20="$(mf "['files']['mp20']")"
[ "$S52" -gt 0 ] && [ "$S20" -gt 0 ] || { echo "[FATAL] k=$K has an empty phase"; exit 1; }
RUN="crev${SEED}_k$(printf %04d "$K")"

train_done () {  # dir steps
  [ -f "$1/DONE.json" ] && [ -f "$1/final_model/model/adapter_model.safetensors" ] &&
  "$PY" -c "import json,sys; d=json.load(open('$1/DONE.json')); s=json.load(open('$1/training_stats.json')); sys.exit(0 if d['steps']==$2==s['global_step'] else 1)"
}
train_phase () {  # run csv steps [init]
  local d="experiments/$1" init="${4:-}"
  if train_done "$d" "$3"; then echo "[$1] trained ($3 steps, verified) -> skip"; return; fi
  if [ -n "$init" ]; then [ -f "$init/adapter_model.safetensors" ] || { echo "[FATAL] missing $init"; exit 1; }; fi
  echo "[$1] training $3 steps on $2 ${init:+(init from $init)}"
  "$PY" training/code_FineTune.py --train_csv "$2" --run_name "$1" --max_steps "$3" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --seed "$SEED" \
    --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_steps "$SAVE_STEPS" --arch "$ARCH" ${init:+--init_adapter "$init"} 2>&1 | tee -a "logs/$1_train.log"
  "$PY" - "$d" "$3" "$2" "$SEED" "$init" <<'PY'
import json, sys, hashlib, pathlib, datetime
d, steps, csv, seed, init = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
json.dump({"steps": steps, "train_csv": csv, "seed": seed,
           "adapter_sha256": h(d / "final_model/model/adapter_model.safetensors"),
           "init_adapter": init or None,
           "init_adapter_sha256": h(pathlib.Path(init) / "adapter_model.safetensors") if init else None,
           "finished": datetime.datetime.now().isoformat(timespec="seconds")}, open(d / "DONE.json", "w"), indent=2)
PY
  train_done "$d" "$3" || { echo "[FATAL] $1 did not reach $3 steps"; exit 1; }
}

echo "######## reverse k=$K: MPTS-52 $S52 steps ($F52) -> MP-20 $S20 steps ($F20)   run=$RUN ########"
train_phase "${RUN}_p1" "$F52" "$S52"
train_phase "$RUN" "$F20" "$S20" "experiments/${RUN}_p1/final_model/model"
echo "[$RUN] DONE."
