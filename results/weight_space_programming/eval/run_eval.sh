#!/bin/bash
# Held-out evaluation: 100 random MPTS-52 test materials (seed 42), N=10, the repo's protocol
# (vLLM, temperature 0.6, top_p 0.9, max_new_tokens 3072, system prompt "sft", seed 42 = rank-sweep defaults).
#   run_eval.sh <tag> <base_model_dir> [adapter_dir]
set -euo pipefail
TAG="$1"; BASE="$2"; AD="${3:-}"
E=/workspace/wsp/eval; REPO=/workspace/LLM_Crystal_CIF; OUT="$E/$TAG"; mkdir -p "$OUT/generated"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN HF_HUB_OFFLINE=1
TOK=/workspace/models/Qwen/Qwen2.5-7B-Instruct
COMMON=(--tokenizer_dir "$TOK" --base_model "$BASE" --csv_path "$E/mpts52_test100_seed42.csv" --output_dir "$OUT/generated"
        --start_ix 0 --stop_ix 100 --ret_seqs 10 --max_new_tokens 3072 --temperature 0.6 --top_p 0.9 --seed 42
        --system_prompt sft --gpu_mem_util "${GPU_MEM_UTIL:-0.30}" --chunk 1000)
cd "$REPO"
if ! ls "$OUT"/generated/generated_cifs_*_10seq_maxtok3072_0_99.csv >/dev/null 2>&1; then
  for try in 1 2 3; do
    if [ -n "$AD" ]; then /venv/vllm/bin/python evaluation/generate_cifs_vllm.py --model_dir "$AD" "${COMMON[@]}" > "$OUT/gen.log" 2>&1 && break
    else /venv/vllm/bin/python "$E/generate_cifs_vllm_base.py" --model_dir "$BASE" "${COMMON[@]}" > "$OUT/gen.log" 2>&1 && break; fi
    echo "[retry $try]"; sleep 5
  done
fi
G="$(ls "$OUT"/generated/generated_cifs_*_10seq_maxtok3072_0_99.csv)"
rm -rf "$OUT/validation"
/venv/py312/bin/python evaluation/cif_structure_validator_mp52.py --input_csv "$G" --output_dir "$OUT/validation" \
  --dataset mp52 --model "$TAG" --precision 16bit --seq_num 10 > "$OUT/val.log" 2>&1 || echo "[repo validator exited non-zero: see $OUT/val.log]"
/venv/py312/bin/python "$E/panel.py" "$TAG"
