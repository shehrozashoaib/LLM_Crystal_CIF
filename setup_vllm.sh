#!/bin/bash
# ============================================================================
# setup_vllm.sh — build the GENERATE/inference env (/venv/vllm)
# ----------------------------------------------------------------------------
# Target: GH200 (aarch64), CUDA 13.0 driver. Mirrors the README's vllm env:
# vllm 0.22 + torch 2.11+cu130. Used only by generate_cifs_vllm.py (inference).
# vLLM cannot train; training stays on /venv/py312.
# ============================================================================
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

VENV=/venv/vllm
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "=== [vllm] create venv (Python 3.12) ==="
uv venv "$VENV" --python 3.12

echo "=== [vllm] install torch 2.11.0+cu130 (aarch64) ==="
uv pip install --python "$VENV" "torch==2.11.0+cu130" --index-url "$TORCH_INDEX"

echo "=== [vllm] install vllm 0.22 + transformers/pandas ==="
# torch pinned to +cu130 so the resolver never swaps it for a CPU build.
uv pip install --python "$VENV" \
  --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match \
  --prerelease=allow \
  "torch==2.11.0+cu130" \
  "vllm==0.22.0" \
  transformers pandas

echo "=== [vllm] smoke test ==="
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
import vllm, pandas
print("vllm", vllm.__version__, "pandas", pandas.__version__)
print("vllm OK")
PY
echo "=== [vllm] DONE ==="
