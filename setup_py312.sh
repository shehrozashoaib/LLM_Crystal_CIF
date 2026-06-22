#!/bin/bash
# ============================================================================
# setup_py312.sh — build the TRAIN/VALIDATE env (/venv/py312)
# ----------------------------------------------------------------------------
# Target: GH200 (aarch64), Ubuntu 22.04, CUDA 12.8 toolkit / 13.0 driver.
# Mirrors the README's py312 env: unsloth + trl 0.24 + torch 2.10+cu128 +
# pymatgen + seaborn. Used by code_FineTune.py (train) and
# cif_structure_validator_mp52.py (validate).
# ============================================================================
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

VENV=/venv/py312
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "=== [py312] create venv (Python 3.12) ==="
uv venv "$VENV" --python 3.12

echo "=== [py312] install torch 2.10.0+cu128 (aarch64) ==="
uv pip install --python "$VENV" "torch==2.10.0+cu128" --index-url "$TORCH_INDEX"

echo "=== [py312] install training + validation stack ==="
# torch pinned again so the resolver never swaps it for a CPU pypi build.
uv pip install --python "$VENV" \
  --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match \
  --prerelease=allow \
  "torch==2.10.0+cu128" \
  unsloth unsloth_zoo \
  "trl==0.24.0" \
  transformers datasets accelerate peft \
  bitsandbytes \
  pymatgen seaborn pandas numpy matplotlib wandb

echo "=== [py312] smoke test ==="
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import unsloth, trl, pymatgen, seaborn
print("unsloth", unsloth.__version__, "trl", trl.__version__)
print("py312 OK")
PY
echo "=== [py312] DONE ==="
