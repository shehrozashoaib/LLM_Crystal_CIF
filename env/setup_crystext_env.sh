#!/bin/bash
# ============================================================================
# env/setup_crystext_env.sh — build the TRAIN env for training/grpo_crystext_reward.py
# ----------------------------------------------------------------------------
# Same stack as env/setup_py312.sh (unsloth + trl 0.24 + torch 2.10+cu128 +
# pymatgen), but targets this box: x86_64, RTX PRO 6000 Blackwell (cc 12.0,
# so CUDA >= 12.8 wheels are mandatory), driver 595 / CUDA 13.
#
# The CrysText reward needs nothing extra beyond pymatgen — the same
# Structure / StructureMatcher / spglib path this repo already used. Unsloth
# is NOT the CrysText FastLanguageModel(fast_inference=True) vLLM path: this
# trainer uses HF generation, so no vllm install is required here.
#
#   bash env/setup_crystext_env.sh          # -> /venv/py312
#   VENV=/venv/other bash env/setup_crystext_env.sh
# ============================================================================
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

VENV="${VENV:-/venv/py312}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PIN="${TORCH_PIN:-torch==2.10.0+cu128}"

echo "=== [crystext-env] create venv (Python 3.12) at $VENV ==="
uv venv "$VENV" --python 3.12

echo "=== [crystext-env] install $TORCH_PIN (cu128 — required for Blackwell cc12.0) ==="
uv pip install --python "$VENV" "$TORCH_PIN" --index-url "$TORCH_INDEX"

echo "=== [crystext-env] install training + reward stack ==="
# torch pinned again so the resolver never swaps it for a CPU pypi build.
uv pip install --python "$VENV" \
  --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match \
  --prerelease=allow \
  "$TORCH_PIN" \
  unsloth unsloth_zoo \
  "trl==0.24.0" \
  transformers datasets accelerate peft \
  bitsandbytes \
  pymatgen seaborn pandas numpy matplotlib wandb

echo "=== [crystext-env] smoke test ==="
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("compute capability", cc)
    assert cc[0] < 10 or int(torch.version.cuda.split(".")[0]) * 10 + int(
        torch.version.cuda.split(".")[1]) >= 128, "Blackwell needs a CUDA>=12.8 build"
    print("matmul ok:", (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).shape)
import trl, transformers, peft, datasets
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
print("trl", trl.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "datasets", datasets.__version__)
import unsloth
print("unsloth", getattr(unsloth, "__version__", "?"))
print("env OK")
PY
echo "=== [crystext-env] DONE -> $VENV ==="
