#!/bin/bash
# ============================================================================
# setup_grpo_vllm.sh — build /venv/grpo_vllm: GRPO training with vLLM generation
# (Unsloth fast_inference=True + TRL use_vllm, colocated) on the RTX PRO 6000
# Blackwell box (sm_120, driver CUDA 13.1). Separate from /venv/py312 (HF generate).
#
# Why these versions (verified 2026-10-16):
#  * unsloth + unsloth_zoo 2026.1.4, trl 0.24.0, transformers 4.57.6 = the py312 reference.
#  * unsloth_zoo 2026.1.4 patches vLLM's LoRA worker manager with the <=0.13 API
#    (create_lora_manager(model)); vLLM >= 0.14 calls create_lora_manager(model, vllm_config)
#    -> TypeError at engine init (tested with vllm 0.17.1). vLLM releases that pin torch 2.10
#    are 0.17.x-0.19.x, so torch 2.10 + unsloth 2026.1.4 + vLLM is impossible;
#    vllm 0.13.0 pins torch 2.9.0 -> torch 2.9.0+cu128 (sm_120 OK, CUDA 12.8 wheels).
#  * torchao 0.16.0: 0.18 (py312's) needs torch>=2.10 (ImportError ScalingType); 0.15 is
#    rejected by peft 0.21.2 ("only versions above 0.16.0").
#  * numpy 2.2.6 (vllm 0.13 -> numba 0.61.2 -> numpy<2.3); py312 has 2.5.3. Reward floats
#    were checked identical across the two envs.
#  * everything else pinned to the py312 versions (lock file grpo_vllm.lock.txt).
#
#   bash setup_grpo_vllm.sh                   # -> /venv/grpo_vllm from the lock file
#   LOCK=0 bash setup_grpo_vllm.sh            # re-resolve from the top-level pins instead
# Runtime env needed by every run on this box:
#   UNSLOTH_DISABLE_DOUBLE_BUFFER=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
# ============================================================================
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/venv/grpo_vllm}"
IDX="https://download.pytorch.org/whl/cu128"

uv venv "$VENV" --python 3.12
if [ "${LOCK:-1}" = "1" ] && [ -f "$HERE/grpo_vllm.lock.txt" ]; then
  uv pip install --python "$VENV/bin/python" -r "$HERE/grpo_vllm.lock.txt" \
    --extra-index-url "$IDX" --index-strategy unsafe-best-match
else
  uv pip install --python "$VENV/bin/python" \
    --extra-index-url "$IDX" --index-strategy unsafe-best-match \
    --prerelease=if-necessary-or-explicit \
    "torch==2.9.0" "vllm==0.13.0" \
    "unsloth==2026.1.4" "unsloth_zoo==2026.1.4" "trl==0.24.0" "transformers==4.57.6" \
    "peft==0.21.2" "accelerate==1.15.0" "datasets==4.3.0" "torchao==0.16.0" \
    "cuda-bindings==12.9.4" "spglib==2.7.0" "pymatgen==2026.9.24" "pymatgen-core==2026.10.2" \
    "scipy==1.18.1" "bitsandbytes==0.50.2" pandas numpy wandb
fi

"$VENV/bin/python" - <<'PY'
import torch, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cc", torch.cuda.get_device_capability(0))
for p in ["vllm", "unsloth", "unsloth_zoo", "trl", "transformers", "peft", "torchao", "numpy", "pymatgen-core", "spglib"]:
    print(p, m.version(p))
import unsloth, vllm
from trl import GRPOTrainer, GRPOConfig
print("grpo_vllm env OK")
PY
