# Running LLM_Crystal_CIF on GH200 (aarch64 / Hopper sm_90)

This file documents the issues hit when bringing this repo up on an **NVIDIA GH200**
(ARM64 host, Hopper `sm_90`, CUDA 12.8 toolkit / 13.0 driver), the fixes applied, and
the exact from-scratch procedure to reproduce a working environment.

Audience: anyone who has to stand this repo up again on a similar Grace-Hopper /
aarch64 box. Read the **"From scratch"** section first if you just want it running.

---

## TL;DR

| Symptom | Root cause | Fix |
|---|---|---|
| CUDA OOM (`Tried to allocate 38.90 GiB`) during training, esp. at larger microbatch; training also slow | No flash-attn / xformers → Unsloth uses PyTorch SDPA, which falls back to the **MATH** attention backend (materializes full N×N scores: ~62 GB at micro_batch=8). MATH is forced because padding-free packing passes a 4D mask **and** Qwen2.5 GQA sets `enable_gqa=True`. | Pin attention to **cuDNN first** with `sdpa_kernel([CUDNN, FLASH, EFFICIENT, MATH], set_priority=True)` around `trainer.train()` in `code_FineTune.py`. ~62 GB → ~2.4 GB, ~1.3× faster. **No new deps.** |
| Considered building `flash-attn` from source | aarch64 + torch 2.10 has no prebuilt wheels → 20–40 min source compile, ABI-risky against bleeding-edge torch | **Don't.** The cuDNN SDPA pin above gives the win with zero build. (An abandoned attempt is in `logs_flashattn_build.log`.) |
| Two CUDA toolchains needed (train vs generate) | Unsloth/trl pinned to torch 2.10+cu128; vLLM 0.22 wants torch 2.11+cu130; they conflict in one venv | **Two venvs**: `/venv/py312` (train + validate), `/venv/vllm` (generate only). See setup scripts. |
| Curriculum phase-2 must continue the *same* LoRA, not start fresh | Default `code_FineTune.py` always called `get_peft_model` (fresh adapter) | Added `--init_adapter <dir>` to load+continue an existing adapter (continued-SFT). Empty = fresh (default). |

---

## Environment (what actually works here)

- GPU: **NVIDIA GH200 480GB**, compute capability **(9, 0)** = Hopper `sm_90`, single visible GPU with ~**94.5 GB** usable.
- Host: **aarch64** (ARM64), Ubuntu 22.04. 64 CPU, ~525 GB RAM.
- Toolkit: `nvcc` 12.8 present; driver exposes CUDA 13.0.
- Package manager: **`uv`** (fast, handles the `+cuXXX` torch wheels cleanly).

Two isolated venvs, built by the scripts in this repo:

| venv | torch | key libs | used by |
|---|---|---|---|
| `/venv/py312` | `2.10.0+cu128` | unsloth, unsloth_zoo, trl 0.24, peft, bitsandbytes, pymatgen, seaborn | `code_FineTune.py` (train), `cif_structure_validator_mp52.py` (validate) |
| `/venv/vllm` | `2.11.0+cu130` | vllm 0.22, transformers, pandas | `generate_cifs_vllm.py` (inference only — vLLM cannot train) |

---

## Issues & fixes in detail

### 1. The attention-backend OOM / slowdown (the big one)

**Symptom.** Training OOM'd with `torch.OutOfMemoryError: CUDA out of memory. Tried to
allocate 38.90 GiB` inside `unsloth/utils/attention_dispatch.py → scaled_dot_product_attention`.
The model header showed `FA [Xformers = None. FA2 = False]` — i.e. no fast attention backend.

**Diagnosis.** Benchmarking the exact attention shape (B=8, 28 q-heads, 4 kv-heads, N=4096,
bf16, +grad) across SDPA backends:

| backend | peak mem | fwd+bwd |
|---|---|---|
| MATH (what ran) | ~62 GB | ~200 ms |
| mem-efficient | fails (no GQA + mask) | — |
| flash | fails (mask not supported) | — |
| **cuDNN** | **~2.4 GB** | **~24 ms** |

Two conditions force MATH: (a) Unsloth's **padding-free packing** hands SDPA a 4D
block-diagonal `attn_mask`, and (b) **Qwen2.5 GQA** (28 q / 4 kv heads) calls SDPA with
`enable_gqa=True`. Only the **cuDNN** fused kernel handles *mask + GQA together*.

**The subtlety that cost an iteration:** simply listing cuDNN in `sdpa_kernel([...])` is not
enough — PyTorch's *default* backend priority reaches MATH before cuDNN, so it still OOM'd.
You must pass **`set_priority=True`** with cuDNN **first** so list order is honored.

**Fix** (in `code_FineTune.py`, wrapping `trainer.train()`):
```python
from torch.nn.attention import sdpa_kernel, SDPBackend
SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                 SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
with sdpa_kernel(SDPA_BACKENDS, set_priority=True):
    trainer_stats = trainer.train(...)
```
FLASH/mem-efficient/MATH remain as fallbacks for other shapes (unmasked causal → FLASH;
odd shapes → MATH).

**Verified.** The micro_batch=16 config that OOM'd now trains clean (loss ~0.45, unchanged →
cuDNN is numerically sound). End-to-end ~**1.3×** faster (2.43 vs ~1.88 samples/s). Peak
training memory dropped to ~**24.6 GB** total (of 94.5).

### 2. flash-attn build — attempted and abandoned

A source build of `flash-attn==2.8.3.post1` was started (no aarch64/torch-2.10 wheels exist).
It is **unnecessary**: cuDNN SDPA already gives the memory + speed win with zero build and no
ABI risk against torch 2.10. Don't go down this path. Remnant: `logs_flashattn_build.log`.

### 3. Two-venv split (train vs generate)

unsloth/trl and vLLM pin incompatible torch builds. Keep them separate (`setup_py312.sh`,
`setup_vllm.sh`). The sweep scripts already invoke the right interpreter per stage
(`PYTHON=/venv/py312/bin/python`, `VLLM_PYTHON=/venv/vllm/bin/python`).

### 4. `--init_adapter` for curriculum phase 2

`code_FineTune.py` now takes `--init_adapter <adapter_dir>`: it loads base+adapter as a
trainable PEFT model and continues training the **same** LoRA weights (continued-SFT), which
is what makes the curriculum / forgetting probe meaningful. Empty → fresh LoRA (default).
`--lora_r/alpha/dropout` are inherited from the loaded adapter when continuing.

---

## Parallelism on one GPU — verdict: **don't (for training)**

Measured on a single training run (8×4, seq 4096, cuDNN fix):
- Peak memory ~**24.6 GB** of 94.5 → memory alone would fit 2–3 concurrent runs.
- **But SM utilization saturates at 100%** during compute; the 0% gaps are
  gradient-offload / optimizer / grad-accum boundaries, not idle compute.

Training is **compute-bound**: two concurrent jobs contend for the same SMs and roughly
time-slice, so aggregate throughput won't approach 2× — and you pay higher fragmentation /
OOM risk and tangled logs. **Run the curriculum conditions sequentially** (the sweep's
default). The only safe overlap would be the CPU-bound validation (pymatgen StructureMatcher)
running alongside the next condition's GPU training, but the sweep doesn't do that and it
isn't worth retrofitting.

---

## From scratch (exact steps)

```bash
# 0. Prereqs: aarch64 GH200 box, nvcc 12.8+, `uv` installed, this repo checked out.
cd /home/ubuntu/LLM_Crystal_CIF

# 1. Build the two venvs (idempotent; each ends with a smoke test).
./setup_py312.sh        # /venv/py312  — train + validate  (torch 2.10+cu128)
./setup_vllm.sh         # /venv/vllm   — generate           (torch 2.11+cu130)

# 2. (Already applied here) the cuDNN SDPA pin in code_FineTune.py around
#    trainer.train(). Without it, training OOMs / is slow. See Issue #1.

# 3. Build datasets if not present (leakage-safe pools + frozen test sets).
/venv/py312/bin/python build_curriculum_datasets.py     # -> Data/curriculum/*

# 4. Sanity smoke (tiny, ~2 min): trains a few steps, generates, validates.
/venv/py312/bin/python code_FineTune.py \
  --train_csv Data/_smoke/time.csv --run_name smoke_chk --max_steps 6 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_seq_length 4096
#   expect: "✅ Done." and train_loss ~0.45 — NOT an OOM.

# 5. Launch the matched-budget curriculum sweep.
#    Budget S defaults to MAX_STEPS=4500 to MATCH the composition sweep (so the
#    runs are comparable, graded on the same 8096 held-out MPTS-52 crystals).
#    Default conditions are `forward reverse` — `mixed` is opt-in because the
#    composition sweep already trains the shuffled union.
./run_curriculum_sweep.sh                 # forward + reverse, sequentially
./run_curriculum_sweep.sh mixed           # add the reference run if you want it
MAX_STEPS=5170 ./run_curriculum_sweep.sh  # override the matched budget
#   resumable: any stage whose output exists is skipped.
#   fast grading pass: TEST_CSV=Data/composition_sweep/test_frozen_mp52_1000.csv TEST_N=1000 ./run_curriculum_sweep.sh
```

Phase-step splits are derived proportionally from the pool sizes (MP-20 = 24,154,
MPTS-52 = 27,380) and always sum to S: at **S=4500** → forward `2109 + 2391`,
reverse `2391 + 2109`. Each pool gets the same #steps in both orders, so only the
*schedule* differs. The matched test set is `Data/source/mp_52_test.csv.gz` (the
same 8096 crystals, verified id-for-id, the composition runs were graded on).

### Throughput knobs (don't touch the science)

`MICRO_BATCH * GRAD_ACCUM` is held at the matched effective batch of **32**, so changing the
shape is a pure throughput knob. Measured (effective batch 32, cuDNN):
`8×4 ≈ 16×2 ≈ 2.43–2.45 samples/s`; `32×1` regresses (2.11, gradient offload). The sweep
default **`8×4`** is already optimal and memory-conservative — leave it.

### Gotchas
- Always run training/validation with `/venv/py312`, generation with `/venv/vllm`.
- If you see `FA2 = False` in the Unsloth header, that's expected here — the cuDNN SDPA
  pin (not flash-attn) is what provides fast/low-memory attention.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is a harmless extra guard against
  fragmentation; the cuDNN fix is what actually removes the OOM.
