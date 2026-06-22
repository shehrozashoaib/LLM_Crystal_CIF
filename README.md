# LLM_Crystal_CIF

Fine-tuning **Qwen2.5-7B-Instruct + LoRA** (via [Unsloth](https://github.com/unslothai/unsloth)) to generate full **CIF crystal structures** from a prompt of *reduced composition + target space-group number*, and a controlled experiment suite to disentangle **data composition, LoRA rank, curriculum, and GRPO**.

This repo is the code + data + results for a paper resubmission. The experimental design, the reviewer concerns it addresses, and the exact runs are documented in **[`experiment_framework.md`](experiment_framework.md)**.

---

## What's here

```
.
├── experiment_framework.md         # the experiment plan (with completion status)
├── README.md                       # this file
│
├── build_composition_datasets.py   # builds the fixed-volume, leakage-free composition sweep datasets
├── code_FineTune.py                # SFT trainer (Unsloth + LoRA), CLI-driven, pinned steps, no early stopping
├── generate_cifs_vllm.py           # fast inference with vLLM (paged attention + continuous batching)
├── generate_cifs_qwen_chat.py      # HuggingFace generate() fallback (slower; same I/O contract)
├── cif_structure_validator_mp52.py # grades generated CIFs vs ground truth (best-of-N match + RMSE panel)
├── run_composition_sweep.sh        # orchestrator: train → generate → validate, one ratio at a time
├── analyze_*.py                    # SFT / GRPO run + per-space-group + step analyses
├── grpo_from_repair_mix_sft_target_aligned_v3.py  # GRPO trainer (StructureMatcher reward)
│
├── Data/
│   ├── source/                     # original MP-20 / MPTS-52 train/val/test splits (gzipped)
│   └── composition_sweep/          # generated sweep datasets (gzipped) + manifest.json
│
└── results/
    ├── comp_mp20_00/               # ratio 0:100 (pure MPTS-52 baseline) — predicted CIFs + validation
    └── comp_mp20_50/               # ratio 50:50 — predicted CIFs + validation
```

All `*.csv.gz` are gzip-compressed (CIF text compresses ~6–7×). `pandas.read_csv` reads `.gz`
transparently; or `gunzip` them first. Model weights / checkpoints are **not** committed.

---

## Environments

Two isolated Python envs (inference pins different torch/transformers than training):

| Env | Path | Used for | Key libs |
|---|---|---|---|
| **py312** | `/venv/py312` | training + validation | unsloth, trl 0.24, torch 2.10+cu128, pymatgen, seaborn |
| **vllm**  | `/venv/vllm`  | inference/generation | vllm 0.22, torch 2.11+cu130 |

GPU used: **NVIDIA RTX PRO 6000 (Blackwell)** → needs CUDA ≥ 12.8 wheels (both envs satisfy this).
vLLM *cannot* train; it is inference only. Training stays on Unsloth.

---

## The pipeline (composition lever)

One command runs the whole sweep, one ratio at a time:

```bash
# build datasets once (writes Data/composition_sweep/ + manifest.json)
/venv/py312/bin/python build_composition_datasets.py

# run any subset of ratios (MP-20% = 00/25/50/75/100): train → generate → validate
./run_composition_sweep.sh 00 25 50 75 100
```

Per ratio `comp_mp20_<tag>` the orchestrator:
1. **Trains** (`code_FineTune.py`, py312) — pinned to `--max_steps 4500`, LoRA r=32/α=64, lr 1e-4, **early stopping disabled** → `experiments/comp_mp20_<tag>/final_model/`.
2. **Generates** (`generate_cifs_vllm.py`, vllm) — 10 CIFs/prompt on the MPTS-52 test set → `generated/comp_mp20_<tag>/`.
3. **Validates** (`cif_structure_validator_mp52.py`, py312) — best-of-10 match + the RMSE panel → `experiments/comp_mp20_<tag>/validation_*/`.

Resumable (completed stages are skipped). Override knobs via env vars (`MAX_STEPS`, `RET_SEQS`,
`TEST_CSV`, `TEST_N`, `GPU_MEM_UTIL`, `GEN_CHUNK`).

---

## The controls that make it causal (see `experiment_framework.md`)

- **Leakage filter** — every MP-20 `material_id` present in the MPTS-52 test∪val set is dropped
  before mixing (2,982 rows). Verified zero leakage in every train set.
- **Matched volume** — every ratio trains on exactly **24,000 unique crystals** (capped by the
  post-filter MP-20 pool of 24,154; that's why 24k, not 27k).
- **Matched steps** — pinned 4,500 optimizer steps, early stopping off, final model used for grading
  (not a best-eval-loss checkpoint) — so every ratio is compared at identical compute.
- **Frozen test set** — graded on the MPTS-52 test set (default full 8,096; a frozen 1,000-sample
  subset is also provided).
- **Metric panel** — per-generation match, best-of-10, **RMSE (strict matched + near-miss)**,
  validity, composition/space-group accuracy.

---

## Results so far

See **[`experiment_framework.md`](experiment_framework.md)** for the full status table.

**Composition sweep — COMPLETE.** Best-of-10 structure match on the full 8,096 MPTS-52 test set,
at constant 24,000-crystal volume and pinned 4,500 steps (only the MP-20:MPTS-52 ratio varies):

| Ratio MP-20:MPTS-52 | Best-of-10 match | Strict matched-RMS (median Å) | Status |
|---|---:|---:|---|
| 0:100 (pure MPTS-52 baseline) | 30.1% | 0.050 | ✅ |
| 25:75                          | **30.4%** | 0.049 | ✅ |
| 50:50                          | 29.5% | 0.046 | ✅ |
| 75:25                          | 28.0% | 0.042 | ✅ |
| 100:0 (pure MP-20)             | 26.6% | 0.039 | ✅ |

**Key finding:** at matched volume **and** matched steps, adding MP-20 does **not** beat the baseline —
match peaks at low MP-20 (0–25%, ~30%) and declines monotonically to 26.6% at pure MP-20. The
published "combined beats baseline" gain therefore came from **volume**, not symmetry composition —
the decisive reviewer concern, answered. (Endpoints differ clearly on 8,096 paired samples; the
0/25/50 middle is within seed noise → needs ≥3 seeds + McNemar to separate.)

Secondary (from the RMSE panel): more MP-20 → **tighter** matches (median RMS 0.050→0.039 Å) but
**fewer** of them — MP-20 teaches precise high-symmetry geometry at the cost of low-symmetry coverage.
Dominant failure mode throughout: right composition (~92%), valid CIF (~76%), but wrong
geometry/symmetry (near-miss RMS clusters at 0.5–1.0 Å, nothing beyond 1.5 Å) — which the old binary
match metric hid.

**Curriculum sweep — COMPLETE.** Same leakage-safe MP-20 + MPTS-52 pool, matched **4,500 steps**
(split ∝ pool size: forward 2109+2391, reverse 2391+2109), graded on the same 8,096 MPTS-52 crystals.
Conditions `forward`/`reverse` (mixed is the composition sweep's shuffled union).

| Condition | MPTS-52 best-of-10 | MP-20 after phase-1 | MP-20 after phase-2 | MP-20 Δ |
|---|---:|---:|---:|---:|
| **Forward** (MP-20 → MPTS-52) | **30.7%** | 65.7% (peak) | 60.4% | **−5.3 pp** (forgetting) |
| **Reverse** (MPTS-52 → MP-20) | **27.5%** | 53.5% (MPTS-52-only) | 69.8% | **+16.3 pp** (recency) |

**Key finding — recency dominates:** each order is best at whatever it trained *last*. Forward ends on
the eval distribution and wins MPTS-52 by **+3.2 pp**; reverse ends on MP-20 and reaches the highest
MP-20 accuracy (69.8%). Sequential MPTS-52 training costs only ~5 pp of MP-20 (modest forgetting).

**LoRA rank sweep — seed 3407 done (seed 1234 running).** Pure MPTS-52, matched 4,500 steps, α=2r;
only rank changes. The % is the fraction of the model optimized (trainable / (base 7.62B + LoRA)):

| LoRA rank | % params optimized | MPTS-52 best-of-10 (seed 3407) |
|---:|---:|---:|
| r=16 | 0.53% | 28.1% |
| r=32 | 1.05% | 29.9% |
| r=64 | 2.08% | 31.3% |
| r=128 | 4.07% | **33.4%** |

**Key finding — no plateau:** match rises monotonically with rank (each doubling +1.4–2.1 pp, biggest
jump at 64→128). Going 0.53% → 4.07% of params buys +5.3 pp and is still climbing — so rank is *not*
the "weakest lever" the paper claimed, at least up to ~4% of params. Seed 1234 adds the error band.
(All prior composition/curriculum runs sit at the **r=32 ≈ 1%** point.)

> **Hardware note.** All results above were produced on an **NVIDIA GH200 (Hopper sm_90, aarch64)**.
> That architecture needs a specific attention-backend fix to train without OOM — if you are running on
> GH200/aarch64, read **[`README_GH200_SETUP.md`](README_GH200_SETUP.md)** before installing/training.
> `code_FineTune.py` selects the right path per architecture via `--arch {auto,a100,gh200}`.
