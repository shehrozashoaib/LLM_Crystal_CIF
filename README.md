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

See **[`experiment_framework.md`](experiment_framework.md)** for the live status table.
Composition sweep, best-of-10 structure match on the full 8,096 MPTS-52 test set:

| Ratio MP-20:MPTS-52 | Best-of-10 match | Status |
|---|---:|---|
| 0:100 (baseline)  | 30.1% | ✅ done |
| 25:75             |   –   | ⏳ training |
| 50:50             | 29.5% | ✅ done |
| 75:25             |   –   | ⬜ pending |
| 100:0             |   –   | ⬜ pending |

Dominant failure mode (from the RMSE panel): **right composition (~92%), valid CIF (~76%), but
wrong geometry/symmetry** — near-miss RMS clusters at 0.5–1.0 Å (nothing beyond 1.5 Å), which the
old binary match metric hid.
