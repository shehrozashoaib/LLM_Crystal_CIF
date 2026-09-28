# LLM_Crystal_CIF

Fine-tuning **Qwen2.5-7B-Instruct + LoRA** (via [Unsloth](https://github.com/unslothai/unsloth)) to generate full **CIF crystal structures** from a prompt of *reduced composition + target space-group number*, and a controlled experiment suite to disentangle **data composition, LoRA rank, curriculum, and GRPO**.

This repo is the code + data + results for a paper resubmission. The experimental design, the reviewer concerns it addresses, and the exact runs are documented in **[`experiment_framework.md`](experiment_framework.md)**.

> **Headline:** at matched volume *and* matched steps, adding MP-20 never beats the pure-MPTS-52
> baseline — match declines monotonically as the MP-20 share rises. The published
> *"combined beats baseline"* gain came from **volume, not composition**. That is the decisive
> reviewer concern, answered.

---

## What's here

```
.
├── experiment_framework.md         # the experiment plan (with completion status)
├── README.md                       # this file
│
│   ── dataset builders ──
├── build_composition_datasets.py   # fixed-volume, leakage-free composition sweep (24,000 crystals)
├── build_curriculum_datasets.py    # curriculum phases; --budget_total caps the union to 24,000
├── build_ratio_datasets.py         # target-emphasis diagonal: data ratio == step ratio
│
│   ── pipeline ──
├── code_FineTune.py                # SFT trainer (Unsloth + LoRA), CLI-driven, pinned steps
├── generate_cifs_vllm.py           # fast inference with vLLM (resumable, shard-based)
├── generate_cifs_qwen_chat.py      # HuggingFace generate() fallback (same I/O contract)
├── cif_structure_validator_mp52.py # grades generated CIFs (best-of-N match + RMSE panel)
├── grpo_from_repair_mix_sft_target_aligned_v3.py  # GRPO trainer (StructureMatcher reward)
├── grpo_crystext_reward.py         # GRPO trainer with CrysText's reward (see below)
├── test_crystext_reward_parity.py  # proves that reward == CrysText's, rung for rung
├── track_crystext_reward.py        # reward-signal record: tiers, parse rate, group variance
├── setup_crystext_env.sh           # builds /venv/py312 on x86_64 + Blackwell
├── run_crystext_grpo.sh            # crash-resume launcher for the CrysText-reward run
├── run_eval_crystext.sh            # full-test eval: vLLM generate → pymatgen validate
├── crystext_hf_upload.py           # pushes the CrysText-run adapter to the HF model repo
│
│   ── orchestrators ──
├── run_composition_sweep.sh        # composition ratios
├── run_curriculum_sweep.sh         # forward / reverse curricula + forgetting probe
├── run_phase_split_sweep.sh        # §4.3.1 switch-point (k) sweep
├── run_ratio_sweep.sh              # §4.3.2 target-emphasis diagonal
├── run_rank_sweep.sh               # LoRA rank sweep
├── run_with_retry.sh               # resumable-sweep watchdog
├── stall_guard.sh                  # kills a wedged GPU stage (see Hardware notes)
│
│   ── analysis ──
├── analyze_ratio_sweep.py          # paired McNemar over the frozen test set
├── analyze_*.py                    # SFT / GRPO run + per-space-group + step analyses
│
├── Data/
│   ├── source/                     # original MP-20 / MPTS-52 train/val/test splits (gzipped)
│   ├── composition_sweep/          # composition datasets + manifest.json
│   ├── curriculum/                 # curriculum phase files + manifest_curriculum.json
│   └── ratio_sweep/                # diagonal datasets + manifest_ratio_sweep.json
│
└── results/                        # one subfolder per sweep; each run holds
    │                               #   predicted_cifs.csv.gz + validation/ panel
    ├── composition_sweep/          # comp_mp20_{00,25,50,75,100}
    ├── curriculum_order/           # curr_fwd / curr_rev + _p1 phase-1 forks (MP-20 probes)
    ├── phase_split/                # psplit_k1000 / k3000 / datamatch (uncapped pools)
    ├── ratio_sweep/                # ratio_1to7 ... ratio_4to3
    │   └── mcnemar_ratio_sweep.csv # paired significance tests
    ├── rank_sweep/                 # rank_r{16,32,64,128}_s{3407,1234}
    ├── grpo/                       # grpo_r32_from3000_{discrete,continuous} and
    │                               #   grpo_crystext_reward/ (2 evaluated checkpoints, plus
    │                               #   val_trace.jsonl (43 held-out checks), reward_trace.jsonl.gz
    │                               #   (26,568 completions), reward_group_trace.jsonl.gz (4,428
    │                               #   groups, within-group variance), reward_record.csv)
    └── training_times.csv          # spans every sweep
```

All `*.csv.gz` are gzip-compressed (CIF text compresses ~6–7×). `pandas.read_csv` reads `.gz`
transparently. Model weights / checkpoints are **not** committed — the LoRA adapters live at
[huggingface.co/shehrozashoaib/LLM_Crystal_CIF](https://huggingface.co/shehrozashoaib/LLM_Crystal_CIF).

---

## Environments

Two isolated Python envs (inference pins different torch/transformers than training):

| Env | Path | Used for | Key libs |
|---|---|---|---|
| **py312** | `/venv/py312` | training + validation | unsloth, trl 0.24, torch 2.10+cu128, pymatgen, seaborn |
| **vllm**  | `/venv/vllm`  | inference/generation | vllm 0.22, torch 2.11+cu130 |

vLLM *cannot* train; it is inference only. Training stays on Unsloth.
Build them with `setup_py312.sh` / `setup_vllm.sh`.

---

## The controls that make it causal

These are what turn "we got a higher number" into "we proved why" (full detail in
[`experiment_framework.md`](experiment_framework.md) §3):

- **Leakage filter** — every MP-20 `material_id` present in the MPTS-52 test∪val set is dropped
  before mixing (2,982 rows). Verified zero leakage in every train set, by assertion at build time.
- **Matched volume** — every run in the composition, rank and diagonal sweeps trains on exactly
  **24,000 unique crystals** (capped by the post-filter MP-20 pool of 24,154 — that's why 24k, not 27k).
- **Matched steps** — pinned **4,500** optimizer steps, early stopping off, effective batch 32,
  final model graded (not a best-eval-loss checkpoint) — so every run is compared at identical compute.
- **Matched exposure** — in the diagonal sweep the data split equals the step split, so **every phase
  runs exactly 6.00 epochs**, the same per-crystal exposure as a single-phase composition run. Without
  this, "more MP-20" silently also means "MP-20 seen more times."
- **Frozen test set** — graded on the full **8,096**-crystal MPTS-52 test set (a frozen 1,000 subset
  is also provided). Every run is graded on the *same* crystals, which is what makes the paired
  statistics below valid.
- **Paired statistics** — McNemar's exact test on per-material best-of-10 outcomes, so sub-1 pp
  differences are *resolved* rather than dismissed as noise.
- **Metric panel** — per-generation match, best-of-10, **RMSE** (strict matched + near-miss),
  validity, composition/space-group accuracy.

---

## Results

Best-of-10 structure match (pymatgen `StructureMatcher`) on the full 8,096-crystal MPTS-52 test set.
"strict-RMS" is the median RMS over matched structures.

### 1. Composition sweep — COMPLETE

Constant 24,000-crystal volume, pinned 4,500 steps; only the MP-20:MPTS-52 **ratio** varies.

| Ratio MP-20:MPTS-52 | best-of-10 | strict-RMS (Å) |
|---|---:|---:|
| 0:100 (pure MPTS-52 baseline) | 30.1% | 0.050 |
| 25:75 | **30.4%** | 0.049 |
| 50:50 | 29.5% | 0.046 |
| 75:25 | 28.0% | 0.042 |
| 100:0 (pure MP-20) | 26.6% | 0.039 |

**Finding.** At matched volume and steps, adding MP-20 does **not** beat the baseline: match peaks
at low MP-20 (0–25%) and falls monotonically to 26.6%. The published gain was **volume, not
composition**.

*Secondary:* more MP-20 → **tighter** matches (0.050 → 0.039 Å) but **fewer** of them. MP-20 teaches
precise high-symmetry geometry at the cost of low-symmetry coverage. Dominant failure mode
throughout: right composition (~92%), valid CIF (~76%), wrong geometry/symmetry — which the old
binary match metric hid.

### 2. Target-emphasis diagonal (§4.3.2) — COMPLETE

The composition sweep varies *what data* the model sees; the curriculum sweep varies *what order*.
This sweep varies **how much of the budget is spent on each domain**, holding everything else fixed.
The MP-20:MPTS-52 split is applied **identically to the data budget and the step budget**:

```
data_mp20  = round(24000 * X/(X+Y))     steps_mp20 = round(4500 * X/(X+Y))
```

Because the data fraction equals the step fraction, every phase runs **6.00 epochs** — identical
per-crystal exposure to the composition sweep. Only *where the emphasis falls* changes.

| Run | warm-up label `k` | MP-20 : MPTS-52 | MP-20 share | steps (P1+P2) | best-of-10 | strict-RMS |
|---|---:|---|---:|---|---:|---:|
| `comp_mp20_00` | 0 | 0 : 24,000 | 0% | 0 + 4500 | 30.1% | 0.050 |
| `ratio_1to7` | 500 | 3,000 : 21,000 | 12.5% | 563 + 3937 | **30.4%** | 0.052 |
| `ratio_2to7` | 1000 | 5,333 : 18,667 | 22.2% | 1000 + 3500 | 30.2% | 0.053 |
| `ratio_3to7` | 1500 | 7,200 : 16,800 | 30.0% | 1350 + 3150 | 29.7% | 0.052 |
| `ratio_3to4` | 2000 | 10,286 : 13,714 | 42.9% | 1929 + 2571 | 27.6% | 0.050 |
| `ratio_4to3` | 2500 | 13,714 : 10,286 | 57.1% | 2571 + 1929 | 28.1% | 0.053 |
| `comp_mp20_100` | 4500 | 24,000 : 0 | 100% | 4500 + 0 | 26.6% | 0.039 |

The `k` column is the warm-up index used on the curriculum figure's x axis, rounded to the
nearest 500 so the axis reads cleanly. **The `steps` column is what actually ran**, and the two
differ for three of the five runs (563 vs 500, 1350 vs 1500, 1929 vs 2000, 2571 vs 2500), because
the step split is set by the data ratio rather than chosen to land on a round number. Each run's
`validation/*_summary.txt` now carries the same mapping so a reader who opens a single result file
can tell which figure point it is. Quote the `steps` value in any text that describes the
experiment, and reserve `k` for referring to a position on the figure.

**Paired significance** (`results/ratio_sweep/mcnemar_ratio_sweep.csv`, McNemar exact, n = 8,096 paired):

| comparison | Δ | p | verdict |
|---|---:|---:|---|
| `ratio_1to7` vs baseline `comp_mp20_00` | +0.32 pp | 0.38 | **not significant** |
| `ratio_3to4` vs baseline | −2.51 pp | 2.0×10⁻¹² | **significant** |
| `ratio_1to7` vs `ratio_4to3` | +2.38 pp | 9.2×10⁻¹¹ | **significant** |
| `ratio_3to4` vs `ratio_4to3` | −0.44 pp | 0.22 | **not significant** |

**Findings.**
1. **A light MP-20 warmup does not help.** `ratio_1to7` (30.4%) is statistically
   indistinguishable from the pure-MPTS-52 baseline (30.1%, p = 0.38). The apparent +0.3 pp is noise.
2. **A heavy MP-20 emphasis genuinely hurts** — −2.5 pp at 42.9% MP-20, p ≈ 10⁻¹².
3. **The decline across the diagonal is real** (1:7 → 4:3, p ≈ 10⁻¹⁰), not an artefact of ordering.
4. **The tail is flat, not rising.** The 4:3 "uptick" over 3:4 is not significant (p = 0.22); the
   curve is best read as levelling off near ~28% rather than turning upward.

This independently reproduces the composition sweep's conclusion through a different lever: at
matched volume, exposure *and* steps, **there is no MP-20 mixture that beats training on the target
domain alone.**

### 3. Curriculum (§4.3) — COMPLETE †

Same leakage-safe pools trained in two orders at matched 4,500 steps (split ∝ pool size).

| Condition | MPTS-52 best-of-10 | strict-RMS (Å) | MP-20 after P1 | MP-20 after P2 | MP-20 Δ |
|---|---:|---:|---:|---:|---:|
| **Forward** (MP-20 → MPTS-52) † | 30.7% | 0.049 | 65.7% | 60.4% | **−5.3 pp** (forgetting) |
| **Reverse** (MPTS-52 → MP-20) † | 27.5% | 0.044 | 53.5% | 69.8% | **+16.3 pp** (recency) |

**Finding — recency dominates.** Each order is best at whatever it trained *last*. Forward ends on
the eval distribution and wins MPTS-52 by +3.2 pp; reverse reaches the highest MP-20 accuracy.
Sequential MPTS-52 training costs only ~5 pp of MP-20 (modest forgetting).

† **These two runs are not volume-matched** — see [Corrections & provenance](#corrections--provenance).

### 4. Curriculum phase-split (§4.3.1) — COMPLETE †

Fix the 4,500-step budget; vary the MP-20→MPTS-52 switch point `k`.

| k (MP-20 steps → then MPTS-52) | best-of-10 | strict-RMS (Å) |
|---:|---:|---:|
| 0 (pure MPTS-52) | 30.1% | 0.050 |
| **1000** † | **32.1%** | 0.049 |
| 2109 (forward) † | 30.7% | 0.049 |
| 3000 † | 29.9% | 0.054 |
| 4500 (pure MP-20) | 26.6% | 0.039 |

† `k = 1000/2109/3000` and the retired `psplit_datamatch` control all trained on **larger pools than
the 24,000-crystal baseline they are compared against**, so the peak at k=1000 is confounded with
volume. See [Corrections & provenance](#corrections--provenance). **The 32.1% result has not been
reproduced at matched volume and should not be quoted as a matched-budget win.**

### 5. LoRA rank sweep — COMPLETE (2 seeds)

Pure MPTS-52, matched 4,500 steps, α = 2r; only rank changes. The % is the fraction of the model
optimized (trainable / (base 7.62 B + LoRA)).

| LoRA rank | % params optimized | best-of-10 (s3407 / s1234 / **mean**) | strict-RMS (Å) |
|---:|---:|---:|---:|
| r=16 | 0.53% | 28.1 / 28.5 / **28.3%** | 0.053 |
| r=32 | 1.05% | 29.9 / 30.2 / **30.0%** | 0.050 |
| r=64 | 2.08% | 31.3 / 31.3 / **31.3%** | 0.048 |
| r=128 | 4.07% | 33.4 / 33.9 / **33.6%** | **0.043** |

**Finding — no plateau, and it's real.** Match rises monotonically with rank (+1.3–2.3 pp per
doubling, biggest jump at 64→128). Going 0.53% → 4.07% of params buys **+5.3 pp** and is still
climbing — so rank is *not* the "weakest lever" the paper claimed, at least to ~4% of params. The
**seed spread is only 0.0–0.5 pp**, far below the inter-rank gaps. Matches also get **tighter**
(0.053 → 0.043 Å).

### 6. GRPO (RL, §4.4) — COMPLETE: negative

Forked from the r=32 SFT checkpoint-3000, 1,500 GRPO steps → 4,500 total (matched to continued-SFT).

| model | reward | group | best-of-10 | strict-RMS (Å) |
|---|---|---:|---:|---:|
| **continued-SFT** (baseline, same fork) | — | — | **29.9%** | **0.050** |
| GRPO final | discrete | 4 | 27.7% | 0.066 |
| GRPO final | continuous | 8 | 28.1% | 0.063 |

**GRPO underperforms continued-SFT by ~2 pp at matched compute — regardless of reward shape or group
size**, and its RMS is *worse* too. Diagnostics explain why: training reward stayed **flat** across
all 1,500 steps with a live advantage signal (`frac_reward_zero_std=0`) and non-trivial policy
movement (KL≈0.25) — RL moved the policy, but the match-reward objective yielded no improvable
gradient from a near-converged SFT prior. The reward *shape* wasn't the bottleneck; the objective is.

**Cross-check with an external reward.** To test whether the reward *design* was the culprit, we
re-ran GRPO with the scoring function from [CrysText](https://github.com/truptimohanty/CrysText)
copied verbatim (and their hyperparameters and batch geometry), from the r=32 SFT adapter:

| model | reward | best-of-10 | strict-RMS (Å) |
|---|---|---:|---:|
| **SFT start point** (`rank_r32_s3407`) | — | **29.9%** | **0.050** |
| GRPO @ step 150 | CrysText, verbatim | 29.8% | 0.053 |
| GRPO @ step 1150 | CrysText, verbatim | 24.3% | 0.108 |

Their reward *preserves* the SFT model early — 29.8% vs 29.9%, inside noise on 8,096 samples —
where our discrete/continuous rewards give up ~2 pp. So reward shape does matter at the margin.
But it still never **beats** SFT, and it degrades the same way with more steps (−5.6 pp by step
1150, with matches twice as loose). A third reward design reaching the same ceiling is independent
support for the conclusion above: **the objective, not the reward shape, is the bottleneck.**
Details, traces and the surviving adapter: the [CrysText-reward section](#grpo-with-the-crystext-reward-external-reward-baseline) below.

---

## GRPO with the CrysText reward (external-reward baseline)

`grpo_crystext_reward.py` runs our GRPO setup — Qwen2.5-7B-Instruct + Unsloth LoRA, val-set
monitor, best-model tracking, JSONL reward traces — with the **reward function from
[truptimohanty/CrysText](https://github.com/truptimohanty/CrysText/blob/main/grpo_training.py)**
instead of ours. This isolates the reward design: same model, same data, same optimiser,
different scoring rule.

**Their reward (verbatim, range `[-2, 3]`):**

| test | points |
|---|---|
| generated CIF parses with pymatgen | +0.5 |
| `structure_validity` (min interatomic dist 0.5 Å, volume > 0.1, space group assignable) | +0.5 |
| reduced formula == reference reduced formula | +0.5 |
| StructureMatcher match, `stol=0.9 ltol=0.7 angle_tol=20` | +0.25 |
| StructureMatcher match, `stol=0.7 ltol=0.5 angle_tol=15` | +0.25 |
| StructureMatcher match, `stol=0.5 ltol=0.3 angle_tol=10` | +1.0 |
| anything raises (unparseable generation) | −2 |

Their tightest rung is exactly our `_MATCHER_VAL`, so `match_tier` in the traces keeps this
repo's `val / med / loose / no_match / not_parseable` names and the files drop straight into
`analyze_grpo_reward_traces.py`.

**Ours, kept unchanged so runs stay comparable:** Qwen2.5 chat-template prompting (CrysText
used an Alpaca string on a Mistral base), tokenizer handling, `Data/source/mp_52_*.csv.gz`,
`HYPERPARAMS`/`GRPO_KW`, and the validation monitor (still StructureMatcher at `stol=0.5`,
*not* the CrysText score, so val match rate is comparable with the continuous/discrete runs).

**Two adaptations at the call site — neither touches the scoring math:**

1. CrysText fed the raw completion to pymatgen because their base model emitted bare CIF.
   Qwen2.5-**Instruct** wraps output in prose and code fences, which pymatgen scores −2
   (verified in the parity test's `fenced` case), so the extracted CIF block is what gets
   scored. `CRYSTEXT_RAW_COMPLETION=1` restores their literal behaviour.
2. Their `Structure.from_str(answer, ...)` on the *reference* CIF sits outside the try block,
   so one bad ground-truth row raises and kills the run. Here it is caught, logged, and
   scored −2 like any other failure.

**Tokenizer (Qwen2.5 / ChatML).** `_configure_qwen_tokenizer()` enforces: `eos = <|im_end|>`
so generation stops at the end of the assistant turn instead of running to
`max_completion_length` (which would clip CIFs and hand the reward truncated text);
`pad = <|endoftext|>`, kept distinct from eos; left padding for batched generation; and a
hard failure if the checkpoint's tokenizer has no chat template. Both terminators are passed
as `eos_token_id` during val generation.

### Results

Run: start from `rank_r32_s3407`, CrysText's reward *and* CrysText's hyperparameters
(lr 1e-6, temperature 1.0, `paged_adamw_8bit`, linear schedule, wd 0.1, warmup 0.1,
betas 0.9/0.99, grad-clip 0.1) and their batch geometry (`per_device_train_batch_size=12`,
`gradient_accumulation_steps=1`, `num_generations=6` → **12 completions per step = 2 crystals ×
6 generations**). Stopped at step 2,184 / 3,000 because held-out match was falling.

| Checkpoint | Best-of-10 (8,096 test) | Strict-RMS (med Å) |
|---|---:|---:|
| SFT start point (`rank_r32_s3407`) | 29.9% | 0.050 |
| step 150 | 29.8% | 0.053 |
| step 1150 | 24.3% | 0.108 |

The decline is monotone in every window of the run — this is the training-side record from
`results/grpo/grpo_crystext_reward/reward_record.csv`, where *val-tier* is the share of completions
matching at `stol=0.5` and *group variance* is the share of crystal groups whose 6 generations
do not all score identically (a group with no spread contributes no GRPO gradient):

| steps | reward mean | val-tier | group variance |
|---|---:|---:|---:|
| 1–400 | 2.137 | 44.8% | 66.9% |
| 801–1200 | 1.915 | 33.6% | 79.3% |
| 1601–2000 | 1.780 | 21.2% | 88.9% |
| 2001+ | 1.714 | 20.4% | 88.7% |

Rising group variance alongside falling match quality is generations *scattering*, not the model
finding better options. Held-out checks (`val_trace.jsonl`, 43 checks) peak at 0.45 per-material
at step 150 and sit in the 0.10–0.25 band thereafter.

**Interpretation.** Their reward pays 1.5–2.0 out of 3.0 for a structure that parses, is valid and
has the right formula but matches *nothing* — so most of the reward mass is reachable without
getting the geometry right, and the gradient pushes toward "safe, plausible, wrong". Ours paid ≈0
for the same output, which is a cleaner signal in principle but produced fewer informative groups.
Neither beats simply stopping after SFT. The untested variant is **their reward at our lr (5e-7)**,
which would separate the reward design from the 2× learning rate.

Artifacts: `results/grpo/grpo_crystext_reward/` (predictions, validation, reward + val traces).
Adapter: [`grpo_crystext_reward/step2150`](https://huggingface.co/shehrozashoaib/LLM_Crystal_CIF)
on the HF model repo — that is the *last* checkpoint (degraded, and not itself evaluated); the two
checkpoints that were evaluated no longer exist as weights, see
[Corrections & provenance #5](#5-best-model-tracking-did-not-survive-a-resume-weights-lost).

### Run it

```bash
bash setup_crystext_env.sh                      # -> /venv/py312 (torch cu128, trl 0.24, unsloth, pymatgen)

# prove the reward is bit-identical to CrysText's before spending GPU-hours
git clone https://github.com/truptimohanty/CrysText ../CrysText
/venv/py312/bin/python test_crystext_reward_parity.py --n 10

# 2-step wiring check on a small model
GRPO_SMOKE=1 GRPO_START_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  /venv/py312/bin/python grpo_crystext_reward.py

# the real run (crash-resume loop, same pattern as grpo_run.sh)
GRPO_START_MODEL=experiments/rank_r32_s3407/checkpoints/checkpoint-3000 \
  bash run_crystext_grpo.sh

# reward-signal record at any point during/after training
/venv/py312/bin/python track_crystext_reward.py --window 50 --csv reward_record.csv

# full 8,096-material test eval: vLLM generate -> pymatgen validate
bash setup_vllm.sh                              # -> /venv/vllm
bash run_eval_crystext.sh best                  # SYSTEM_PROMPT=sft for comparability with results/
```

`test_crystext_reward_parity.py` lifts both scorers out of their source files (neither is
importable — both build a model at import time) and runs them head-to-head on real MPTS-52
CIFs plus mutations that span every rung: **180/180 identical, rewards spanning
`{-2.0, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0}`**. Note that rigid translation and uniform volume
scaling are *not* useful mutations — StructureMatcher normalises both — so the test perturbs
sites individually and applies anisotropic strain.

**Env overrides:** `GRPO_START_MODEL`, `GRPO_START_TOKENIZER`, `GRPO_OUTPUT_DIR`,
`CRYSTEXT_RAW_COMPLETION=1`, `CRYSTEXT_HPARAMS=1` (their lr `1e-6`, `temperature 1.0`,
`num_generations 6`, `weight_decay 0.1`, `warmup_ratio 0.1`, `paged_adamw_8bit`, linear schedule,
grad-clip 0.1), `GRPO_PDBS` / `GRPO_GRAD_ACCUM` / `GRPO_NUM_GEN` (batch geometry),
`GRPO_MAX_STEPS`, `GRPO_SAVE_STEPS`, `GRPO_BETA`, `GRPO_SMOKE=1`.

> **Reward scale.** CrysText's range is `[-2, 3]`, ours was `[-0.5, 1.05]`. GRPO normalises
> advantages by within-group std, so the scale itself is not the lever — the *shape* is. When
> comparing runs, read `has_match_variance` in `reward_group_trace.jsonl`, not `reward_mean`:
> mean reward is dominated by the partial credit their ladder pays for parseable-but-unmatched
> structures, and it moved far less than test-set match did (2.14 → 1.71 while best-of-10 fell
> 29.8% → 24.3%).

---

## Corrections & provenance

Issues found while preparing the diagonal sweep and the CrysText GRPO run. They are recorded here
rather than quietly patched, because several of them affect numbers already reported.

### 1. The curriculum family is not volume-matched

Reconstructed from git history of `Data/curriculum/`:

| committed data | at 2026-06-22 / 07-02 (when curriculum results were produced) | at 2026-07-15 (current) |
|---|---:|---:|
| `train_phase_mp20` | 24,154 | 11,249 |
| `train_phase_mp52` | 27,380 | 12,751 |
| `train_mixed` | **51,534** | **24,000** |

`curr_fwd`, `curr_rev`, `psplit_mp20base`, `psplit_k1000`, `psplit_k3000` therefore trained on the
**uncapped 51,534-crystal union — 2.15× the 24,000 crystals of the baseline they are compared
against.** `psplit_datamatch` used 7,823 + 27,380 = 35,203. At a pinned 4,500 steps (144,000
example-presentations) that is ~2.8 epochs versus the baseline's 6.0, i.e. **more unique data at the
same compute** — the very volume confound this study exists to remove.

The 24k-capped datasets were uploaded on 2026-07-15, *after* all of those runs, and **no committed
run has ever consumed them.** All affected numbers are marked † above.

**Not affected:** the composition sweep (24,000 unique since 2026-06-16, never changed), the rank
sweep (reads `Data/composition_sweep/train_mp20_00.csv.gz`), and GRPO (forked from an r=32 SFT
checkpoint trained on the 24k composition data).

### 2. `psplit_datamatch` is superseded

`psplit_datamatch` (31.2%) was intended as the "data ratio = step ratio 2:7" control, but it kept
MPTS-52 at full size, so it trained on 35,203 crystals. **`ratio_2to7` (30.2%) is the correct
control**: same 1000:3500 step split, at a genuinely matched 24,000 crystals. The earlier claim that
warmup benefit comes from MP-20 *diversity* rather than repetition rests on the confounded number
and is not supported at matched volume.

**Still open:** whether `psplit_k1000`'s 32.1% survives volume matching. The diagonal's `ratio_2to7`
shares its step split but also shrinks the MP-20 *pool*, so it cannot separate "full pool, few
steps" from "small pool, few steps." Settling that needs a k=1000 run on the capped 24k pools
(`Data/curriculum/`, now regenerable — see below). It is the outstanding gap in this study.

### 3. The curriculum builder could not reproduce its own data

`build_curriculum_datasets.py` generated the full 51,534-crystal union while `Data/curriculum/`
shipped the 24k-capped files — code and data disagreed, and the capping script was never committed.
Fixed: `--budget_total` (default **24000**) caps the union, subsampled proportional to the
leakage-safe pools. Verified to regenerate the committed files exactly (11,249 / 12,751 / 24,000).
Pass `--budget_total 0` to reproduce the older uncapped datasets.

### 4. Every GRPO run was evaluated under a prompt it was not trained on

`code_FineTune.py` (SFT) and both generation scripts use the system message
`"You are an expert in materials science and crystallography."` — while **all three** `grpo_*.py`
trainers append `" Return only one complete CIF file and nothing else."` So `grpo_r32_from3000_discrete`,
`grpo_r32_from3000_continuous` and the CrysText run were all *optimised* on one prompt and *scored*
on another. `generate_cifs_vllm.py` now takes `--system_prompt {sft,grpo}`, defaulting to `sft` so
every number already in `results/` is unchanged; the CrysText-run evals used `grpo`. Re-running an
older GRPO eval with `--system_prompt grpo` would quantify what the mismatch cost.

### 5. Best-model tracking did not survive a resume (weights lost)

`GRPOValidationCallback` initialised `best_val_match_rate = 0.0`, and a resumed run builds a fresh
callback — so on resuming the CrysText run from step 1150, the first validation check overwrote
`best_model/` with a *worse* checkpoint, and `save_total_limit=10` later evicted checkpoint-1150.
The weights of both evaluated checkpoints (steps 150 and 1150) are therefore gone; their
predictions and validation panels are published in full under `results/grpo/grpo_crystext_reward/`, so
the reported numbers remain verifiable, but those adapters cannot be re-run. Fixed in
`grpo_crystext_reward.py`: the callback restores the previous best from `best_info.json`, so
`best_model/` can only ever improve. **Port that fix before resuming any other `grpo_*.py` trainer.**

### Provenance of the diagonal runs

`ratio_1to7`, `ratio_2to7`, `ratio_3to7` were produced before `build_ratio_datasets.py` was
committed. The builder reproduces their splits **exactly** (3,000/21,000 · 5,333/18,667 ·
7,200/16,800 and the matching step splits), so the design is identical — but their original builder
was not preserved, so the specific crystals sampled may differ from a fresh rebuild.
`ratio_3to4` and `ratio_4to3` were produced by the committed builder.

`ratio_3to4` phase 1 ran on the stock xformers/SDPA attention path; phase 2 onward and all of
`ratio_4to3` used `--arch gh200` (cuDNN SDPA pinned). Same mathematics, different kernels.

Wall-clock training times are **not** reported for the diagonal runs: they were interrupted and
resumed repeatedly on unstable hardware, so `training_stats.json` records only the final resumed
segment. See `results/training_times.csv` for the runs where the figure is meaningful.

---

## Reproduce it

```bash
# 1. build datasets (writes Data/<sweep>/ + a manifest with leakage + volume assertions)
/venv/py312/bin/python build_composition_datasets.py
/venv/py312/bin/python build_curriculum_datasets.py            # --budget_total 24000 by default
/venv/py312/bin/python build_ratio_datasets.py 1:7 2:7 3:7 3:4 4:3

# 2. run any sweep: train -> generate -> validate, resumable end to end
./run_composition_sweep.sh 00 25 50 75 100
./run_ratio_sweep.sh 3:4 4:3
./run_rank_sweep.sh 16 32 64 128

# 3. paired significance over the frozen test set
/venv/py312/bin/python analyze_ratio_sweep.py
```

Every stage is skipped if its output already exists, so a killed sweep resumes where it stopped.
Generation is checkpointed per 1,000-prompt shard, and a partial output CSV is **regenerated rather
than graded** (a row-count check, not a file-existence check — an earlier version of this pipeline
could silently validate a truncated test set).

Knobs: `MAX_STEPS`, `RET_SEQS`, `TEST_CSV`, `TEST_N`, `GPU_MEM_UTIL`, `GEN_CHUNK`, `SAVE_STEPS`,
`MICRO_BATCH`/`GRAD_ACCUM` (product pinned to the matched effective batch of 32), `ARCH`.

---

## Hardware notes

Results were produced on two machines. Nothing about the science depends on which, but the setup does:

**NVIDIA GH200 (Hopper sm_90, aarch64)** — composition, rank, curriculum, phase-split, GRPO, and the
1:7/2:7/3:7 diagonal points. Needs a specific attention-backend fix to train without OOM; read
**[`README_GH200_SETUP.md`](README_GH200_SETUP.md)** before installing. `code_FineTune.py` selects
the path via `--arch {auto,a100,gh200}`.

**NVIDIA RTX PRO 6000 Blackwell (sm_120, x86_64)** — `ratio_3to4`, `ratio_4to3`. Caveats found there:

- **vLLM's FlashInfer path fails on sm_120 with a CUDA < 12.9 toolkit**, reporting the misleading
  `FlashInfer requires GPUs with sm75 or higher`. Root cause: FlashInfer resolves the CUDA version
  from the *system* `nvcc` (12.8 in this image) rather than torch's bundled runtime (cu130), then
  refuses SM 12.x — the sm75 message is vLLM re-raising it. It bites in two places, so both need
  routing around: `VLLM_USE_FLASHINFER_SAMPLER=0` for the sampler and
  `VLLM_ATTENTION_BACKEND=FLASH_ATTN` for attention (the runners set both automatically).
  Installing a CUDA ≥ 12.9 toolkit also fixes it.
- **Intermittent segmentation faults** hit training *and* inference across three independent stacks
  (unsloth+xformers, unsloth+cuDNN, and vLLM on a different torch/CUDA build) — most consistent with
  a driver/hardware issue rather than a library bug. The pipeline is built to survive them:
  `--save_steps` keeps checkpoints tighter than the mean time-to-failure, generation resumes per
  shard, and `stall_guard.sh` kills a stage whose GPU sits idle (a segfaulting vLLM `EngineCore` can
  leave the parent process blocked forever, which a plain exit-code watchdog will never notice).
- For long unattended runs, drive the sweep from a process supervisor rather than `nohup` — see the
  `run_with_retry.sh` + `stall_guard.sh` pairing.

---

## Authors

- **Shehroz Ahmad Shoaib** — [shehrozashoaib@gmail.com](mailto:shehrozashoaib@gmail.com)
- **Dr. Burhan SaifAddin** — [burhan.saifaddin@kfupm.edu.sa](mailto:burhan.saifaddin@kfupm.edu.sa)
