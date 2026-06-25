# Experiment Framework — LLM-Driven CIF Generation
### Disentangling Data Composition, LoRA Rank, Curriculum, and GRPO

> Purpose: a single reference for the resubmission experiments. For each lever it states **what the paper claimed**, **what the reviewers objected to**, **what to hold fixed vs. change**, **the exact runs**, and **what each run proves**.

---

## 0. Current status (last updated 2026-06-16)

Pipeline implemented and verified end-to-end (dataset build → SFT train → vLLM generate → validate w/ RMSE panel). Runs use the **full 8,096-sample MPTS-52 test set**, **24,000 unique training crystals**, **pinned 4,500 steps**, final model (no early stopping), leakage-filtered.

| Lever | Run | Status | Best-of-10 |
|---|---|---|---:|
| **Composition** | 0:100 (baseline) | ✅ done | **30.1%** |
| **Composition** | 25:75 | ✅ done | **30.4%** |
| **Composition** | 50:50 | ✅ done | **29.5%** |
| **Composition** | 75:25 | ✅ done | **28.0%** |
| **Composition** | 100:0 | ✅ done | **26.6%** |
| → *finding* | composition sweep COMPLETE | ✅ | match ↓ monotonically as MP-20 ↑ at matched volume/steps → gain was **volume, not composition** |
| Volume control (#2) | combined-54k vs oversampled-MPTS | ⬜ not started | – |
| **LoRA rank (#3)** | r=16/32/64/128 × 2 seeds (0.53/1.05/2.08/4.07% params) | ✅ done | **28.3 / 30.0 / 31.3 / 33.6%** (mean) |
| → *finding* | monotonic, no plateau (+1.3–2.3pp/doubling); seed spread ≤0.5pp | ✅ | rank is **not** the weakest lever up to ~4% params |
| **Curriculum (#4)** | forward (MP-20→MPTS-52) @ S=4500 | ✅ done | **30.7%** |
| **Curriculum (#5)** | reverse (MPTS-52→MP-20) @ S=4500 | ✅ done | **27.5%** |
| → *forgetting* | MP-20: fwd 65.7→60.4 (−5.3) · rev 53.5→69.8 (+16.3) | ✅ | recency-dominated (best at last-trained set) |
| GRPO (#6–9) | best-of-N, RAFT, cont-SFT, KL sweep | ⬜ not started | – |

Legend: ✅ done · ⏳ in progress · ⬜ pending. See §5 for the full run table.

---

## 1. Background: the paper and why it was rejected

**Paper.** Fine-tune **Qwen2.5-7B-Instruct + LoRA** to emit a full CIF from a prompt of *reduced composition + target space-group number*. Three levers studied: LoRA rank, curriculum vs. combined data, and GRPO (discrete vs. continuous reward). Central claim: **"Data composition beats curriculum and GRPO."**

**Published results (best-of-10 structure match on 1000 held-out MPTS-52, pymatgen StructureMatcher):**

| Variant | Best-of-10 match |
|---|---:|
| SFT r=32 (baseline) | 24.4% |
| SFT r=64 | 26.4% |
| Curriculum (MP-20 → MPTS-52) | 30.3% |
| Combined (MP-20 + MPTS-52) | 36.1% |
| GRPO on baseline (per-gen 12.2 → 14.0–14.1) | 25.6–25.9% |

Compute (effective batch 32 = batch 4 × 8 grad-accum): baseline/r64 = 4,275 steps; combined = 5,170 (~21% over baseline); curriculum = 7,156 (5,076 MP-20 + 2,080 MPTS-52). Noise: best-of-10 varies ~0.6% across seeds, ~2% across test-split draws.

**Reject — the five reviewer/AC concerns:**
1. **Central claim unproven (decisive).** The combined model also saw *more total data* than the MPTS-52 baseline, so the win may be **volume/mixture**, not symmetry composition. Curriculum uses the same two sets but *sequentially*. Volume × composition × schedule are confounded. They explicitly asked for **matched-size / duplicated / resampled** controls and **matched steps**.
2. **No RMSE; no stability.** Best-of-N match is binary — no sense of *how wrong* a miss is. No e_above_hull / DFT energy → downstream usefulness unshown.
3. **Diffusion comparison not head-to-head.** CDVAE (21%) / DiffCSP (34%) numbers were taken from Jiao et al. at N=20, different split (chronological 27,380/5,000/8,096) and different conditioning (LLM gets composition **+** space group; DiffCSP is composition-only). "Exceeds DiffCSP at half the budget" is therefore not a fair comparison.
4. **No LLM baselines** (CrystaLLM, Gruver et al., CrysText).
5. **Downstream value not demonstrated.**

---

## 2. Datasets (real counts — `*_reduced_withmpids` splits)

| | Train | Val | Test | Total | Atoms (max / median) |
|---|---:|---:|---:|---:|---|
| **MP-20** (easy / high-symmetry) | 27,136 | 9,047 | 9,046 | 45,229 | 20 / ~10 |
| **MPTS-52** (hard / low-symmetry) | 27,380 | 5,000 | 8,096 | 40,476 | 52 / ~15–24 |

**Three facts that constrain every design below:**
- **Train ceilings.** You can only train on the *train* split. Max unique MPTS-52 training crystals = **27,380**; max unique MP-20 = **27,136**. A 50k *pure-MPTS-52 training set is impossible* without duplication (only ~27k unique exist). "MPTS-52 ≈ 50k" was the *total* (train+val+test); ~13k of it is val/test you must never train on.
- **Leakage trap (ID overlap).** MP-20 and MPTS-52 share material IDs partially. So some MP-20 *train* crystals are the same materials as MPTS-52 *test* crystals. **Before any mixing, drop from the MP-20 portion every `material_id` present in the MPTS-52 test (and val) set.** Otherwise adding MP-20 leaks the eval answers. *(TODO: compute exact overlap once the data folder is connected.)*
- **Token asymmetry.** MP-20 CIFs (~10 atoms) are shorter than MPTS-52 CIFs (median ~1,092 tokens on MPTS-52 train). So a 50/50 mix *by crystal count* is **not** 50/50 *by tokens*. Decide upfront whether to balance by count or by tokens. *(TODO: compute MP-20 token lengths for a token-balanced table.)*

---

## 3. Cross-cutting rules (apply to ALL experiments)

These are the controls that turn "we got a higher number" into "we proved why."

1. **Frozen, leakage-free test set.** Carve out the MPTS-52 test crystals once, freeze them, grade *every* run on the **same 1000-sample subset**. No test/val crystal ever enters any training set (including via MP-20 ID overlap).
2. **Match the budget.** When two runs are compared, hold **total optimizer steps + total example-presentations + total tokens** equal. State which budget metric is matched in every comparison.
3. **Pin the schedule.** Use explicit `max_steps`; **disable per-phase early stopping** for controlled runs (early stopping is what made the published curriculum drift to 7,156 steps). Report actual steps.
4. **Replicate + paired stats.** ≥3 SFT seeds per condition. Evaluate every variant on the *same* fixed split (cross-split noise then cancels). Use **McNemar's test / paired bootstrap** over the 1000 per-item binary matches → lets you resolve sub-2% effects instead of discarding them.
5. **Report a metric panel, not one number**, for every run:
   - per-generation (single-shot) match,
   - **best-of-N curve** (N = 1, 2, 5, 10, 20),
   - **RMSE** (StructureMatcher RMS distribution for matches + near-misses),
   - validity (parseable / valid elements / inferable SG),
   - **sample diversity** (distinct structures among the N, or mean pairwise RMS).

---

## 4. Experiment categories

### 4.1 Composition (Data) — the decisive experiment

| | |
|---|---|
| **Paper claim** | Combined (MP-20+MPTS-52) beats baseline; gains in low-symmetry (triclinic/monoclinic). |
| **Reviewer concern** | Could be volume, not composition (combined saw more data). |
| **Hold fixed** | Total training size, total steps, seeds, eval split. |
| **Change** | Only the MP-20 : MPTS-52 **ratio**. |

**Design — fixed-size SWAP sweep (no duplication needed).** Fix the training set at **27,000 crystals** (both pools exceed this, so all ratios are unique), and *replace* hard with easy rather than adding on top:

| Ratio MP-20 : MPTS-52 | MP-20 | MPTS-52 | Notes |
|---|---:|---:|---|
| 0 : 100 (= baseline) | 0 | 27,000 | anchors to your published 24.4% |
| 25 : 75 | 6,750 | 20,250 | unique |
| 50 : 50 | 13,500 | 13,500 | unique |
| 75 : 25 | 20,250 | 6,750 | unique |
| 100 : 0 | 27,000 | 0 | unique |

- **Proves:** at *constant volume/steps*, does composition move accuracy? A rising-then-falling curve = composition effect that volume cannot explain.
- **Volume control (mirrors the reviewers' "duplication/resampling").** One extra run: the full ~54k pooled corpus ("combined", as in the paper) **vs.** an *oversampled pure-MPTS-52* set duplicated to ~54k. If combined > duplicated-MPTS → composition; if they tie → volume.
- **Gotchas:** apply leakage filter; pick count- *or* token-balanced (state which); ≥3 seeds; McNemar vs. baseline.

---

### 4.2 Training Parameters (LoRA rank)

| | |
|---|---|
| **Paper claim** | Doubling rank (32→64) gives only ~+1.2%; least useful lever. |
| **Reviewer concern** | (Implicit) is the ranking real or within noise? |
| **Hold fixed** | Same data (pure MPTS-52), same steps, same eval split. |
| **Change** | LoRA rank only. |

**Design.** Sweep **r ∈ {16, 32, 64, 128}** (α = 2r), pure MPTS-52, identical budget, ≥3 seeds each.
- **Proves:** the *shape* of the rank→accuracy curve and whether the +1.2% (32→64) clears the seed band. Brackets the claim "more parameters is the weakest lever" with a curve instead of a single point.
- **Gotchas:** keep dropout, LR, sequence length fixed; only rank/α change.

**Trainable parameters per rank** (Qwen2.5-7B + LoRA on q,k,v,o,gate,up,down; % = trainable / (base 7.62B + LoRA), as Unsloth reports). Use this to read the rank axis as "fraction of the model optimized" — it doubles with r:

| LoRA rank | α (=2r) | Trainable params | % of model optimized |
|---:|---:|---:|---:|
| 16 | 32 | 40,370,176 | 0.53% |
| 32 | 64 | 80,740,352 | 1.05% |
| 64 | 128 | 161,480,704 | 2.08% |
| 128 | 256 | 322,961,408 | 4.07% |

**Result (2-seed mean, MPTS-52 best-of-10 / strict-RMS median Å):** r16 28.3% / 0.053 · r32 30.0% / 0.050 · r64 31.3% / 0.048 · r128 33.6% / 0.043 — monotonic, no plateau (+1.3–2.3 pp per doubling), matches tighten with rank, seed spread ≤0.5 pp (so the ordering is real, not noise). Per-seed: r16 28.1/28.5 · r32 29.9/30.2 · r64 31.3/31.3 · r128 33.4/33.9.

---

### 4.3 Curriculum (Schedule)

| | |
|---|---|
| **Paper claim** | MP-20 → MPTS-52 curriculum beats rank scaling (+6%), below combined. |
| **Reviewer concern** | Curriculum saw more data **and** more steps (7,156) than combined (5,170); confounds schedule with volume/compute. |
| **Hold fixed** | Same crystal pool as the mixed run, same **total** steps, same per-crystal exposure, same eval split. |
| **Change** | Only the **order** (interleaved vs. sequential). |

**Design — matched-budget schedule comparison** (on the leakage-safe MP-20 + MPTS-52 union, graded on the same held-out MPTS-52 crystals as the composition sweep):
- **Budget `S` = 4,500** — chosen to MATCH the composition sweep (`run_composition_sweep.sh`, `--max_steps 4500`) so curriculum runs are directly comparable to the committed composition results. (The paper's combined run used 5,170 and its curriculum drifted to 7,156; we pin to our own 4,500 instead. Override with `MAX_STEPS`.)
- **Mixed (reference):** union, shuffled, `S` steps. Opt-in only here — the composition sweep already trains the shuffled union, so the curriculum sweep defaults to `forward reverse`.
- **Curriculum-forward:** MP-20 then MPTS-52, with `S₁ + S₂ = S`. Split proportional to pool crystal count (MP-20 = 24,154 / MPTS-52 = 27,380) → **2,109 + 2,391 = 4,500**. **Disable early stopping**; pin `max_steps`.
- **Curriculum-reverse:** MPTS-52 then MP-20, same per-pool step counts (**2,391 + 2,109**).
- **Result:** forward 30.7% vs reverse 27.5% on MPTS-52 (+3.2 pp for ending on the eval set); forgetting probe MP-20 fwd 65.7→60.4 (−5.3), rev 53.5→69.8 (+16.3) → **recency dominates**.
- **Forgetting probe:** grade MP-20 accuracy *before and after* the second phase.

- **Proves:** mixed vs. forward at matched budget = pure **order** effect. Forward vs. reverse = is it "easy-to-hard" or just **recency** (whatever was trained last)? Forgetting probe = how much sequential training overwrites; mixed is the no-forgetting reference.
- **Key narrative:** forward ends *on the eval distribution* (an advantage), yet combined still wins on *fewer* steps — strong once budgets are matched.

---

### 4.4 GRPO (Reinforcement Learning)

| | |
|---|---|
| **Paper finding** | GRPO on the r=32 SFT: per-gen 12.2 → 14.0–14.1; best-of-10 24.4 → 25.6–25.9 (marginal). Discrete ≈ continuous. Continuous reward ~25× denser signal but **held-out flat** (drift). |
| **Reviewer concern** | Practical value of the whole approach unclear; this lever looks ineffective. |
| **Problem** | The test wasn't built to decide it: **no GRPO seed variance measured**, only the weak prior tried, no RL-free baseline, only the binary best-of-10 metric. |

**Step 0 — define "helpful for what?"** GRPO *helped single-shot* (per-gen +1.8) but not *best-of-10*. Report and conclude **per regime**.

**Step 1 — the deciding figure: best-of-N curve + diversity.** Sample 20 CIFs/prompt; plot match at N = 1, 2, 5, 10, 20 for SFT vs. each GRPO variant on the frozen set, with seed bands, plus a diversity metric.
- whole curve lifts → genuinely better;
- N=1 up but curves converge by N=10 → **diversity collapse** (better-but-similar samples) — the likely mechanism behind the flat best-of-10, and a publishable finding;
- curve left-shifts → helpful as **sample efficiency** (same accuracy, fewer samples).

**Step 2 — the controls GRPO must beat (matched compute):**
1. **Continued SFT** — train SFT on ground-truth CIFs for the same extra steps GRPO used. If "just train longer" matches GRPO, RL added nothing.
2. **Rejection-sampling fine-tuning (RAFT/STaR/best-of-N)** — sample from SFT, keep the matching/high-reward CIFs, SFT on them. **The single most important baseline:** if it ties GRPO, the group-relative machinery buys nothing over filtered imitation. Reviewers will ask for this by name.

GRPO is "helpful" only if it beats **both** beyond the seed band.

**Step 3 — replicate.** ≥3 GRPO seeds + repeated SFT evals → noise band. The +1.3 best-of-10 currently sits between seed (~0.6%) and split (~2%) noise; McNemar on paired items.

**Step 4 — tune enough that a null is credible:**
- **KL coefficient** (leash to the SFT reference) — the continuous-reward "reward up, held-out down" is the textbook symptom of *too weak* a KL; sweep it.
- **Group size / rollouts** — 4 is noisy and collapses to zero advantage when all 4 land in one reward tier; try **8–16**.
- **Learning rate** — 5e-7 is tiny; try ±1 step.

**Step 5 — apply GRPO to BOTH priors** (r=32 baseline *and* combined). If GRPO only helps the weak prior and is redundant on the good-data model, that *strengthens* the "data wins" thesis.

**Step 6 — RMSE + drift plot.** Report StructureMatcher RMS (GRPO, esp. continuous reward, may tighten near-misses without flipping a match). Plot training-reward vs. held-out-match over steps — the gap *is* the drift, quantified.

- **Decision rule:** GRPO is helpful **iff** it beats continued-SFT *and* rejection-sampling-FT, beyond the seed band, in the named regime — and the best-of-N + diversity figure says *why*.

---

## 5. Consolidated run table

| # | Run | Lever | Held fixed | Changed | Kills which concern |
|---|---|---|---|---|---|
| 1 | Composition swap sweep (0/25/50/75/100% MP-20 @ **24k**) ✅ *(all 5 done — match ↓ with MP-20; see §0)* | Data | size, steps, seeds, split | ratio | #1 (volume vs composition) |
| 2 | Combined-54k vs oversampled-MPTS-54k | Data | size, steps | mixture vs duplication | #1 (volume control) |
| 3 | LoRA rank sweep r∈{16,32,64,128} | Params | data, steps | rank | ranking realism |
| 4 | Mixed vs curriculum-forward @ matched S | Schedule | pool, steps, exposure | order | #1 (schedule confound) |
| 5 | Curriculum reverse + forgetting probe | Schedule | budget | order direction | recency vs easy-to-hard |
| 6 | GRPO best-of-N curve + diversity (both priors) | RL | inference N, split | RL on/off, prior | "GRPO useless?" → regime + mechanism |
| 7 | Rejection-sampling-FT vs GRPO @ matched compute | RL | compute | RL vs filtered-SFT | is RL needed at all |
| 8 | Continued-SFT vs GRPO @ matched compute | RL | compute | RL vs more SFT | trivial-explanation control |
| 9 | GRPO KL / group-size / LR mini-sweep | RL | data, prior | hyperparams | "you under-tuned" rebuttal |

All runs: same frozen 1000-sample MPTS-52 test set, ≥3 seeds, report the full metric panel (per-gen, best-of-N, RMSE, validity, diversity), McNemar for pairwise significance.

---

## 6. Positioning track (separate from the causal experiments)

- **Fair diffusion comparison:** rerun DiffCSP (and/or CDVAE) on *your* 1000-sample subset with matched conditioning (composition + space group) and matched N — or, if infeasible, state the split/conditioning/budget differences explicitly and stop claiming "exceeds DiffCSP."
- **LLM baselines:** at least one of CrystaLLM / Gruver et al. / CrysText on the same eval.
- **Downstream signal:** report e_above_hull or DFT-relaxed energy on a sample of generated CIFs to move from "matches a reference" toward "physically plausible."

---

## 7. Suggested order of execution

1. **Plumbing first:** frozen split + leakage filter + the metric panel (best-of-N curve, RMSE, diversity, McNemar). Everything else depends on it.
2. **Composition swap sweep (#1)** + volume control (#2) — answers the decisive concern.
3. **Curriculum matched-budget (#4–5)** — cheap once #1 plumbing exists.
4. **Rank sweep (#3)**.
5. **GRPO battery (#6–9)** — rejection-sampling baseline (#7) and best-of-N curve (#6) first; they do most of the work.
6. **Positioning track (§6)** in parallel.

---

### Open TODOs (need the data/code folder connected)
- [ ] Compute MP-20 ↔ MPTS-52-test/val `material_id` overlap (leakage filter size).
- [ ] Compute MP-20 token lengths → token-balanced composition table.
- [ ] Confirm effective batch / step counts from the training scripts to finalize matched-step numbers.
