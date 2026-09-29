# paper_figures

Scripts that produce the figures in the manuscript. Each one is self-contained,
reads only files already published in `results/`, and writes a PDF and a
300-dpi PNG beside itself.

| script | figure | what it answers |
|---|---|---|
| `fig_best_of_n.py` | best-of-N curves, N = 1…10 | whether a lever improves the mode or just widens sampling |
| `fig_grpo_reward_variance.py` | within-group reward spread, and its source | why the median and mean spread disagree, and how often the advantage carries no match signal |
| `fig_levers_by_system.py` | per-crystal-system change with bootstrap CIs | which per-system differences are resolved and which are not |

`analyze_composition_sweep.py`, `analyze_curriculum_sweep.py` and
`analyze_levers_per_spg_24k.py` (the sweep curves and the per-space-group panel)
live alongside these and follow the same conventions.

## Input data

The scripts read per-material grader output and per-rollout reward traces from
a local `data/` directory. Populate it from the published results:

```bash
mkdir -p paper_figures/data
cp results/composition_sweep/comp_mp20_00/validation/per_material_results.csv.gz  paper_figures/data/base.csv.gz
cp results/composition_sweep/comp_mp20_25/validation/per_material_results.csv.gz  paper_figures/data/mix25.csv.gz
cp results/ratio_sweep/ratio_1to7/validation/*_16bit.csv                          paper_figures/data/curr.csv
cp results/rank_sweep/rank_r128_s3407/validation/*_16bit.csv                      paper_figures/data/r128.csv
cp results/grpo/grpo_r32_from3000_continuous/final/validation/*_16bit.csv         paper_figures/data/grpo.csv
cp results/grpo/grpo_r32_from3000_discrete/reward_trace.jsonl.gz                  paper_figures/data/rt.jsonl.gz
cp results/grpo/grpo_r32_from3000_continuous/reward_trace.jsonl.gz                paper_figures/data/rt_cont.jsonl.gz
```

`data/` is deliberately not committed: every file in it is a verbatim copy of
something already in `results/`, and duplicating it would add tens of megabytes
to a repository that is already large.

## Conventions

Colours are Okabe-Ito. `pdf.use14corefonts` keeps the PDFs to base-14 fonts so
they embed cleanly in LaTeX. Matched RMS is StructureMatcher's displacement
normalized by `(V/N_sites)^(1/3)`, so it is dimensionless and no axis carries an
angstrom unit.

Best-of-N uses the unbiased estimator of Chen et al. (2021),
`1 - C(n-c, N)/C(n, N)` averaged over materials, with `n = 10` generations and
`c` the number that matched. At `N = 10` it reproduces the reported best-of-10
exactly; at `N = 1` it equals the per-generation match rate.

Bootstrap intervals are percentile intervals over 2,000 resamples of materials,
drawn in pairs so a lever and the baseline always see the same crystals, which
is the same pairing the McNemar tests use.
