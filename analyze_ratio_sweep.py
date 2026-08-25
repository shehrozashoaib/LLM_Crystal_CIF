#!/usr/bin/env python3
"""
analyze_ratio_sweep.py — paired significance testing for the target-emphasis
diagonal sweep (§4.3.2 of experiment_framework.md).

Every run is graded on the SAME frozen 8,096-crystal MPTS-52 test set, so the
per-material best-of-10 outcomes are paired and McNemar's test applies directly.
That is what lets sub-1 pp differences be resolved instead of hand-waved as
"within seed noise" — framework §3.4.

Per-material best-of-10 success = OR over Match_generation_1..N in a run's
validation `*_16bit.csv`.

Usage:
    python analyze_ratio_sweep.py                          # default comparisons
    python analyze_ratio_sweep.py --pairs A:B C:D          # explicit pairs
    python analyze_ratio_sweep.py --out results/mcnemar_ratio_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import sys
from pathlib import Path

csv.field_size_limit(10 ** 9)

TRUE = {"1", "true", "True", "TRUE", "yes", "Y"}

# Default comparisons: the two questions the sweep actually raises.
DEFAULT_PAIRS = [
    ("ratio_3to4", "ratio_4to3"),    # is the tail uptick real?
    ("ratio_3to4", "comp_mp20_00"),  # is the drop below baseline real?
    ("ratio_1to7", "comp_mp20_00"),  # does a light warmup beat pure MPTS-52?
    ("ratio_1to7", "ratio_4to3"),    # ends of the diagonal
]


def find_16bit(run: str, results_dir: Path) -> Path:
    """Locate a run's per-material grader output.

    Two conventions exist in results/: the composition/rank sweeps wrote
    `per_material_results.csv.gz`, later sweeps wrote `*_16bit.csv`. Both carry
    the same material_id + Match_generation_* schema.
    """
    vdir = results_dir / run / "validation"
    for pattern in ("*_16bit.csv", "per_material_results.csv.gz", "*_16bit.csv.gz"):
        hits = sorted(glob.glob(str(vdir / pattern)))
        if hits:
            return Path(hits[0])
    sys.exit(f"[FATAL] no per-material grader CSV under {vdir}")


def _open(path: Path):
    return (gzip.open(path, "rt", newline="") if path.suffix == ".gz"
            else open(path, newline=""))


def load_outcomes(path: Path) -> dict[str, int]:
    """material_id -> 1 if ANY generation matched (best-of-N), else 0."""
    out: dict[str, int] = {}
    with _open(path) as fh:
        rdr = csv.DictReader(fh)
        match_cols = [c for c in (rdr.fieldnames or []) if c.startswith("Match_generation_")]
        if not match_cols:
            sys.exit(f"[FATAL] no Match_generation_* columns in {path}")
        for row in rdr:
            mid = row["material_id"]
            out[mid] = int(any(str(row.get(c, "")).strip() in TRUE for c in match_cols))
    return out


def mcnemar(a: dict[str, int], b: dict[str, int]) -> dict:
    """Exact (binomial) McNemar on the paired per-material outcomes."""
    from scipy.stats import binomtest

    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("[FATAL] runs share no material_ids — not a paired comparison")
    b_only = sum(1 for m in shared if a[m] == 1 and b[m] == 0)   # A wins
    c_only = sum(1 for m in shared if a[m] == 0 and b[m] == 1)   # B wins
    n_disc = b_only + c_only
    p = binomtest(b_only, n_disc, 0.5).pvalue if n_disc else 1.0
    acc_a = sum(a[m] for m in shared) / len(shared)
    acc_b = sum(b[m] for m in shared) / len(shared)
    return {
        "n_paired": len(shared),
        "acc_a": round(100 * acc_a, 2),
        "acc_b": round(100 * acc_b, 2),
        "delta_pp": round(100 * (acc_a - acc_b), 2),
        "a_only_wins": b_only,
        "b_only_wins": c_only,
        "n_discordant": n_disc,
        "p_value": p,
        "significant_at_0.05": "yes" if p < 0.05 else "no",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="comparisons as RUN_A:RUN_B (default: the sweep's key questions)")
    ap.add_argument("--out", default="results/mcnemar_ratio_sweep.csv")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    pairs = ([tuple(p.split(":", 1)) for p in args.pairs] if args.pairs else DEFAULT_PAIRS)

    cache: dict[str, dict[str, int]] = {}
    rows = []
    for run_a, run_b in pairs:
        for r in (run_a, run_b):
            if r not in cache:
                cache[r] = load_outcomes(find_16bit(r, rdir))
        res = mcnemar(cache[run_a], cache[run_b])
        res = {"run_a": run_a, "run_b": run_b, **res}
        rows.append(res)
        print(f"{run_a:>14} vs {run_b:<14}  "
              f"{res['acc_a']:5.2f}% vs {res['acc_b']:5.2f}%  "
              f"Δ={res['delta_pp']:+5.2f} pp  "
              f"discordant {res['a_only_wins']}/{res['b_only_wins']}  "
              f"p={res['p_value']:.4g}  "
              f"{'SIGNIFICANT' if res['p_value'] < 0.05 else 'not significant'}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {outp}")


if __name__ == "__main__":
    main()
