#!/usr/bin/env python3
"""
evaluation/track_crystext_reward.py — reward-signal record for the CrysText-reward GRPO run.

Reads the JSONL traces the trainer writes and reports how the CrysText reward is
behaving over training: the reward distribution, which rung of their ladder each
completion landed on, the parse rate, and — the number that decides whether GRPO
learns anything — the fraction of crystal groups carrying within-group variance.
A group whose 6 generations all score the same contributes zero advantage.

    /venv/py312/bin/python evaluation/track_crystext_reward.py [--exp experiments/grpo_crystext_reward]
                                                    [--csv record.csv] [--window 50]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# CrysText's ladder, for reference in the printed record
LADDER = [
    ("parses (pymatgen)",                 +0.5),
    ("structure_validity",                +0.5),
    ("reduced formula == reference",      +0.5),
    ("match stol=0.9 ltol=0.7 ang=20",   +0.25),
    ("match stol=0.7 ltol=0.5 ang=15",   +0.25),
    ("match stol=0.5 ltol=0.3 ang=10",   +1.0),
    ("exception (unparseable)",           -2.0),
]
TIERS = ["val", "med", "loose", "no_match", "not_parseable"]


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # trainer may be mid-write on the last line
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments/grpo_crystext_reward")
    ap.add_argument("--window", type=int, default=50,
                    help="steps per bucket in the over-time table")
    ap.add_argument("--csv", default=None, help="also write the per-bucket table here")
    args = ap.parse_args()

    exp = Path(args.exp)
    rows = _read_jsonl(exp / "reward_trace.jsonl")
    groups = _read_jsonl(exp / "reward_group_trace.jsonl")
    vals = _read_jsonl(exp / "val_trace.jsonl")

    if not rows:
        raise SystemExit(f"no reward_trace.jsonl under {exp} yet")

    df = pd.DataFrame(rows)
    gdf = pd.DataFrame(groups)

    # completions arrive in generation-batch order; one batch == one step
    n_per_step = int(df.groupby("batch_hash").size().mode()[0])
    df["step"] = np.arange(len(df)) // max(1, n_per_step) + 1
    gdf["step"] = np.arange(len(gdf)) // max(1, len(gdf) // df["step"].max() or 1) + 1

    print("=" * 74)
    print("CrysText reward — signal record")
    print("=" * 74)
    print(f"experiment      {exp}")
    print(f"completions     {len(df)}   crystals {len(gdf)}   steps {df['step'].max()}")
    print(f"per step        {n_per_step} completions = "
          f"{n_per_step // 6} crystals x 6 generations")
    print()
    print("reward ladder (theirs, verbatim):")
    for name, pts in LADDER:
        print(f"   {pts:+5.2f}  {name}")
    print()

    r = df["final_reward"].astype(float)
    print(f"reward   mean {r.mean():+.3f}   std {r.std():.3f}   "
          f"min {r.min():+.2f}   max {r.max():+.2f}")
    print(f"         observed values: {sorted(r.unique().tolist())}")
    print(f"parse rate      {100 * df['parse_ok'].mean():.1f}%")
    print(f"clipped         {int(df['is_clipped'].sum())} / {len(df)}"
          f"  (completions that hit max_completion_length)")
    print()

    print("tier composition (all completions):")
    c = Counter(df["match_tier"])
    for t in TIERS:
        n = c.get(t, 0)
        print(f"   {t:<15} {n:>6}  {100 * n / len(df):5.1f}%  "
              f"{'#' * int(40 * n / len(df))}")
    print()

    if not gdf.empty and "has_reward_variance" in gdf:
        frac = gdf["has_reward_variance"].mean()
        print(f"GROUPS WITH REWARD VARIANCE   {gdf['has_reward_variance'].sum()}"
              f"/{len(gdf)}  ({100 * frac:.1f}%)")
        print("   ^ groups whose generations all score the same give GRPO no gradient")
        print()

    # ---- over time -------------------------------------------------------
    w = args.window
    df["bucket"] = ((df["step"] - 1) // w) * w + 1
    agg = df.groupby("bucket").agg(
        completions=("final_reward", "size"),
        reward_mean=("final_reward", "mean"),
        reward_std=("final_reward", "std"),
        parse_rate=("parse_ok", "mean"),
        val_rate=("match_val", "mean"),
    )
    agg["parse_rate"] *= 100
    agg["val_rate"] *= 100
    if not gdf.empty and "has_reward_variance" in gdf:
        gdf["bucket"] = ((gdf["step"] - 1) // w) * w + 1
        agg["grp_var_%"] = gdf.groupby("bucket")["has_reward_variance"].mean() * 100

    print(f"over time (buckets of {w} steps):")
    print(agg.round(3).to_string())
    print()

    if vals:
        vdf = pd.DataFrame(vals)
        keep = [c for c in ["step", "val_rate", "material_rate", "val_matches",
                            "not_parseable", "total_gens"] if c in vdf]
        print("validation monitor (StructureMatcher @ stol=0.5 on held-out val):")
        print(vdf[keep].round(4).to_string(index=False))
    else:
        print("validation monitor: no checks yet")

    if args.csv:
        agg.to_csv(args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
