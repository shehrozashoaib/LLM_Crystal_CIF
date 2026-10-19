#!/usr/bin/env python3
"""
release/grpo_step_stats.py — per-step training curves of a finished GRPO run
============================================================================
Writes into OUT (the run's results dir):

  * reward_per_step.csv          one row per optimizer step, from reward_trace.jsonl /
                                 reward_group_trace.jsonl (every completion of every step):
                                 reward mean / std / variance over the step's completions,
                                 mean within-prompt-group std (the spread GRPO's advantages
                                 come from), fraction of groups with zero reward std,
                                 eval-tolerance match / parse / clipped rates, completion length
  * train_log_history.jsonl.gz   the trainer's log_history (logging_steps=5 averages: reward,
                                 reward_std, kl, loss, grad_norm, learning_rate, lengths, ...)
                                 from the last checkpoint's trainer_state.json

The traces carry a batch_hash per step but no step number. Steps are recovered from the order
of batches, keeping each batch's LAST occurrence (a resume regenerates the batches after the
checkpoint, and the regenerated ones are what was trained on).

    python release/grpo_step_stats.py experiments/<RUN> results/<...>/<RUN>
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics as stats
import sys
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def by_step(rows: list[dict]) -> list[list[dict]]:
    """Group consecutive rows by batch_hash; a repeated batch replaces its earlier occurrence."""
    runs: list[tuple[str, list[dict]]] = []
    for r in rows:
        if runs and runs[-1][0] == r["batch_hash"]:
            runs[-1][1].append(r)
        else:
            runs.append((r["batch_hash"], [r]))
    last = {h: i for i, (h, _) in enumerate(runs)}
    return [g for i, (h, g) in enumerate(runs) if last[h] == i]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def write(exp: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []

    comp = by_step(load_jsonl(exp / "reward_trace.jsonl"))
    groups = by_step(load_jsonl(exp / "reward_group_trace.jsonl"))
    if len(comp) != len(groups):
        raise SystemExit(f"{exp}: {len(comp)} steps in reward_trace vs {len(groups)} in reward_group_trace")
    cols = ["step", "n_completions", "n_prompts", "reward_mean", "reward_std", "reward_var",
            "reward_min", "reward_max", "group_reward_std_mean", "frac_groups_zero_std",
            "match_val_rate", "parse_rate", "clipped_rate", "token_len_mean"]
    p = out / "reward_per_step.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for step, (c, g) in enumerate(zip(comp, groups), start=1):
            r = [x["final_reward"] for x in c]
            w.writerow([step, len(c), len(g), mean(r), stats.pstdev(r), stats.pvariance(r), min(r), max(r),
                        mean(x["reward_std"] for x in g), mean(x["reward_std"] == 0 for x in g),
                        mean(x["match_val"] for x in c), mean(x["parse_ok"] for x in c),
                        mean(bool(x["is_clipped"]) for x in c), mean(x["token_len"] for x in c)])
    written.append(p)

    cks = [c for c in (exp / "checkpoints").glob("checkpoint-*") if c.name.split("-")[-1].isdigit()]
    if cks:
        ts = json.load(open(max(cks, key=lambda c: int(c.name.split("-")[-1])) / "trainer_state.json"))
        p = out / "train_log_history.jsonl.gz"
        with gzip.GzipFile(p, "wb", mtime=0) as gz:
            for rec in ts["log_history"]:
                gz.write((json.dumps(rec) + "\n").encode())
        written.append(p)
    return written


if __name__ == "__main__":
    for p in write(Path(sys.argv[1]), Path(sys.argv[2])):
        print(p)
