#!/usr/bin/env python3
"""collect_timings.py — wall-clock timing for the curriculum + rank experiments.

Records, per run, the three pipeline stages the user asked for:
    train  : SFT training        -> experiments/<run>/training_stats.json (train_runtime)
    test   : generation on the   -> logs/<...>_gen.log  (first..last vLLM "INFO MM-DD HH:MM:SS")
             MPTS-52 test set        captures vLLM init + generation wall-clock
    val    : best-of-N + RMSE     -> experiments/<run>/validation_*/validation_log_*.log
             scoring                 (first..last "YYYY-MM-DD HH:MM:SS,mmm" line)

All sources are INTERNAL timestamps, so durations stay correct even when a stage was
resumed days after its sibling (file mtimes would be wrong there).

Writes experiments/timing_summary.csv and prints a table. Re-run any time to pick up
newly finished runs (e.g. rank seed 1234). Blank cell = stage not finished yet.

Usage:  /venv/py312/bin/python collect_timings.py
"""
import csv
import glob
import json
import os
import re
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
EXP_DIR = os.path.join(ROOT, "experiments")

# vLLM gen log:  "... INFO 06-18 20:28:20 ..."  (no year -> assume 2026, the run year)
_GEN_TS = re.compile(r"INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
# validator inner log line start: "2026-06-18 20:28:23,291 - INFO - ..."
_VAL_TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),\d+")
YEAR = 2026  # gen log omits the year; all runs are 2026


def _secs(a, b):
    return None if (a is None or b is None) else (b - a).total_seconds()


def train_secs(*run_dirs):
    """Sum train_runtime across one or more phase dirs (curriculum has 2 phases)."""
    total, found = 0.0, False
    for d in run_dirs:
        p = os.path.join(EXP_DIR, d, "training_stats.json")
        if os.path.isfile(p):
            try:
                rt = json.load(open(p)).get("train_runtime")
                if rt is not None:
                    total += float(rt); found = True
            except (ValueError, OSError):
                pass
    return total if found else None


def gen_secs(gen_log):
    """First..last vLLM INFO timestamp in a generation log."""
    p = os.path.join(LOG_DIR, gen_log)
    if not os.path.isfile(p):
        return None
    first = last = None
    with open(p, errors="ignore") as f:
        for line in f:
            m = _GEN_TS.search(line)
            if m:
                mo, d, h, mi, s = map(int, m.groups())
                ts = datetime(YEAR, mo, d, h, mi, s)
                first = first or ts
                last = ts
    return _secs(first, last)


def val_secs(run, dataset="mp52", subdir=""):
    """First..last line timestamp of the inner validation_log for a run/dataset."""
    base = os.path.join(EXP_DIR, run, subdir) if subdir else os.path.join(EXP_DIR, run)
    logs = glob.glob(os.path.join(base, f"validation_*{dataset}*", "validation_log_*.log"))
    if not logs:
        return None
    p = sorted(logs)[-1]  # most recent validation of this dataset
    first = last = None
    with open(p, errors="ignore") as f:
        for line in f:
            m = _VAL_TS.match(line)
            if m:
                y, mo, d, h, mi, s = map(int, m.groups())
                ts = datetime(y, mo, d, h, mi, s)
                first = first or ts
                last = ts
    return _secs(first, last)


def hms(sec):
    if sec is None:
        return ""
    sec = int(round(sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def main():
    rows = []

    # ---- Curriculum: forward / reverse (train = sum of both phases) ----------
    for cond, p1, p2 in [("forward", "curr_fwd_p1", "curr_fwd"),
                         ("reverse", "curr_rev_p1", "curr_rev")]:
        if not (os.path.isdir(os.path.join(EXP_DIR, p1)) or os.path.isdir(os.path.join(EXP_DIR, p2))):
            continue
        rows.append({
            "experiment": "curriculum", "run": cond, "variant": cond, "seed": "",
            "train_sec": train_secs(p1, p2),
            "test_sec": gen_secs(f"{p2}_gen.log"),          # main MPTS-52 generation
            "val_sec": val_secs(p2, "mp52"),                # main MPTS-52 validation
            "forget_test_sec": gen_secs(f"{p2}_mp20_gen.log"),
            "forget_val_sec": val_secs(p2, "mp20", subdir="forgetting"),
        })

    # ---- Rank sweep: rank_r<R>_s<S> -----------------------------------------
    for d in sorted(glob.glob(os.path.join(EXP_DIR, "rank_r*_s*"))):
        run = os.path.basename(d)
        m = re.match(r"rank_r(\d+)_s(\d+)", run)
        if not m:
            continue
        r, seed = m.groups()
        rows.append({
            "experiment": "rank", "run": run, "variant": f"r{r}", "seed": seed,
            "train_sec": train_secs(run),
            "test_sec": gen_secs(f"{run}_gen.log"),
            "val_sec": val_secs(run, "mp52"),
            "forget_test_sec": None, "forget_val_sec": None,
        })

    # ---- Write CSV -----------------------------------------------------------
    cols = ["experiment", "run", "variant", "seed",
            "train_sec", "test_sec", "val_sec", "forget_test_sec", "forget_val_sec"]
    out = os.path.join(EXP_DIR, "timing_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["train_hms", "test_hms", "val_hms"])
        for row in rows:
            w.writerow([row[c] if row.get(c) is not None else "" for c in cols]
                       + [hms(row["train_sec"]), hms(row["test_sec"]), hms(row["val_sec"])])

    # ---- Print table ---------------------------------------------------------
    print(f"\n{'experiment':<11} {'run':<16} {'train':>10} {'test(gen)':>10} {'val':>10}"
          f"  {'forget_test':>11} {'forget_val':>10}")
    print("-" * 86)
    for row in rows:
        print(f"{row['experiment']:<11} {row['run']:<16} "
              f"{hms(row['train_sec']):>10} {hms(row['test_sec']):>10} {hms(row['val_sec']):>10}"
              f"  {hms(row['forget_test_sec']):>11} {hms(row['forget_val_sec']):>10}")
    print(f"\nwrote {out}  ({len(rows)} runs; blank = stage not finished yet)")


if __name__ == "__main__":
    main()
