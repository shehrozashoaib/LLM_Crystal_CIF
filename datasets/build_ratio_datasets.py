#!/usr/bin/env python3
"""
datasets/build_ratio_datasets.py
=======================
Build the DIAGONAL (target-emphasis) curriculum datasets — the family where the
MP-20 : MPTS-52 split is applied IDENTICALLY to the data budget and the step
budget, at the composition sweep's fixed 24,000-crystal volume.

    data_mp20  = round(TOTAL_TRAIN * X/(X+Y))     data_mp52 = TOTAL_TRAIN - data_mp20
    steps_mp20 = round(TOTAL_STEPS * X/(X+Y))     steps_mp52 = TOTAL_STEPS - steps_mp20

Because the data fraction equals the step fraction, EVERY phase runs the same
number of epochs as the composition sweep's single-phase runs:

    steps * eff_batch / data  =  TOTAL_STEPS * eff_batch / TOTAL_TRAIN  =  6.00

That invariant is the point: it removes the volume and per-crystal-exposure
confounds, so the only thing separating a diagonal run from `comp_mp20_00` is
WHERE in training the MP-20 crystals appear.

This rule reproduces the existing ratio_sweep runs exactly:
    1:7 -> 3,000 / 21,000 crystals, 563 / 3,937 steps
    2:7 -> 5,333 / 18,667 crystals, 1,000 / 3,500 steps
    3:7 -> 7,200 / 16,800 crystals, 1,350 / 3,150 steps

Controls (identical philosophy to datasets/build_curriculum_datasets.py):
  1. LEAKAGE FILTER (mandatory): every MP-20 material_id present in the MPTS-52
     test OR val set is dropped from the MP-20 pool BEFORE sampling.
  2. NESTED SUBSETS: each pool is shuffled ONCE under a fixed seed and then
     truncated, so the MP-20 set for a small X is a strict prefix (subset) of the
     MP-20 set for a larger X. Ratios therefore differ only by what is ADDED, not
     by an independent redraw.
  3. DETERMINISM: MASTER_SEED matches datasets/build_curriculum_datasets.py (3407) and the
     same per-pool seed offsets (+1 MP-20, +2 MPTS-52) are reused.

Val sets are NOT rebuilt — the existing Data/curriculum/val_mp20.csv.gz and
val_mp52.csv.gz are reused (they are logging-only; pinned-step training never
gates on eval), so every curriculum-family run shares one val set.

Usage:
    python datasets/build_ratio_datasets.py 3:4 4:3
    python datasets/build_ratio_datasets.py 1:7 2:7 3:7      # regenerate the zip's family
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SRC_DIR = "Data/source"
DEFAULT_OUT_DIR = "Data/ratio_sweep"
TOTAL_TRAIN = 24_000          # matches datasets/build_composition_datasets.py
TOTAL_STEPS = 4_500           # matches every committed sweep
EFF_BATCH = 32                # micro_batch * grad_accum, held fixed everywhere
MASTER_SEED = 3407

COLS = ["material_id", "instruction", "input", "output"]
FILES = {
    "mp20_train": "mp_20_train.csv.gz",
    "mp52_train": "mp_52_train.csv.gz",
    "mp52_val":   "mp_52_val.csv.gz",
    "mp52_test":  "mp_52_test.csv.gz",
}


def _load(src_dir: Path, key: str) -> pd.DataFrame:
    p = src_dir / FILES[key]
    df = pd.read_csv(p)
    missing = set(COLS) - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] {p} missing columns: {sorted(missing)}")
    return df[COLS].copy()


def _ids(df: pd.DataFrame) -> set:
    return set(df["material_id"].tolist())


def _ids_hash(df: pd.DataFrame) -> str:
    joined = "|".join(sorted(map(str, df["material_id"].tolist())))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def _round_half_up(x: float) -> int:
    """Round .5 UP, not to-even.

    Python's round() is banker's rounding: round(562.5) == 562. The published
    1:7 run used 563 steps, so half-to-even would silently disagree with the
    committed results by one step. math.floor(x + 0.5) matches them exactly.
    """
    return math.floor(x + 0.5)


def parse_ratio(s: str) -> tuple[int, int]:
    try:
        x, y = s.split(":")
        x, y = int(x), int(y)
    except Exception:
        sys.exit(f"[FATAL] bad ratio {s!r} — expected X:Y, e.g. 3:4")
    if x < 0 or y < 0 or x + y == 0:
        sys.exit(f"[FATAL] bad ratio {s!r}")
    return x, y


def build(src_dir: Path, out_dir: Path, ratios: list[str],
          total_train: int, total_steps: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    mp20_train = _load(src_dir, "mp20_train")
    mp52_train = _load(src_dir, "mp52_train")
    mp52_val   = _load(src_dir, "mp52_val")
    mp52_test  = _load(src_dir, "mp52_test")

    # --- Leakage filter -----------------------------------------------------
    mpts_eval_ids = _ids(mp52_test) | _ids(mp52_val)
    mp20_clean = mp20_train[~mp20_train["material_id"].isin(mpts_eval_ids)].copy()
    dropped = len(mp20_train) - len(mp20_clean)
    print(f"[leakage] MP-20 train: dropped {dropped} -> {len(mp20_clean)} usable")

    # --- Shuffle each pool ONCE (nested-subset property) --------------------
    pool20 = mp20_clean[COLS].sample(frac=1.0, random_state=MASTER_SEED + 1).reset_index(drop=True)
    pool52 = mp52_train[COLS].sample(frac=1.0, random_state=MASTER_SEED + 2).reset_index(drop=True)

    runs = {}
    for spec in ratios:
        x, y = parse_ratio(spec)
        tag = f"{x}to{y}"
        f = x / (x + y)
        d20 = _round_half_up(total_train * f); d52 = total_train - d20
        s20 = _round_half_up(total_steps * f); s52 = total_steps - s20

        if d20 > len(pool20):
            sys.exit(f"[FATAL] {spec}: need {d20} MP-20, pool has {len(pool20)}")
        if d52 > len(pool52):
            sys.exit(f"[FATAL] {spec}: need {d52} MPTS-52, pool has {len(pool52)}")

        p20 = pool20.head(d20).reset_index(drop=True)
        p52 = pool52.head(d52).reset_index(drop=True)

        # leakage assertions — must be ZERO
        assert len(_ids(p20) & _ids(mp52_test)) == 0, f"{spec}: MP-20 phase leaks MPTS-52 test!"
        assert len(_ids(p20) & _ids(mp52_val)) == 0,  f"{spec}: MP-20 phase leaks MPTS-52 val!"
        assert len(_ids(p52) & _ids(mp52_test)) == 0, f"{spec}: MPTS-52 phase leaks MPTS-52 test!"

        f20 = out_dir / f"train_phase_mp20_{tag}.csv.gz"
        f52 = out_dir / f"train_phase_mp52_{tag}.csv.gz"
        p20.to_csv(f20, index=False)
        p52.to_csv(f52, index=False)

        ep20 = s20 * EFF_BATCH / d20 if d20 else 0.0
        ep52 = s52 * EFF_BATCH / d52 if d52 else 0.0
        runs[spec] = {
            "tag": tag,
            "fraction_mp20": round(f, 6),
            "data": {"mp20": d20, "mp52": d52, "total": d20 + d52},
            "steps": {"mp20": s20, "mp52": s52, "total": s20 + s52},
            "epochs_per_phase": {"mp20": round(ep20, 3), "mp52": round(ep52, 3)},
            "ids_hash": {"mp20": _ids_hash(p20), "mp52": _ids_hash(p52)},
            "files": {"mp20": str(f20), "mp52": str(f52)},
        }
        print(f"[{spec:>4}] data {d20:>6} + {d52:>6} = {d20+d52}   "
              f"steps {s20:>4} + {s52:>4} = {s20+s52}   "
              f"epochs {ep20:.2f} / {ep52:.2f}")

    manifest = {
        "rule": "data_frac = step_frac = X/(X+Y); phases therefore run equal epochs",
        "params": {
            "total_train": total_train,
            "total_steps": total_steps,
            "effective_batch": EFF_BATCH,
            "master_seed": MASTER_SEED,
            "reference_epochs": round(total_steps * EFF_BATCH / total_train, 3),
        },
        "leakage_filter": {
            "mpts52_eval_ids": len(mpts_eval_ids),
            "mp20_train_dropped": dropped,
            "mp20_train_usable": len(mp20_clean),
        },
        "pools_available": {"mp20": len(pool20), "mp52": len(pool52)},
        "val_sets": {
            "mp20": "Data/curriculum/val_mp20.csv.gz",
            "mp52": "Data/curriculum/val_mp52.csv.gz",
            "note": "reused from the curriculum build; logging only, never gates training",
        },
        "runs": runs,
    }
    man = out_dir / "manifest_ratio_sweep.json"
    with open(man, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[manifest] -> {man}")
    print("[OK] ratio datasets built, leakage assertions passed.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ratios", nargs="+", help="ratios as X:Y (MP-20:MPTS-52), e.g. 3:4 4:3")
    ap.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--total_train", type=int, default=TOTAL_TRAIN)
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()
    build(Path(args.src_dir), Path(args.out_dir), args.ratios,
          args.total_train, args.total_steps)


if __name__ == "__main__":
    main()
