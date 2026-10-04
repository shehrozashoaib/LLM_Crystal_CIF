#!/usr/bin/env python3
"""
datasets/build_curriculum_k_sweep.py
====================================
Build the datasets for the forward-curriculum SWITCH-POINT sweep (MP-20 -> MPTS-52)
under a single seed, with the data split applied IDENTICALLY to the step split
(the §4.3.2 diagonal rule, on a ninths grid):

    k (MP-20 steps)      in {0, 500, 1000, 1500, 2000, 2500, 4500}   of TOTAL_STEPS = 4500
    data_mp20  = round_half_up(TOTAL_TRAIN * k / TOTAL_STEPS)        data_mp52 = TOTAL_TRAIN - data_mp20
    steps_mp52 = TOTAL_STEPS - k

e.g. k=500 -> 1/9 of the budget on MP-20: 2,667 MP-20 crystals for 500 steps, then
21,333 MPTS-52 crystals for 4,000 steps. Every non-empty phase runs 6.00 epochs
(steps * eff_batch / data), the same exposure as a single-phase 24k / 4500-step run.

What this fixes relative to datasets/build_ratio_datasets.py (which is left untouched so
the committed ratio_sweep files still reproduce):

  1. CROSS-PHASE DUPLICATES. 8,850 MP-20 train crystals are the SAME materials
     (same material_id, byte-identical CIF) as MPTS-52 train crystals. The old builder
     only removed MP-20 ids found in MPTS-52 test/val, so a crystal could be drawn into
     BOTH phases of one run (835-1,928 per committed ratio run; "24,000 unique" was
     really 22,072-23,165). Here the MP-20 phase of each run additionally excludes every
     id that run's MPTS-52 phase drew, so each run has EXACTLY 24,000 unique crystals.
     (MP-20 crystals that also exist in MPTS-52 train but were not drawn by this run
     remain eligible — that is the agreed design, "option 2".)
  2. LEAKAGE checked against BOTH MPTS-52 test and val, for BOTH phases (the old
     builder never checked the MPTS-52 phase against val).
  3. ENDPOINTS. k=0 (pure MPTS-52) and k=4500 (pure MP-20) are built from the same
     seeded pools as the interior points, as single-phase runs (the old runner crashed
     on a 0-row / 0-step phase).
  4. ONE SEED. Both pools are shuffled with --seed (no hidden +1/+2 offsets).

Sampling. Each pool is shuffled ONCE under --seed:
  * MPTS-52 phase = first data_mp52 rows of the shuffled MPTS-52 train pool, so a
    smaller MPTS-52 set is a strict prefix (subset) of a larger one.
  * MP-20 phase   = first data_mp20 rows of the shuffled leakage-safe MP-20 pool after
    removing ids drawn by THIS run's MPTS-52 phase.

Usage:
    python datasets/build_curriculum_k_sweep.py --seed 123456
    python datasets/build_curriculum_k_sweep.py --seed 123456 --ks 0 500 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SRC_DIR = "Data/source"
TOTAL_TRAIN = 24_000          # matches the composition / ratio sweeps
TOTAL_STEPS = 4_500           # matches every committed sweep
EFF_BATCH = 32                # micro_batch * grad_accum, held fixed everywhere
DEFAULT_KS = [0, 500, 1000, 1500, 2000, 2500, 4500]

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
    if df["material_id"].duplicated().any():
        sys.exit(f"[FATAL] {p} has duplicate material_ids")
    if df[COLS].isna().any().any():
        sys.exit(f"[FATAL] {p} has null values in {COLS}")
    return df[COLS].copy()


def _ids(df: pd.DataFrame) -> set:
    return set(df["material_id"].tolist())


def _ids_hash(df: pd.DataFrame) -> str:
    joined = "|".join(sorted(map(str, df["material_id"].tolist())))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def _round_half_up(x: float) -> int:
    # Same rule as datasets/build_ratio_datasets.py (Python's round() is half-to-even).
    return math.floor(x + 0.5)


def build(src_dir: Path, out_dir: Path, ks: list[int], seed: int,
          total_train: int, total_steps: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    mp20_train = _load(src_dir, "mp20_train")
    mp52_train = _load(src_dir, "mp52_train")
    mp52_val   = _load(src_dir, "mp52_val")
    mp52_test  = _load(src_dir, "mp52_test")
    test_ids, val_ids = _ids(mp52_test), _ids(mp52_val)

    # MPTS-52's own splits must already be disjoint.
    assert not (_ids(mp52_train) & (test_ids | val_ids)), "MPTS-52 train overlaps its own test/val!"

    # --- Leakage filter (MP-20 vs MPTS-52 eval sets) --------------------------
    mp20_safe = mp20_train[~mp20_train["material_id"].isin(test_ids | val_ids)]
    leak_dropped = len(mp20_train) - len(mp20_safe)
    shared_with_mp52_train = len(_ids(mp20_safe) & _ids(mp52_train))
    print(f"[leakage] MP-20 train: dropped {leak_dropped} ids in MPTS-52 test/val -> {len(mp20_safe)} usable")
    print(f"[overlap] {shared_with_mp52_train} of those are also MPTS-52 train crystals "
          f"(excluded per run when that run's MPTS-52 phase draws them)")

    # --- Shuffle each pool ONCE under the single seed -------------------------
    pool20 = mp20_safe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool52 = mp52_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    runs = {}
    for k in ks:
        if not 0 <= k <= total_steps:
            sys.exit(f"[FATAL] k={k} outside [0, {total_steps}]")
        tag = f"k{k:04d}"
        d20 = _round_half_up(total_train * k / total_steps)
        d52 = total_train - d20
        s20, s52 = k, total_steps - k
        if (d20 == 0) != (s20 == 0) or (d52 == 0) != (s52 == 0):
            sys.exit(f"[FATAL] k={k}: a phase has data without steps or vice versa")

        if d52 > len(pool52):
            sys.exit(f"[FATAL] k={k}: need {d52} MPTS-52, pool has {len(pool52)}")
        p52 = pool52.head(d52).reset_index(drop=True)

        # MP-20 candidates: leakage-safe pool minus whatever THIS run's MPTS-52 phase drew.
        cand20 = pool20[~pool20["material_id"].isin(_ids(p52))]
        if d20 > len(cand20):
            sys.exit(f"[FATAL] k={k}: need {d20} MP-20, only {len(cand20)} disjoint candidates")
        p20 = cand20.head(d20).reset_index(drop=True)

        # --- Assertions: all must hold, or nothing is written -----------------
        i20, i52 = _ids(p20), _ids(p52)
        assert len(p20) == d20 and len(p52) == d52
        assert len(i20) == d20 and len(i52) == d52, f"k={k}: duplicate ids inside a phase"
        assert not (i20 & i52), f"k={k}: a crystal appears in BOTH phases"
        assert len(i20 | i52) == total_train, f"k={k}: run is not {total_train} unique crystals"
        for name, ids in (("MP-20", i20), ("MPTS-52", i52)):
            assert not (ids & test_ids), f"k={k}: {name} phase leaks MPTS-52 test!"
            assert not (ids & val_ids),  f"k={k}: {name} phase leaks MPTS-52 val!"

        files = {}
        for phase, df, n in (("mp20", p20, d20), ("mp52", p52, d52)):
            if n:
                f = out_dir / f"train_{phase}_{tag}.csv.gz"
                df.to_csv(f, index=False)
                files[phase] = str(f)
            else:
                files[phase] = None

        ep20 = s20 * EFF_BATCH / d20 if d20 else 0.0
        ep52 = s52 * EFF_BATCH / d52 if d52 else 0.0
        runs[str(k)] = {
            "tag": tag,
            "fraction_mp20": round(k / total_steps, 6),
            "data": {"mp20": d20, "mp52": d52, "total": d20 + d52, "unique_total": len(i20 | i52)},
            "steps": {"mp20": s20, "mp52": s52, "total": s20 + s52},
            "epochs_per_phase": {"mp20": round(ep20, 3), "mp52": round(ep52, 3)},
            "mp20_in_mp52_train": len(i20 & _ids(mp52_train)),
            "ids_hash": {"mp20": _ids_hash(p20) if d20 else None, "mp52": _ids_hash(p52) if d52 else None},
            "files": files,
        }
        print(f"[k={k:>4}] data {d20:>6} + {d52:>6} = {d20 + d52} unique   "
              f"steps {s20:>4} + {s52:>4}   epochs {ep20:.3f} / {ep52:.3f}   "
              f"(MP-20 phase holds {runs[str(k)]['mp20_in_mp52_train']} crystals that are also in MPTS-52 train)")

    manifest = {
        "rule": "data_frac = step_frac = k/total_steps; each non-empty phase runs equal epochs",
        "sampling": ("pools shuffled once with `seed`; MPTS-52 phase = prefix of shuffled MPTS-52 train; "
                     "MP-20 phase = prefix of shuffled leakage-safe MP-20 train after removing ids "
                     "drawn by this run's MPTS-52 phase"),
        "params": {
            "seed": seed,
            "total_train": total_train,
            "total_steps": total_steps,
            "effective_batch": EFF_BATCH,
            "reference_epochs": round(total_steps * EFF_BATCH / total_train, 3),
        },
        "leakage_filter": {
            "mpts52_eval_ids": len(test_ids | val_ids),
            "mp20_train_dropped": leak_dropped,
            "mp20_train_usable": len(mp20_safe),
            "mp20_usable_also_in_mp52_train": shared_with_mp52_train,
        },
        "runs": runs,
    }
    man = out_dir / "manifest.json"
    with open(man, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[manifest] -> {man}")
    print("[OK] datasets built; leakage, disjointness and uniqueness assertions passed.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="single sampling seed for both pools")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="MP-20 step counts")
    ap.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    ap.add_argument("--out_dir", default="", help="default: Data/curriculum_k_sweep_s<seed>")
    ap.add_argument("--total_train", type=int, default=TOTAL_TRAIN)
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or f"Data/curriculum_k_sweep_s{args.seed}")
    build(Path(args.src_dir), out_dir, args.ks, args.seed, args.total_train, args.total_steps)


if __name__ == "__main__":
    main()
