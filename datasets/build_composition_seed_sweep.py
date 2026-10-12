#!/usr/bin/env python3
"""
datasets/build_composition_seed_sweep.py
========================================
Composition (data-mix) sweep for ONE seed, drawn from the SAME seeded pools as
datasets/build_curriculum_k_sweep.py, so a seed's composition and curriculum
sweeps share their endpoints:

    pct (MP-20 share)  in {25, 50, 75} by default (0 and 100 are the k-sweep endpoints)
    n_mp20 = round_half_up(TOTAL_TRAIN * pct / 100)      n_mp52 = TOTAL_TRAIN - n_mp20

Each run is ONE shuffled mixture trained single-phase for 4,500 steps (6.00
epochs), exactly the design of the seed-3407 composition sweep
(datasets/build_composition_datasets.py / training/run_composition_sweep.sh).

Sampling (identical rules to build_curriculum_k_sweep.py):
  * pools shuffled ONCE with --seed: MPTS-52 train, and leakage-safe MP-20 train
    (MP-20 ids in MPTS-52 test or val removed);
  * MPTS-52 part = first n_mp52 rows of the shuffled MPTS-52 pool;
  * MP-20 part   = first n_mp20 rows of the shuffled MP-20 pool after removing ids
    this run's MPTS-52 part drew (so every run is exactly 24,000 unique crystals);
  * the mixture is shuffled once with --seed.

Endpoint equivalence: for pct=0 and pct=100 this rule selects exactly the crystal
sets of the k-sweep's k=0 (pure MPTS-52) and k=4500 (pure MP-20) runs. The builder
recomputes both and asserts they match Data/curriculum_k_sweep_s<seed>/ (when
present), recording the result in the manifest — that is what licenses reusing
those runs as the 0% / 100% composition points. (Row ORDER differs — those files are
pool-prefix order — but the trainer reshuffles with the same seed either way.)

Usage:
    python datasets/build_composition_seed_sweep.py --seed 25478
    python datasets/build_composition_seed_sweep.py --seed 25478 --pcts 25 50 75
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
TOTAL_TRAIN = 24_000
TOTAL_STEPS = 4_500
EFF_BATCH = 32
DEFAULT_PCTS = [25, 50, 75]

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


def _ids_hash(ids) -> str:
    return hashlib.sha1("|".join(sorted(map(str, ids))).encode()).hexdigest()[:16]


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def build(src_dir: Path, out_dir: Path, pcts: list[int], seed: int, total_train: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp20_train = _load(src_dir, "mp20_train")
    mp52_train = _load(src_dir, "mp52_train")
    test_ids = _ids(_load(src_dir, "mp52_test"))
    val_ids = _ids(_load(src_dir, "mp52_val"))
    assert not (_ids(mp52_train) & (test_ids | val_ids)), "MPTS-52 train overlaps its own test/val!"

    mp20_safe = mp20_train[~mp20_train["material_id"].isin(test_ids | val_ids)]
    print(f"[leakage] MP-20 train: dropped {len(mp20_train) - len(mp20_safe)} ids in MPTS-52 "
          f"test/val -> {len(mp20_safe)} usable")

    # Same two shuffles as build_curriculum_k_sweep.py — this is what ties the sweeps together.
    pool20 = mp20_safe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool52 = mp52_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def draw(pct: int):
        n20 = _round_half_up(total_train * pct / 100)
        n52 = total_train - n20
        p52 = pool52.head(n52)
        cand20 = pool20[~pool20["material_id"].isin(_ids(p52))]
        if n20 > len(cand20):
            sys.exit(f"[FATAL] {pct}%: need {n20} MP-20, only {len(cand20)} disjoint candidates")
        return cand20.head(n20), p52

    # --- endpoint equivalence with the k-sweep ---------------------------------
    kdir = Path(f"Data/curriculum_k_sweep_s{seed}")
    endpoints = {}
    if (kdir / "manifest.json").exists():
        km = json.load(open(kdir / "manifest.json"))["runs"]
        for pct, k, part, idx in ((0, "0", "mp52", 1), (100, "4500", "mp20", 0)):
            mine = _ids(draw(pct)[idx])
            theirs = _ids(pd.read_csv(km[k]["files"][part]))
            assert mine == theirs, f"{pct}% composition != k-sweep k={k} crystal set"
            endpoints[str(pct)] = {"equals": f"k-sweep k={k}", "file": km[k]["files"][part],
                                   "ids_hash": _ids_hash(mine)}
            print(f"[endpoint] {pct:>3}% == k-sweep k={k} ({len(mine)} crystals, identical set)")
    else:
        print(f"[endpoint] {kdir} not found — endpoint equivalence not checked")

    runs = {}
    for pct in pcts:
        if not 0 < pct < 100:
            sys.exit(f"[FATAL] pct={pct}: build interior points only (0/100 are the k-sweep endpoints)")
        p20, p52 = draw(pct)
        mix = pd.concat([p20, p52], ignore_index=True)[COLS]
        mix = mix.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        i20, i52, im = _ids(p20), _ids(p52), _ids(mix)
        assert len(mix) == total_train and len(im) == total_train, f"{pct}%: not {total_train} unique crystals"
        assert not (i20 & i52), f"{pct}%: a crystal is in both parts"
        assert not (im & test_ids), f"{pct}%: leaks MPTS-52 test!"
        assert not (im & val_ids), f"{pct}%: leaks MPTS-52 val!"
        tag = f"mp20_{pct:03d}"
        f = out_dir / f"train_{tag}.csv.gz"
        mix.to_csv(f, index=False)
        runs[str(pct)] = {
            "tag": tag,
            "pct_mp20": pct,
            "data": {"mp20": len(p20), "mp52": len(p52), "total": len(mix), "unique_total": len(im)},
            "steps": TOTAL_STEPS,
            "epochs": round(TOTAL_STEPS * EFF_BATCH / len(mix), 3),
            "mp20_in_mp52_train": len(i20 & _ids(mp52_train)),
            "ids_hash": {"mp20": _ids_hash(i20), "mp52": _ids_hash(i52), "mix": _ids_hash(im)},
            "file": str(f),
        }
        print(f"[{pct:>3}%] {len(p20):>6} MP-20 + {len(p52):>6} MPTS-52 = {len(mix)} unique -> {f}")

    manifest = {
        "rule": "single shuffled mixture, 24,000 unique crystals, 4,500 steps; MP-20 share = pct",
        "sampling": ("pools shuffled once with `seed` (as build_curriculum_k_sweep.py); MPTS-52 part = "
                     "prefix of shuffled MPTS-52 train; MP-20 part = prefix of shuffled leakage-safe "
                     "MP-20 train after removing ids drawn by this run's MPTS-52 part; mixture shuffled "
                     "with `seed`"),
        "params": {"seed": seed, "total_train": total_train, "total_steps": TOTAL_STEPS,
                   "effective_batch": EFF_BATCH},
        "endpoints": endpoints,
        "runs": runs,
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[manifest] -> {out_dir / 'manifest.json'}")
    print("[OK] composition datasets built; leakage, disjointness, uniqueness and endpoint checks passed.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pcts", type=int, nargs="+", default=DEFAULT_PCTS, help="MP-20 percentages")
    ap.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    ap.add_argument("--out_dir", default="", help="default: Data/composition_sweep_s<seed>")
    ap.add_argument("--total_train", type=int, default=TOTAL_TRAIN)
    args = ap.parse_args()
    build(Path(args.src_dir), Path(args.out_dir or f"Data/composition_sweep_s{args.seed}"),
          args.pcts, args.seed, args.total_train)


if __name__ == "__main__":
    main()
