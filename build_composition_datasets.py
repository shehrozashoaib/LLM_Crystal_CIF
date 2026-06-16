#!/usr/bin/env python3
"""
build_composition_datasets.py
=============================
Generate the data for the COMPOSITION (Data) lever — §4.1 of experiment_framework.md.

Produces a fixed-volume SWAP sweep: a constant-size training set whose only
varying property is the MP-20 : MPTS-52 ratio. This isolates *composition* from
*volume* and *steps* (steps are pinned to 4500 in the trainer), answering the
decisive reviewer concern that the published "combined" win could be volume.

What it builds (into Data/composition_sweep/):
  - train_mp20_{00,25,50,75,100}.csv   5 train sets, each TOTAL_TRAIN rows
  - val_mp20_{00,25,50,75,100}.csv      composition-matched val sets
  - test_frozen_mp52_1000.csv           ONE frozen pure-MPTS-52 test set,
                                         graded for EVERY run (§3.1 rule)
  - manifest.json                        counts, leakage, seeds, checks

Controls enforced here (so the experiment is causal, not just "a higher number"):
  1. LEAKAGE FILTER (§2, mandatory): every MP-20 material_id present in the
     MPTS-52 test OR val set is dropped from the MP-20 pools BEFORE mixing.
     Otherwise adding MP-20 leaks held-out answers.
  2. CONSTANT VOLUME: every ratio has exactly TOTAL_TRAIN unique crystals.
  3. UNIQUENESS / cross-pool dedup: no material_id appears twice inside a set
     (the two train pools share ~8.8k materials), and no val material appears in
     that run's train set. So "constant volume" means constant *unique* crystals.
  4. COUNT-BALANCED: the ratio is by crystal count (the user's choice). Token
     balance is reported in the manifest for transparency but not enforced.
  5. DETERMINISM: a fixed master seed; per-ratio derived seeds.

NOTE ON TOTAL_TRAIN = 24,000 (not 27,000):
  The framework targeted ~27k, but the mandatory leakage filter leaves only
  24,154 usable MP-20 train crystals. The 100:0 (pure-MP-20) point therefore
  cannot reach 27k while staying leakage-free. 24,000 is the largest clean,
  constant size that ALL five ratios can hit. Change TOTAL_TRAIN below to
  override (a value <= the post-filter MP-20 pool size for the 100:0 point).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config (override via CLI)
# ----------------------------------------------------------------------------
DEFAULT_DATA_DIR = "."
DEFAULT_OUT_DIR = "Data/composition_sweep"
TOTAL_TRAIN = 24_000          # constant unique crystals per ratio (see header note)
TOTAL_VAL = 4_000             # constant unique crystals per matched val set
TEST_FROZEN_N = 1_000         # frozen MPTS-52 test subset graded for every run
MASTER_SEED = 3407
RATIOS_MP20_PCT = [0, 25, 50, 75, 100]   # MP-20 percentage of each train set

COLS = ["material_id", "instruction", "input", "output"]

FILES = {
    "mp20_train": "mp_20_train_cifs_description_reduced_withmpids.csv",
    "mp20_val":   "mp_20_val_cifs_description_reduced_withmpids.csv",
    "mp52_train": "mp_52_train_cifs_description_reduced_withmpids.csv",
    "mp52_val":   "mp_52_val_cifs_description_reduced_withmpids.csv",
    "mp52_test":  "mp_52_test_cifs_description_reduced_withmpids.csv",
}


def _load(data_dir: Path, key: str) -> pd.DataFrame:
    p = data_dir / FILES[key]
    df = pd.read_csv(p)
    missing = set(COLS) - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] {p} is missing required columns: {sorted(missing)}")
    return df


def _ids(df: pd.DataFrame) -> set:
    return set(df["material_id"].tolist())


def _ids_hash(df: pd.DataFrame) -> str:
    """Stable hash of the sorted material_id list — lets us prove the frozen
    test set is byte-identical across runs."""
    joined = "|".join(sorted(map(str, df["material_id"].tolist())))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def _output_char_len(df: pd.DataFrame) -> dict:
    """Cheap proxy for sequence length (real token stats handled separately)."""
    lens = df["output"].astype(str).str.len()
    return {
        "median_chars": float(lens.median()),
        "mean_chars": float(lens.mean()),
        "p95_chars": float(lens.quantile(0.95)),
        "max_chars": int(lens.max()),
    }


def _sample(df: pd.DataFrame, n: int, seed: int, exclude_ids: set | None = None) -> pd.DataFrame:
    """Deterministically sample n unique-material rows, optionally excluding ids."""
    pool = df
    if exclude_ids:
        pool = pool[~pool["material_id"].isin(exclude_ids)]
    if len(pool) < n:
        raise RuntimeError(
            f"Not enough rows to sample: need {n}, pool has {len(pool)} "
            f"(after excluding {len(exclude_ids) if exclude_ids else 0} ids)."
        )
    return pool.sample(n=n, random_state=seed)


def build(data_dir: Path, out_dir: Path, total_train: int = TOTAL_TRAIN,
          total_val: int = TOTAL_VAL) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    mp20_train = _load(data_dir, "mp20_train")
    mp20_val   = _load(data_dir, "mp20_val")
    mp52_train = _load(data_dir, "mp52_train")
    mp52_val   = _load(data_dir, "mp52_val")
    mp52_test  = _load(data_dir, "mp52_test")

    # --- Split-disjointness sanity checks (data integrity) ------------------
    checks = {}
    checks["mp52_train_x_test"] = len(_ids(mp52_train) & _ids(mp52_test))
    checks["mp52_train_x_val"]  = len(_ids(mp52_train) & _ids(mp52_val))
    checks["mp20_train_x_val"]  = len(_ids(mp20_train) & _ids(mp20_val))
    for k, v in checks.items():
        if v != 0:
            print(f"[WARN] split overlap {k} = {v} (expected 0) — investigate.")

    # --- Leakage filter (§2): drop MP-20 materials that live in MPTS-52 eval -
    mpts_eval_ids = _ids(mp52_test) | _ids(mp52_val)
    n_mp20_train_before = len(mp20_train)
    n_mp20_val_before = len(mp20_val)
    mp20_train_clean = mp20_train[~mp20_train["material_id"].isin(mpts_eval_ids)].copy()
    mp20_val_clean   = mp20_val[~mp20_val["material_id"].isin(mpts_eval_ids)].copy()
    dropped_train = n_mp20_train_before - len(mp20_train_clean)
    dropped_val   = n_mp20_val_before - len(mp20_val_clean)
    print(f"[leakage] MP-20 train: dropped {dropped_train} -> {len(mp20_train_clean)} usable")
    print(f"[leakage] MP-20 val  : dropped {dropped_val} -> {len(mp20_val_clean)} usable")

    # --- Feasibility guard for the binding (100:0) point --------------------
    max_mp20_needed = max(round(total_train * p / 100) for p in RATIOS_MP20_PCT)
    if max_mp20_needed > len(mp20_train_clean):
        sys.exit(
            f"[FATAL] total_train={total_train} needs up to {max_mp20_needed} "
            f"MP-20 crystals but only {len(mp20_train_clean)} survive the leakage "
            f"filter. Lower TOTAL_TRAIN to <= {len(mp20_train_clean)}."
        )

    # --- Frozen test set (§3.1): one pure-MPTS-52 1000-sample subset --------
    test_frozen = mp52_test.sample(n=TEST_FROZEN_N, random_state=MASTER_SEED)[COLS]
    test_path = out_dir / "test_frozen_mp52_1000.csv"
    test_frozen.to_csv(test_path, index=False)
    frozen_test_ids = _ids(test_frozen)
    print(f"[frozen test] {len(test_frozen)} rows -> {test_path} (ids_hash={_ids_hash(test_frozen)})")

    manifest = {
        "params": {
            "total_train": total_train,
            "total_val": total_val,
            "test_frozen_n": TEST_FROZEN_N,
            "master_seed": MASTER_SEED,
            "ratios_mp20_pct": RATIOS_MP20_PCT,
            "balance": "count",
        },
        "split_disjointness_checks": checks,
        "leakage_filter": {
            "mpts52_eval_ids": len(mpts_eval_ids),
            "mp20_train_dropped": dropped_train,
            "mp20_train_usable": len(mp20_train_clean),
            "mp20_val_dropped": dropped_val,
            "mp20_val_usable": len(mp20_val_clean),
        },
        "frozen_test": {
            "file": str(test_path),
            "n": len(test_frozen),
            "ids_hash": _ids_hash(test_frozen),
            "source": FILES["mp52_test"],
            "token_proxy": _output_char_len(test_frozen),
        },
        "runs": [],
    }

    # --- Per-ratio train + matched val -------------------------------------
    for i, pct in enumerate(RATIOS_MP20_PCT):
        tag = f"{pct:02d}"
        seed_tr = MASTER_SEED + 100 + i
        seed_va = MASTER_SEED + 200 + i

        # TRAIN: pick MPTS-52 first, then MP-20 excluding those ids (uniqueness)
        n_mp20 = round(total_train * pct / 100)
        n_mp52 = total_train - n_mp20
        tr52 = _sample(mp52_train, n_mp52, seed_tr) if n_mp52 else mp52_train.iloc[0:0]
        chosen52 = _ids(tr52)
        tr20 = _sample(mp20_train_clean, n_mp20, seed_tr + 1, exclude_ids=chosen52) if n_mp20 else mp20_train_clean.iloc[0:0]
        train = pd.concat([tr52, tr20], ignore_index=True)[COLS]
        train = train.sample(frac=1.0, random_state=seed_tr + 2).reset_index(drop=True)
        train_ids = _ids(train)
        assert len(train) == total_train, f"train size {len(train)} != {total_train}"
        assert train["material_id"].nunique() == len(train), "duplicate material_id in train set!"

        # VAL: composition-matched, exclude this run's train ids + leakage
        nv_mp20 = round(total_val * pct / 100)
        nv_mp52 = total_val - nv_mp20
        va52 = _sample(mp52_val, nv_mp52, seed_va, exclude_ids=train_ids) if nv_mp52 else mp52_val.iloc[0:0]
        chosen_va52 = _ids(va52)
        va20 = (_sample(mp20_val_clean, nv_mp20, seed_va + 1,
                        exclude_ids=train_ids | chosen_va52)
                if nv_mp20 else mp20_val_clean.iloc[0:0])
        val = pd.concat([va52, va20], ignore_index=True)[COLS]
        val = val.sample(frac=1.0, random_state=seed_va + 2).reset_index(drop=True)
        assert len(val) == total_val, f"val size {len(val)} != {total_val}"
        assert val["material_id"].nunique() == len(val), "duplicate material_id in val set!"

        # Cross-leakage assertions (the whole point — must be ZERO)
        assert len(train_ids & frozen_test_ids) == 0, "TRAIN leaks frozen-test ids!"
        assert len(train_ids & _ids(val)) == 0, "VAL overlaps TRAIN ids!"
        assert len(_ids(val) & frozen_test_ids) == 0, "VAL leaks frozen-test ids!"

        train_path = out_dir / f"train_mp20_{tag}.csv"
        val_path = out_dir / f"val_mp20_{tag}.csv"
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)

        run = {
            "ratio_mp20_pct": pct,
            "tag": tag,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "train": {"n_total": len(train), "n_mp20": int(n_mp20), "n_mp52": int(n_mp52),
                      "unique_ids": int(train["material_id"].nunique()),
                      "token_proxy": _output_char_len(train)},
            "val": {"n_total": len(val), "n_mp20": int(nv_mp20), "n_mp52": int(nv_mp52),
                    "unique_ids": int(val["material_id"].nunique())},
            "seeds": {"train": seed_tr, "val": seed_va},
        }
        manifest["runs"].append(run)
        print(f"[ratio {tag}] train={len(train)} (MP20={n_mp20}/MPTS52={n_mp52})  "
              f"val={len(val)} (MP20={nv_mp20}/MPTS52={nv_mp52})  -> {train_path.name}, {val_path.name}")

    man_path = out_dir / "manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[manifest] -> {man_path}")
    print("[OK] All datasets built, leakage assertions passed.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="dir holding the source *_reduced_withmpids.csv files")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="output dir for the sweep datasets")
    ap.add_argument("--total_train", type=int, default=TOTAL_TRAIN)
    ap.add_argument("--total_val", type=int, default=TOTAL_VAL)
    args = ap.parse_args()

    build(Path(args.data_dir), Path(args.out_dir),
          total_train=args.total_train, total_val=args.total_val)


if __name__ == "__main__":
    main()
