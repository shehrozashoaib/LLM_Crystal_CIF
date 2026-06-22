#!/usr/bin/env python3
"""
build_curriculum_datasets.py
============================
Build the data for the CURRICULUM (Schedule) lever — §4.3 of experiment_framework.md.

The curriculum experiment compares the SAME crystal pool trained in three orders,
at a MATCHED total step budget, so the only thing that moves is the *schedule*:

  * mixed     : MP-20 + MPTS-52 shuffled together, trained for S steps  (reference)
  * forward   : MP-20 phase (S1 steps) -> MPTS-52 phase (S2 steps),  S1+S2 = S
  * reverse   : MPTS-52 phase (S1 steps) -> MP-20 phase (S2 steps),  same budget
  * forgetting probe : grade MP-20 accuracy before vs after the 2nd phase, on a
                       frozen MP-20 test set.

Controls (identical philosophy to build_composition_datasets.py):
  1. LEAKAGE FILTER (§2, mandatory): every MP-20 material_id present in the
     MPTS-52 test OR val set is dropped from the MP-20 pools BEFORE anything.
     The MP-20 *forgetting-probe* test set is additionally filtered against the
     MPTS-52 *train* set, so MP-20 grading never lands on a crystal the MPTS-52
     phase trained on.
  2. SAME MULTISET across conditions: train_mixed is exactly the concatenation of
     the two phase files (MP-20 phase + MPTS-52 phase), shuffled. So mixed and
     curriculum see the identical set of crystals — only the ORDER differs.
  3. DETERMINISM: a fixed master seed; derived per-set seeds.

Note on the union size (51,534, not the framework's 54,516):
  54,516 = 27,136 (MP-20 train) + 27,380 (MPTS-52 train) ignores the mandatory
  leakage filter. After dropping the 2,982 MP-20 train crystals that live in the
  MPTS-52 eval set, the leakage-safe union is 24,154 + 27,380 = 51,534. We do NOT
  cross-pool dedup (a material shared by both train pools appears once per phase,
  hence twice in mixed) so that mixed and curriculum remain the same multiset.

Outputs (into Data/curriculum/):
  - train_mixed.csv            shuffled leakage-safe union (the reference run)
  - train_phase_mp20.csv       MP-20 phase  (leakage-filtered MP-20 train)
  - train_phase_mp52.csv       MPTS-52 phase (full MPTS-52 train)
  - val_mixed.csv / val_mp20.csv / val_mp52.csv   matched val sets (logging only)
  - test_frozen_mp20_1000.csv  frozen MP-20 forgetting-probe test set
  - manifest_curriculum.json   counts, leakage, seeds, suggested step split

The frozen MPTS-52 test set is reused from the composition build
(Data/composition_sweep/test_frozen_mp52_1000.csv) so curriculum and composition
runs are graded on the identical held-out MPTS-52 crystals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SRC_DIR = "Data/source"
DEFAULT_OUT_DIR = "Data/curriculum"
TOTAL_VAL_EACH = 4_000        # rows per matched val set (logging only; eval doesn't gate)
TEST_FROZEN_MP20_N = 1_000    # frozen MP-20 forgetting-probe subset
MASTER_SEED = 3407

COLS = ["material_id", "instruction", "input", "output"]

# gzipped source splits shipped in the repo (Data/source/*.csv.gz)
FILES = {
    "mp20_train": "mp_20_train.csv.gz",
    "mp20_val":   "mp_20_val.csv.gz",
    "mp20_test":  "mp_20_test.csv.gz",
    "mp52_train": "mp_52_train.csv.gz",
    "mp52_val":   "mp_52_val.csv.gz",
    "mp52_test":  "mp_52_test.csv.gz",
}


def _load(src_dir: Path, key: str) -> pd.DataFrame:
    p = src_dir / FILES[key]
    df = pd.read_csv(p)
    missing = set(COLS) - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] {p} is missing required columns: {sorted(missing)}")
    return df[COLS].copy()


def _ids(df: pd.DataFrame) -> set:
    return set(df["material_id"].tolist())


def _ids_hash(df: pd.DataFrame) -> str:
    joined = "|".join(sorted(map(str, df["material_id"].tolist())))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def _output_char_len(df: pd.DataFrame) -> dict:
    lens = df["output"].astype(str).str.len()
    return {
        "median_chars": float(lens.median()),
        "mean_chars": float(lens.mean()),
        "p95_chars": float(lens.quantile(0.95)),
        "max_chars": int(lens.max()),
    }


def _sample(df: pd.DataFrame, n: int, seed: int, exclude_ids: set | None = None) -> pd.DataFrame:
    pool = df
    if exclude_ids:
        pool = pool[~pool["material_id"].isin(exclude_ids)]
    if len(pool) < n:
        raise RuntimeError(f"need {n} rows, pool has {len(pool)} after excludes")
    return pool.sample(n=n, random_state=seed)


def build(src_dir: Path, out_dir: Path, total_step_budget: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    mp20_train = _load(src_dir, "mp20_train")
    mp20_val   = _load(src_dir, "mp20_val")
    mp20_test  = _load(src_dir, "mp20_test")
    mp52_train = _load(src_dir, "mp52_train")
    mp52_val   = _load(src_dir, "mp52_val")
    mp52_test  = _load(src_dir, "mp52_test")

    # --- Split-disjointness sanity checks -----------------------------------
    checks = {
        "mp52_train_x_test": len(_ids(mp52_train) & _ids(mp52_test)),
        "mp52_train_x_val":  len(_ids(mp52_train) & _ids(mp52_val)),
        "mp20_train_x_val":  len(_ids(mp20_train) & _ids(mp20_val)),
        "mp20_train_x_test": len(_ids(mp20_train) & _ids(mp20_test)),
    }
    for k, v in checks.items():
        if v != 0:
            print(f"[WARN] split overlap {k} = {v} (expected 0) — investigate.")

    # --- Leakage filter (§2): drop MP-20 materials that live in MPTS-52 eval -
    mpts_eval_ids = _ids(mp52_test) | _ids(mp52_val)
    mp20_train_clean = mp20_train[~mp20_train["material_id"].isin(mpts_eval_ids)].copy()
    mp20_val_clean   = mp20_val[~mp20_val["material_id"].isin(mpts_eval_ids)].copy()
    dropped_train = len(mp20_train) - len(mp20_train_clean)
    dropped_val   = len(mp20_val) - len(mp20_val_clean)
    print(f"[leakage] MP-20 train: dropped {dropped_train} -> {len(mp20_train_clean)} usable")
    print(f"[leakage] MP-20 val  : dropped {dropped_val} -> {len(mp20_val_clean)} usable")

    # --- Phase files (the same multiset that mixed is built from) ------------
    phase_mp20 = mp20_train_clean[COLS].sample(frac=1.0, random_state=MASTER_SEED + 1).reset_index(drop=True)
    phase_mp52 = mp52_train[COLS].sample(frac=1.0, random_state=MASTER_SEED + 2).reset_index(drop=True)
    n_mp20, n_mp52 = len(phase_mp20), len(phase_mp52)

    # --- Mixed = shuffled concatenation of the two phases (SAME multiset) ----
    mixed = pd.concat([phase_mp20, phase_mp52], ignore_index=True)[COLS]
    mixed = mixed.sample(frac=1.0, random_state=MASTER_SEED + 3).reset_index(drop=True)
    assert len(mixed) == n_mp20 + n_mp52

    # --- Matched val sets (logging only; pinned-step training, no gating) ----
    val_mp20 = _sample(mp20_val_clean, min(TOTAL_VAL_EACH, len(mp20_val_clean)),
                       MASTER_SEED + 11)[COLS].reset_index(drop=True)
    val_mp52 = _sample(mp52_val, min(TOTAL_VAL_EACH, len(mp52_val)),
                       MASTER_SEED + 12)[COLS].reset_index(drop=True)
    nmix20 = round(TOTAL_VAL_EACH * n_mp20 / (n_mp20 + n_mp52))
    val_mixed = pd.concat([
        _sample(mp20_val_clean, nmix20, MASTER_SEED + 13),
        _sample(mp52_val, TOTAL_VAL_EACH - nmix20, MASTER_SEED + 14),
    ], ignore_index=True)[COLS].sample(frac=1.0, random_state=MASTER_SEED + 15).reset_index(drop=True)

    # --- Frozen MP-20 forgetting-probe test set -----------------------------
    # Also drop MP-20 test ids present in the MPTS-52 train set, so MP-20 grading
    # never lands on a crystal the MPTS-52 phase trained on.
    mp52_train_ids = _ids(mp52_train)
    mp20_test_clean = mp20_test[~mp20_test["material_id"].isin(mp52_train_ids)].copy()
    dropped_mp20_test = len(mp20_test) - len(mp20_test_clean)
    test_mp20 = _sample(mp20_test_clean, TEST_FROZEN_MP20_N, MASTER_SEED)[COLS].reset_index(drop=True)
    print(f"[forgetting test] MP-20 test: dropped {dropped_mp20_test} (in MPTS-52 train) "
          f"-> sampled {len(test_mp20)} (ids_hash={_ids_hash(test_mp20)})")

    # --- Cross-leakage assertions (must be ZERO) ----------------------------
    union_train_ids = _ids(phase_mp20) | _ids(phase_mp52)
    frozen_mp52 = None
    fp = Path("Data/composition_sweep/test_frozen_mp52_1000.csv")
    if fp.exists():
        frozen_mp52 = pd.read_csv(fp)
        assert len(union_train_ids & _ids(frozen_mp52)) == 0, "TRAIN union leaks frozen MPTS-52 test!"
    assert len(_ids(phase_mp20) & _ids(test_mp20)) == 0, "MP-20 phase leaks MP-20 forgetting test!"
    assert len(_ids(phase_mp52) & _ids(test_mp20)) == 0, "MPTS-52 phase leaks MP-20 forgetting test!"

    # --- Suggested matched-budget step split (proportional to phase size) ----
    S = total_step_budget
    s1_forward = round(S * n_mp20 / (n_mp20 + n_mp52))   # forward phase-1 = MP-20
    s2_forward = S - s1_forward                          # forward phase-2 = MPTS-52
    # reverse: phase-1 = MPTS-52, phase-2 = MP-20 (same per-pool step counts)
    s1_reverse = s2_forward
    s2_reverse = s1_forward

    # --- Write everything ---------------------------------------------------
    paths = {}
    for name, df in [
        ("train_mixed", mixed),
        ("train_phase_mp20", phase_mp20),
        ("train_phase_mp52", phase_mp52),
        ("val_mixed", val_mixed),
        ("val_mp20", val_mp20),
        ("val_mp52", val_mp52),
        ("test_frozen_mp20_1000", test_mp20),
    ]:
        p = out_dir / f"{name}.csv.gz"
        df.to_csv(p, index=False)
        paths[name] = str(p)

    manifest = {
        "params": {
            "total_step_budget_S": S,
            "effective_batch": 32,
            "master_seed": MASTER_SEED,
            "total_val_each": TOTAL_VAL_EACH,
            "test_frozen_mp20_n": TEST_FROZEN_MP20_N,
        },
        "split_disjointness_checks": checks,
        "leakage_filter": {
            "mpts52_eval_ids": len(mpts_eval_ids),
            "mp20_train_dropped": dropped_train,
            "mp20_train_usable": len(mp20_train_clean),
            "mp20_val_dropped": dropped_val,
            "mp20_val_usable": len(mp20_val_clean),
            "mp20_test_dropped_in_mp52_train": dropped_mp20_test,
        },
        "union": {
            "n_mp20_phase": n_mp20,
            "n_mp52_phase": n_mp52,
            "n_union_mixed": n_mp20 + n_mp52,
            "note": "leakage-safe union; not cross-pool deduped (same multiset as the two phases)",
            "token_proxy_mixed": _output_char_len(mixed),
        },
        "step_split_suggested": {
            "S": S,
            "forward": {"phase1_mp20_steps": s1_forward, "phase2_mp52_steps": s2_forward},
            "reverse": {"phase1_mp52_steps": s1_reverse, "phase2_mp20_steps": s2_reverse},
            "rule": "proportional to phase crystal count; phases sum to S",
        },
        "frozen_test": {
            "mpts52": str(fp) if frozen_mp52 is not None else "(missing — run build_composition_datasets.py)",
            "mpts52_ids_hash": _ids_hash(frozen_mp52) if frozen_mp52 is not None else None,
            "mp20_forgetting": paths["test_frozen_mp20_1000"],
            "mp20_forgetting_ids_hash": _ids_hash(test_mp20),
        },
        "files": paths,
    }
    man_path = out_dir / "manifest_curriculum.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[union] mixed={len(mixed)} (MP20={n_mp20} + MPTS52={n_mp52})")
    print(f"[steps] S={S} -> forward(MP20 {s1_forward} + MPTS52 {s2_forward}); "
          f"reverse(MPTS52 {s1_reverse} + MP20 {s2_reverse})")
    print(f"[manifest] -> {man_path}")
    print("[OK] curriculum datasets built, leakage assertions passed.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", default=DEFAULT_SRC_DIR, help="dir with the gzipped source splits")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--step_budget", type=int, default=4500,
                    help="total matched step budget S (mixed=S; curriculum phases sum to S). "
                         "Default 4500 MATCHES the composition sweep (run_composition_sweep.sh / "
                         "README) so curriculum runs are comparable to the committed composition "
                         "results. Only feeds step_split_suggested in the manifest; the data CSVs "
                         "are independent of S.")
    args = ap.parse_args()
    build(Path(args.src_dir), Path(args.out_dir), args.step_budget)


if __name__ == "__main__":
    main()
