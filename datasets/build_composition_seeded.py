#!/usr/bin/env python3
"""
datasets/build_composition_seeded.py
====================================
Composition (Data) lever, §4.1, rebuilt under ONE seed so it shares its pools with the
curriculum k-sweep (datasets/build_curriculum_k_sweep.py). Each set is a single
SHUFFLED MIX of 24,000 unique crystals, trained in one 4,500-step phase:

    data_mp20 = round_half_up(24,000 * pct/100)        data_mp52 = 24,000 - data_mp20

Sampling is IDENTICAL to the k-sweep builder (same seed, same pools, same rule):
  * MPTS-52 part = first data_mp52 rows of the MPTS-52 train pool shuffled under --seed
  * MP-20 part   = first data_mp20 rows of the leakage-safe MP-20 pool shuffled under
                   --seed, after removing ids this set's MPTS-52 part drew
                   (8,850 MP-20 train crystals are the same materials as MPTS-52 train ones)
  * the two parts are concatenated and shuffled under --seed.
So pct=0 holds exactly the crystals of the k-sweep's k=0 run and pct=100 exactly those of
its k=4500 run: the curriculum endpoints ARE the 0% / 100% composition points (same model,
hyperparameters, 4,500 steps, 24,000 unique crystals), and only 25/50/75 need training.
This builder asserts that equivalence against the k-sweep manifest when it is present.

Writes a manifest the k-sweep runner understands (runs[pct].files.mixed / steps.mixed),
so training/run_curriculum_k_sweep.sh drives these runs with all of its verification.

Usage:
    python datasets/build_composition_seeded.py --seed 123456              # 25 50 75
    python datasets/build_composition_seeded.py --seed 123456 --pcts 0 25 50 75 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_curriculum_k_sweep import (COLS, DEFAULT_SRC_DIR, EFF_BATCH, TOTAL_STEPS,  # noqa: E402
                                      TOTAL_TRAIN, _ids, _ids_hash, _load, _round_half_up)


def build(src_dir: Path, out_dir: Path, pcts: list[int], seed: int,
          total_train: int, total_steps: int, ksweep_manifest: Path | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    mp20_train = _load(src_dir, "mp20_train")
    mp52_train = _load(src_dir, "mp52_train")
    test_ids, val_ids = _ids(_load(src_dir, "mp52_test")), _ids(_load(src_dir, "mp52_val"))
    assert not (_ids(mp52_train) & (test_ids | val_ids)), "MPTS-52 train overlaps its own test/val!"

    mp20_safe = mp20_train[~mp20_train["material_id"].isin(test_ids | val_ids)]
    leak_dropped = len(mp20_train) - len(mp20_safe)
    print(f"[leakage] MP-20 train: dropped {leak_dropped} ids in MPTS-52 test/val -> {len(mp20_safe)} usable")

    # Same shuffles as build_curriculum_k_sweep.py — this is what ties the two sweeps together.
    pool20 = mp20_safe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool52 = mp52_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    kman = json.loads(ksweep_manifest.read_text()) if ksweep_manifest and ksweep_manifest.exists() else None
    if kman and kman["params"]["seed"] != seed:
        sys.exit(f"[FATAL] k-sweep manifest seed {kman['params']['seed']} != --seed {seed}")

    runs = {}
    for pct in pcts:
        if not 0 <= pct <= 100:
            sys.exit(f"[FATAL] pct={pct} outside [0, 100]")
        tag = f"mp20_{pct:02d}"
        d20 = _round_half_up(total_train * pct / 100)
        d52 = total_train - d20
        p52 = pool52.head(d52)
        cand20 = pool20[~pool20["material_id"].isin(_ids(p52))]
        if d20 > len(cand20):
            sys.exit(f"[FATAL] pct={pct}: need {d20} MP-20, only {len(cand20)} disjoint candidates")
        p20 = cand20.head(d20)
        mixed = pd.concat([p52, p20], ignore_index=True)[COLS] \
                  .sample(frac=1.0, random_state=seed).reset_index(drop=True)

        i20, i52, im = _ids(p20), _ids(p52), _ids(mixed)
        assert len(mixed) == total_train and len(im) == total_train, f"pct={pct}: not {total_train} unique"
        assert not (i20 & i52), f"pct={pct}: a crystal is in both parts"
        assert not (im & test_ids), f"pct={pct}: leaks MPTS-52 test!"
        assert not (im & val_ids), f"pct={pct}: leaks MPTS-52 val!"

        # Endpoint equivalence with the k-sweep (pct 0 == k=0, pct 100 == k=4500).
        for kpct, kkey, phase in ((0, "0", "mp52"), (100, "4500", "mp20")):
            if kman and pct == kpct:
                want = kman["runs"][kkey]["ids_hash"][phase]
                assert _ids_hash(mixed) == want, f"pct={pct}: crystals differ from k-sweep k={kkey}"
                print(f"[pct={pct}] identical crystal set to k-sweep k={kkey} (ids_hash {want})")

        f = out_dir / f"train_{tag}.csv.gz"
        mixed.to_csv(f, index=False)
        runs[str(pct)] = {
            "tag": tag,
            "fraction_mp20": round(pct / 100, 6),
            "data": {"mp20": d20, "mp52": d52, "total": len(mixed), "unique_total": len(im)},
            "steps": {"mp20": 0, "mp52": 0, "mixed": total_steps, "total": total_steps},
            "epochs": round(total_steps * EFF_BATCH / total_train, 3),
            "mp20_in_mp52_train": len(i20 & _ids(mp52_train)),
            "ids_hash": {"mixed": _ids_hash(mixed), "mp20": _ids_hash(p20) if d20 else None,
                         "mp52": _ids_hash(p52) if d52 else None},
            "files": {"mixed": str(f), "mp20": None, "mp52": None},
        }
        print(f"[pct={pct:>3}] {d20:>6} MP-20 + {d52:>6} MPTS-52 = {len(im)} unique, shuffled; "
              f"{total_steps} steps ({runs[str(pct)]['epochs']} epochs) -> {f.name}")

    manifest = {
        "rule": "single shuffled mix; data_mp20 = round_half_up(total_train*pct/100); one phase of total_steps",
        "sampling": ("pools shuffled once with `seed` exactly as build_curriculum_k_sweep.py; MPTS-52 part = "
                     "prefix of shuffled MPTS-52 train; MP-20 part = prefix of shuffled leakage-safe MP-20 "
                     "train after removing ids drawn by the MPTS-52 part; parts concatenated and shuffled"),
        "params": {"seed": seed, "total_train": total_train, "total_steps": total_steps,
                   "effective_batch": EFF_BATCH},
        "endpoints": {"0": "k-sweep k=0 (identical crystals)", "100": "k-sweep k=4500 (identical crystals)"},
        "leakage_filter": {"mpts52_eval_ids": len(test_ids | val_ids), "mp20_train_dropped": leak_dropped,
                           "mp20_train_usable": len(mp20_safe)},
        "runs": runs,
    }
    man = out_dir / "manifest.json"
    man.write_text(json.dumps(manifest, indent=2))
    print(f"\n[manifest] -> {man}\n[OK] composition datasets built; assertions passed.")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pcts", type=int, nargs="+", default=[25, 50, 75], help="MP-20 percentages")
    ap.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    ap.add_argument("--out_dir", default="", help="default: Data/composition_sweep_s<seed>")
    ap.add_argument("--ksweep_manifest", default="", help="default: Data/curriculum_k_sweep_s<seed>/manifest.json")
    ap.add_argument("--total_train", type=int, default=TOTAL_TRAIN)
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    a = ap.parse_args()
    build(Path(a.src_dir), Path(a.out_dir or f"Data/composition_sweep_s{a.seed}"), a.pcts, a.seed,
          a.total_train, a.total_steps,
          Path(a.ksweep_manifest or f"Data/curriculum_k_sweep_s{a.seed}/manifest.json"))


if __name__ == "__main__":
    main()
