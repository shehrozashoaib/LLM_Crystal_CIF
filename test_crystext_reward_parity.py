#!/usr/bin/env python3
"""
Parity test: the reward in grpo_crystext_reward.py == CrysText's reward.

Both scorers are lifted out of their source files by text slicing (neither file
is importable: ours pulls in unsloth and loads a model at import time, theirs
loads a model too), exec'd in a clean namespace, and run head-to-head on real
MPTS-52 CIFs plus a battery of mutations that exercise every rung of the ladder.

    /venv/py312/bin/python test_crystext_reward_parity.py \
        [--crystext ../CrysText/grpo_training.py] [--n 25]

Exits non-zero on any disagreement.
"""
from __future__ import annotations

import argparse
import gzip
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

HERE = Path(__file__).resolve().parent


FUNCS = ["get_safe_space_group_number", "structure_validity",
         "compare_structures", "validate_structure"]


def _extract_defs(path: Path, names) -> str:
    """Pull the named top-level functions out of a source file, in order.

    Neither file can be imported (both build a model at import time), so the
    scorer is recovered textually: from `def name(` up to the next top-level
    statement, i.e. the next non-blank line at column 0.
    """
    lines = path.read_text().splitlines()
    starts = {}
    for i, line in enumerate(lines):
        for n in names:
            if line.startswith(f"def {n}(") and n not in starts:
                starts[n] = i
    missing = [n for n in names if n not in starts]
    if missing:
        raise SystemExit(f"could not locate {missing} in {path}")

    out = []
    for n in names:
        i = starts[n]
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.strip() and not line[0].isspace():
                break
            j += 1
        out.append("\n".join(lines[i:j]))
    return "\n\n".join(out)


def _load(path: Path, label: str):
    ns = {"np": np, "pd": pd, "Structure": Structure,
          "StructureMatcher": StructureMatcher}
    code = _extract_defs(path, FUNCS)
    exec(compile(code, f"<{label}>", "exec"), ns)
    return ns["validate_structure"]


# ---------------------------------------------------------------- mutations
# NOTE: StructureMatcher is invariant to rigid translation AND (with the default
# scale=True) to uniform volume change, so neither makes a useful mutation —
# both keep scoring 3.0. What walks a structure down the ladder is *random
# per-site displacement* and *anisotropic strain*.
def mutate_perturb(cif: str, distance: float, seed: int = 0) -> str:
    """Random per-site displacement of `distance` A — the ladder's main dial."""
    s = Structure.from_str(cif, fmt="cif")
    s.perturb(distance, seed=seed)
    return s.to(fmt="cif")


def mutate_strain(cif: str, eps: float) -> str:
    """Anisotropic strain: changes cell shape, which volume normalisation
    cannot undo, so tolerance (ltol/angle_tol) actually bites."""
    s = Structure.from_str(cif, fmt="cif")
    s.apply_strain([eps, 0.0, -eps / 2])
    return s.to(fmt="cif")


def mutate_swap_element(cif: str) -> str:
    """Same lattice, wrong chemistry -> the reduced-formula test must fail."""
    s = Structure.from_str(cif, fmt="cif")
    sp = sorted({str(x.symbol) for x in s.composition.elements})
    s.replace_species({sp[0]: "Xe"})
    return s.to(fmt="cif")


def mutate_drop_site(cif: str) -> str:
    """Remove an atom -> composition and matching both change."""
    s = Structure.from_str(cif, fmt="cif")
    if len(s) > 1:
        s.remove_sites([0])
    return s.to(fmt="cif")


def mutate_collapse(cif: str) -> str:
    """All atoms on one site -> structure_validity / parsing must fail."""
    s = Structure.from_str(cif, fmt="cif")
    for i in range(len(s)):
        s[i] = s[i].species, [0.0, 0.0, 0.0]
    return s.to(fmt="cif")


def build_cases(cifs):
    cases = []
    for k, cif in enumerate(cifs):
        cases.append(("identical", cif, cif))
        for d in (0.05, 0.2, 0.4, 0.7, 1.2, 2.0):
            cases.append((f"perturb_{d}A", mutate_perturb(cif, d, seed=k), cif))
        for e in (0.05, 0.15, 0.35):
            cases.append((f"strain_{e}", mutate_strain(cif, e), cif))
        cases.append(("perturb+strain", mutate_strain(mutate_perturb(cif, 0.3, seed=k), 0.12), cif))
        cases.append(("swap_elem", mutate_swap_element(cif), cif))
        cases.append(("drop_site", mutate_drop_site(cif), cif))
        cases.append(("collapsed", mutate_collapse(cif), cif))
        cases.append(("garbage", "this is not a cif at all", cif))
        cases.append(("empty", "", cif))
        cases.append(("truncated", cif[: len(cif) // 3], cif))
        # raw chat-model output: shows why the CIF block is extracted before
        # scoring unless CRYSTEXT_RAW_COMPLETION=1
        cases.append(("fenced", f"Here is the CIF:\n```cif\n{cif}\n```", cif))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(HERE / "grpo_crystext_reward.py"))
    ap.add_argument("--crystext", default=str(HERE.parent / "CrysText" / "grpo_training.py"))
    ap.add_argument("--csv", default=str(HERE / "Data/source/mp_52_val.csv.gz"))
    ap.add_argument("--n", type=int, default=25, help="reference structures to test")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    ours_path, theirs_path = Path(args.ours), Path(args.crystext)
    if not theirs_path.exists():
        raise SystemExit(
            f"CrysText source not found at {theirs_path}\n"
            "clone it:  git clone https://github.com/truptimohanty/CrysText"
        )

    ours = _load(ours_path, "grpo_crystext_reward")
    theirs = _load(theirs_path, "CrysText")

    df = pd.read_csv(args.csv)
    random.seed(args.seed)
    cifs = [str(c) for c in df["output"].sample(n=min(args.n, len(df)),
                                               random_state=args.seed)]

    cases = build_cases(cifs)
    print(f"{len(cases)} cases from {len(cifs)} reference CIFs "
          f"({len(cases)//len(cifs)} mutations each)\n")

    mismatches = []
    tally = {}
    for kind, gen, ref in cases:
        r_ours = ours(gen, ref)          # our copy takes an optional _trace kwarg
        r_theirs = theirs(gen, ref)
        tally.setdefault(kind, []).append(float(r_theirs))
        if float(r_ours) != float(r_theirs):
            mismatches.append((kind, r_ours, r_theirs))

    width = max(len(k) for k in tally)
    for kind, vals in tally.items():
        uniq = sorted(set(vals))
        print(f"  {kind:<{width}}  n={len(vals):>3}  reward(s)={uniq}")

    # the ladder must actually be exercised, or the test proves nothing
    seen = {v for vals in tally.values() for v in vals}
    print(f"\ndistinct reward values seen: {sorted(seen)}")
    if len(seen) < 5:
        print("WARNING: mutations did not span the ladder; parity is weakly tested")

    if mismatches:
        print(f"\nFAIL — {len(mismatches)} mismatch(es):")
        for kind, a, b in mismatches[:20]:
            print(f"  {kind}: ours={a} crystext={b}")
        sys.exit(1)

    print(f"\nPASS — all {len(cases)} cases identical to CrysText.")


if __name__ == "__main__":
    main()
