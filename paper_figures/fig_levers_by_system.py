"""
fig_levers_by_system.py
-----------------------
Per-crystal-system match-rate change for each lever, with bootstrap confidence
intervals.

The per-space-group panel it replaces showed 184 bars, most of them resting on
a handful of test crystals, so a full-height bar could mean a single structure
flipping. Reviewers asked for uncertainty or sample counts. Aggregating to the
seven crystal systems puts every bar on hundreds of crystals and lets a
paired bootstrap over the shared test set carry the error, which is the honest
resolution at which these differences can be read.

Intervals are percentile bootstrap over materials, resampled in pairs so the
lever and the baseline always see the same crystals; that is the same pairing
the McNemar tests use. An interval crossing zero means the sign of that bar is
not resolved.

Inputs are the per-material grader CSVs under data/, copied verbatim from
results/<sweep>/<run>/validation/ in the code release.

Run from inside paper_figures/:
    python fig_levers_by_system.py
"""
from pathlib import Path
import csv
import gzip

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

csv.field_size_limit(10 ** 9)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TRUE = {"1", "1.0", "true", "True", "TRUE", "yes", "Y"}
RNG = np.random.default_rng(3407)
N_BOOT = 2000

BASE = ("Baseline SFT (r=32)", "base.csv.gz")
LEVERS = [
    ("LoRA rank r=128",          "r128.csv", "#009E73"),
    ("Curriculum (k=500)",       "curr.csv", "#0072B2"),
    ("GRPO (continuous)",        "grpo.csv", "#D55E00"),
]

SYSTEMS = [("Triclinic", 1, 2), ("Monoclinic", 3, 15), ("Orthorhombic", 16, 74),
           ("Tetragonal", 75, 142), ("Trigonal", 143, 167),
           ("Hexagonal", 168, 194), ("Cubic", 195, 230)]


def load(path):
    """material_id -> (matched 0/1, space group)."""
    op = gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="")
    out = {}
    with op as fh:
        rdr = csv.DictReader(fh)
        cols = [c for c in rdr.fieldnames if c.startswith("Match_generation_")]
        for row in rdr:
            try:
                spg = int(float(row["Groundtruth SPG"]))
            except (TypeError, ValueError):
                continue
            hit = int(any(str(row.get(c, "")).strip() in TRUE for c in cols))
            out[row["material_id"]] = (hit, spg)
    return out


def main():
    base = load(DATA / BASE[1])
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    width = 0.26
    xs = np.arange(len(SYSTEMS))

    for j, (label, fname, colour) in enumerate(LEVERS):
        lev = load(DATA / fname)
        ids = sorted(set(base) & set(lev))
        spg = np.array([base[i][1] for i in ids])
        b = np.array([base[i][0] for i in ids], dtype=float)
        v = np.array([lev[i][0] for i in ids], dtype=float)

        deltas, los, his, ns = [], [], [], []
        for name, lo, hi in SYSTEMS:
            m = (spg >= lo) & (spg <= hi)
            bb, vv = b[m], v[m]
            d = 100 * (vv.mean() - bb.mean())
            idx = RNG.integers(0, len(bb), size=(N_BOOT, len(bb)))
            boot = 100 * (vv[idx].mean(axis=1) - bb[idx].mean(axis=1))
            lo_, hi_ = np.percentile(boot, [2.5, 97.5])
            deltas.append(d); los.append(d - lo_); his.append(hi_ - d); ns.append(m.sum())

        ax.bar(xs + (j - 1) * width, deltas, width, color=colour,
               edgecolor="black", linewidth=0.5, label=label, zorder=3)
        ax.errorbar(xs + (j - 1) * width, deltas, yerr=[los, his], fmt="none",
                    ecolor="#333333", elinewidth=1.0, capsize=2.5, zorder=4)
        print(f"\n-- {label}")
        for name, n, d, l, h in zip([s[0] for s in SYSTEMS], ns, deltas, los, his):
            flag = "" if (d - l) * (d + h) > 0 else "   (crosses 0)"
            print(f"   {name:14s} n={n:5d}  {d:+6.2f} pp  "
                  f"[{d-l:+6.2f}, {d+h:+6.2f}]{flag}")

    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s[0]}\n(n={sum((np.array([base[i][1] for i in sorted(base)]) >= s[1]) & (np.array([base[i][1] for i in sorted(base)]) <= s[2]))})"
                        for s in SYSTEMS], fontsize=10.5)
    ax.set_ylabel("Match rate change vs baseline\n(percentage points)", fontsize=13)
    ax.set_title("Per-crystal-system change, with paired bootstrap 95% intervals",
                 fontsize=15, fontweight="bold", loc="left", pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, ncol=3, loc="upper left")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = HERE / f"fig_levers_by_system.{ext}"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print("Saved", p)


if __name__ == "__main__":
    main()
