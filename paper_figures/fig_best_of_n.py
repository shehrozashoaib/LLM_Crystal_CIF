"""
fig_best_of_n.py
----------------
Best-of-N match rate as a function of N, for every lever, on the frozen
8,096-crystal MPTS-52 test split.

Reviewers asked for this because best-of-10 conflates single-sample quality
with sampling breadth: a lever can raise the headline either by improving the
mode or by widening the distribution, and one number at N=10 cannot tell those
apart.

Each run generated exactly 10 CIFs per prompt, so rather than truncating to the
first N (which throws away samples and is sensitive to generation order) we use
the standard unbiased estimator of Chen et al. (2021):

    best-of-N  =  mean over materials of  1 - C(n-c, N) / C(n, N)

with n = 10 generations and c = the number of those that matched. At N = n this
reduces exactly to the reported best-of-10, and at N = 1 it equals the
per-generation match rate.

Inputs are the per-material grader CSVs under data/, copied verbatim from
results/<sweep>/<run>/validation/ in the code release.

Run from inside paper_figures/:
    python fig_best_of_n.py
"""
from pathlib import Path
from math import comb
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
N_GEN = 10

# label, file, colour, marker
RUNS = [
    ("LoRA rank r=128",            "r128.csv",    "#009E73", "^"),
    ("Curriculum (k=500)",         "curr.csv",    "#0072B2", "o"),
    ("Data composition (25:75)",   "mix25.csv.gz", "#56B4E9", "D"),
    ("Baseline SFT (r=32)",        "base.csv.gz", "#7F7F7F", "s"),
    ("GRPO (continuous)",          "grpo.csv",    "#D55E00", "v"),
]


def match_counts(path):
    """material -> number of its 10 generations that matched."""
    op = gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="")
    counts = []
    with op as fh:
        rdr = csv.DictReader(fh)
        cols = [c for c in rdr.fieldnames if c.startswith("Match_generation_")]
        for row in rdr:
            counts.append(sum(str(row.get(c, "")).strip() in TRUE for c in cols))
    return np.array(counts, dtype=int)


def best_of_n(counts, n_gen=N_GEN):
    """Unbiased best-of-N curve for N = 1..n_gen."""
    out = []
    for k in range(1, n_gen + 1):
        # probability that a random k-subset contains at least one match
        p = [1.0 - (comb(n_gen - c, k) / comb(n_gen, k) if n_gen - c >= k else 0.0)
             for c in counts]
        out.append(100.0 * float(np.mean(p)))
    return np.array(out)


def main():
    ks = np.arange(1, N_GEN + 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    curves = {}
    print(f"{'run':28s} " + "  ".join(f"N={k:<2d}" for k in ks))
    for label, fname, colour, marker in RUNS:
        counts = match_counts(DATA / fname)
        curve = best_of_n(counts)
        curves[label] = curve
        ax.plot(ks, curve, "-", marker=marker, color=colour, lw=2.2, markersize=7,
                markeredgecolor="black", markeredgewidth=0.6, label=label, zorder=3)
        print(f"{label:28s} " + "  ".join(f"{v:5.1f}" for v in curve))

    ax.set_xticks(ks)
    ax.set_xlabel("N (CIFs sampled per prompt)", fontsize=13)
    ax.set_ylabel("Best-of-N match on MPTS-52 (%)", fontsize=13)
    ax.set_title("Sampling budget dominates every training lever",
                 fontsize=15, fontweight="bold", loc="left", pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11.5)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")

    # What N buys on one fixed model, against what the best lever buys at fixed N.
    base = curves["Baseline SFT (r=32)"]
    gain_n = base[-1] - base[0]
    best_lever = max(c[-1] for c in curves.values()) - base[-1]
    ax.set_xlim(0.35, 10.5)
    ax.annotate("", xy=(0.66, base[0]), xytext=(0.66, base[-1]),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.2))
    ax.text(0.84, (base[0] + base[-1]) / 2,
            f"same model,\nN=1 to 10:\n+{gain_n:.1f} pp",
            fontsize=9.5, color="#444444", va="center", style="italic")
    ax.text(10.45, base[-1] - 1.1,
            f"best lever at N=10: +{best_lever:.1f} pp",
            fontsize=9.5, color="#444444", ha="right", va="top", style="italic")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = HERE / f"fig_best_of_n.{ext}"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print("Saved", p)


if __name__ == "__main__":
    main()
