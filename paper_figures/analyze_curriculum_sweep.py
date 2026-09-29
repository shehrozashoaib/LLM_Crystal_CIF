"""
analyze_curriculum_sweep.py
---------------------------
Curriculum warm-up sweep at a MATCHED 24,000-crystal training budget.

Panel (A) best-of-10 match rate on the held-out MPTS-52 test split vs the
number of MP-20 warm-up steps k (the remaining 4,500 - k steps run on
MPTS-52).  Panel (B) the median strict-matched RMS of the same runs.

The x axis is LINEAR in k over the full 0-4,500 budget, so the horizontal
spacing reflects the real step counts.  The sweep samples k densely below
2,500 and not at all between 2,500 and 4,500; the segment spanning that
unsampled interval is drawn dashed.

Every point trains on exactly 24,000 unique crystals at 4,500 optimizer
steps, r=32, and is graded on the same frozen 8,096-crystal MPTS-52 test
set, so the only thing that moves across the curve is where the budget is
spent.  The earlier uncapped-pool runs (51,534 crystals: psplit_k1000,
psplit_k3000, curr_fwd, curr_rev, psplit_datamatch) are deliberately NOT
plotted -- they are not volume-matched to the baseline they sit next to.

RMS is StructureMatcher's displacement normalized by
(V/N_sites)^(1/3); it is dimensionless, so no angstrom unit is given.

Numbers are transcribed from each run's validation *_summary.txt in
github.com/shehrozashoaib/LLM_Crystal_CIF/results/<run>/validation/.

Run from inside the LLM_materials_files directory:
    python analyze_curriculum_sweep.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# k (plotted, rounded), exact k, source run, best-of-10 (%), strict matched-RMS
# `k` is the rounded warm-up length used for plotting; `k_exact` is the step
# count the run actually used (set by the data:step ratio of that point).
POINTS = [
    (0,    0,    "comp_mp20_00",  30.0, 0.050),
    (500,  563,  "ratio_1to7",    30.8, 0.052),
    (1000, 1000, "ratio_2to7",    30.4, 0.053),
    (1500, 1350, "ratio_3to7",    29.7, 0.052),
    (2000, 1929, "ratio_3to4",    28.6, 0.050),
    (2500, 2571, "ratio_4to3",    28.1, 0.053),
    (4500, 4500, "comp_mp20_100", 26.6, 0.039),
]

S_TOTAL = 4500                                  # full step budget = x-axis range
XTICKS  = [0, 1000, 2000, 3000, 4000, 4500]     # sparse; k=3000/4000 unsampled
GAP     = 750                                   # any wider jump is drawn dashed

BLUE   = "#0072B2"   # panel A, matching the other SFT figures
ORANGE = "#E69F00"   # panel B
GREY   = "#7F7F7F"

XLABEL = "MP-20 warm-up steps k  (then MPTS-52 to 4,500)"


def edge_align(i, n):
    """Anchor the first/last point labels inward so they clear the axes."""
    if i == 0:
        return "left", (-5, 12)
    if i == n - 1:
        return "right", (1, 12)
    return "center", (0, 12)


def draw_series(ax, xs, ys, color, marker, msize):
    """Line + markers, with any unsampled stretch drawn dashed.

    Drawing the 2,500 -> 4,500 jump dashed keeps the linear axis honest:
    nothing was measured in between.
    """
    for i in range(len(xs) - 1):
        dashed = (xs[i + 1] - xs[i]) > GAP
        ax.plot(xs[i:i + 2], ys[i:i + 2], color=color, lw=2.5, zorder=2,
                ls="--" if dashed else "-", alpha=0.55 if dashed else 1.0)
    ax.plot(xs, ys, marker, color=color, ls="none", markersize=msize,
            markeredgecolor="black", markeredgewidth=0.7, zorder=3)


def label_series(ax, xs, ys, fmt):
    for i, (x, y) in enumerate(zip(xs, ys)):
        ha, off = edge_align(i, len(xs))
        ax.annotate(fmt(y), (x, y), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=11, fontweight="bold")


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.35, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.set_xticks(XTICKS)
    ax.set_xlim(-190, S_TOTAL + 190)
    ax.set_xlabel(XLABEL, fontsize=13)


def main():
    xs  = np.array([p[0] for p in POINTS], dtype=float)
    acc = np.array([p[3] for p in POINTS], dtype=float)
    rms = np.array([p[4] for p in POINTS], dtype=float)

    print("Curriculum warm-up sweep @ 24,000 crystals / 4,500 steps / r=32")
    for (k, k_exact, run, a, r) in POINTS:
        print(f"  k={k:>4d} (exact {k_exact:>4d})  {run:14s}  {a:4.1f}%   RMS {r:.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # ---------------- (A) best-of-10 accuracy ----------------
    draw_series(axA, xs, acc, BLUE, "o", 10)

    # baseline reference: pure MPTS-52, the value every warm-up must beat
    axA.axhline(acc[0], color=GREY, lw=1.1, ls="--", alpha=0.75, zorder=1)
    axA.text(S_TOTAL, acc[0] + 0.10, "pure MPTS-52 baseline",
             ha="right", va="bottom", fontsize=10, color=GREY, style="italic")

    label_series(axA, xs, acc, lambda v: f"{v:.1f}%")

    lo, hi = acc.min(), acc.max()
    axA.set_ylim(lo - 1.7, hi + 1.5)
    axA.text(0, lo - 1.45, "pure\nMPTS-52", ha="left", va="bottom",
             fontsize=10, color=GREY)
    axA.text(S_TOTAL, lo - 1.45, "pure\nMP-20", ha="right", va="bottom",
             fontsize=10, color=GREY)
    axA.set_ylabel("Best-of-10 match on MPTS-52 (%)", fontsize=14)
    axA.set_title("(A)  Best-of-10 accuracy", fontsize=15,
                  fontweight="bold", loc="left", pad=10)

    # ---------------- (B) matched RMS ----------------
    draw_series(axB, xs, rms, ORANGE, "s", 9)
    label_series(axB, xs, rms, lambda v: f"{v:.3f}")

    rlo, rhi = rms.min(), rms.max()
    pad = (rhi - rlo) * 0.30
    axB.set_ylim(rlo - pad, rhi + pad)
    axB.set_ylabel("Matched RMS (median, normalized)", fontsize=14)
    axB.set_title("(B)  Matched RMS", fontsize=15,
                  fontweight="bold", loc="left", pad=10)

    for ax in (axA, axB):
        style(ax)

    fig.suptitle("Curriculum warm-up sweep at 24,000 crystals / 4,500 steps / r=32",
                 fontsize=17, fontweight="bold", y=0.99)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pdf = HERE / "fig_curriculum_24k.pdf"
    png = HERE / "fig_curriculum_24k.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
