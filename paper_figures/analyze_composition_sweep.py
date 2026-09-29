"""
analyze_composition_sweep.py
----------------------------
Composition sweep: what the model sees, at a fixed training-set size.

Panel (A) best-of-10 match rate on the held-out MPTS-52 test split vs the
MP-20 fraction of the training corpus.  Panel (B) the median strict-matched
RMS of the same runs.

Every point trains on exactly 24,000 unique crystals at 4,500 optimizer
steps, r=32, and is graded on the same frozen 8,096-crystal MPTS-52 test
set; only the MP-20:MPTS-52 ratio of the single-stage corpus changes.  This
is the "what data" lever -- the companion figure
(analyze_curriculum_sweep.py) moves the ordering instead.

RMS is StructureMatcher's displacement normalized by
(V/N_sites)^(1/3); it is dimensionless, so no angstrom unit is given.

Numbers are transcribed from each run's validation *_summary.txt in
github.com/shehrozashoaib/LLM_Crystal_CIF/results/<run>/validation/.

Run from inside the LLM_materials_files directory:
    python analyze_composition_sweep.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# MP-20 % of corpus, source run, MP-20:MPTS-52 counts, best-of-10 (%), matched-RMS
POINTS = [
    (0,   "comp_mp20_00",  (0,      24000), 30.0, 0.050),
    (25,  "comp_mp20_25",  (6000,   18000), 30.4, 0.049),
    (50,  "comp_mp20_50",  (12000,  12000), 29.5, 0.046),
    (75,  "comp_mp20_75",  (18000,   6000), 28.0, 0.042),
    (100, "comp_mp20_100", (24000,      0), 26.6, 0.039),
]

X_MAX  = 100                          # x axis is the MP-20 percentage
XTICKS = [0, 25, 50, 75, 100]
GAP    = 40                           # wider than any sampled spacing -> no dashes

BLUE   = "#0072B2"   # panel A, matching the other SFT figures
ORANGE = "#E69F00"   # panel B
GREY   = "#7F7F7F"

XLABEL = "MP-20 fraction of the training set (%)"


def edge_align(i, n):
    """Anchor the first/last point labels inward so they clear the axes."""
    if i == 0:
        return "left", (-3, 12)
    if i == n - 1:
        return "right", (3, 12)
    return "center", (0, 12)


def draw_series(ax, xs, ys, color, marker, msize):
    """Line + markers, with any unsampled stretch drawn dashed."""
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
    ax.set_xlim(-X_MAX * 0.042, X_MAX * 1.042)
    ax.set_xlabel(XLABEL, fontsize=13)


def main():
    xs  = np.array([p[0] for p in POINTS], dtype=float)
    acc = np.array([p[3] for p in POINTS], dtype=float)
    rms = np.array([p[4] for p in POINTS], dtype=float)

    print("Composition sweep @ 24,000 crystals / 4,500 steps / r=32")
    for (pct, run, (n20, n52), a, r) in POINTS:
        print(f"  MP-20 {pct:>3d}%  {run:14s}  {n20:>5d}:{n52:<5d}  {a:4.1f}%   RMS {r:.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # ---------------- (A) best-of-10 accuracy ----------------
    draw_series(axA, xs, acc, BLUE, "o", 10)

    # baseline reference: pure MPTS-52, the value every mixture must beat
    axA.axhline(acc[0], color=GREY, lw=1.1, ls="--", alpha=0.75, zorder=1)
    axA.text(X_MAX, acc[0] + 0.10, "pure MPTS-52 baseline",
             ha="right", va="bottom", fontsize=10, color=GREY, style="italic")

    label_series(axA, xs, acc, lambda v: f"{v:.1f}%")

    lo, hi = acc.min(), acc.max()
    axA.set_ylim(lo - 1.7, hi + 1.5)
    axA.text(0, lo - 1.45, "pure\nMPTS-52", ha="left", va="bottom",
             fontsize=10, color=GREY)
    axA.text(X_MAX, lo - 1.45, "pure\nMP-20", ha="right", va="bottom",
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

    fig.suptitle("Composition sweep at 24,000 crystals / 4,500 steps / r=32",
                 fontsize=17, fontweight="bold", y=0.99)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pdf = HERE / "fig_composition.pdf"
    png = HERE / "fig_composition.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
