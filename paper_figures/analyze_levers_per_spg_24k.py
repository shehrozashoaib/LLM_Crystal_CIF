"""
analyze_levers_per_spg_24k.py
-----------------------------
Per-space-group accuracy DELTA for the training levers, stacked against one
common SFT baseline.

Panels (top to bottom):
  1. Curriculum      ratio_1to7      k=563 MP-20 warm-up, then 3,937 MPTS-52
  2. LoRA rank       rank_r128_s3407 r=128 (alpha=256), pure MPTS-52
  3. GRPO            grpo_cont_final continuous reward, forked from ckpt-3000

The layout adapts to len(LEVERS), so adding or removing a panel needs no
other change.

Baseline for every panel: `comp_mp20_00` -- r=32 SFT on pure MPTS-52, 24,000
crystals, 4,500 steps, 30.1% best-of-10. A multi-way comparison needs ONE
reference or the panels are not comparable to each other. The rank and GRPO
families were originally quoted against `rank_r32_s3407` (29.9%), the same
recipe at a different seed; the 0.23 pp gap between the two is far below the
per-space-group noise here.

Rank note: the sweep tops out at r=128. There is no r=256 run in the repo,
so the rank lever is shown at its strongest available setting.

All panels share one symmetric y-scale so bar heights mean the same thing
across levers. Bar opacity scales with the number of test crystals in that
space group; the +/-100 pp bars are space groups holding one or two crystals.

Inputs live in curriculum_24k_results/, mirrored from
github.com/shehrozashoaib/LLM_Crystal_CIF/results/<run>/validation/.

Run from inside the LLM_materials_files directory:
    python analyze_levers_per_spg_24k.py
"""
from pathlib import Path
import gzip
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "curriculum_24k_results"

BASE_LABEL = "Baseline (r=32 SFT, pure MPTS-52, 24k)"
BASE_FILE  = DATA / "comp_mp20_00_per_material.csv.gz"

# panel label, file
LEVERS = [
    ("Curriculum  (k=500 warm-up, 24k matched)", DATA / "ratio_1to7_16bit.csv"),
    ("LoRA rank  (r=32 to r=128)",               DATA / "rank_r128_s3407_16bit.csv"),
    ("GRPO  (continuous reward)",                DATA / "grpo_cont_final_16bit.csv"),
]

CRYSTAL_SYSTEMS = [
    ("Triclinic",    1,   2),
    ("Monoclinic",   3,  15),
    ("Orthorhombic", 16, 74),
    ("Tetragonal",   75, 142),
    ("Trigonal",     143, 167),
    ("Hexagonal",    168, 194),
    ("Cubic",        195, 230),
]

POS, NEG = "#1A8F3F", "#C03030"

# Figure geometry (shared with draw_brackets so label fitting is exact).
FIG_W    = 11.5
GS_LEFT  = 0.085
GS_RIGHT = 0.985


def load_run(path):
    if not path.is_file():
        print("[ERROR] missing:", path, file=sys.stderr)
        sys.exit(1)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as fh:
            return pd.read_csv(fh, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def per_spg(df):
    mc = [c for c in df.columns if c.startswith("Match_generation_")]
    df = df.copy()
    df["b"] = df[mc].fillna(0).any(axis=1).astype(int)
    df["s"] = pd.to_numeric(df["Groundtruth SPG"], errors="coerce").astype("Int64")
    g = (df.dropna(subset=["s"])
           .groupby("s")
           .agg(total=("b", "size"), matched=("b", "sum")))
    g["pct"] = 100 * g["matched"] / g["total"]
    return g.reindex(range(1, 231))


def draw_brackets(ax, sg_to_pos, fs=13, axis_width_in=10.0):
    """Crystal-system brackets under the stack.

    The bracket bar sits low in its own axes and the names hang below it, so
    the system labels cannot collide with the space-group tick numbers on the
    panel above. Each name is shrunk to fit its own bracket, and rotated if
    even the floor size will not fit -- Triclinic spans two space groups and
    always rotates.
    """
    n = len(sg_to_pos)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    by, ty = 0.62, 0.40          # bracket bar + tick height
    label_y_h, label_y_v = 0.30, 0.26

    for name, lo, hi in CRYSTAL_SYSTEMS:
        sgs = [sg for sg in range(lo, hi + 1) if sg in sg_to_pos]
        if not sgs:
            continue
        x_lo = sg_to_pos[min(sgs)] - 0.5
        x_hi = sg_to_pos[max(sgs)] + 0.5
        ax.plot([x_lo, x_hi], [by, by], color="black", lw=1.2, clip_on=False)
        ax.plot([x_lo, x_lo], [by, ty], color="black", lw=1.2, clip_on=False)
        ax.plot([x_hi, x_hi], [by, ty], color="black", lw=1.2, clip_on=False)
        cx = (x_lo + x_hi) / 2
        w  = x_hi - x_lo
        avail_in = (w / n) * axis_width_in
        fit_fs = avail_in * 72.0 / (len(name) * 0.60)
        if w <= 2.5 or fit_fs < 7.0:
            ax.text(cx, label_y_v, name, ha="center", va="top",
                    fontsize=fs - 3, rotation=90)
        else:
            ax.text(cx, label_y_h, name, ha="center", va="top",
                    fontsize=min(fs, fit_fs))


def report_by_system(populated, counts, base_pct, lever_pcts):
    print("\nPer-crystal-system delta (pp, weighted by test crystals):")
    head = "  {:14s} {:>9}".format("system", "crystals")
    for label, _ in LEVERS:
        head += " {:>12}".format(label.split("  ")[0][:12])
    print(head)
    pos = {sg: i for i, sg in enumerate(populated)}
    for name, lo, hi in CRYSTAL_SYSTEMS:
        sgs = [sg for sg in populated if lo <= sg <= hi]
        if not sgs:
            continue
        w = counts[[pos[sg] for sg in sgs]]
        b = np.array([base_pct.loc[sg] for sg in sgs], dtype=float)
        bw = np.nansum(b * w) / w.sum()
        row = "  {:14s} {:9d}".format(name, int(w.sum()))
        for lp in lever_pcts:
            v = np.array([lp.loc[sg] for sg in sgs], dtype=float)
            vw = np.nansum(v * w) / w.sum()
            row += " {:+12.2f}".format(vw - bw)
        print(row)


def main():
    base = per_spg(load_run(BASE_FILE))
    base_pct = base["pct"]
    base_tot = base["total"].fillna(0).astype(int)

    populated = base_tot[base_tot > 0].index.astype(int).tolist()
    sg_to_pos = {sg: i for i, sg in enumerate(populated)}
    n_pop = len(populated)
    counts = np.array([base_tot.loc[sg] for sg in populated], dtype=float)
    alphas = np.clip(0.25 + 0.75 * np.log1p(counts) / np.log1p(counts.max()),
                     0.25, 1.0)

    base_overall = 100 * base["matched"].sum() / base["total"].sum()
    print("{:44s} {:.2f}%  ({} populated SGs)".format(
        BASE_LABEL, base_overall, n_pop))

    lever_pcts, deltas = [], []
    for label, path in LEVERS:
        g = per_spg(load_run(path))
        lever_pcts.append(g["pct"])
        deltas.append(np.array([(g["pct"] - base_pct).loc[sg] for sg in populated],
                               dtype=float))
        ov = 100 * g["matched"].sum() / g["total"].sum()
        print("{:44s} {:.2f}%  ({:+.2f} pp)".format(label, ov, ov - base_overall))

    report_by_system(populated, counts, base_pct, lever_pcts)

    # One symmetric scale for every panel so heights are comparable.
    ylim = 1.06 * max(np.nanmax(np.abs(d)) for d in deltas)
    print("\nshared y-limit +/-{:.0f} pp".format(ylim))

    n_lev = len(LEVERS)
    fig = plt.figure(figsize=(FIG_W, 2.15 * n_lev + 1.15))
    gs = fig.add_gridspec(n_lev + 1, 1,
                          height_ratios=[1] * n_lev + [0.60],
                          hspace=0.13, top=0.94, bottom=0.05,
                          left=GS_LEFT, right=GS_RIGHT)
    axes = [fig.add_subplot(gs[i]) for i in range(n_lev)]
    axb  = fig.add_subplot(gs[n_lev], sharex=axes[0])

    xs = np.arange(n_pop)
    for i, (ax, (label, _), delta) in enumerate(zip(axes, LEVERS, deltas)):
        v = ~np.isnan(delta)
        bars = ax.bar(xs[v], delta[v], width=0.85,
                      color=[POS if x > 0 else NEG for x in delta[v]],
                      edgecolor="black", linewidth=0.25)
        for b, a in zip(bars, alphas[v]):
            b.set_alpha(float(a))
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(-0.5, n_pop - 0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_yticks([-100, -50, 0, 50, 100])

        # Label inside the panel: cheaper in vertical space than set_title.
        ax.text(0.004, 0.95, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=12.5, fontweight="bold")

        if i < n_lev - 1:
            ax.tick_params(axis="x", labelbottom=False, length=0)

    # Tick numbers on the last bar panel only; thinned so they do not crowd.
    step = max(1, n_pop // 14)
    tick_pos = list(range(0, n_pop, step))
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels([str(populated[i]) for i in tick_pos], fontsize=10.5)
    axes[-1].tick_params(axis="x", pad=1.5)

    # Brackets get their own strip, and carry the axis label underneath, so
    # nothing sits between the tick numbers and the system names.
    draw_brackets(axb, sg_to_pos, fs=13,
                  axis_width_in=FIG_W * (GS_RIGHT - GS_LEFT))
    axb.set_xlabel("Space group number", fontsize=14, labelpad=1)

    fig.supylabel("Match rate change (percentage points)", fontsize=14, x=0.018)
    fig.suptitle("Per-space-group accuracy change vs the r=32 SFT baseline  (MPTS-52)",
                 fontsize=16.5, y=0.985, fontweight="bold")

    pdf = HERE / "fig_levers_per_spg_24k.pdf"
    png = HERE / "fig_levers_per_spg_24k.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
