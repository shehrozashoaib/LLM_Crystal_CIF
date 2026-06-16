"""Per-space-group accuracy DELTA vs 1% (r=32) SFT baseline.

Three stacked bar charts (2x params, curriculum, combined). Only populated
space groups appear; bar alpha scales with sample count; crystal-system
brackets are drawn UNDER the panels.

Run from inside LLM_materials_files:
    python analyze_sft_per_spg.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

RUNS = {
    "Baseline (r=32)":
        HERE / "plain_sft_baseline_extracted"
             / "validation_20251220_125842_mp52_16bit_10sequences_mp52_Qwen_2.5_7B",
    "2x params":
        HERE / "experiments"
             / "validation_20260427_162747_mp52_16bit_1sequences_mp52_Qwen2.5-7B-Instruct_r64",
    "Curriculum (MP-20 -> MP-52)":
        HERE / "1_percent_parameters_0.0001_lr_curriculum_learning_validation_results"
             / "validation_20260129_233847_mp52_16bit_10sequences_mp52_Qwen2.5-7B-Instruct",
    "Combined (MP-20 + MP-52)":
        HERE / "1_percent_parameters_0.0001_lr_actual_validation_outputs_mp_combined_test_mp52_only"
             / "validation_20260217_203701_mp52_16bit_1sequences_mp52_Qwen2.5-7B-Instruct",
}

CRYSTAL_SYSTEMS = [
    ("Triclinic",    1,   2),
    ("Monoclinic",   3,  15),
    ("Orthorhombic", 16, 74),
    ("Tetragonal",   75, 142),
    ("Trigonal",     143, 167),
    ("Hexagonal",    168, 194),
    ("Cubic",        195, 230),
]


def load_run(folder):
    csvs = [p for p in folder.iterdir()
            if p.name.endswith("_16bit.csv") and "summary" not in p.name]
    return pd.read_csv(csvs[0], low_memory=False)


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


def draw_brackets(ax, sg_to_pos, fs=14):
    """Brackets at the top of the strip; labels horizontal underneath."""
    n = len(sg_to_pos)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    by, ty = 0.92, 0.74          # bracket bar / vertical-tick endpoints
    label_y_h = 0.62             # horizontal-label baseline (top, va='top')
    label_y_v = 0.58             # vertical-label anchor (Triclinic), va='top'

    for name, lo, hi in CRYSTAL_SYSTEMS:
        sgs = [sg for sg in range(lo, hi+1) if sg in sg_to_pos]
        if not sgs:
            continue
        x_lo = sg_to_pos[min(sgs)] - 0.5
        x_hi = sg_to_pos[max(sgs)] + 0.5
        ax.plot([x_lo, x_hi], [by, by], color="black", lw=1.2, clip_on=False)
        ax.plot([x_lo, x_lo], [by, ty], color="black", lw=1.2, clip_on=False)
        ax.plot([x_hi, x_hi], [by, ty], color="black", lw=1.2, clip_on=False)
        cx = (x_lo + x_hi) / 2
        w  = x_hi - x_lo

        # Triclinic is the only bracket narrow enough to need rotation.
        if w <= 2.5:
            ax.text(cx, label_y_v, name, ha="center", va="top",
                    fontsize=fs - 2, rotation=90)
        else:
            # Everything else fits horizontally; centre the label.
            local_fs = fs - (1.5 if w < 6 else 0)
            ax.text(cx, label_y_h, name, ha="center", va="top",
                    fontsize=local_fs)


def main():
    data = {}
    for label, folder in RUNS.items():
        if not folder.is_dir():
            print("[ERROR]", label, folder, file=sys.stderr); sys.exit(1)
        data[label] = per_spg(load_run(folder))
        n = data[label].dropna(subset=["pct"]).shape[0]
        print(label, ":", n, "populated SGs")

    base_pct = data["Baseline (r=32)"]["pct"]
    base_tot = data["Baseline (r=32)"]["total"].fillna(0).astype(int)
    populated = base_tot[base_tot > 0].index.astype(int).tolist()
    sg_to_pos = {sg: i for i, sg in enumerate(populated)}
    n_pop = len(populated)
    counts = np.array([base_tot.loc[sg] for sg in populated], dtype=float)
    alphas = np.clip(0.25 + 0.75 * np.log1p(counts) / np.log1p(counts.max()),
                     0.25, 1.0)

    deltas = {}
    for label in ["2x params",
                  "Curriculum (MP-20 -> MP-52)",
                  "Combined (MP-20 + MP-52)"]:
        d = data[label]["pct"] - base_pct
        deltas[label] = np.array([d.loc[sg] for sg in populated], dtype=float)

    all_v = np.concatenate(list(deltas.values()))
    abs_max = np.nanmax(np.abs(all_v))
    ylim = 1.10 * abs_max
    print("y-limit +/-", round(ylim, 1), "pp ;  populated SGs:", n_pop)

    fig = plt.figure(figsize=(11.5, 10.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 3, 3, 1.4],
                          hspace=0.40, top=0.87, bottom=0.07,
                          left=0.10, right=0.97)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
    axb = fig.add_subplot(gs[3], sharex=ax1)
    data_axes = [ax1, ax2, ax3]

    POS, NEG = "#1A8F3F", "#C03030"
    xs = np.arange(n_pop)

    for ax, (lbl, d) in zip(data_axes, deltas.items()):
        v = ~np.isnan(d)
        cols = [POS if x > 0 else NEG for x in d[v]]
        bars = ax.bar(xs[v], d[v], width=0.85, color=cols,
                      edgecolor="black", linewidth=0.25)
        for b, a in zip(bars, alphas[v]):
            b.set_alpha(float(a))
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(lbl, loc="left", fontsize=15, fontweight="bold", pad=8)
        ax.set_ylabel("Match rate change\n(percentage points)", fontsize=13.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(-0.5, n_pop - 0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=11)

    step = max(1, n_pop // 18)
    tick_pos = list(range(0, n_pop, step))
    tick_lab = [str(populated[i]) for i in tick_pos]
    for ax in data_axes:
        ax.set_xticks(tick_pos)
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax3.set_xticklabels(tick_lab)
    ax3.set_xlabel("Space group number", fontsize=15.5)

    draw_brackets(axb, sg_to_pos, fs=14)

    fig.suptitle("Per-space-group accuracy change vs SFT baseline  (MPTS-52)",
                 fontsize=17, y=0.965, fontweight="bold")

    pdf = HERE / "fig_sft_per_spg.pdf"
    png = HERE / "fig_sft_per_spg.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
