"""
analyze_grpo_per_spg.py
-----------------------
Per-space-group accuracy DELTA: GRPO (discrete reward) vs the SFT baseline.
Single panel; bar opacity scales with sample count; crystal-system brackets
are drawn UNDER the panel.

Run from inside the LLM_materials_files directory:
    python analyze_grpo_per_spg.py
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
    "GRPO (discrete reward)":
        HERE / "experiments_generated_cifs_grpo_from_sft_target_aligned_discrete_reward_function_v5_10seq_maxtok3372_0_749"
             / "validation_20260422_062544_mp52_16bit_10sequences_mp52_Qwen_2.5_7B",
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
    n = len(sg_to_pos)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    by, ty = 0.92, 0.74
    label_y_h = 0.62
    label_y_v = 0.58

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
        if w <= 2.5:
            ax.text(cx, label_y_v, name, ha="center", va="top",
                    fontsize=fs - 2, rotation=90)
        else:
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

    label = "GRPO (discrete reward)"
    d = data[label]["pct"] - base_pct
    delta = np.array([d.loc[sg] for sg in populated], dtype=float)

    abs_max = np.nanmax(np.abs(delta))
    ylim = 1.10 * abs_max
    print("y-limit +/-", round(ylim, 1), "pp ;  populated SGs:", n_pop)

    fig = plt.figure(figsize=(11.5, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.4],
                          hspace=0.45, top=0.84, bottom=0.10,
                          left=0.10, right=0.97)
    ax  = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=ax)

    POS, NEG = "#1A8F3F", "#C03030"
    xs = np.arange(n_pop)

    v = ~np.isnan(delta)
    cols = [POS if x > 0 else NEG for x in delta[v]]
    bars = ax.bar(xs[v], delta[v], width=0.85, color=cols,
                  edgecolor="black", linewidth=0.25)
    for b, a in zip(bars, alphas[v]):
        b.set_alpha(float(a))
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title(label, loc="left", fontsize=15, fontweight="bold", pad=8)
    ax.set_ylabel("Match rate change\n(percentage points)", fontsize=15.5)
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
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_xlabel("Space group number", fontsize=15.5)

    draw_brackets(axb, sg_to_pos, fs=14)

    fig.suptitle("Per-space-group accuracy change vs SFT baseline  (MPTS-52)",
                 fontsize=17, y=0.965, fontweight="bold")

    pdf = HERE / "fig_grpo_discrete_per_spg.pdf"
    png = HERE / "fig_grpo_discrete_per_spg.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
