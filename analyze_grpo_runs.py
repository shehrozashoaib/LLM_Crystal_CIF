"""
analyze_grpo_runs.py
--------------------
Bar chart of best-of-10 match rate on MPTS-52 for:
    Baseline SFT  vs  GRPO (discrete reward)  vs  GRPO (continuous reward)

Run from inside the LLM_materials_files directory:
    python analyze_grpo_runs.py
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
    "Baseline\nSFT (r=32)":
        HERE / "plain_sft_baseline_extracted"
             / "validation_20251220_125842_mp52_16bit_10sequences_mp52_Qwen_2.5_7B",

    "GRPO\n(discrete reward)":
        HERE / "experiments_generated_cifs_grpo_from_sft_target_aligned_discrete_reward_function_v5_10seq_maxtok3372_0_749"
             / "validation_20260422_062544_mp52_16bit_10sequences_mp52_Qwen_2.5_7B",

    "GRPO\n(continuous reward)":
        HERE / "experiments_generated_cifs_grpo_from_sft_target_aligned_v6_continuous_rewrad_function_10seq_maxtok3372_0_749"
             / "validation_20260424_084203_mp52_16bit_10sequences_mp52_Qwen_2.5_7B",
}

COLORS = ["#0072B2", "#D55E00", "#CC79A7"]  # baseline, discrete, continuous


def load_run(folder):
    csvs = [p for p in folder.iterdir()
            if p.name.endswith("_16bit.csv") and "summary" not in p.name]
    if not csvs:
        raise FileNotFoundError("No *_16bit.csv in " + str(folder))
    return pd.read_csv(csvs[0], low_memory=False)


def best_of_n(df):
    mc = [c for c in df.columns if c.startswith("Match_generation_")]
    matched = df[mc].fillna(0).any(axis=1).sum()
    total = len(df)
    return 100.0 * matched / total, int(matched), total


def main():
    rows = []
    for label, folder in RUNS.items():
        if not folder.is_dir():
            print("[ERROR]", label, folder, file=sys.stderr); sys.exit(1)
        df = load_run(folder)
        pct, matched, total = best_of_n(df)
        rows.append((label, pct, matched, total))
        print("{:30s}  {:5.1f}%   ({}/{})".format(
            label.replace("\n", " "), pct, matched, total))

    labels = [r[0] for r in rows]
    pcts   = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, pcts, color=COLORS[: len(labels)],
                  edgecolor="black", linewidth=0.7, width=0.6)

    for x, v in zip(xs, pcts):
        ax.text(x, v + 0.6, "{:.1f}%".format(v),
                ha="center", va="bottom", fontsize=12.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Best-of-10 match rate on MPTS-52 (%)", fontsize=13.5)
    ax.set_ylim(0, max(pcts) * 1.20)
    ax.set_title("GRPO reward designs vs the SFT baseline",
                 fontsize=14, pad=12)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.35, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    pdf = HERE / "fig_grpo_runs.pdf"
    png = HERE / "fig_grpo_runs.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
