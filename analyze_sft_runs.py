"""
analyze_sft_runs.py
-------------------
Reads the four SFT validation CSVs (baseline r=32, r=64, curriculum,
combined-tested-on-MP52), computes best-of-10 match rate against the
MPTS-52 test split, and plots a bar chart.

Run from inside the LLM_materials_files directory:
    python analyze_sft_runs.py
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Locate runs.  Paths are RELATIVE to this script.
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

RUNS = {
    # label                              -> folder containing validation_..._16bit.csv
    "Baseline\nSFT (r=32)":
        HERE / "plain_sft_baseline_extracted"
             / "validation_20251220_125842_mp52_16bit_10sequences_mp52_Qwen_2.5_7B",

    "2x params\nSFT (r=64)":
        HERE / "experiments"
             / "validation_20260427_162747_mp52_16bit_1sequences_mp52_Qwen2.5-7B-Instruct_r64",

    "Curriculum\n(MP-20 -> MP-52)":
        HERE / "1_percent_parameters_0.0001_lr_curriculum_learning_validation_results"
             / "validation_20260129_233847_mp52_16bit_10sequences_mp52_Qwen2.5-7B-Instruct",

    "Combined\n(MP-20 + MP-52)":
        HERE / "1_percent_parameters_0.0001_lr_actual_validation_outputs_mp_combined_test_mp52_only"
             / "validation_20260217_203701_mp52_16bit_1sequences_mp52_Qwen2.5-7B-Instruct",
}

# Color per bar (color-blind safe Okabe-Ito-ish)
COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00"]


def load_run(folder: Path) -> pd.DataFrame:
    """Find the per-material validation CSV inside `folder` and load it."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Run folder not found: {folder}")
    csv_files = [
        p for p in folder.iterdir()
        if p.name.endswith("_16bit.csv") and "summary" not in p.name
    ]
    if not csv_files:
        raise FileNotFoundError(f"No *_16bit.csv in {folder}")
    return pd.read_csv(csv_files[0], low_memory=False)


def best_of_n(df: pd.DataFrame) -> tuple[float, int, int]:
    """Return (best-of-N percentage, matched count, total)."""
    match_cols = [c for c in df.columns if c.startswith("Match_generation_")]
    if not match_cols:
        raise ValueError("No Match_generation_* columns in CSV")
    matched = df[match_cols].fillna(0).any(axis=1).sum()
    total = len(df)
    return 100.0 * matched / total, int(matched), total


def main():
    rows = []
    for label, folder in RUNS.items():
        try:
            df = load_run(folder)
            pct, matched, total = best_of_n(df)
            rows.append((label, pct, matched, total))
            print(f"{label.replace(chr(10),' '):35s}  best-of-10 = "
                  f"{pct:5.1f}%   ({matched}/{total})")
        except FileNotFoundError as e:
            print(f"[WARN] skipping {label}: {e}")

    if not rows:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    labels = [r[0] for r in rows]
    pcts = [r[1] for r in rows]

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, pcts,
                  color=COLORS[: len(labels)],
                  edgecolor="black", linewidth=0.7, width=0.62)

    # Annotate values on top of each bar
    for x, v in zip(xs, pcts):
        ax.text(x, v + 0.9, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Best-of-10 match rate on MPTS-52 (%)", fontsize=13)
    ax.set_ylim(0, max(pcts) * 1.18)
    ax.set_title("SFT variants on the MPTS-52 test split",
                 fontsize=13.5, pad=10)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.35, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_pdf = HERE / "fig_sft_only.pdf"
    out_png = HERE / "fig_sft_only.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
