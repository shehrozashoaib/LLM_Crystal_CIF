"""
analyze_sft_steps.py
--------------------
Bar chart of total optimizer steps used to train each of the four SFT
variants. Curriculum learning is shown as a stacked bar (MP-20
pretraining + MPTS-52 fine-tuning).

Run from inside the LLM_materials_files directory:
    python analyze_sft_steps.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Step budgets per run (manually entered).
RUNS = [
    # label,                              segments dict   ,  bar_color
    ("Baseline\nSFT (r=32)",
        [("MP-52", 4275, "#0072B2")]),
    ("2x params\nSFT (r=64)",
        [("MP-52", 4275, "#56B4E9")]),
    ("Curriculum\n(MP-20 -> MP-52)",
        [("MP-20\npretraining", 5076, "#7BB28E"),  # bottom segment, lighter green
         ("MPTS-52",            2080, "#009E73")]),# top segment, darker green
    ("Combined\n(MP-20 + MP-52)",
        [("MP-20 + MP-52", 5170, "#E69F00")]),
]


def main():
    labels = [r[0] for r in RUNS]
    totals = [sum(seg[1] for seg in r[1]) for r in RUNS]
    print("Step totals:")
    for lbl, tot in zip(labels, totals):
        print("  ", lbl.replace("\n", " "), "=", tot)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    xs = np.arange(len(labels))

    bar_w = 0.62
    for x, (label, segs) in zip(xs, RUNS):
        bottom = 0
        for seg_name, seg_steps, seg_color in segs:
            ax.bar(x, seg_steps, bottom=bottom, color=seg_color,
                   edgecolor="black", linewidth=0.7, width=bar_w)
            # In-bar label for stacked segments (only when there is more than one segment)
            if len(segs) > 1:
                ax.text(x, bottom + seg_steps / 2, seg_name + "\n" + str(seg_steps),
                        ha="center", va="center", fontsize=10.5, color="white",
                        fontweight="bold")
            bottom += seg_steps

        # Total label on top of every bar (sum if stacked)
        total = sum(s[1] for s in segs)
        ax.text(x, total + max(totals) * 0.025,
                str(total) + " steps",
                ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Optimizer steps", fontsize=14)
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_title("Training-step budget per SFT variant",
                 fontsize=16, pad=12, fontweight="bold")

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", lw=0.35, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    pdf = HERE / "fig_sft_steps.pdf"
    png = HERE / "fig_sft_steps.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)


if __name__ == "__main__":
    main()
