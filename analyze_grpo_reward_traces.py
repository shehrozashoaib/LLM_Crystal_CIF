"""GRPO reward-signal analysis: discrete vs continuous reward.

Reads six JSONL traces dropped in this folder:

  reward_group_trace_v5 (1).jsonl    - per-prompt group stats, discrete reward
  reward_group_trace_v6 (1).jsonl    - per-prompt group stats, continuous reward
  reward_trace_v5_discrete.jsonl     - per-rollout sample, discrete reward
  reward_trace_v6_continuous.jsonl   - per-rollout sample, continuous reward
  val_trace_discrete.jsonl           - held-out validation curve, discrete
  val_trace_continuous.jsonl         - held-out validation curve, continuous

Produces a 2x2 figure that contrasts the two reward designs:
 (A) within-group reward standard deviation over training,
 (B) fraction of prompt-groups carrying match variance over training,
 (C) held-out best-of-4 validation match rate over training,
 (D) match-tier composition across all training rollouts.

Run from inside LLM_materials_files:
    python analyze_grpo_reward_traces.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

GROUP_FILES = {
    "Discrete":   HERE / "reward_group_trace_v5 (1).jsonl",
    "Continuous": HERE / "reward_group_trace_v6 (1).jsonl",
}
SAMPLE_FILES = {
    "Discrete":   HERE / "reward_trace_v5_discrete.jsonl",
    "Continuous": HERE / "reward_trace_v6_continuous.jsonl",
}
VAL_FILES = {
    "Discrete":   HERE / "val_trace_discrete.jsonl",
    "Continuous": HERE / "val_trace_continuous.jsonl",
}

# Discrete in red, continuous in green
COLOR = {"Discrete": "#C03030", "Continuous": "#1A8F3F"}


def load_jsonl(p):
    if not p.exists():
        print("[ERROR] missing", p, file=sys.stderr); sys.exit(1)
    return [json.loads(l) for l in p.open()]


def to_step(rows):
    """Replace batch_hash by an integer training-step index in the order
    the batch first appears."""
    seen = {}
    idx = []
    for r in rows:
        bh = r["batch_hash"]
        if bh not in seen:
            seen[bh] = len(seen)
        idx.append(seen[bh])
    return np.asarray(idx)


def rolling(y, win):
    y = np.asarray(y, dtype=float)
    if len(y) < win or win < 2:
        return y
    c = np.convolve(y, np.ones(win)/win, mode="valid")
    pad = np.full(win - 1, np.nan)
    return np.concatenate([pad, c])


def main():
    group_data = {k: load_jsonl(p) for k, p in GROUP_FILES.items()}
    sample_data = {k: load_jsonl(p) for k, p in SAMPLE_FILES.items()}
    val_data    = {k: load_jsonl(p) for k, p in VAL_FILES.items()}

    print("Loaded:")
    for k in group_data:
        n_g = len(group_data[k])
        n_s = len(sample_data[k])
        n_v = len(val_data[k])
        n_b = len({r["batch_hash"] for r in group_data[k]})
        print(f"  {k:12s}  steps={n_b:3d}  groups={n_g:4d}  "
              f"samples={n_s:5d}  val_pts={n_v}")

    # ----- Figure --------------------------------------------------------
    fig = plt.figure(figsize=(15.5, 11.0))
    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.985,
        top=0.91, bottom=0.10,
        wspace=0.24, hspace=0.42,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---------- (A) within-group reward std vs training step --------------
    win = 24
    for label, rows in group_data.items():
        step = to_step(rows)
        std  = np.array([r["reward_std"] for r in rows])
        steps = np.unique(step)
        per_step = np.array([std[step == s].mean() for s in steps])
        smooth = rolling(per_step, win=min(win, max(2, len(per_step)//8)))
        axA.plot(steps, per_step, color=COLOR[label], alpha=0.18, lw=0.9)
        axA.plot(steps, smooth,   color=COLOR[label], lw=2.6, label=label)
    axA.set_xlabel("Training step", fontsize=14, labelpad=8)
    axA.set_ylabel("Within-group reward std", fontsize=14, labelpad=8)
    axA.set_title("(A)  Reward variance inside each prompt-group",
                  loc="left", fontsize=15, fontweight="bold", pad=14)
    axA.tick_params(axis="both", labelsize=12)
    axA.grid(axis="y", lw=0.3, alpha=0.4); axA.set_axisbelow(True)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)
    yA0, yA1 = axA.get_ylim()
    axA.set_ylim(yA0, yA1 * 1.18)
    axA.legend(loc="upper right", fontsize=12, frameon=True,
               framealpha=0.95, facecolor="white", edgecolor="lightgray")

    # ---------- (B) fraction of groups with match-variance ---------------
    for label, rows in group_data.items():
        step = to_step(rows)
        hmv  = np.array([1.0 if r["has_match_variance"] else 0.0 for r in rows])
        steps = np.unique(step)
        frac  = np.array([hmv[step == s].mean() for s in steps])
        smooth = rolling(frac, win=min(win, max(2, len(frac)//8)))
        axB.plot(steps, frac,   color=COLOR[label], alpha=0.18, lw=0.9)
        axB.plot(steps, smooth, color=COLOR[label], lw=2.6, label=label)
    axB.set_ylim(0, 1.02)
    axB.set_xlabel("Training step", fontsize=14, labelpad=8)
    axB.set_ylabel("Fraction of groups with match variance",
                   fontsize=14, labelpad=8)
    axB.set_title("(B)  Groups that actually deliver a learning signal",
                  loc="left", fontsize=15, fontweight="bold", pad=14)
    axB.tick_params(axis="both", labelsize=12)
    axB.grid(axis="y", lw=0.3, alpha=0.4); axB.set_axisbelow(True)
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)
    axB.legend(loc="lower right", fontsize=12, frameon=True,
               framealpha=0.95, facecolor="white", edgecolor="lightgray")

    # ---------- (C) held-out validation match rate during training -------
    # Each entry: 4 generations per held-out material x 20 materials.
    # We plot the best-of-4 per-material match rate (analogue of the test
    # best-of-N metric used elsewhere in the paper).
    for label, rows in val_data.items():
        steps = np.array([r["step"] for r in rows])
        mat   = 100 * np.array([r["material_rate"] for r in rows])
        c = COLOR[label]
        axC.plot(steps, mat, color=c, lw=2.6, marker="o", markersize=7,
                 label=label)
    axC.set_xlabel("Training step", fontsize=14, labelpad=8)
    axC.set_ylabel("Held-out validation match rate (%)",
                   fontsize=14, labelpad=8)
    axC.set_title("(C)  Validation match rate during training",
                  loc="left", fontsize=15, fontweight="bold", pad=14)
    axC.set_ylim(0, max(45, axC.get_ylim()[1] * 1.10))
    axC.tick_params(axis="both", labelsize=12)
    axC.grid(axis="y", lw=0.3, alpha=0.4); axC.set_axisbelow(True)
    for sp in ("top", "right"): axC.spines[sp].set_visible(False)
    axC.legend(loc="lower right", fontsize=12, frameon=True,
               framealpha=0.95, facecolor="white", edgecolor="lightgray")

    # ---------- (D) match-tier composition (stacked horizontal bar) ------
    tiers_order = ["val", "med", "close", "loose", "no_match", "not_parseable"]
    tier_colors = {
        "val":            "#1A8F3F",
        "med":            "#7FBF6E",
        "close":          "#C9E0A6",
        "loose":          "#F0D27A",
        "no_match":       "#C03030",
        "not_parseable":  "#7F7F7F",
    }
    labels = list(sample_data.keys())
    counts = {t: [] for t in tiers_order}
    for label in labels:
        rows = sample_data[label]
        for t in tiers_order:
            counts[t].append(sum(1 for r in rows if r["match_tier"] == t))
    totals = np.array([sum(counts[t][i] for t in tiers_order)
                       for i in range(len(labels))], dtype=float)
    bottoms = np.zeros(len(labels))
    y = np.arange(len(labels))
    for t in tiers_order:
        v = np.asarray(counts[t], dtype=float)
        pct = 100 * v / totals
        axD.barh(y, pct, left=bottoms, height=0.55,
                 color=tier_colors[t], edgecolor="black", linewidth=0.4,
                 label=t)
        for yi, p, w in zip(y, pct, bottoms):
            if p >= 4:
                axD.text(w + p/2, yi, f"{p:.0f}%", ha="center", va="center",
                         fontsize=12, color="black")
        bottoms += pct
    axD.set_yticks(y); axD.set_yticklabels(labels, fontsize=13)
    axD.set_xlim(0, 100)
    axD.set_xlabel("Share of rollouts (%)", fontsize=14, labelpad=8)
    axD.set_title("(D)  Match tier composition across all rollouts",
                  loc="left", fontsize=15, fontweight="bold", pad=14)
    axD.tick_params(axis="x", labelsize=12)
    for sp in ("top", "right"): axD.spines[sp].set_visible(False)
    axD.legend(frameon=False, fontsize=12, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.32), handlelength=1.6,
               columnspacing=1.4)

    fig.suptitle(
        "GRPO reward-signal evolution: discrete vs continuous reward",
        fontsize=18, fontweight="bold", y=0.965,
    )

    pdf = HERE / "fig_grpo_reward_signal.pdf"
    png = HERE / "fig_grpo_reward_signal.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    print("Saved", pdf)
    print("Saved", png)

    # ----- A few numbers to copy into the paper --------------------------
    print("\n=== Summary stats for the paper ===")
    for label, rows in group_data.items():
        stds = [r["reward_std"] for r in rows]
        hmv  = sum(1 for r in rows if r["has_match_variance"])
        print(f"{label:12s}  groups={len(rows):4d}  "
              f"median_std={np.median(stds):.4f}  "
              f"mean_std={np.mean(stds):.4f}  "
              f"variance_groups={hmv}/{len(rows)} ({100*hmv/len(rows):.1f}%)")
    for label, rows in sample_data.items():
        from collections import Counter
        c = Counter(r["match_tier"] for r in rows)
        n = sum(c.values())
        order = ["val", "med", "close", "loose", "no_match", "not_parseable"]
        print(f"{label:12s} tier % -> " +
              "  ".join(f"{t}={100*c.get(t,0)/n:4.1f}" for t in order))
    for label, rows in val_data.items():
        mat = [100 * r["material_rate"] for r in rows]
        print(f"{label:12s} val_curve  best-of-4 mean={np.mean(mat):.1f}%  "
              f"max={max(mat):.1f}%  end={mat[-1]:.1f}%")


if __name__ == "__main__":
    main()
