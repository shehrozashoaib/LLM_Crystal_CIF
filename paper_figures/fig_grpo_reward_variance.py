"""
fig_grpo_reward_variance.py
---------------------------
Replacement for the within-group reward-variance panel, and a diagnosis of
where that variance comes from.

Two problems with the old panel. First, the text quoted a *median* within-group
reward standard deviation (about 0.001 discrete, 0.087 continuous) while the
figure plotted a rolling *mean*, and the mean is dominated by the small number
of groups that do straddle a tier boundary. The two curves therefore sat on top
of each other near 0.1 and appeared to contradict the text. Both statistics are
reproduced here so the discrepancy is visible rather than confusing.

Second, and more important: the reward is a match term plus a chemistry
tiebreaker capped at 0.03. A group whose rollouts all land in the same match
tier still has a non-zero reward spread, purely from that tiebreaker. Since the
advantage is normalised by the within-group standard deviation, such a group
still contributes a full-size gradient, one that carries no information about
structural correctness. Panel B separates the two by recomputing the spread
from the match term alone.

Inputs are the per-rollout reward traces shipped with the code release,
results/grpo/grpo_r32_from3000_{discrete,continuous}/reward_trace.jsonl.gz,
copied verbatim into data/.

Run from inside paper_figures/:
    python fig_grpo_reward_variance.py
"""
from pathlib import Path
from collections import defaultdict
import gzip
import json
import statistics as st

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

RUNS = [("Discrete (group 4)", "rt.jsonl.gz", "#C03030"),
        ("Continuous (group 8)", "rt_cont.jsonl.gz", "#1A8F3F")]
WINDOW = 50          # groups per rolling window


def group_spreads(path):
    """Per prompt-group: spread of the full reward, and of the match term alone.

    Groups are returned in the order they were logged, which is training order.
    """
    groups, first_seen = defaultdict(list), {}
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            key = (d["batch_hash"], d["row_id"])
            groups[key].append(d)
            first_seen.setdefault(key, i)
    full, match = [], []
    for key in sorted(groups, key=lambda k: first_seen[k]):
        g = groups[key]
        if len(g) < 2:
            continue
        full.append(st.pstdev([x["final_reward"] for x in g]))
        match.append(st.pstdev([x["r_match"] for x in g]))
    return np.array(full), np.array(match)


def rolling(x, w, fn):
    return np.array([fn(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    stats = {}

    for label, fname, colour in RUNS:
        full, match = group_spreads(DATA / fname)
        xs = np.arange(len(full))
        stats[label] = dict(
            n=len(full),
            med_full=float(np.median(full)), mean_full=float(np.mean(full)),
            med_match=float(np.median(match)),
            zero_match=float(np.mean(match == 0) * 100),
            only_tb=float(np.mean((full > 0) & (match == 0)) * 100),
        )
        axA.plot(xs, rolling(full, WINDOW, np.mean), color=colour, lw=1.0,
                 alpha=0.45, zorder=2)
        axA.plot(xs, rolling(full, WINDOW, np.median), color=colour, lw=2.2,
                 zorder=3, label=label)

    axA.set_yscale("log")
    axA.set_xlabel("prompt-group (training order)", fontsize=12.5)
    axA.set_ylabel("within-group reward spread", fontsize=12.5)
    axA.set_title("(A)  Median vs mean spread", fontsize=14,
                  fontweight="bold", loc="left", pad=8)
    axA.text(0.98, 0.04,
             "bold = rolling median   ·   faint = rolling mean",
             transform=axA.transAxes, ha="right", fontsize=9.5,
             color="#555555", style="italic")
    axA.legend(fontsize=10, frameon=False, loc="upper right")

    # ---- Panel B: where the spread comes from ----
    labels = [r[0] for r in RUNS]
    only_tb = [stats[l]["only_tb"] for l in labels]
    real = [100 - v for v in only_tb]
    y = np.arange(len(labels))
    axB.barh(y, real, color="#1A8F3F", edgecolor="black", linewidth=0.6,
             label="group spans >1 match tier")
    axB.barh(y, only_tb, left=real, color="#D9D9D9", edgecolor="black",
             linewidth=0.6, label="tiebreaker only, no match signal")
    for i, (r, t) in enumerate(zip(real, only_tb)):
        axB.text(r / 2, i, f"{r:.0f}%", ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color="white")
        axB.text(r + t / 2, i, f"{t:.0f}%", ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color="#333333")
    axB.set_yticks(y); axB.set_yticklabels(labels, fontsize=11.5)
    axB.set_xlim(0, 100)
    axB.set_xlabel("share of prompt-groups (%)", fontsize=12.5)
    axB.set_title("(B)  Where the advantage signal comes from", fontsize=14,
                  fontweight="bold", loc="left", pad=8)
    axB.legend(fontsize=9.5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.36), ncol=2)
    axB.invert_yaxis()

    for ax in (axA, axB):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x" if ax is axB else "y", lw=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)

    print(f"{'run':22s} {'groups':>7} {'med(full)':>10} {'mean(full)':>11} "
          f"{'med(match)':>11} {'no match signal':>16}")
    for l in labels:
        s = stats[l]
        print(f"{l:22s} {s['n']:7d} {s['med_full']:10.4f} {s['mean_full']:11.4f} "
              f"{s['med_match']:11.4f} {s['only_tb']:15.1f}%")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = HERE / f"fig_grpo_reward_variance.{ext}"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print("Saved", p)


if __name__ == "__main__":
    main()
