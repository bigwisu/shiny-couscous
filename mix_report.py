"""The training class mix at each data stage, against the corpus design, as a table and a plot.

For the screening task there is one binary label: include ("none") vs exclude ("excluded").
The design clean share is the corpus-wide positive rate (~16.4%).
"""

import collections
import os
import sys
from typing import Any

from finetune_data import DESIGN_CLEAN_SHARE, labels

# For a binary task the only type-level check is include vs exclude balance.
# We reuse TYPE_BAND to check whether the exclude share sits within a sensible range.
TYPE_BAND = (0.5, 1.5)
CLEAN_TOLERANCE = 0.05
SURFACE, INK, INK_2, MUTED, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a"
BEFORE, AFTER = "#3987e5", "#d95926"

Stage = tuple[str, list[dict[str, Any]], bool]  # (name, rows, use each row's weight)


def shares(rows: list[dict[str, Any]], weighted: bool, reference: str = "planted") -> dict[str, float]:
    """Each label's share of total training weight."""
    total, out = 0.0, collections.defaultdict(float)
    for r in rows:
        w = r.get("weight", 1.0) if weighted else 1.0
        out[labels(r, reference)["verdict"]] += w
        total += w
    return {k: v / total for k, v in out.items()}


def problems(mix: dict[str, float]) -> list[str]:
    """What is off target; empty when the mix is safe to train on."""
    found = []
    include_share = mix.get("include", 0.0)
    if abs(include_share - DESIGN_CLEAN_SHARE) > CLEAN_TOLERANCE:
        found.append(f"include share {include_share:.1%}, design {DESIGN_CLEAN_SHARE:.1%}")
    return found


def print_table(stages: list[Stage]) -> None:
    width = max(len(name) for name, _, _ in stages) + 2
    print(f"{'stage':{width}s} {'rows':>6} {'include':>8} {'exclude':>8}")
    for name, rows, weighted in stages:
        mix = shares(rows, weighted)
        print(f"{name:{width}s} {len(rows):>6} {mix.get('include', 0.0):>7.1%} {mix.get('exclude', 0.0):>8.1%}")
    print(f"{'design':{width}s} {'':>6} {DESIGN_CLEAN_SHARE:>7.1%} {1 - DESIGN_CLEAN_SHARE:>8.1%}")


def _clean_panel(ax, stages: list[Stage]) -> None:
    include_pct = [shares(rows, w).get("include", 0.0) * 100 for _, rows, w in stages]
    ax.bar(range(len(stages)), include_pct,
           color=[AFTER if w else BEFORE for _, _, w in stages], width=0.6)
    for i, v in enumerate(include_pct):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", color=INK, fontsize=13)
    ax.axhline(DESIGN_CLEAN_SHARE * 100, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(-0.3, DESIGN_CLEAN_SHARE * 100 + 2.5, f"design: {DESIGN_CLEAN_SHARE:.1%}",
            color=MUTED, ha="left")
    ax.set_ylim(0, DESIGN_CLEAN_SHARE * 100 + 12)
    ax.set_xticks(range(len(stages)), [name for name, _, _ in stages], rotation=20, ha="right")
    ax.set_ylabel("include (positive) %, of training weight")
    ax.set_title("Include vs exclude balance", color=INK, loc="left", fontsize=15)


def plot(stages: list[Stage], path: str | None = None, compare: tuple[int, int] = (0, -1)):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                         "ytick.color": INK_2, "axes.edgecolor": GRID, "font.size": 12})
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    _clean_panel(ax, stages)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def preflight(rows: list[dict], mix: str, out: str | None, allow_skew: bool) -> None:
    """Show the class mix this run will train on, and refuse a skewed one unless told otherwise.

    When --mix balanced is active the gradient weights are intentionally 50/50, which
    is off the 16.4% corpus design by construction.  That is not a data problem, so the
    design-skew check is skipped for balanced runs.
    """
    import matplotlib
    matplotlib.use("Agg")

    stages = [("as loaded", rows, False), (f"as trained (--mix {mix})", rows, mix != "none")]
    print_table(stages)
    if out:
        os.makedirs(out, exist_ok=True)
        plot(stages, os.path.join(out, "mix.png"))
    # Skip design-skew check when balanced weighting is intentionally applied
    if mix == "balanced":
        return
    found = problems(shares(rows, weighted=mix != "none"))
    if found and not allow_skew:
        sys.exit("mix is off the design, not training:\n  " + "\n  ".join(found)
                 + "\nfix it with --mix balanced, or pass --allow-skew")
