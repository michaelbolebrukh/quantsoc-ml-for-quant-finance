"""Charts for the Workshop 3 notebook, kept here so the notebook cells stay short."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QS_NAVY, QS_GOLD, QS_GREY, QS_RED, QS_GREEN = "#14213d", "#fca311", "#8d99ae", "#d62828", "#2a9d8f"


def periods_timeline(periods, ax=None):
    """One bar per study period: training block in grey, trading block in gold."""
    ax = ax or plt.subplots(figsize=(11, 0.28 * len(periods) + 1))[1]
    for p in periods:
        ax.barh(p.number, (p.train_end - p.train_start).days, left=p.train_start, color=QS_GREY, height=0.7)
        ax.barh(p.number, (p.trade_end - p.trade_start).days, left=p.trade_start, color=QS_GOLD, height=0.7)
    ax.invert_yaxis()
    ax.set_ylabel("study period")
    ax.set_title("Walk-forward: train on 750 days (grey), trade the next 250 (gold), slide forward")
    return ax.figure


def bar(series: pd.Series, title: str, ylabel: str = "", color=None, ax=None, pct: bool = False):
    ax = ax or plt.subplots(figsize=(11, 3.6))[1]
    vals = series * 100 if pct else series
    colors = color or [QS_GREEN if v >= 0 else QS_RED for v in vals]
    ax.bar([str(i) for i in vals.index], vals.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel + (" (%)" if pct else ""))
    plt.setp(ax.get_xticklabels(), rotation=60 if len(vals) > 12 else 0, ha="right" if len(vals) > 12 else "center")
    return ax.figure


def equity(curves: Mapping[str, pd.Series], title: str, log: bool = True, ax=None):
    """Growth of $1 for each daily return series."""
    ax = ax or plt.subplots(figsize=(11, 4.2))[1]
    palette = [QS_NAVY, QS_GOLD, QS_RED, QS_GREEN, QS_GREY, "#6a4c93"]
    for (name, r), c in zip(curves.items(), palette):
        ax.plot((1 + r.fillna(0)).cumprod(), label=name, color=c, lw=1.4)
    if log:
        ax.set_yscale("log")
    ax.axhline(1, color="black", lw=0.6)
    ax.set_title(title)
    ax.set_ylabel("growth of $1" + (" (log scale)" if log else ""))
    ax.legend(loc="upper left")
    return ax.figure


def importance_compare(mdi: pd.Series, perm: pd.Series, title: str = "What the forest uses vs what actually helps"):
    order = perm.sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(order, mdi.reindex(order), color=QS_GREY)
    axes[0].set_title("MDI: impurity removed by its splits (training data)")
    axes[1].barh(order, perm.reindex(order), color=[QS_GREEN if v > 0 else QS_RED for v in perm.reindex(order)])
    axes[1].set_title("Permutation: AUC lost when shuffled (test data)")
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- inside the models

def _box(ax, x, y, w, h, text, fc, tc="white", fs=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="none"))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=QS_NAVY, lw=1.4))


def pipeline_diagram():
    """One trading day, from prices to orders: 31 features, three models, the average, the ranking."""
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off"); ax.grid(False)
    _box(ax, 1.1, 2.2, 1.9, 1.3, "Prices\nup to today's\nclose", QS_GREY)
    _box(ax, 3.3, 2.2, 1.9, 1.3, "31 features\nper stock\nR1 ... R240", QS_NAVY)
    names = [("Random forest\n150 independent trees,\naveraged", 3.6),
             ("Gradient boosting\n100 small trees,\neach fixing the last", 2.2),
             ("Neural network\n31 > 31 > 10 > 5 > 1", 0.8)]
    for text, y in names:
        _box(ax, 6.0, y, 2.4, 1.05, text, QS_GOLD, tc=QS_NAVY, fs=9)
        _arrow(ax, 4.25, 2.2, 4.8, y)
        _arrow(ax, 7.2, y, 7.75, 2.2)
    _box(ax, 8.55, 2.2, 1.6, 1.5, "Average of the\n3 probabilities\nP(beats the\nmedian)", QS_NAVY, fs=9)
    _box(ax, 10.8, 3.1, 1.9, 0.9, "Top 10:\nbuy", QS_GREEN)
    _box(ax, 10.8, 1.3, 1.9, 0.9, "Bottom 10:\nsell short", QS_RED)
    _arrow(ax, 2.05, 2.2, 2.35, 2.2)
    _arrow(ax, 9.35, 2.5, 9.85, 3.0)
    _arrow(ax, 9.35, 1.9, 9.85, 1.4)
    ax.text(10.8, 2.2, "rank ~100 stocks", ha="center", va="center", fontsize=9, color=QS_NAVY)
    ax.set_title("Every trading day, for every stock: features, three models, one average, one ranking")
    return fig


def mlp_diagram(layers=(31, 31, 10, 5, 1), show=8):
    """The network's shape. Wide layers are drawn with `show` neurons and a '...'."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off"); ax.grid(False)
    labels = ["31 inputs\n(R1 ... R240,\nstandardised)", "hidden 1\n31 neurons", "hidden 2\n10 neurons",
              "hidden 3\n5 neurons", "output\nP(beats median)"]
    pos = []
    for i, n in enumerate(layers):
        k = min(n, show)
        ys = np.linspace(-1, 1, k) if k > 1 else np.array([0.0])
        pos.append([(i, y) for y in ys])
        for x, y in pos[-1]:
            ax.add_patch(plt.Circle((x, y), 0.055, color=QS_GOLD if i in (0, len(layers) - 1) else QS_NAVY, zorder=3))
        if n > show:
            ax.text(i, 0, "...", ha="center", va="center", fontsize=14, color="white", zorder=4,
                    bbox=dict(boxstyle="round", fc=QS_NAVY, ec="none"))
        ax.text(i, -1.45, labels[i], ha="center", va="top", fontsize=9)
    for a, b in zip(pos[:-1], pos[1:]):
        for x0, y0 in a:
            for x1, y1 in b:
                ax.plot([x0, x1], [y0, y1], color=QS_GREY, lw=0.4, alpha=0.6, zorder=1)
    ax.set_xlim(-0.5, len(layers) - 0.5); ax.set_ylim(-2.3, 1.3)
    ax.set_title("The neural network: every neuron takes a weighted sum of the layer before, then ReLU")
    return fig


def agreement_figure(rank_corr: pd.DataFrame, spread: pd.Series):
    """Left: how alike the models' daily rankings are. Right: how widely each model's probabilities spread."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(rank_corr.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(rank_corr)), rank_corr.columns)
    ax.set_yticks(range(len(rank_corr)), rank_corr.index)
    for i in range(len(rank_corr)):
        for j in range(len(rank_corr)):
            v = rank_corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else QS_NAVY)
    ax.set_title("Rank correlation of the daily stock rankings")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    ax.bar(spread.index, spread.values, color=[QS_GREY, QS_GOLD, QS_NAVY][:len(spread)])
    ax.set_title("Typical spread of each model's probabilities on a day\n(std across stocks)")
    ax.set_ylabel("standard deviation")
    fig.tight_layout()
    return fig
