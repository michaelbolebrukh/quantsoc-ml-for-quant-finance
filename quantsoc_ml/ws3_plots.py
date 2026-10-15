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
    axes[0].set_title("MDI: how often a feature splits (training data)")
    axes[1].barh(order, perm.reindex(order), color=[QS_GREEN if v > 0 else QS_RED for v in perm.reindex(order)])
    axes[1].set_title("Permutation: AUC lost when shuffled (test data)")
    axes[0].invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
