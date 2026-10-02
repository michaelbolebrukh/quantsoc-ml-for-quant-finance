"""Plot helpers for the notebook. Every figure has a title, labelled axes with units, and returns the Figure.

Returns are shown in percent (0.01 = 1%) because decimals like 0.0012 are hard to read on an
axis. Dates are always on the x axis of time plots. These are the repetitive plotting parts
only; nothing here hides a learning objective.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "figure.figsize": (10, 4.5)})

_RETURN_LIKE = ("ret_", "vol_", "hi52", "LEAKY", "target")


def _equity(x) -> pd.Series:
    return x.equity if hasattr(x, "equity") else pd.Series(x)


def price_panel(prices_wide: pd.DataFrame, symbols: Optional[Iterable[str]] = None,
                title: str = "Adjusted close prices") -> plt.Figure:
    """Each symbol's adjusted close rebased to 100 on its first date, on a log scale so % moves compare."""
    symbols = list(symbols) if symbols is not None else list(prices_wide.columns[:6])
    fig, ax = plt.subplots()
    for i, s in enumerate(symbols):
        p = prices_wide[s].dropna()
        ax.plot(p.index, p / p.iloc[0] * 100, lw=1, color=PALETTE[i % len(PALETTE)], label=s)
    ax.set_yscale("log")
    ax.set_title(f"{title} (first date = 100, log scale)")
    ax.set_xlabel("date")
    ax.set_ylabel("price index (first date = 100)")
    ax.legend(loc="upper left", ncol=min(len(symbols), 6))
    return fig


def feature_hist(features: pd.DataFrame, name: str, bins: int = 80, clip_pct: float = 0.5) -> plt.Figure:
    """Histogram of one feature over all (date, symbol) rows; the extreme clip_pct% each side is cut for readability."""
    v = features[name].dropna()
    pct = name.startswith(_RETURN_LIKE)
    vals = v * 100 if pct else v
    lo, hi = np.percentile(vals, [clip_pct, 100 - clip_pct])
    fig, ax = plt.subplots()
    ax.hist(vals.clip(lo, hi), bins=bins, color=PALETTE[0], alpha=0.85)
    ax.axvline(0, color="grey", lw=1)
    ax.set_title(f"Distribution of {name} over {len(v):,} stock-days")
    ax.set_xlabel(f"{name} ({'%' if pct else 'standard deviations' if name == 'volume_z' else 'value'})")
    ax.set_ylabel("number of stock-days")
    return fig


def split_timeline(train_df: pd.DataFrame, test_df: pd.DataFrame, prices_wide: Optional[pd.DataFrame] = None,
                   symbol: Optional[str] = None) -> plt.Figure:
    """Shaded train / embargo / test periods over a price line (one symbol, or the equal-weight average)."""
    tr = train_df.index.get_level_values("date")
    te = test_df.index.get_level_values("date")
    fig, ax = plt.subplots()
    if prices_wide is not None:
        if symbol is None:
            line = (prices_wide / prices_wide.iloc[0]).mean(axis=1) * 100
            label = "average of all symbols (first date = 100)"
        else:
            line = prices_wide[symbol] / prices_wide[symbol].iloc[0] * 100
            label = f"{symbol} (first date = 100)"
        ax.plot(line.index, line, color="black", lw=1, label=label)
        ax.set_ylabel("price index (first date = 100)")
    else:
        ax.set_yticks([])
        ax.set_ylabel("")
    ax.axvspan(tr.min(), tr.max(), color=PALETTE[0], alpha=0.15, label=f"train: {tr.min().date()} to {tr.max().date()}")
    ax.axvspan(tr.max(), te.min(), color="grey", alpha=0.5, label="embargo (no rows used)")
    ax.axvspan(te.min(), te.max(), color=PALETTE[1], alpha=0.2, label=f"test: {te.min().date()} to {te.max().date()}")
    ax.set_title("Split by time: the model learns from the past and is tested on the future")
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    return fig


def decile_bar(table: pd.DataFrame, title: str = "Mean 5-day return by forecast decile (test set)") -> plt.Figure:
    """Bars of mean realised target per forecast decile (from evaluate.decile_table), with 95% error bars."""
    fig, ax = plt.subplots()
    y = table["mean_target"] * 100
    err = 1.96 * table["target_se"] * 100 if "target_se" in table else None
    colours = [PALETTE[3] if v < 0 else PALETTE[2] for v in y]
    ax.bar(table.index.astype(str), y, yerr=err, color=colours, capsize=3,
           label="mean, with 95% interval" if err is not None else "mean")
    ax.axhline(0, color="grey", lw=1)
    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("forecast decile (1 = lowest forecasts, 10 = highest)")
    ax.set_ylabel("mean realised 5-day return (%)")
    return fig


def train_vs_test_bars(table: pd.DataFrame, title: str = "Fit on training data vs test data") -> plt.Figure:
    """Grouped bars of train_r2 and test_r2 (r2 vs predicting zero, in %) per row of the table."""
    fig, ax = plt.subplots()
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["train_r2"] * 100, width=0.4, color=PALETTE[0], label="train")
    ax.bar(x + 0.2, table["test_r2"] * 100, width=0.4, color=PALETTE[1], label="test")
    ax.set_xticks(x, [str(i) for i in table.index])
    ax.axhline(0, color="grey", lw=1)
    ax.set_title(title)
    ax.set_xlabel("model")
    ax.set_ylabel("R squared vs predicting zero (%)")
    ax.legend(loc="upper right")
    return fig


def importance_bar(table: pd.DataFrame, title: str = "Permutation importance on the test set") -> plt.Figure:
    """Horizontal bars: rise in test MSE when each feature is shuffled (from permutation_importance_table)."""
    t = table.sort_values("mse_increase")
    fig, ax = plt.subplots()
    scale = 1e4  # MSE of decimal returns is tiny; show it in (return in %) squared
    ax.barh(t["feature"], t["mse_increase"] * scale, xerr=t["mse_increase_std"] * scale, color=PALETTE[4], capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_title(title)
    ax.set_xlabel("rise in mean squared error when shuffled ((% return) squared)")
    ax.set_ylabel("feature")
    return fig


def equity_curve(strategy, benchmark=None, title: str = "Strategy vs equal-weight benchmark, after costs") -> plt.Figure:
    """Growth of 100 for the strategy (a BacktestResult or an equity Series) and, optionally, the benchmark."""
    fig, ax = plt.subplots()
    s = _equity(strategy)
    ax.plot(s.index, s * 100, color=PALETTE[0], lw=1.5, label="strategy (long/short)")
    if benchmark is not None:
        b = _equity(benchmark)
        ax.plot(b.index, b * 100, color=PALETTE[1], lw=1.2, label="benchmark (equal weight, long only)")
    ax.axhline(100, color="grey", lw=1)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("value of 100 invested")
    ax.legend(loc="upper left")
    return fig


def drawdown_area(strategy, benchmark=None, title: str = "Drawdown: % below the previous peak") -> plt.Figure:
    """Shaded drawdown (equity / running peak - 1, in %) for the strategy, and a line for the benchmark."""
    fig, ax = plt.subplots()
    s = _equity(strategy)
    dd = (s / np.maximum(s.cummax(), 1.0) - 1) * 100
    ax.fill_between(dd.index, dd, 0, color=PALETTE[3], alpha=0.4, label="strategy")
    if benchmark is not None:
        b = _equity(benchmark)
        ax.plot(b.index, (b / np.maximum(b.cummax(), 1.0) - 1) * 100, color=PALETTE[1], lw=1, label="benchmark")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown (%)")
    ax.legend(loc="lower left")
    return fig
