"""Charts for the Workshop 4 notebook (Avellaneda & Lee), kept here so the notebook cells stay short."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ws3_plots import QS_GOLD, QS_GREEN, QS_GREY, QS_NAVY, QS_RED

SECTOR_COLOURS = {"XLK": "#4c72b0", "XLF": "#dd8452", "XLV": "#55a868", "XLE": "#c44e52", "XLI": "#8172b3",
                  "XLY": "#937860", "XLP": "#da8bc3", "XLU": "#8c8c8c", "XLB": "#ccb974", "XLRE": "#64b5cd",
                  "XLC": "#2f4b7c"}


def simulate_ou(kappa: float, m: float, sigma: float, days: int, x0: float = 0.0, seed: int = 0) -> np.ndarray:
    """Exact daily simulation of dX = kappa (m - X) dt + sigma dW, with dt = 1/252."""
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa / 252)
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa)) if kappa > 0 else sigma / np.sqrt(252)
    x = np.empty(days)
    x[0] = x0
    for i in range(1, days):
        x[i] = m + b * (x[i - 1] - m) + sd * rng.normal()
    return x


def ou_gallery(kappas: Sequence[float] = (0.0, 8.4, 40.0), sigma: float = 0.3, days: int = 252, seed: int = 0):
    """The same random shocks under different speeds of mean reversion (kappa = 0 is a random walk)."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    colours = [QS_GREY, QS_GOLD, QS_NAVY]
    for k, c in zip(kappas, colours):
        if k == 0:
            rng = np.random.default_rng(seed)
            x = np.concatenate([[0.0], np.cumsum(sigma / np.sqrt(252) * rng.normal(size=days - 1))])
            label = "kappa = 0: random walk, never comes back"
        else:
            x = simulate_ou(k, 0.0, sigma, days, seed=seed)
            label = f"kappa = {k:g}: half-life {np.log(2) / k * 252:.0f} days"
        ax.plot(x, color=c, lw=1.4, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("trading day"); ax.set_ylabel("X(t)")
    ax.set_title("Ornstein-Uhlenbeck: the stronger the pull (kappa), the faster X returns to its mean")
    ax.legend(loc="upper left", fontsize=9)
    return fig


def spectrum(vals: np.ndarray, n_days: int, title: str = "Eigenvalues of the correlation matrix"):
    """Eigenvalues as % of variance, with the Marchenko-Pastur edge: the largest eigenvalue pure noise would give."""
    n = len(vals)
    edge = (1 + np.sqrt(n / n_days)) ** 2
    fig, ax = plt.subplots(figsize=(11, 3.6))
    shown = min(30, n)
    colours = [QS_NAVY if v > edge else QS_GREY for v in vals[:shown]]
    ax.bar(range(1, shown + 1), vals[:shown] / n * 100, color=colours)
    ax.axhline(edge / n * 100, color=QS_RED, lw=1.2, ls="--", label=f"largest eigenvalue from pure noise ({edge / n:.1%})")
    ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("% of total variance")
    ax.set_title(title); ax.legend()
    return fig


def eigenvector_bars(v: pd.Series, sectors: pd.Series, title: str):
    """One eigenvector's weights, sorted, coloured by the stock's sector fund."""
    v = v.sort_values()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(range(len(v)), v.values, color=[SECTOR_COLOURS.get(sectors.get(s, ""), QS_GREY) for s in v.index])
    for i in list(range(5)) + list(range(len(v) - 5, len(v))):
        ax.text(i, v.values[i], v.index[i], rotation=90, fontsize=7,
                va="bottom" if v.values[i] > 0 else "top", ha="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SECTOR_COLOURS.values()]
    ax.legend(handles, SECTOR_COLOURS.keys(), ncol=6, fontsize=8, loc="upper left")
    ax.set_xticks([]); ax.set_title(title); ax.axhline(0, color="black", lw=0.6)
    return fig


def sscore_chart(s: pd.Series, cfg, positions: Optional[pd.Series] = None, title: str = ""):
    """An s-score through time with the paper's bands, and the trade the rules would hold."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, color=QS_NAVY, lw=1.1, label="s-score")
    for y, c, lab in [(cfg.s_open, QS_RED, "open short above +1.25"), (-cfg.s_open, QS_GREEN, "open long below -1.25"),
                      (cfg.s_close_short, QS_RED, "close short below +0.75"),
                      (-cfg.s_close_long, QS_GREEN, "close long above -0.50")]:
        ax.axhline(y, color=c, lw=1, ls="-" if abs(y) == cfg.s_open else ":", label=lab)
    if positions is not None:
        p = positions.reindex(s.index).fillna(0)
        ax.fill_between(s.index, -4, 4, where=p > 0, color=QS_GREEN, alpha=0.12, step="post", label="holding long")
        ax.fill_between(s.index, -4, 4, where=p < 0, color=QS_RED, alpha=0.12, step="post", label="holding short")
    ax.set_ylim(-3.5, 3.5); ax.set_title(title); ax.legend(ncol=3, fontsize=8, loc="lower left")
    return fig
