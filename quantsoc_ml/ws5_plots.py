"""Charts for the Workshop 5 notebook (Jiang, Kelly & Xiu), kept here so the notebook cells stay short."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ws3_plots import QS_GOLD, QS_GREEN, QS_GREY, QS_NAVY, QS_RED


def show(img: np.ndarray, ax=None, title: str = "", grid: bool = False):
    """One chart picture exactly as the network sees it: black background, white pixels, no smoothing."""
    ax = ax or plt.subplots(figsize=(3, 3 * img.shape[0] / img.shape[1]))[1]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="equal")
    if grid:
        ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#444", lw=0.3)
        ax.tick_params(which="minor", length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return ax


def gallery(images: Sequence[np.ndarray], titles: Sequence[str] = (), ncols: int = 8, size: float = 1.4,
            suptitle: str = ""):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    h, w = images[0].shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size * h / w + 0.35) + 0.5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            show(images[i], ax, titles[i] if i < len(titles) else "")
        else:
            ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def ohlc_chart(close, open_, high, low, ma, volume, dates=None, title: str = "", ax=None):
    """The same days drawn the ordinary way (as on Yahoo Finance), to compare with the network's picture."""
    if ax is None:
        fig, (ax, axv) = plt.subplots(2, 1, figsize=(7, 4), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    else:
        axv = None
    x = np.arange(len(close))
    for i in x:
        c = QS_GREEN if close[i] >= open_[i] else QS_RED
        ax.vlines(i, low[i], high[i], color=c, lw=1.6)
        ax.hlines(open_[i], i - 0.3, i, color=c, lw=1.6)
        ax.hlines(close[i], i, i + 0.3, color=c, lw=1.6)
    ax.plot(x, ma, color=QS_GOLD, lw=1.4, label="moving average")
    ax.set_title(title); ax.legend(fontsize=8, loc="upper left")
    if axv is not None:
        axv.bar(x, volume, color=QS_GREY)
        axv.set_ylabel("volume")
        if dates is not None:
            step = max(1, len(x) // 6)
            axv.set_xticks(x[::step]); axv.set_xticklabels([pd.Timestamp(d).strftime("%d %b") for d in dates[::step]])
    return ax.figure


def matrix(img: np.ndarray, rows: Optional[slice] = None) -> str:
    """The picture as the grid of numbers it really is (1 = white pixel)."""
    sub = img[rows] if rows is not None else img
    return "\n".join("".join("#" if v else "." for v in row) for row in sub)


def conv_demo(image: np.ndarray, filters: Mapping[str, np.ndarray], outputs: Mapping[str, np.ndarray],
              pooled: Mapping[str, np.ndarray]):
    """Figure 5 of the paper: an input, two filters, what each finds, and what pooling keeps."""
    fig, axes = plt.subplots(len(filters), 4, figsize=(11, 2.7 * len(filters)))
    for r, name in enumerate(filters):
        panels = [(image, "input"), (filters[name], f"{name}"), (outputs[name], "convolution output"),
                  (pooled[name], "after 2x2 max pooling")]
        for c, (a, t) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(a, cmap="gray" if c == 0 else "coolwarm", vmin=None if c == 0 else -abs(a).max() - 1e-9,
                      vmax=None if c == 0 else abs(a).max() + 1e-9, interpolation="nearest")
            for (i, j), v in np.ndenumerate(a):
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=7 if a.size > 9 else 9,
                        color="black" if c else ("black" if v > 128 else "white"))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=9)
    fig.tight_layout()
    return fig


def feature_maps(maps: np.ndarray, n: int = 8, title: str = ""):
    """The first n channels of one layer's output for one picture."""
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2.6))
    for k, ax in enumerate(axes):
        ax.imshow(maps[k], cmap="magma", interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"filter {k}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def training_curves(history: pd.DataFrame, title: str = "", up_share: Optional[float] = None):
    """Loss per epoch for each network in the ensemble; the dot marks the epoch that was kept.
    up_share: share of 'up' labels, to draw what always predicting that share would score."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for seed, h in history.groupby("seed"):
        axes[0].plot(h["epoch"], h["train_loss"], color=QS_GREY, lw=1)
        axes[0].plot(h["epoch"], h["val_loss"], color=QS_NAVY, lw=1.2)
        best = h.loc[h["val_loss"].idxmin()]
        axes[0].plot(best["epoch"], best["val_loss"], "o", color=QS_GOLD)
        axes[1].plot(h["epoch"], h["val_acc"] * 100, color=QS_NAVY, lw=1.2)
    axes[0].axhline(np.log(2), color=QS_RED, ls="--", lw=1, label="a coin flip (ln 2)")
    if up_share is not None:
        q = up_share
        axes[0].axhline(-(q * np.log(q) + (1 - q) * np.log(1 - q)), color=QS_RED, ls=":", lw=1.2,
                        label=f"always say {q:.0%} up")
        axes[1].axhline(max(q, 1 - q) * 100, color=QS_RED, ls=":", lw=1.2)
    axes[0].plot([], [], color=QS_GREY, label="training loss"); axes[0].plot([], [], color=QS_NAVY, label="validation loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend(fontsize=8)
    axes[1].axhline(50, color=QS_RED, ls="--", lw=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy (%)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def cam_grid(images: Sequence[np.ndarray], cams: Sequence[Sequence[np.ndarray]], titles: Sequence[str], suptitle: str = ""):
    """Paper Figure 13: each column a picture, then its Grad-CAM heatmap after each building block."""
    n, layers = len(images), len(cams[0])
    h, w = images[0].shape
    fig, axes = plt.subplots(layers + 1, n, figsize=(n * 1.6, (layers + 1) * 1.6 * h / w + 0.5), squeeze=False)
    for j in range(n):
        show(images[j], axes[0, j], titles[j])
        for i in range(layers):
            axes[i + 1, j].imshow(cams[j][i], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            axes[i + 1, j].set_xticks([]); axes[i + 1, j].set_yticks([])
            if j == 0:
                axes[i + 1, j].set_ylabel(f"block {i + 1}", fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def deciles(table: pd.DataFrame, title: str):
    """Average holding-period return of each decile (columns = strategies), annualised."""
    ax = table.plot.bar(figsize=(11, 3.6), color=[QS_NAVY, QS_GOLD, QS_GREY, QS_RED, QS_GREEN][:table.shape[1]], width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("decile of the signal (1 = lowest, 10 = highest)"); ax.set_ylabel("annualised return (%)")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=0)
    return ax.figure
