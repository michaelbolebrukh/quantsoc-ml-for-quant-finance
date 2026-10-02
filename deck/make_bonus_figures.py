"""Draw the round 5 deck figures: two feature figures on real data and the bonus learning-theory figures.

Usage:  python3 deck/make_bonus_figures.py            (all figures, about 30 seconds on 4 cores)
        python3 deck/make_bonus_figures.py --only B3  (one figure; keys F5 F6 B1 ... B9)

Writes deck/figures/<KEY>_*.png (200 dpi, about 16:10, dark canvas) and docs/bonus-results.md, the table of
every number these figures measure. With --only, the rows of the other figures already in that file are kept.

Real-data figures (F5, F6, B3) load data/prices.csv.gz through quantsoc_ml exactly as the notebook does
(build_features, rank_normalise, make_dataset, time_split with train_end 2014-12-31 and test_start 2015-01-12).
Synthetic figures (B1, B2, B4, B6, B8) say "synthetic" in their subtitle; B5, B7 and B9 are diagrams.
Every random draw uses a fixed seed, so a rerun gives the same pictures and numbers.
Nothing is tuned to look like the textbook: B3 draws whatever the depth sweep measures.
The look copies quantsoc_ml.notebook_support.make_deck_figures(dark=True): same tokens, rcParams and Inter.
Exit code 0 on success; any exception propagates (nonzero exit).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
FONT_DIR = HERE / "assets" / "fonts"
RESULTS_MD = ROOT / "docs" / "bonus-results.md"
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

for _ttf in FONT_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(_ttf))

# QuantSoc dark tokens (next-pwa/app/styles/tokens.css, .dark); contrast against #0a0a0a in brackets.
BG, FG, MUTED, RULE, BRAND = "#0a0a0a", "#dae0e7", "#acb6c3", "#717a88", "#61a6fa"   # 14.9, 9.7, 4.6, 7.9
WARN = "#f26464"     # --destructive (6.4)
AMBER = "#fccb4f"    # --warning (13.0)
GREEN = "#12d393"    # --success (10.2)
HAIR = "#3a3f49"
STYLE = {"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
         "axes.edgecolor": HAIR, "axes.labelcolor": MUTED, "text.color": FG, "xtick.color": MUTED,
         "ytick.color": MUTED, "grid.color": FG, "grid.alpha": 0.12, "legend.facecolor": "#141515",
         "legend.edgecolor": HAIR, "legend.framealpha": 1.0, "font.size": 13, "axes.titlesize": 14,
         "axes.titleweight": "semibold", "axes.titlecolor": FG, "font.family": ["Inter", "DejaVu Sans"],
         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.grid": True,
         "axes.axisbelow": True, "axes.unicode_minus": True}
SIZE = (12, 7.5)       # 16:10
WIDE = (12, 5.45)      # about 2.2:1, for the full-width slots (F5, B5, B9)
# Figures shown full width on a slide get projector-sized type: ticks 15pt, axis titles 18pt, main title 20pt.
BIG = {"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 15,
       "ytick.labelsize": 15, "legend.fontsize": 15}
BIG_KEYS = {"F5", "B3", "B4", "B5", "B9"}
DPI = 200
TRAIN_END, TEST_START = "2014-12-31", "2015-01-12"

RESULTS: dict = {}     # key -> (value, how it was computed)


def rec(key, value, how):
    RESULTS[key] = (value, how)
    return value


def titles(fig, title, subtitle=None, y=0.985, size=None, sub=None):
    big = plt.rcParams["font.size"] >= 16
    size, sub = size or (20 if big else 16), sub or (15 if big else 13)
    fig.suptitle(title, x=0.5, y=y, fontsize=size, fontweight="semibold", color=FG)
    if subtitle:
        gap = size * 1.5 / 72 / fig.get_figheight()      # one title line, in figure fractions
        fig.text(0.5, y - gap, subtitle, ha="center", va="top", fontsize=sub, color=MUTED)


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# ------------------------------------------------------------------ real data, loaded once
_DATA = {}


def data():
    if not _DATA:
        from quantsoc_ml import data as D, features as F, splits as SP
        long = D.load_prices()
        prices, volume = D.to_wide(long, "adj_close"), D.to_wide(long, "volume")
        raw = F.build_features(prices, volume)
        ranked = F.rank_normalise(raw)
        ds = F.make_dataset(ranked, F.forward_return(prices))
        train, test = SP.time_split(ds, TRAIN_END, TEST_START)
        _DATA.update(prices=prices, volume=volume, raw=raw, ranked=ranked, train=train, test=test)
    return _DATA


# ------------------------------------------------------------------ F5: the eight features, one stock
def fig_F5():
    d = data()
    sym = "JPM"
    f = d["raw"].xs(sym, level="symbol").loc["2008-01-01":"2010-12-31"]
    panels = [("ret_1", "1-day return", 100), ("ret_5", "5-day return", 100), ("ret_21", "21-day return", 100),
              ("ret_63", "63-day return", 100), ("vol_21", "21-day volatility", 100),
              ("vol_63", "63-day volatility", 100), ("volume_z", "volume z-score", 1),
              ("hi52", "below 52-week high", 100)]
    fig, axes = plt.subplots(2, 4, figsize=WIDE, sharex=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.77, bottom=0.08, wspace=0.30, hspace=0.42)
    for ax, (col, label, scale) in zip(axes.ravel(), panels):
        v = f[col] * scale
        ax.plot(v.index, v, color=BRAND, lw=1.1)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.set_title(label, fontsize=18)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("'%y"))
        rec(f"F5.{col}_min", round(float(f[col].min()), 4), f"min of raw {col} for {sym}, 2008-01-01 to 2010-12-31")
        rec(f"F5.{col}_max", round(float(f[col].max()), 4), f"max of raw {col} for {sym}, 2008-01-01 to 2010-12-31")
    rec("F5.symbol", sym, "the stock shown")
    rec("F5.n_days", int(len(f)), "trading days shown")
    titles(fig, "The eight workshop features for one stock: JPMorgan Chase (JPM), 2008 to 2010",
           "real data, raw values before ranking; all in % except volume z-score (standard deviations)", y=0.99)
    save(fig, "F5_features_one_stock.png")


# ------------------------------------------------------------------ F6: ranking keeps order, drops scale
def fig_F6():
    d = data()
    date = "2008-10-10"
    raw = d["raw"].xs(date, level="date")["ret_21"].dropna()
    rk = d["ranked"].xs(date, level="date")["ret_21"].reindex(raw.index)
    order = raw.sort_values().index
    raw, rk = raw[order] * 100, rk[order]
    x = np.arange(len(raw))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.12, wspace=0.25)
    a1.bar(x, raw.to_numpy(), color=BRAND, width=0.8)
    a1.set_title("raw 21-day return (%)")
    a1.set_ylabel("21-day return (%)")
    a2.bar(x, rk.to_numpy(), color=BRAND, width=0.8)
    a2.set_title("the same, rank-normalised on that date")
    a2.set_ylabel("rank value (lowest -0.5, highest +0.5)")
    a2.set_ylim(-0.6, 0.6)
    for ax, vals in ((a1, raw), (a2, rk)):
        ax.axhline(0, color=RULE, lw=0.9)
        ax.set_xticks([])
        ax.set_xlabel(f"the {len(raw)} stocks, sorted by raw 21-day return", fontsize=12)
        for i in (0, len(vals) - 1):
            v = vals.iloc[i]
            ax.annotate(vals.index[i], (i, v), xytext=(0, -16 if v < 0 else 6), textcoords="offset points",
                        ha="center", fontsize=11, color=FG)
    rec("F6.date", date, "the one date shown (the week of the October 2008 crash, picked to make the scale visible)")
    rec("F6.n_stocks", int(len(raw)), "stocks with a ret_21 value on that date")
    rec("F6.raw_mean_pct", round(float(raw.mean()), 2), "mean raw 21-day return across the stocks, %")
    rec("F6.raw_min_pct", round(float(raw.min()), 2), f"lowest raw 21-day return, % ({raw.index[0]})")
    rec("F6.raw_max_pct", round(float(raw.max()), 2), f"highest raw 21-day return, % ({raw.index[-1]})")
    rec("F6.n_negative_raw", int((raw < 0).sum()), "stocks with a negative raw 21-day return that day")
    rec("F6.rank_min", round(float(rk.min()), 4), "lowest rank value")
    rec("F6.rank_max", round(float(rk.max()), 4), "highest rank value")
    rec("F6.rank_mean", round(float(rk.mean()), 6), "mean rank value (0 by construction)")
    rec("F6.order_kept", bool((np.diff(rk.to_numpy()) >= 0).all()), "rank order equals raw order")
    titles(fig, f"Ranking keeps the order and removes the scale: 21-day return on {pd_date(date)}",
           f"real data; {int((raw < 0).sum())} of {len(raw)} stocks fell that month, "
           "so the raw values mostly say 'the market fell'; ranks say who fell least")
    save(fig, "F6_ranking.png")


def pd_date(s):
    import pandas as pd
    return pd.Timestamp(s).strftime("%-d %B %Y")


# ------------------------------------------------------------------ B1: under, good, over fit
def _sine(x):
    return np.sin(2 * np.pi * x)


def fig_B1():
    from numpy.polynomial import Polynomial
    rng = np.random.default_rng(1)
    n, sd = 30, 0.3
    x = np.sort(rng.uniform(0, 1, n))
    y = _sine(x) + rng.normal(0, sd, n)
    xt = rng.uniform(0, 1, 2000)
    yt = _sine(xt) + rng.normal(0, sd, 2000)
    grid = np.linspace(0, 1, 600)
    fig, axes = plt.subplots(1, 3, figsize=SIZE, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.12, wspace=0.08)
    names = {1: "underfit: degree 1", 4: "good fit: degree 4", 15: "overfit: degree 15"}
    for ax, deg in zip(axes, (1, 4, 15)):
        p = Polynomial.fit(x, y, deg)
        tr, te = float(np.mean((p(x) - y) ** 2)), float(np.mean((p(xt) - yt) ** 2))
        rec(f"B1.train_mse_degree_{deg}", round(tr, 4), f"mean squared error on the {n} training points")
        rec(f"B1.test_mse_degree_{deg}", round(te, 4), "mean squared error on 2,000 fresh points from the same process")
        ax.plot(grid, _sine(grid), color=MUTED, lw=1.4, ls="--", label="true curve")
        ax.plot(grid, np.clip(p(grid), -3, 3), color=BRAND, lw=2.2, label="fitted polynomial")
        ax.scatter(x, y, s=28, color=FG, zorder=3, label="30 training points")
        ax.set_ylim(-2, 2)
        ax.set_title(names[deg])
        ax.set_xlabel("x")
        ax.text(0.04, 0.05, f"train MSE {tr:,.3f}\ntest MSE {te:,.3f}", transform=ax.transAxes, fontsize=17,
                color=FG, va="bottom", bbox=dict(boxstyle="round,pad=0.35", fc="#141515", ec=HAIR))
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=14)
    rec("B1.noise_mse", sd ** 2, "irreducible error: the noise variance (sd 0.3)")
    titles(fig, "Underfit, good fit, overfit: the same 30 points, three polynomial degrees",
           "synthetic: y = sin(2πx) + noise (sd 0.3); test MSE measured on 2,000 fresh points; noise alone gives 0.09")
    save(fig, "B1_fit_three.png")


# ------------------------------------------------------------------ B2: dartboards
def fig_B2():
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(2, 2, figsize=SIZE)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.78, bottom=0.02, wspace=0.0, hspace=0.12)
    cases = {(0, 0): (0.0, 0.10), (0, 1): (0.0, 0.33), (1, 0): (0.5, 0.10), (1, 1): (0.5, 0.33)}
    for (r, c), (bias, spread) in cases.items():
        ax = axes[r, c]
        ax.set_aspect("equal")
        ax.grid(False)
        for rad, col in ((1.0, "#141a26"), (0.75, "#18243a"), (0.5, "#1c2e4c"), (0.25, "#22396b")):
            ax.add_patch(Circle((0, 0), rad, fc=col, ec=RULE, lw=1))
        ax.add_patch(Circle((0, 0), 0.06, fc=AMBER, ec="none"))
        pts = rng.normal(0, spread, (14, 2)) + np.array([bias * 0.71, bias * 0.71])
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts[rad > 1.05] *= (1.05 / rad[rad > 1.05])[:, None]   # keep every dot on the board
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=FG, ec=BG, lw=0.8, zorder=3)
        m = pts.mean(0)
        ax.scatter([m[0]], [m[1]], marker="X", s=170, color=WARN, ec=BG, lw=1.0, zorder=4)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for c, lab in enumerate(("low variance", "high variance")):
        axes[0, c].set_title(lab, fontsize=15, pad=6)
    for r, lab in enumerate(("low bias", "high bias")):
        axes[r, 0].text(-1.32, 0, lab, rotation=90, ha="center", va="center", fontsize=15, color=FG,
                        fontweight="semibold")
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=8, color=FG, label="one model, fitted to one redrawn training set"),
               plt.Line2D([], [], ls="none", marker="X", ms=11, color=WARN, label="the average model"),
               plt.Line2D([], [], ls="none", marker="o", ms=10, color=AMBER, label="the truth")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3, fontsize=16, frameon=False)
    titles(fig, "Bias is how far the average shot lands from the centre; variance is how scattered the shots are",
           "illustration with synthetic shots", y=0.975)
    save(fig, "B2_dartboard.png")


# ------------------------------------------------------------------ B3: one tree, depth sweep, our data
def fig_B3():
    from sklearn.tree import DecisionTreeRegressor
    from quantsoc_ml import evaluate as E, models as M
    d = data()
    train, test = d["train"], d["test"]
    Xtr, ytr = M.xy(train)
    Xte, yte = M.xy(test)
    depths = list(range(1, 21))
    tr_r2, te_r2 = [], []
    for k in depths:
        m = DecisionTreeRegressor(max_depth=k, random_state=0).fit(Xtr, ytr)
        tr_r2.append(E.score(ytr, m.predict(Xtr))["r2_vs_zero"] * 100)
        te_r2.append(E.score(yte, m.predict(Xte))["r2_vs_zero"] * 100)
    const = E.score(yte, np.full(len(yte), ytr.mean()))["r2_vs_zero"] * 100
    best = int(depths[int(np.argmax(te_r2))])
    rec("B3.train_rows", int(len(train)), "all training rows (2000-12-29 to 2014-12-31), rank features, no sampling")
    rec("B3.test_rows", int(len(test)), "test rows from 2015-01-12")
    rec("B3.best_test_depth", best, "depth with the highest test R squared vs predicting zero")
    rec("B3.best_test_r2_pct", round(max(te_r2), 3), "that highest test R squared, %")
    for k in (1, 2, 3, 4, 6, 8, 10, 15, 20):
        rec(f"B3.test_r2_at_depth_{k}", round(te_r2[k - 1], 3), "test R squared vs predicting zero, %")
        rec(f"B3.train_r2_at_depth_{k}", round(tr_r2[k - 1], 3), "train R squared vs predicting zero, %")
    rec("B3.const_mean_test_r2_pct", round(const, 3), "constant forecast = training mean target, test R squared, %")
    rec("B3.n_depths_test_above_zero", int(sum(v > 0 for v in te_r2)), "depths (of 20) whose test R squared > 0")
    rec("B3.n_depths_test_above_const", int(sum(v > const for v in te_r2)),
        "depths (of 20) whose test R squared beats the constant training-mean forecast")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.77, bottom=0.12, wspace=0.30)
    a1.plot(depths, tr_r2, color=BRAND, lw=2.2, marker="o", ms=4, label="training data")
    a1.plot(depths, te_r2, color=AMBER, lw=2.2, marker="o", ms=4, label="test, 2015 to 2019")
    a1.axhline(0, color=RULE, lw=1)
    a1.set_title("depths 1 to 20")
    a1.set_xlabel("max_depth of the tree")
    a1.set_ylabel("R² vs predicting zero (%)")
    a1.legend(loc="upper left", fontsize=15)
    a2.plot(depths, te_r2, color=AMBER, lw=2.2, marker="o", ms=5, label="test, 2015 to 2019")
    a2.axhline(0, color=RULE, lw=1, label="predicting zero")
    a2.axhline(const, color=MUTED, lw=1.2, ls="--", label=f"training mean ({const:+.2f}%)")
    lo = min(te_r2[:8])
    a2.set_ylim(min(lo * 1.15, -0.5), max(max(te_r2), const) + 0.4)
    a2.set_xlim(0.5, 8.5)
    a2.set_title("test only, depths 1 to 8 (zoomed)")
    a2.set_xlabel("max_depth of the tree")
    a2.set_ylabel("R² vs predicting zero (%)")
    a2.legend(loc="lower left", fontsize=15)
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    falling = bool(all(np.diff(te_r2) < 0))
    rec("B3.test_falls_every_step", falling, "test R squared lower at every depth than at the depth before")
    if best == 1 and falling:
        head = "On our data the test score is best at depth 1 and falls with every extra level"
    elif best <= 3:
        head = f"On our data the test score peaks at depth {best}, then falls: deeper trees fit noise"
    else:
        head = f"On our data the test score peaks at depth {best}, then falls away"
    beats = sum(v > const for v in te_r2)
    titles(fig, head, "real data, rank features, one tree per depth, train to 2014, test 2015 to 2019;\n"
                      + ("no depth beats" if beats == 0 else f"{beats} depths beat")
                      + f" the constant training-mean forecast ({const:+.2f}%)")
    save(fig, "B3_depth_ours.png")
    return head


# ------------------------------------------------------------------ B4: the computed U-curve
def fig_B4():
    from numpy.polynomial import Polynomial
    rng = np.random.default_rng(4)
    n, sd, reps = 30, 0.3, 2000
    degrees = list(range(1, 13))
    x = np.linspace(0, 1, n)                      # fixed design: the x values stay, the noise is redrawn
    x0 = np.linspace(0.0, 1.0, 200)
    f0 = _sine(x0)
    Y = _sine(x)[None, :] + rng.normal(0, sd, (reps, n))
    bias2, var, total_meas = [], [], []
    for deg in degrees:
        preds = np.array([Polynomial.fit(x, y, deg, domain=[0, 1])(x0) for y in Y])
        mean = preds.mean(0)
        bias2.append(float(np.mean((mean - f0) ** 2)))
        var.append(float(np.mean(preds.var(0))))
        ynew = f0 + rng.normal(0, sd, preds.shape)
        total_meas.append(float(np.mean((preds - ynew) ** 2)))
    noise = sd ** 2
    total = [b + v + noise for b, v in zip(bias2, var)]
    best = degrees[int(np.argmin(total))]
    rec("B4.n_train", n, "points per training set at fixed, evenly spaced x on [0, 1]; y = sin(2 pi x) + N(0, 0.3^2)")
    rec("B4.n_training_sets", reps, "training sets per degree (the noise is redrawn each time)")
    rec("B4.best_degree", best, "degree with the lowest total expected test error")
    rec("B4.noise", noise, "irreducible error, sd^2")
    rec("B4.variance_rises_every_step", bool(all(np.diff(var) > 0)), "variance higher at every degree than the one before")
    for deg in sorted({1, 3, best, 8, 12}):
        rec(f"B4.bias2_degree_{deg}", float(f"{bias2[deg - 1]:.3g}"), "squared bias averaged over 200 test x in [0, 1]")
        rec(f"B4.variance_degree_{deg}", round(var[deg - 1], 5), "variance of the fit across training sets, averaged over x")
        rec(f"B4.total_degree_{deg}", round(total[deg - 1], 5), "bias^2 + variance + noise")
        rec(f"B4.measured_test_mse_degree_{deg}", round(total_meas[deg - 1], 5),
            "check: mean squared error against fresh noisy y at the same x (should match total)")
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.12)
    ax.plot(degrees, total, color=FG, lw=2.8, marker="o", ms=5, label="total expected test error")
    ax.plot(degrees, bias2, color=BRAND, lw=2.2, marker="o", ms=4, label="bias²")
    ax.plot(degrees, var, color=AMBER, lw=2.2, marker="o", ms=4, label="variance")
    ax.axhline(noise, color=MUTED, lw=1.5, ls="--", label=f"noise (irreducible) = {noise:.2f}")
    ax.axvline(best, color=RULE, lw=1, ls=":")
    ax.annotate(f"lowest total at degree {best}", (best, total[best - 1]), xytext=(14, 34),
                textcoords="offset points", fontsize=16, color=FG, arrowprops=dict(arrowstyle="-", color=RULE))
    ax.set_yscale("log")
    ax.set_xticks(degrees)
    ax.set_xlabel("model complexity: polynomial degree")
    ax.set_ylabel("error (log scale)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=16)
    titles(fig, "U-curve, computed: bias² drops then flattens, variance rises, the total bottoms out",
           f"synthetic, by simulation: {reps:,} redrawn training sets of {n} points per degree, "
           "y = sin(2πx) + noise (sd 0.3)")
    save(fig, "B4_ucurve_textbook.png")


# ------------------------------------------------------------------ B5: cross-validation schemes
def fig_B5():
    rng = np.random.default_rng(5)
    N, K = 48, 4
    purge, embargo = 2, 2
    C = {"train": BRAND, "test": AMBER, "purged": WARN, "embargo": RULE, "unused": None}
    fig, axes = plt.subplots(3, 1, figsize=WIDE)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.115, hspace=0.62)

    def draw(ax, rows, title):
        ax.grid(False)
        for r, row in enumerate(rows):
            for t, kind in enumerate(row):
                y = len(rows) - 1 - r
                if C[kind] is None:
                    ax.add_patch(Rectangle((t + 0.06, y + 0.1), 0.88, 0.8, fc="none", ec=HAIR, lw=0.8))
                else:
                    ax.add_patch(Rectangle((t + 0.06, y + 0.1), 0.88, 0.8, fc=C[kind], ec="none"))
        ax.set_xlim(0, N)
        ax.set_ylim(0, len(rows))
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels([f"fold {len(rows) - i}" for i in range(len(rows))], fontsize=14)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, loc="left", fontsize=17, pad=5)

    perm = rng.permutation(N)
    rows = [["test" if t in set(perm[k::K]) else "train" for t in range(N)] for k in range(K)]
    draw(axes[0], rows, "shuffled k-fold: test days scattered among training days")
    rows = []
    block = N // (K + 1)
    for k in range(K):
        end = block * (k + 1)
        rows.append(["train" if t < end else "test" if t < end + block else "unused" for t in range(N)])
    draw(axes[1], rows, "walk-forward: train on the past, test on the next block")
    rows = []
    b = N // K
    for k in range(K):
        lo, hi = k * b, (k + 1) * b
        rows.append(["test" if lo <= t < hi else "purged" if lo - purge <= t < lo
                     else "embargo" if hi <= t < hi + embargo else "train" for t in range(N)])
    draw(axes[2], rows, "purged k-fold with embargo: drop training labels that overlap the test block")
    handles = [Patch(fc=BRAND, label="train"), Patch(fc=AMBER, label="test"),
               Patch(fc=WARN, label="purged (overlaps test labels)"),
               Patch(fc=RULE, label="embargo (just after test)"),
               Patch(fc="none", ec=RULE, label="not used")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=1.4)
    fig.text(0.99, 0.085, "time \u2192", ha="right", va="center", fontsize=15, color=MUTED)
    titles(fig, "Three ways to cross-validate a time series",
           "diagram; one cell is one period; purging and embargo after L\u00f3pez de Prado (2018), AFML ch. 7",
           y=0.99)
    save(fig, "B5_cv_schemes.png")


# ------------------------------------------------------------------ B6: ridge and lasso paths
def fig_B6():
    from sklearn.linear_model import Ridge, lasso_path
    rng = np.random.default_rng(6)
    n, p = 120, 20
    cov = 0.3 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X = rng.multivariate_normal(np.zeros(p), cov, n)
    X = (X - X.mean(0)) / X.std(0)
    beta = np.zeros(p)
    true_idx = [2, 9, 15]
    beta[true_idx] = [3.0, -2.0, 1.5]
    y = X @ beta + rng.normal(0, 2.0, n)
    y = y - y.mean()
    alphas_r = np.logspace(-2, 4.5, 80)
    ridge = np.array([Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_ for a in alphas_r])
    alphas_l, lasso, _ = lasso_path(X, y, alphas=np.logspace(-3, 0.7, 80))
    lasso = lasso.T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.21, wspace=0.08)
    sig_col = {2: BRAND, 9: AMBER, 15: GREEN}
    for ax, al, path, name in ((a1, alphas_r, ridge, "ridge (L2 penalty)"), (a2, alphas_l, lasso, "lasso (L1 penalty)")):
        for j in range(p):
            if j in sig_col:
                ax.plot(al, path[:, j], color=sig_col[j], lw=2.6, zorder=3,
                        label=f"true signal, coefficient {beta[j]:+.1f}")
            else:
                ax.plot(al, path[:, j], color=RULE, lw=1.0, label="the 17 noise features" if j == 0 else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("penalty strength (log scale, stronger to the right)")
    a1.set_ylabel("fitted coefficient")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.53, 0.0))
    # measured facts
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    rec("B6.n", n, "rows")
    rec("B6.p", p, "features: 3 true signals (+3.0, -2.0, +1.5) among 20, AR(1) correlation 0.3, noise sd 2.0")
    rec("B6.ridge_exact_zeros_any_alpha", int((np.abs(ridge) == 0).sum()), "ridge coefficients exactly 0, any penalty")
    zeros = (lasso == 0).sum(1)
    nz = p - zeros
    rec("B6.lasso_max_exact_zeros", int(zeros.max()), "most lasso coefficients exactly 0 at one penalty on the path")
    three = [i for i in range(len(alphas_l)) if nz[i] == 3]
    if three:
        i = three[0]
        kept = sorted(np.flatnonzero(lasso[i]).tolist())
        rec("B6.lasso_alpha_3_nonzero", round(float(alphas_l[i]), 4),
            "largest penalty on the path with exactly 3 nonzero lasso coefficients")
        rec("B6.lasso_3_kept_are_true", kept == true_idx, f"those 3 are the true signals (kept {kept})")
    mid = int(np.argmin(np.abs(alphas_l - 0.3)))
    rec("B6.lasso_nonzero_at_alpha_0.3", int(nz[mid]), f"nonzero lasso coefficients at penalty {alphas_l[mid]:.3f}")
    rec("B6.ols_max_abs_noise_coef", round(float(np.abs(np.delete(ols, true_idx)).max()), 3),
        "largest |coefficient| on a noise feature with no penalty (least squares)")
    kept_txt = (f"at penalty {RESULTS['B6.lasso_alpha_3_nonzero'][0]:.2f} the lasso keeps only the 3 true ones"
                if RESULTS.get("B6.lasso_3_kept_are_true", (False,))[0] else "the lasso drops features one by one")
    titles(fig, "Regularisation shrinks coefficients; the lasso sets most of them to exactly zero",
           f"synthetic, measured: {n} rows, 20 features, 3 true signals; {kept_txt}; ridge zeroes none")
    save(fig, "B6_regularisation.png")


# ------------------------------------------------------------------ B7: flexibility vs interpretability
def fig_B7():
    # In the source figure (ISLR 2nd ed., fig. 2.7): solid markers, placed as there.
    src = [("subset selection", 0.10, 0.93), ("lasso", 0.10, 0.86), ("least squares", 0.30, 0.72),
           ("generalised additive models", 0.50, 0.53), ("trees", 0.50, 0.46),
           ("bagging, boosting", 0.74, 0.29), ("support vector machines", 0.80, 0.19),
           ("deep learning", 0.93, 0.07)]
    # Not in the source figure: hollow markers, placed by us.
    ours = [("ridge", 0.21, 0.69), ("k-nearest neighbours", 0.60, 0.38), ("random forests", 0.66, 0.23)]
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.83, bottom=0.12)
    ax.grid(False)
    for group, hollow in ((src, False), (ours, True)):
        for name, fx, iy in group:
            ax.scatter(fx, iy, s=100, zorder=3, facecolor="none" if hollow else BRAND,
                       edgecolor=AMBER if hollow else BRAND, linewidth=2.0)
            ha = "right" if (fx > 0.85 or name == "ridge") else "left"
            dx = 0.02 if ha == "left" else -0.02
            ax.text(fx + dx, iy, name, fontsize=14, color=MUTED if hollow else FG, va="center", ha=ha)
    ax.scatter([], [], s=100, color=BRAND, label="in the source figure")
    ax.scatter([], [], s=100, facecolor="none", edgecolor=AMBER, linewidth=2.0,
               label="placed by us, not in the source figure")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0.03, 1.0])
    ax.set_xticklabels(["low", "high"])
    ax.set_yticks([0.03, 0.98])
    ax.set_yticklabels(["low", "high"])
    ax.set_xlabel("flexibility")
    ax.set_ylabel("interpretability")
    titles(fig, "The more flexible the model, the harder it is to read",
           "qualitative, after James et al. (2021) fig. 2.7; hollow markers added by us")
    save(fig, "B7_flex_interp.png")


# ------------------------------------------------------------------ B8: double descent
def fig_B8():
    rng = np.random.default_rng(8)
    n, d, sd, reps = 100, 5, 0.3, 20
    ps = sorted(set(list(range(10, 80, 10)) + list(range(80, 121, 4)) + [130, 150, 175, 200, 250, 300, 400,
                                                                         500, 700, 1000, 1400, 2000]))
    beta = rng.normal(0, 1, d) / np.sqrt(d)

    def f(X):
        return np.tanh(2 * X @ beta) + 0.5 * np.sin(2 * X[:, 0])

    errs = np.empty((reps, len(ps)))
    for r in range(reps):
        X = rng.normal(0, 1, (n, d))
        y = f(X) + rng.normal(0, sd, n)
        Xt = rng.normal(0, 1, (2000, d))
        W = rng.normal(0, 1, (d, max(ps)))
        W /= np.linalg.norm(W, axis=0)
        A, T = np.maximum(0, X @ W), np.maximum(0, Xt @ W)     # random ReLU features
        ft = f(Xt)
        for j, p in enumerate(ps):
            coef = np.linalg.lstsq(A[:, :p], y, rcond=None)[0]   # minimum-norm least squares when p > n
            errs[r, j] = np.mean((T[:, :p] @ coef - ft) ** 2) + sd ** 2
    med = np.median(errs, 0)
    peak_j = int(np.argmax(med))
    under = [j for j, p in enumerate(ps) if p < n]
    over = [j for j, p in enumerate(ps) if p > n]
    best_under = min(under, key=lambda j: med[j])
    best_over = min(over, key=lambda j: med[j])
    rec("B8.n_train", n, "training points, x ~ N(0, I) in 5 dimensions, y = tanh(2 x.b) + 0.5 sin(2 x1) + N(0, 0.3^2)")
    rec("B8.repetitions", reps, "independent redraws of data and random ReLU features; median plotted")
    rec("B8.peak_p", int(ps[peak_j]), "number of features with the highest median expected test MSE")
    rec("B8.peak_test_mse", round(float(med[peak_j]), 3), "that median expected test MSE (error vs truth + noise variance)")
    rec("B8.best_under_p", int(ps[best_under]), "best p below the threshold p = n")
    rec("B8.best_under_test_mse", round(float(med[best_under]), 4), "its median expected test MSE")
    rec("B8.best_over_p", int(ps[best_over]), "best p above the threshold")
    rec("B8.best_over_test_mse", round(float(med[best_over]), 4), "its median expected test MSE")
    rec("B8.test_mse_p_2000", round(float(med[-1]), 4), "median expected test MSE with 2,000 features")
    rec("B8.over_beats_under", bool(med[best_over] < med[best_under]),
        "does the best over-parameterised fit beat the best classical one here")
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.11)
    ax.plot(ps, med, color=BRAND, lw=2.4, marker="o", ms=3.5, label=f"test error, median of {reps} runs")
    ax.axvline(n, color=AMBER, lw=1.6, ls="--", label=f"interpolation threshold p = n = {n}")
    ax.axhline(sd ** 2, color=MUTED, lw=1.2, ls=":", label="noise level (0.09)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("number of random features p (log scale)")
    ax.set_ylabel("expected test mean squared error (log scale)")
    ax.legend(loc="upper left", fontsize=16)
    ax.annotate(f"peak at p = {ps[peak_j]}", (ps[peak_j], med[peak_j]), xytext=(18, -6), textcoords="offset points",
                fontsize=16, color=FG)
    ax.text(ps[best_under], med[best_under] * 0.78, "classical regime", fontsize=17, color=FG,
            ha="center", va="top")
    tail = ("and here beats the best small model" if med[best_over] < med[best_under]
            else "but here stays worse than the best small model")
    ax.text(ps[-1], med[-1] * 3.2, f"past the threshold, the minimum-norm\nfit improves again, {tail.split(' ', 1)[0]}\n"
            f"{tail.split(' ', 1)[1]}", fontsize=17, color=FG, ha="right", va="bottom")
    ax.set_ylim(sd ** 2 * 0.8, med[peak_j] * 2.5)
    titles(fig, "Double descent: test error peaks where the model can just fit every training point",
           f"synthetic, measured: minimum-norm least squares on random ReLU features, n = {n}, after Belkin et al. (2019)")
    save(fig, "B8_double_descent.png")


# ------------------------------------------------------------------ B9: overlapping labels
def uniqueness(T, h):
    """AFML ch. 4 average uniqueness for T daily labels each spanning the next h days of a plain calendar."""
    conc = np.zeros(T + h + 1)
    for t in range(T):
        conc[t + 1:t + 1 + h] += 1
    u = np.array([np.mean(1.0 / conc[t + 1:t + 1 + h]) for t in range(T)])
    return u, conc


def fig_B9():
    h = 5
    u, _ = uniqueness(1000, h)
    rec("B9.avg_uniqueness_5d", round(float(u.mean()), 4),
        "mean over 1,000 daily labels of mean(1 / concurrency) across each label's 5 days (AFML ch. 4)")
    rec("B9.effective_n_1000", round(float(u.sum()), 1), "sum of uniqueness: about how many independent labels")
    rec("B9.interior_uniqueness", round(float(u[500]), 4), "uniqueness of a label away from the ends (1/5)")
    rec("B9.shared_days", h - 1, "days two neighbouring 5-day labels share")
    T = 8
    last = T + h                                   # label i (made on day i) measures days i+1 .. i+h
    days = np.arange(1, last + 1)
    conc = np.array([sum(i + 1 <= dd <= i + h for i in range(1, T + 1)) for dd in days])
    u4 = float(np.mean([1.0 / conc[dd - 1] for dd in range(5, 5 + h)]))   # label made on day 4
    fig, (a1, a2) = plt.subplots(2, 1, figsize=WIDE, gridspec_kw={"height_ratios": [2.4, 1]}, sharex=True)
    fig.subplots_adjust(left=0.145, right=0.99, top=0.80, bottom=0.13, hspace=0.12)
    a1.grid(False)
    for i in range(1, T + 1):
        y = T - i
        a1.add_patch(Rectangle((i + 1, y + 0.14), h, 0.72, fc=AMBER if i in (4, 5) else BRAND, ec=BG, lw=1))
        a1.plot([i + 0.5], [y + 0.5], marker="o", color=FG, ms=7)
    a1.set_ylim(0, T)
    a1.set_yticks(np.arange(T) + 0.5)
    a1.set_yticklabels([f"made on day {T - k}" for k in range(T)], fontsize=14)
    a1.tick_params(axis="y", length=0)
    a1.text(last + 1.0, T + 0.1,
            "dot = day a label is made;\nbar = the 5 days it measures\n\nuniqueness of one label =\nthe average, over its 5 days, of\n1 / (labels covering that day)\n\n"
            f"label made on day 4: {u4:.2f}\n(the amber pair share 4 of 5 days)",
            fontsize=15, color=FG, va="top")
    a2.bar(days + 0.5, conc, width=0.85, color=MUTED)
    a2.set_ylabel("labels on\nthat day", fontsize=15)
    a2.set_yticks([0, 5])
    a2.set_xlabel("trading day", fontsize=16)
    a2.set_xlim(0.8, last + 9.5)
    a2.set_xticks(days + 0.5)
    a2.set_xticklabels([str(dd) for dd in days], fontsize=15)
    for s_ in ("top", "right"):
        a1.spines[s_].set_visible(False)
    rec("B9.example_label_day4_uniqueness", round(u4, 4), "the drawn label made on day 4, in the 8-label picture")
    titles(fig, f"1,000 overlapping 5-day labels hold about {u.sum():.0f} labels' worth of information",
           f"diagram; average uniqueness of 1,000 daily labels on a plain daily calendar = {u.mean():.3f} "
           "(AFML ch. 4)", y=0.99)
    save(fig, "B9_overlap.png")


FIGS = {"F5": fig_F5, "F6": fig_F6, "B1": fig_B1, "B2": fig_B2, "B3": fig_B3, "B4": fig_B4, "B5": fig_B5,
        "B6": fig_B6, "B7": fig_B7, "B8": fig_B8, "B9": fig_B9}

HOW = {"F5": "features.build_features on data/prices.csv.gz, raw values, JPM, 2008-01-01 to 2010-12-31",
       "F6": "build_features then rank_normalise; ret_21 cross-section on one date",
       "B1": "numpy Polynomial.fit (least squares), seed 1, n = 30, x uniform on [0, 1]",
       "B2": "illustration only, seed 2; no measured numbers",
       "B3": "rank_normalise -> make_dataset -> time_split(2014-12-31, 2015-01-12); sklearn DecisionTreeRegressor"
             "(max_depth=k, random_state=0) on all training rows; evaluate.score r2_vs_zero",
       "B4": "seed 4; fixed evenly spaced x (n = 30), noise redrawn 2,000 times; Polynomial.fit; bias^2 and variance at 200 x",
       "B5": "diagram; seed 5 for the shuffled fold assignment; 50 time cells, 5 folds, purge 2, embargo 2",
       "B6": "seed 6; sklearn Ridge (fit_intercept False) and lasso_path on standardised X",
       "B7": "qualitative positions after James et al. (2021) fig. 2.7 (solid); ridge, k-NN, random forests hollow, placed by us; no measured numbers",
       "B8": "seed 8; features max(0, x.w), w uniform on the unit sphere in 5 dimensions; coef = lstsq (minimum norm); expected test MSE on 2,000 fresh x",
       "B9": "exact computation, no randomness"}


def write_results(only):
    old = {}
    if only and RESULTS_MD.exists():
        for line in RESULTS_MD.read_text().splitlines():
            m = re.match(r"^\| `([A-Z]\d\.[^`]+)` \| (.*?) \| (.*) \|$", line)
            if m and m.group(1).split(".")[0] not in only:
                old[m.group(1)] = (m.group(2), m.group(3))
    rows = {k: (str(v), h) for k, (v, h) in RESULTS.items()}
    merged = {**old, **rows}
    order = list(FIGS)
    keys = sorted(merged, key=lambda k: (order.index(k.split(".")[0]), list(merged).index(k)))
    lines = ["# Bonus and feature figures: measured numbers", "",
             "Written by `deck/make_bonus_figures.py` (run `python3 deck/make_bonus_figures.py`). Every value below",
             "was produced by running that script; nothing is estimated. Percentages are given as %, R squared is",
             "r2_vs_zero (1 - MSE / MSE of predicting zero) as in quantsoc_ml.evaluate. Seeds are fixed, so a rerun",
             "reproduces every value.", "", "## How each figure was computed", ""]
    lines += [f"- **{k}**: {HOW[k]}" for k in order]
    lines += ["", "## Values", "", "| key | value | how |", "|---|---|---|"]
    lines += [f"| `{k}` | {merged[k][0]} | {merged[k][1]} |" for k in keys]
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS_MD.relative_to(ROOT)} ({len(keys)} values)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=list(FIGS), help="draw only these figures")
    args = ap.parse_args()
    keys = args.only or list(FIGS)
    FIG.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for k in keys:
            t0 = time.perf_counter()
            with plt.rc_context(BIG if k in BIG_KEYS else {}):
                FIGS[k]()
            print(f"  {k}: {time.perf_counter() - t0:.1f} s")
    write_results(set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
