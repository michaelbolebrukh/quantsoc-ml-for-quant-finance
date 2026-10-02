"""Notebook plumbing: environment checks, friendly errors, Reference boxes, timers, tables and the export checkpoint.

Nothing here hides a learning objective: these are the repetitive display and bookkeeping parts.
The Reference citations are copied verbatim from docs/CONTRACT.md ("Full citations");
tests/test_notebook_runs.py checks that every one still appears there word for word.
"""
from __future__ import annotations

import html
import os
import platform
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .exercises import ExerciseIncomplete

CITATIONS: Dict[str, str] = {
    "lopez_de_prado_2018_jpm": 'López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." The Journal of Portfolio Management, 44(6), 120-133. https://doi.org/10.3905/jpm.2018.44.6.120 (free copy: SSRN 3104816)',
    "afml": 'López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 2 "Financial Data Structures"; Chapter 7 "Cross-Validation in Finance"; Chapter 11 "The Dangers of Backtesting".',
    "jegadeesh_titman_1993": 'Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." The Journal of Finance, 48(1), 65-91.',
    "jegadeesh_1990": 'Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security Returns." The Journal of Finance, 45(3), 881-898.',
    "kaufman_2012": 'Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). "Leakage in Data Mining: Formulation, Detection, and Avoidance." ACM Transactions on Knowledge Discovery from Data, 6(4), Article 15.',
    "gu_kelly_xiu_2020": 'Gu, S., Kelly, B. and Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." The Review of Financial Studies, 33(5), 2223-2273.',
    "breiman_2001": 'Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.',
    "krauss_2017": 'Krauss, C., Do, X. A. and Huck, N. (2017). "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500." European Journal of Operational Research, 259(2), 689-702.',
    "bailey_2014": 'Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." Notices of the AMS, 61(5), 458-471. (open access at ams.org)',
    "shumway_1997": 'Shumway, T. (1997). "The Delisting Bias in CRSP Data." The Journal of Finance, 52(1), 327-340.',
    "arnott_2019": 'Arnott, R., Harvey, C. R. and Markowitz, H. (2019). "A Backtesting Protocol in the Era of Machine Learning." The Journal of Financial Data Science, 1(1), 64-74.',
    "deflated_sharpe_2014": 'Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." The Journal of Portfolio Management, 40(5), 94-107',
}


# --------------------------------------------------------------------------- environment

def is_colab() -> bool:
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def runtime_name() -> str:
    return "Google Colab" if is_colab() else "Kaggle" if is_kaggle() else "local Jupyter"


def setup_ipython() -> None:
    """Short messages instead of long tracebacks for the two errors a participant meets.

    ExerciseIncomplete -> "[stop] ...". ValueError (the package raises it with one plain sentence,
    for example when two split dates are too close) -> "[x] ..." plus one line of advice that fits
    the message. The cell still counts as failed.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return

        def handler(shell, etype, evalue, tb, tb_offset=None):
            if issubclass(etype, ExerciseIncomplete):
                print(f"\n[stop] {evalue}\n")
            else:
                print(f"\n[x] {evalue}\n    {value_error_advice(str(evalue))} "
                      "(Full details: run %tb in a new cell.)\n")
            return []
        ip.set_custom_exc((ExerciseIncomplete, ValueError), handler)
    except Exception:
        pass


STALE_MODEL_ADVICE = ("A task was changed after section 7 fitted your model. Run section 7 again from its "
                      "refit cell, then this cell.")
TASK_ADVICE = ("If this is a task cell, fix the value marked '# <- change this' and run the cell again; "
               "otherwise see docs/troubleshooting.md.")


def value_error_advice(message: str) -> str:
    """The one line printed under a ValueError: what to do next, true for the message given."""
    if "these knobs give" in message:              # export_bundle: the model and the knobs disagree
        return STALE_MODEL_ADVICE
    return TASK_ADVICE


def environment_report() -> pd.DataFrame:
    import matplotlib
    import sklearn
    rows = [("python", platform.python_version()), ("numpy", np.__version__), ("pandas", pd.__version__),
            ("scikit-learn", sklearn.__version__), ("matplotlib", matplotlib.__version__),
            ("CPU cores", str(os.cpu_count())), ("runtime", runtime_name())]
    return pd.DataFrame(rows, columns=["component", "version"]).set_index("component")


def runtime_loss_notice() -> None:
    print("Reminder: Colab and Kaggle runtimes are temporary. Files made here (your exported strategy, the ZIP)\n"
          "disappear when the runtime disconnects. Saving the notebook does NOT save them.\n"
          "Section 8 downloads a ZIP of your strategy: do that before you leave.")


def find_repo_root(start: Optional[Path] = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "quantsoc_ml" / "__init__.py").exists() and (p / "data").exists():
            return p
    raise FileNotFoundError("could not find the workshop folder (quantsoc_ml/ and data/). Run the Setup cell first.")


# --------------------------------------------------------------------------- display

def reading_box(citations: Union[str, Sequence[str]], note: str = ""):
    """An HTML callout: the full citation(s) and one line on what this section uses from them.

    `citations` are keys of CITATIONS (or full citation strings, used as given).
    """
    from IPython.display import HTML
    if isinstance(citations, str):
        citations = [citations]
    items = "".join(f"<li style='margin:2px 0'>{html.escape(CITATIONS.get(c, c))}</li>" for c in citations)
    note_html = f"<div style='margin-top:6px'><b>What we use from it:</b> {html.escape(note)}</div>" if note else ""
    return HTML("<div style='border-left:4px solid #3b82f6;background:rgba(59,130,246,0.08);"
                "padding:8px 12px;margin:6px 0;border-radius:4px'>"
                f"<div style='font-weight:600'>📖 Reference</div><ul style='margin:4px 0 0 18px;padding:0'>{items}</ul>"
                f"{note_html}</div>")


@contextmanager
def timed(label: str):
    """Print how long the block took, so every slow step is labelled with a measured time."""
    t0 = time.perf_counter()
    yield
    print(f"[time] {label}: {time.perf_counter() - t0:.1f} s")


def pct(x, digits: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{digits}f}%"


# --------------------------------------------------------------------------- research helpers

def quarter_sample(df: pd.DataFrame, frac: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """A stratified sample: the same share of rows from EVERY date, so no year is left out."""
    out = df.groupby(level="date", group_keys=False).sample(frac=frac, random_state=seed)
    out = out.sort_index()
    out.attrs.update(df.attrs)
    return out


@contextmanager
def quiet_constant_input():
    """Silence scipy's ConstantInputWarning (a forecast that is the same for every stock on a day).

    Small forests (20 trees of depth 2, say) do that on some days; the IC is then undefined and the
    tables show "n/a" instead, so the warning adds nothing. Only score_table uses this.
    """
    with warnings.catch_warnings():
        try:
            from scipy.stats import ConstantInputWarning
            warnings.simplefilter("ignore", ConstantInputWarning)
        except ImportError:                        # older scipy: the warning has no class of its own
            warnings.filterwarnings("ignore", message=".*constant.*")
        yield


FOREST_TREES = (20, 500)
FOREST_DEPTH = (2, 12)


def make_forest(X, y, n_estimators: int, max_depth: int):
    """fit_forest with the Task 4 ranges checked FIRST, so a refused value never costs a slow fit.

    20 to 500 trees and a max_depth of 2 to 12; anything else raises a ValueError with one plain
    sentence (the notebook prints it as "[x] ..."). The fit itself is models.fit_forest, unchanged.
    """
    from .exercises import _is_int
    from .models import fit_forest
    if max_depth is None:
        raise ValueError("max_depth=None lets every tree grow until it memorises single days; section 5 shows "
                         "exactly that, live. Here pick a max_depth from 2 to 12")
    if not _is_int(n_estimators) or not _is_int(max_depth):
        raise ValueError(f"n_estimators and max_depth must be whole numbers, like 200 and 6, not "
                         f"{n_estimators!r} and {max_depth!r}")
    if not FOREST_TREES[0] <= int(n_estimators) <= FOREST_TREES[1]:
        raise ValueError(f"choose from {FOREST_TREES[0]} to {FOREST_TREES[1]} trees, not {int(n_estimators)} "
                         "(more than 500 is slow and adds almost nothing)")
    if not FOREST_DEPTH[0] <= int(max_depth) <= FOREST_DEPTH[1]:
        raise ValueError(f"choose a max_depth from {FOREST_DEPTH[0]} to {FOREST_DEPTH[1]}, not {int(max_depth)}")
    return fit_forest(X, y, n_estimators=int(n_estimators), max_depth=int(max_depth))


def score_table(predictions: Mapping[str, np.ndarray], test: pd.DataFrame) -> pd.DataFrame:
    """Scores per model on `test`: r2 vs zero, pooled IC, and the day-by-day IC a long/short book trades on.

    An IC that is undefined (a forecast that never varies, or no day where it varies) shows "n/a".
    """
    from .evaluate import ic_by_date, score
    y = test["target"].to_numpy(dtype=float)
    dates = test.index.get_level_values("date")
    rows = {}
    with quiet_constant_input():
        for name, p in predictions.items():
            p = np.asarray(p, dtype=float)
            s = score(y, p)
            ic = s["ic"]
            row = {"r2 vs zero": pct(s["r2_vs_zero"]),
                   "IC, all rows pooled": f"{ic:+.3f}" if ic is not None and np.isfinite(ic) else "n/a"}
            daily = ic_by_date(p, y, dates).dropna() if np.std(p) > 0 else pd.Series(dtype=float)
            if len(daily) and np.isfinite(daily.mean()):
                row["IC per day (mean)"] = f"{daily.mean():+.4f}"
                row["days with IC > 0"] = f"{(daily > 0).mean():.1%}"
            else:
                row["IC per day (mean)"], row["days with IC > 0"] = "n/a", "n/a"
            rows[name] = row
    out = pd.DataFrame(rows).T
    out.index.name = "forecast"
    return out


def feature_ic_table(dataset: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """For each column: its rank correlation with the target, day by day, averaged (and how often it is positive).

    Used for the leakage demo: an honest feature scores a few hundredths at most; a leaky one scores far more.
    """
    from .evaluate import ic_by_date
    y = dataset["target"].to_numpy(dtype=float)
    dates = dataset.index.get_level_values("date")
    rows = {}
    for c in columns:
        daily = ic_by_date(dataset[c].to_numpy(dtype=float), y, dates).dropna()
        rows[c] = {"correlation with the target (mean per day)": f"{daily.mean():+.3f}",
                   "days with a positive correlation": f"{(daily > 0).mean():.0%}"}
    out = pd.DataFrame(rows).T
    out.index.name = "feature"
    return out


def model_scores(pairs: Mapping[str, tuple]) -> pd.DataFrame:
    """Scores for named (model, test_df) pairs: r2 vs zero, IC per day, and top-minus-bottom decile spread.

    The spread is the mean realised 5-day return of the 10% highest forecasts minus that of the 10% lowest.
    """
    from .evaluate import decile_table
    from .models import xy
    rows = []
    for name, (model, df) in pairs.items():
        X, y = xy(df)
        p = np.asarray(model.predict(X), dtype=float)
        t = score_table({name: p}, df)
        d = decile_table(p, y)
        t["top minus bottom decile (5 days)"] = pct(d["mean_target"].iloc[-1] - d["mean_target"].iloc[0])
        rows.append(t)
    return pd.concat(rows)


def stats_view(results: Mapping[str, object]) -> pd.DataFrame:
    """The backtest headline numbers, formatted for reading (one column per result)."""
    labels = {"total_return": "total return", "annual_return": "return per year", "annual_vol": "volatility per year",
              "sharpe": "Sharpe ratio", "max_drawdown": "max drawdown", "avg_turnover": "turnover per rebalance",
              "n_rebalances": "rebalances"}
    cols = {}
    for name, r in results.items():
        s = r.stats
        cols[name] = {labels["total_return"]: pct(s["total_return"], 1), labels["annual_return"]: pct(s["annual_return"], 1),
                      labels["annual_vol"]: f"{s['annual_vol'] * 100:.1f}%", labels["sharpe"]: f"{s['sharpe']:+.2f}",
                      labels["max_drawdown"]: f"{s['max_drawdown'] * 100:.1f}%",
                      labels["avg_turnover"]: f"{s['avg_turnover']:.2f}", labels["n_rebalances"]: int(s["n_rebalances"])}
    return pd.DataFrame(cols)


MARKET_LABEL = "the market: all 40 stocks, equal weight, long only (for scale)"


def label_market(fig, label: str = MARKET_LABEL):
    """Rename the second line of an equity figure: it is the market, shown for scale."""
    ax = fig.axes[0]
    if len(ax.lines) >= 2:
        ax.lines[1].set_label(label)
    ax.legend(loc="upper left")
    return fig


def equity_figure(net, gross=None, market=None, title: str = "Your strategy after costs, test period"):
    """viz.equity_curve (strategy after costs vs the market) plus, when given, the same strategy before costs."""
    from . import viz
    fig = viz.equity_curve(net, market, title=title)
    ax = fig.axes[0]
    ax.lines[0].set_label("your strategy, after costs")
    if market is not None:
        ax.lines[1].set_label(MARKET_LABEL)
    if gross is not None:
        g = gross.equity if hasattr(gross, "equity") else gross
        ax.plot(g.index, g * 100, color=viz.PALETTE[0], lw=1, ls="--", label="your strategy, before costs")
    ax.legend(loc="upper left")
    return fig


def drawdown_figure(net, market=None, title: str = "Drawdown: how far below its previous peak (after costs)"):
    """viz.drawdown_area with the market line labelled as the market, for scale."""
    from . import viz
    fig = viz.drawdown_area(net, market, title=title)
    ax = fig.axes[0]
    if market is not None and ax.lines:
        ax.lines[0].set_label(MARKET_LABEL)
    for c in ax.collections:
        c.set_label("your strategy")
    ax.legend(loc="lower left")
    return fig


def cost_per_year(turnover_per_rebalance: float, rebalance_days: int, cost_bps: float = 5.0) -> float:
    """Approximate share of capital paid in costs per year: turnover x cost x rebalances per year (252 days)."""
    return float(turnover_per_rebalance) * float(cost_bps) / 1e4 * 252 / int(rebalance_days)


# --------------------------------------------------------------------------- backtests, counted

_BACKTESTS_RUN = 0


def run_backtest(pred_panel, returns_wide, config):
    """backtest.run_backtest, counted: the checklist reports how many backtests this session ran.

    The notebook imports run_backtest from here, so every backtest a participant runs (re-runs
    included) is counted. run_benchmark (the market line) is not a try and is not counted.
    """
    global _BACKTESTS_RUN
    from .backtest import run_backtest as _run
    result = _run(pred_panel, returns_wide, config)
    _BACKTESTS_RUN += 1
    return result


def backtests_run() -> int:
    """How many strategy backtests this session has run through run_backtest above."""
    return _BACKTESTS_RUN


def backtest_checklist(config, used_reference: bool, n_backtests_run: Optional[int] = None,
                       test_end: str = "2019-12-20") -> pd.DataFrame:
    """A short checklist adapted from Arnott, Harvey and Markowitz (2019), marked for this run.

    n_backtests_run defaults to the session's count (backtests_run()).
    """
    if n_backtests_run is None:
        n_backtests_run = backtests_run()
    shallow = int(config.max_depth) <= 6
    years = (pd.Timestamp(test_end) - pd.Timestamp(config.test_start)).days / 365.25
    rows = [
        ("An economic reason before the data", True,
         "the features come from published research: momentum, reversal, volatility"),
        ("Test period chosen before seeing results", True,
         f"you fixed {config.test_start} onwards in Task 3, before any model was scored"),
        ("Every try counted and reported", False,
         f"you have run {n_backtests_run} backtest{'s' if n_backtests_run != 1 else ''} in this session "
         "and tried several models today; "
         "Part 2 adjusts the Sharpe ratio for the number of tries"),
        ("Data free of survivorship bias", False, "only companies that still trade in 2019 are included"),
        ("Trading costs included", True, f"{config.cost_bps:g} bps per unit traded (but no borrowing fee for shorts)"),
        ("Model kept simple", shallow, f"max_depth {config.max_depth}" + (" with leaves of 50 rows" if shallow else ", deeper than the reference 6")),
        ("More than one test period", False,
         f"one test window of about {years:.0f} years (from {config.test_start}); Part 2 uses purged cross-validation"),
        ("Model refreshed as the world changes", False, "fitted once and never retrained during the test"),
        ("Own choices, not someone else's", not used_reference,
         "all five knobs are yours" if not used_reference else "some knobs are still the reference"),
    ]
    return pd.DataFrame([{"check": c, "this run": "✔ yes" if ok else "✘ no", "why": why} for c, ok, why in rows]).set_index("check")


# --------------------------------------------------------------------------- export and paper preview

_LAST_EXPORT: Optional[Path] = None


def export_checkpoint(bundle_dir, zip_path, download: bool = True) -> Path:
    """Zip the bundle, check the ZIP, and in Colab start the browser download.

    It records the bundle as this session's export, so run_paper_preview can tell a fresh bundle
    from one left over from an earlier session.
    """
    global _LAST_EXPORT
    from .export import check_zip, zip_bundle
    _LAST_EXPORT = Path(bundle_dir).resolve()
    zip_path = Path(zip_path)
    zip_bundle(bundle_dir, zip_path)
    report = check_zip(zip_path)
    print(f"ZIP written: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    print(f"contents check: {'OK' if report.get('ok') else 'PROBLEM'}")
    if not report.get("ok"):
        print(report)
    if download and is_colab():
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as e:
            print(f"Automatic download failed ({e}); use the Files pane on the left to download {zip_path}.")
    elif download and is_kaggle():
        print("Kaggle: download the ZIP from the Output panel on the right (it is under /kaggle/working).")
    elif download:
        print("Not running in Colab: the ZIP is saved in the folder shown above.")
    return zip_path


def run_paper_preview(root: Path, bundle_dir: Path, timeout: int = 120) -> int:
    """Run `python run_paper.py --preview --dry-data data/prices.csv.gz --bundle <dir>` and print its output.

    Offline: the runner uses a MOCK broker and the committed price file, needs no keys, sends nothing.
    """
    root, bundle_dir = Path(root).resolve(), Path(bundle_dir).resolve()
    if not (bundle_dir / "strategy.py").is_file():
        print(f"[x] There is no exported strategy in {bundle_dir} yet. Run the export cell above until it prints "
              "'contents check: OK', then run this cell again.")
        return 3
    if _LAST_EXPORT != bundle_dir:
        print(f"[!] The export cell has not finished in this session, so this previews the strategy exported "
              f"earlier into {bundle_dir}, not the model above. Run the export cell first to preview this one.\n")
    cmd = [sys.executable, "run_paper.py", "--preview", "--dry-data", "data/prices.csv.gz", "--bundle", str(bundle_dir)]
    print(f"(in {root})\n$ python " + " ".join(cmd[1:]) + "\n")
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr)
    print(f"[runner exit code {proc.returncode}]")
    return proc.returncode


# --------------------------------------------------------------------------- instructor only

def make_deck_figures(out_dir, prices, volume, returns, dpi: int = 200, dark: bool = False) -> Dict[str, Path]:
    """Write the deck's four result figures, ALWAYS from the reference strategy (never a participant's knobs).

    F1_decile.png  decile bar of the reference forest as section 4 fits it (quarter of the training rows) on the 2015-2019 test
    F2_noise.png   train vs test r2, unbraked forest (no depth limit, leaf 5, 50 trees, 20,000 rows), with and without noise
    F3_equity.png  reference strategy, 21-day rebalance, before and after 5 bps costs, against the market (for scale)
    F4_split.png   the reference split timeline
    dark=True draws the same data and titles for the deck's dark canvas: background #0a0a0a, off-white text,
    brand-blue series, Inter where matplotlib can find it, and type a size larger, since the slide shows each
    figure at about two thirds of its width. The default (False) is the notebook's own light look.
    Takes about 30 seconds on 4 cores (one full forest fit and the unbraked noise experiment).
    """
    import contextlib
    from dataclasses import replace

    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from . import backtest as B, evaluate as E, features as F, models as M, splits as SP, viz
    from .solutions import reference_strategy

    # QuantSoc dark tokens (next-pwa/app/styles/tokens.css); contrast against #0a0a0a in brackets.
    fg, muted, rule, brand = "#dae0e7", "#acb6c3", "#717a88", "#61a6fa"   # 14.9, 9.7, 4.6, 7.9 to 1
    recolour = {"#1f77b4": brand, "#ff7f0e": muted, "#2ca02c": brand, "#d62728": "#f26464",  # viz.PALETTE
                "#000000": fg, "#808080": rule}  # split_timeline's black price line; the grey zero lines
    style = {"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a", "savefig.facecolor": "#0a0a0a",
             "axes.edgecolor": "#3a3f49", "axes.labelcolor": muted, "text.color": fg, "xtick.color": muted,
             "ytick.color": muted, "grid.color": "#dae0e7", "grid.alpha": 0.12, "legend.facecolor": "#141515",
             "legend.edgecolor": "#3a3f49", "legend.framealpha": 1.0, "font.size": 13, "axes.titlesize": 14,
             "axes.titleweight": "semibold", "axes.titlecolor": fg,
             "font.family": ["Inter", "DejaVu Sans"]}

    def _darken(fig):
        """Swap the light palette for the dark one on every artist, keeping each one's alpha."""
        for a in fig.findobj():
            for get, put in (("get_color", "set_color"), ("get_facecolor", "set_facecolor"),
                             ("get_edgecolor", "set_edgecolor"), ("get_markerfacecolor", "set_markerfacecolor"),
                             ("get_markeredgecolor", "set_markeredgecolor")):
                if not (hasattr(a, get) and hasattr(a, put)) or a is fig:
                    continue
                try:
                    c = getattr(a, get)()
                    rgba = mcolors.to_rgba_array(c)
                except (TypeError, ValueError):
                    continue
                if len(rgba) != 1:
                    continue
                new = recolour.get(mcolors.to_hex(rgba[0][:3]))
                if new:
                    getattr(a, put)(mcolors.to_rgba(new, rgba[0][3]))
        return fig

    def _save(fig, path):
        if dark:
            _darken(fig)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    with (plt.rc_context(style) if dark else contextlib.nullcontext()):
        cfg = reference_strategy()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        feats = F.rank_normalise(F.build_features(prices, volume, names=cfg.feature_names))
        ds = F.make_dataset(feats, F.forward_return(prices))
        train, test = SP.time_split(ds, cfg.train_end, cfg.test_start)
        # The same forest the notebook's section 4 fits (a quarter of the training rows, for speed), so the
        # slide and the room show the same picture.
        model = M.fit_forest(*M.xy(quarter_sample(train)), n_estimators=cfg.n_estimators, max_depth=cfg.max_depth)
        Xte, yte = M.xy(test)
        paths = {}
        fig = viz.decile_bar(E.decile_table(model.predict(Xte), yte),
                             title="Reference forest (as in section 4): mean 5-day return by forecast decile, test 2015 to 2019")
        paths["F1"] = out / "F1_decile.png"
        _save(fig, paths["F1"])
        noise = E.noise_experiment(train.sample(n=20000, random_state=0), test, n_noise=30, seed=0,
                                   n_estimators=50, max_depth=None, min_samples_leaf=5)
        noise.index = ["real features", "real + 30 noise columns"]
        fig = viz.train_vs_test_bars(noise, title="Unbraked forest (no depth limit): fit on training vs test data")
        paths["F2"] = out / "F2_noise.png"
        _save(fig, paths["F2"])
        panel = M.predict_panel(model, test)
        res = B.run_backtest(panel, returns, cfg)
        gross = B.run_backtest(panel, returns, replace(cfg, cost_bps=0.0))
        mkt = B.run_benchmark(returns, panel.index, cfg)
        fig = equity_figure(res, gross, mkt, title="Reference strategy (21-day rebalance) vs the market, test 2015 to 2019")
        fig.axes[0].lines[0].set_label("reference strategy, after 5 bps costs")
        fig.axes[0].lines[-1].set_label("reference strategy, before costs")
        fig.axes[0].legend(loc="upper left")
        paths["F3"] = out / "F3_equity.png"
        _save(fig, paths["F3"])
        fig = viz.split_timeline(train, test, prices)
        paths["F4"] = out / "F4_split.png"
        _save(fig, paths["F4"])
        for k, p in paths.items():
            print(f"wrote {p}")
        return paths
