"""Generate workshop3/workshop3.ipynb from the cells below (outputs cleared).

Usage:  python workshop3/build_notebook.py
Edit the text here rather than in the notebook, so the two never drift apart.
Meme slots are invisible HTML comments: replace one with ![](link-to-image) to show a meme there.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "workshop3.ipynb"

CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


def meme(where):
    md(f"<!-- MEME SLOT: {where}. Replace this comment with ![](https://...) to show an image. -->")


# ------------------------------------------------------------------------------------------- intro
md("""
# Hands-on machine learning for trading
### QuantSoc · Workshop 3 · [quant-soc.com](https://quant-soc.com)

Today we replicate a well-known paper, test it honestly on fresh data, and put the strategy on a paper-trading
account. No slides: we walk through this notebook together. Run each cell when we reach it.

> **Reference.** Krauss, C., Do, X. A. and Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random
> forests: Statistical arbitrage on the S&P 500.* European Journal of Operational Research 259(2), 689-702.
> [Free working paper](https://www.econstor.eu/handle/10419/130166) ·
> [published version](https://doi.org/10.1016/j.ejor.2016.10.031)

**Plan for the two hours**

| Part | What we do |
|---|---|
| 0 | Setup |
| 1 | The paper in five lines |
| 2 | Data: the S&P 100, 2000 to yesterday |
| 3 | Features and target, exactly as in the paper |
| 4 | Signal research before any ML |
| 5 | Walk-forward training, one study period live |
| 6 | Did the models learn anything? |
| 7 | The backtest, before and after costs |
| 8 | When does it work? Regimes |
| 9 | What did the forest learn? Feature importance |
| 10 | Cross-validation done right: purging |
| 11 | How many backtests did we run? The deflated Sharpe ratio |
| 12 | Making it better: your turn |
| 13 | Going live: the paper-trading dashboard |

> **Disclosure.** Prices are real (Yahoo Finance via `yfinance`) but the universe is **today's** S&P 100, so it is
> survivorship-biased: companies that failed or left the index are missing. Teaching only, not investment advice.
""")
meme("title, e.g. 'stonks'")

# ------------------------------------------------------------------------------------------- setup
md("""
## 0. Setup

Run the next two cells. The first copies the workshop files into this runtime (Colab) or finds them on your own
computer. The second loads the code.
""")
code('''
# @title Setup (run once; safe to run again)
OWNER, REPO, REVISION = "michaelbolebrukh", "quantsoc-ml-for-quant-finance", "workshop-3"

import io, shutil, sys, urllib.request, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
FILES = ["quantsoc_ml", "workshop3", "requirements.txt", ".env.example"]


def fetch(root: Path):
    marker, wanted = root / ".release", f"{OWNER}/{REPO}@{REVISION}"
    if marker.exists() and marker.read_text().strip() == wanted:
        print("Workshop files are already here.")
        return
    url = f"https://github.com/{OWNER}/{REPO}/archive/{REVISION}.zip"
    print("Downloading", url)
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())) as zf:
        tmp = root.parent / "_workshop_src"
        shutil.rmtree(tmp, ignore_errors=True)
        zf.extractall(tmp)
    src = tmp / zf.namelist()[0].split("/")[0]
    root.mkdir(parents=True, exist_ok=True)
    for item in FILES:
        s, d = src / item, root / item
        if s.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(s, d)
        elif s.exists():
            shutil.copy(s, d)
    marker.write_text(wanted)
    print("Done.")


if IN_COLAB:
    ROOT = Path("/content/workshop")
    fetch(ROOT)
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in [here, *here.parents]
                 if (p / "quantsoc_ml" / "statarb.py").exists() and (p / "workshop3").is_dir()), None)
    if ROOT is None:
        raise SystemExit("Cannot find the workshop folder (it holds quantsoc_ml/ and workshop3/).")
for p in (ROOT, ROOT / "workshop3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
print("Workshop folder:", ROOT)
''')
code('''
%matplotlib inline
import time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from quantsoc_ml import statarb as sa
from quantsoc_ml import research as rs
from quantsoc_ml import ws3_plots as plots

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 40)
pct = lambda x: f"{x:+.2%}"
print("Ready.")
''')

# ------------------------------------------------------------------------------------------- 1 paper
md("""
## 1. The paper in five lines

1. **Universe:** every stock in the S&P 500, using the index members **as they were** on each date.
2. **Features:** 31 past returns per stock: over the last 1, 2, ..., 20 days, then 40, 60, ..., 240 days.
3. **Target:** will this stock beat the **median** stock tomorrow? A yes/no question, so it is classification.
4. **Models:** a deep neural net, gradient-boosted trees, a random forest, and their average (the "ensemble").
5. **Trading:** every day, buy the 10 stocks most likely to beat the median and short the 10 least likely.
   Close everything the next day and repeat.

**Their result:** large returns before trading costs in the 1990s, much weaker after 2001, a spike in the 2008
crisis, and close to nothing after costs from 2010. The paper itself says the edge decays.

**Our question:** does the recipe find anything on large US stocks from 2004 to **yesterday**?

*Discuss:* why a target of "beats the median" rather than "goes up"? What happens to market crashes?
""")
meme("paper section, e.g. 'reading the abstract vs reading the results table'")

# ------------------------------------------------------------------------------------------- 2 data
md("""
## 2. The data

Daily prices for today's S&P 100 members from January 2000 to the last close before the workshop. `adj_close`
is adjusted for splits and dividends, so its percentage changes are the returns a holder actually earned.
""")
code('''
long = sa.load_prices()
prices = sa.wide(long, "adj_close")           # dates down, one column per stock
print(f"{prices.shape[1]} stocks, {len(prices):,} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")
display(long.tail(3))
listed = prices.notna().sum(axis=1)
ax = listed.plot(figsize=(11, 3), title="How many of today's S&P 100 have prices on each date", color=plots.QS_NAVY)
ax.set_ylabel("stocks");
''')
md("""
The line rises because some of today's members did not exist yet (Meta 2012, Tesla 2010, Uber 2019...).
That rising line is **survivorship bias** in a picture: in 2004 we only see companies that went on to become huge.
""")

# ------------------------------------------------------------------------------------------- 3 features
md("""
## 3. Features and target, exactly as in the paper

For each stock and day $t$, the feature $R(m)$ is the return over the last $m$ days:

$$R_t(m) = \\frac{P_t}{P_{t-m}} - 1, \\qquad m \\in \\{1, \\dots, 20\\} \\cup \\{40, 60, \\dots, 240\\}$$

Everything uses prices **up to and including** day $t$, so it is known at the close of $t$. The target looks one day
ahead: $y_t = 1$ if the stock's return from $t$ to $t+1$ beats the median of all stocks that day, else $0$.
""")
code('''
with_time = time.time()
panel = sa.make_panel(prices)                 # features R1..R240, next-day return `fwd`, label `y`
print(f"{len(panel):,} stock-days, {len(sa.FEATURES)} features, built in {time.time() - with_time:.1f}s")
print("Features:", ", ".join(sa.FEATURES))
display(panel.xs("AAPL", level="symbol").tail(3).round(4))
''')
code('''
# The label is balanced by construction: half the stocks beat the median every day
print("share of 1s in the label:", round(panel["y"].mean(), 4))
''')
md("""
*Discuss:* the label is 50/50 by construction. So what accuracy should a model that knows nothing get? What
accuracy would be impressive here? (Hint: in this game 51% can be a lot.)
""")

# ------------------------------------------------------------------------------------------- 4 research
md("""
## 4. Signal research before any ML

Before training anything, ask each feature on its own: does it predict tomorrow's **relative** return?

The **information coefficient (IC)** is the rank correlation, across stocks on one day, between a signal and the
next day's returns. We average it over all days. IC = 0.02 sounds tiny, but repeated every day across 100 stocks
it is a real signal.
""")
code('''
research = panel.loc["2004":"2022"]           # keep 2023 onwards untouched for the final test
ic = pd.Series({f: sa.daily_ic(research[f], research["fwd"]).mean() for f in sa.FEATURES})
plots.bar(ic, "Mean daily IC of each feature, 2004 to 2022 (negative = past winners lose tomorrow)", "IC");
''')
md("""
The short lags have **negative** IC: stocks that went up over the last few days tend to do **worse** than the
median tomorrow. This is **short-term reversal**, documented long before ML (Jegadeesh 1990; Lehmann 1990).

The decile chart sorts stocks each day by their 5-day return and shows each group's average next-day return
**relative to that day's average stock** (so the market's general drift does not hide the pattern).
""")
code('''
dec = sa.decile_returns(research["R5"], research["fwd"])
plots.bar(dec, "Next-day return by decile of the last 5 days' return (1 = biggest losers)", "next-day return vs the day's average",
          pct=True);
''')
md("""
*Discuss:* if a one-line rule (buy last week's losers, short the winners) already captures this, what can a random
forest add? Possible answers: combining many lags at once, non-linear effects, interactions such as "big drop
**and** a long uptrend".
""")
meme("signal research, e.g. 'buy high sell low' / mean reversion")

# ------------------------------------------------------------------------------------------- 5 walk forward
md("""
## 5. Walk-forward training

The paper never shuffles time. It cuts history into **study periods**: train on 750 trading days (about three
years), then trade the next 250 days with that frozen model, then slide forward 250 days and retrain. Every
trading day is predicted by a model that never saw it.
""")
code('''
periods = sa.study_periods(panel.index.get_level_values("date").unique())
for p in periods[:2] + periods[-2:]:
    print(p)
plots.periods_timeline(periods);
''')
md("""
Let's train **one** study period live: the random forest on period 20 (trading late 2022 to late 2023).
""")
code('''
p = periods[19]
t0 = time.time()
live_pred, fitted = sa.fit_predict_period(panel, p, ["forest"])
print(p, f"\\ntrained and predicted in {time.time() - t0:.1f}s, {len(live_pred):,} predictions")
display(live_pred.head())
''')
md("""
Training all 23 periods for four models takes about 4 minutes on a laptop, longer on Colab. So we load
predictions made earlier by the **same code** (`workshop3/build_cache.py`) and check them against the period we
just trained live. If someone had edited the cache, this check would catch it.
""")
code('''
RETRAIN_ALL = False          # set True to retrain every period yourself (about 4 to 10 minutes)

if RETRAIN_ALL:
    pred = sa.walk_forward(panel, models=sa.MODEL_NAMES)
else:
    pred = sa.load_cached_predictions()
    cached = pred.loc[live_pred.index, "forest"]
    gap = (cached - live_pred["forest"]).abs().max()
    print(f"cache vs live retrain, period {p.number}: largest difference {gap:.5f}",
          "(match)" if gap < 1e-3 else "(DIFFERENT: different library versions give slightly different trees)")
print(f"{len(pred):,} out-of-sample predictions, models: {[c for c in pred.columns if c != 'period']}")
''')

# ------------------------------------------------------------------------------------------- 6 evaluation
md("""
## 6. Did the models learn anything?

Three checks per model on every out-of-sample day: **accuracy** (is P > 0.5 right?), **AUC** (does a higher P
mean a higher chance of beating the median?), and the mean daily **IC** with the next-day return.
""")
code('''
from sklearn.metrics import accuracy_score, roc_auc_score

oos = pred.join(panel[["fwd", "y"]]).dropna(subset=["y"])
models = ["logit", "forest", "boosting", "mlp", "ensemble"]
rows = {}
for m in models:
    rows[m] = {"accuracy": accuracy_score(oos["y"], oos[m] > 0.5), "AUC": roc_auc_score(oos["y"], oos[m]),
               "mean daily IC": sa.daily_ic(oos[m], oos["fwd"]).mean()}
quality = pd.DataFrame(rows).T
display(quality.style.format("{:.4f}"))
''')
md("""
Accuracy just above 50% and AUC just above 0.5 are normal in this game, so do not expect 60%. The interesting
comparison is **logit** (a straight-line model) against the trees: is the extra flexibility worth anything?
""")
code('''
plots.bar(sa.decile_returns(oos["ensemble"], oos["fwd"]),
          "Next-day return by decile of the ensemble's probability (10 = most likely to beat the median)",
          "next-day return vs the day's average", pct=True);
''')
meme("model evaluation, e.g. '51% accuracy' reactions")

# ------------------------------------------------------------------------------------------- 7 backtest
md("""
## 7. The backtest, before and after costs

The paper's rule with $k = 10$: each day, long the 10 highest probabilities and short the 10 lowest, equal weights.
Like the paper we count **\\$1 long and \\$1 short**, so the daily return is the long leg's average return minus the
short leg's.

**Costs:** the paper charges 5 basis points (0.05%) per half-turn, i.e. on every dollar bought or sold. We charge
the same on the turnover that actually happens.
""")
code('''
K, COST_BPS = 10, 5.0
results = {m: sa.long_short(pred[m], panel["fwd"], k=K, cost_bps=COST_BPS) for m in models}
table = pd.DataFrame({m: r.stats() for m, r in results.items()}).T
display(table.style.format({c: "{:.4f}" for c in table.columns}))
''')
code('''
ens = results["ensemble"]
plots.equity({"ensemble, before costs": ens.gross, "ensemble, after 5 bps costs": ens.net},
             "Krauss et al. replication on the S&P 100, every day out of sample");
plots.bar(pd.DataFrame({"before costs": sa.yearly(ens.gross), "after costs": sa.yearly(ens.net)})["after costs"],
          "Ensemble return per calendar year after costs", "return", pct=True);
''')
md("""
**How to read this.** Before costs the ensemble earns a few basis points a day, with a Sharpe ratio around 0.9 in
our run. But it trades more than twice its book every day, so 5 bps per half-turn costs more than it earns in most
years. The profitable years cluster in **2007 to 2011**, around the financial crisis, just as in the paper.

*Discuss:* the paper found about 0.45% a day before costs on the whole S&P 500 in the 1990s. Ours is far smaller.
Name three reasons. (Universe: 100 mega-caps are the most efficiently priced stocks in the world; period: the
anomaly was published and traded away; model size: ours are smaller.)
""")
meme("backtest, e.g. 'before costs vs after costs'")

# ------------------------------------------------------------------------------------------- 8 regimes
md("""
## 8. When does it work? Regimes

Reversal pays when markets are panicking: forced sellers push prices too far and liquidity providers get paid for
taking the other side. Split the out-of-sample days by how volatile the market was **the day before** (known at the
time, so no peeking).
""")
code('''
cuts = {"2004-2007 before the crisis": ("2004-01-01", "2008-08-31"), "2008-2009 crisis": ("2008-09-01", "2009-12-31"),
        "2010-2019": ("2010-01-01", "2019-12-31"), "2020 Covid": ("2020-01-01", "2020-12-31"),
        "2021-2022": ("2021-01-01", "2022-12-31"), "2023 to today (final test)": ("2023-01-01", "2030-01-01")}
display(sa.by_period(ens.gross, cuts)[["mean daily", "Sharpe"]].rename(columns=lambda c: c + " before costs")
        .join(sa.by_period(ens.net, cuts)[["annual return", "Sharpe", "max drawdown"]]).style.format("{:.4f}"))
''')
code('''
vol = sa.market_volatility(prices).shift(1).reindex(ens.gross.index)        # yesterday's 20-day market volatility
bucket = pd.qcut(vol, 5, labels=["calmest", "calm", "middle", "volatile", "most volatile"])
by_vol = pd.DataFrame({"before costs": ens.gross.groupby(bucket).mean(), "after costs": ens.net.groupby(bucket).mean()})
by_vol.plot.bar(figsize=(9, 3.5), color=[plots.QS_NAVY, plots.QS_GOLD], rot=0,
                title="Mean daily return by yesterday's market volatility (fifths of all days)")
plt.axhline(0, color="black", lw=0.8);
''')
md("""
*Discuss:* if the edge lives in volatile markets, a **volatility filter** (only trade when yesterday's volatility is
high) should help. That is one of the improvements in part 12. Is choosing the threshold after seeing this chart a
problem? (Yes: see part 11.)
""")

# ------------------------------------------------------------------------------------------- 9 importance
md("""
## 9. What did the forest learn? Feature importance

Two ways to ask a random forest which features matter (Lopez de Prado 2018, *Advances in Financial Machine
Learning*, ch. 8):

* **MDI** (mean decrease in impurity): how much each feature's splits cleaned up the **training** data. Fast, but
  biased towards noisy, many-valued features and blind to whether the split generalises.
* **Permutation importance**: shuffle one feature in **test** data and measure how much AUC drops. Slower, honest.

We use the forest trained live in part 5 and the 250 days it traded.
""")
code('''
from sklearn.inspection import permutation_importance

forest = fitted["forest"]
dates = panel.index.get_level_values("date")
test = panel[(dates >= p.trade_start) & (dates <= p.trade_end)].dropna(subset=["y"])
test = test.sample(n=min(len(test), 12000), random_state=0)       # a sample keeps this under a minute
t0 = time.time()
perm = permutation_importance(forest, test[sa.FEATURES].to_numpy(), test["y"].astype(int), scoring="roc_auc",
                              n_repeats=3, random_state=0, n_jobs=-1)
print(f"permutation importance in {time.time() - t0:.0f}s")
mdi = pd.Series(forest.feature_importances_, index=sa.FEATURES)
plots.importance_compare(mdi, pd.Series(perm.importances_mean, index=sa.FEATURES));
''')
md("""
*Discuss:* do the short lags (R1 to R5) win on permutation importance, i.e. is the forest mostly a fancy reversal
signal? Which features have high MDI but zero or negative permutation importance? Those are what the forest
**fitted** in training but which do not **help** out of sample.
""")
meme("feature importance")

# ------------------------------------------------------------------------------------------- 10 CV
md("""
## 10. Cross-validation done right: purging

Walk-forward is clean, but you will want cross-validation to tune models. Textbook k-fold shuffles rows, which
breaks in finance when **labels overlap**. Our one-day label barely overlaps, so let's make the problem visible:
predict whether a stock beats the median over the next **20** days. Today's label and tomorrow's share 19 of 20 days.

* **Shuffled k-fold:** neighbouring days land in train and test, so the model is tested on near-copies of its
  training rows.
* **Purged k-fold** (Lopez de Prado 2018, ch. 7): folds are blocks of consecutive dates, and training dates whose
  label window overlaps the test block are removed, plus an embargo after it.
""")
code('''
from sklearn.ensemble import HistGradientBoostingClassifier

H = 20
y20 = rs.forward_median_target(prices, H).stack(future_stack=True).rename("y20")
y20.index.names = ["date", "symbol"]
cv_data = panel.loc["2010":"2019", sa.FEATURES].join(y20).dropna()
cv_data = cv_data[cv_data.index.get_level_values("date").isin(cv_data.index.get_level_values("date").unique()[::2])]
X, y = cv_data[sa.FEATURES].to_numpy(), cv_data["y20"].to_numpy().astype(int)
make = lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, random_state=0)
dates_cv = cv_data.index.get_level_values("date")
scores = {
    "shuffled k-fold": rs.cv_scores(make, X, y, rs.shuffled_kfold(len(y), 5), roc_auc_score),
    "date blocks, no purge": rs.cv_scores(make, X, y, rs.purged_kfold(dates_cv, 5, 0, 0), roc_auc_score),
    "purged + embargo": rs.cv_scores(make, X, y, rs.purged_kfold(dates_cv, 5, label_horizon=H // 2 + 1, embargo=H // 2 + 1), roc_auc_score),
}
display(pd.DataFrame(scores, index=[f"fold {i + 1}" for i in range(5)]).T.assign(mean=lambda d: d.mean(axis=1)).round(4))
''')
md("""
`label_horizon` and `embargo` are counted in **rows of the calendar we kept** (every second date), so 11 of them
cover the 20-day label window.

*Discuss:* which estimate would you have trusted when deciding whether to trade this? In our run shuffled k-fold
reports an AUC well above 0.5 while both date-based versions sit near 0.5: the "skill" was the model recognising
near-copies of its training rows. Most of the leak here comes from shuffling; purging and the embargo remove what
is left at the block edges, which matters more with fewer, longer folds or longer labels.
""")
meme("cross-validation, e.g. 'data leakage'")

# ------------------------------------------------------------------------------------------- 11 DSR
md("""
## 11. How many backtests did we run? The deflated Sharpe ratio

So far we have tried 5 models. Add a few values of $k$ and a few holding rules and we have dozens of backtests.
Pick the best and its Sharpe ratio is biased upwards: some of it is luck.

The **deflated Sharpe ratio** (Bailey and Lopez de Prado 2014, *Journal of Portfolio Management* 40(5)) asks:
what is the probability the true Sharpe is above zero, given that we picked the best of N tries, and given the
returns are skewed and fat-tailed? The usual bar is 0.95.
""")
code('''
trials = {}
for m in models:
    for k in (5, 10, 20):
        for exit_k in (None, 2 * k):
            name = f"{m} k={k}" + (f" exit={exit_k}" if exit_k else "")
            trials[name] = sa.long_short(pred[m], panel["fwd"], k=k, exit_k=exit_k)
dsr = {}
for label, attr in [("before costs", "gross"), ("after costs", "net")]:
    series = {n: getattr(r, attr) for n, r in trials.items()}
    sharpes = pd.Series({n: rs.sharpe(x) for n, x in series.items()})
    best = sharpes.idxmax()
    dsr[f"best {label}: {best}"] = rs.deflated_sharpe(series[best], sharpes.values)
print(f"{len(trials)} backtests run")
display(pd.DataFrame(dsr).round(4))
''')
md("""
`hurdle_annual` is the Sharpe ratio the **best of this many zero-skill strategies** would show by luck. Our best
must clear it, not zero. Read the two columns separately: before costs, is the signal still convincing after
deflating for 30 tries? After costs, is there anything left to deflate?

*Discuss:* every chart you looked at before choosing counts as a trial, including part 8.
""")
meme("deflated Sharpe, e.g. 'p-hacking'")

# ------------------------------------------------------------------------------------------- 12 improve
md("""
## 12. Making it better: your turn

Costs are the main enemy. Ideas, roughly from easiest to hardest:

1. **Trade less.** `exit_k` keeps a stock until it falls out of the top (bottom) `exit_k`, instead of replacing it
   every day. Example below.
2. **Volatility filter.** Only hold positions when yesterday's market volatility is above a threshold (part 8).
3. **More stocks.** Reversal is stronger in smaller, less liquid stocks, but costs are higher there too.
4. **Sector neutral.** Rank stocks within their sector, so the book is not one big bet on tech against energy.
5. **New features.** Volume, volatility, or the 52-week high from Workshop 2.

Whatever you try, test it on 2023 onwards **once**, at the end, and count every variant as a trial for part 11.
""")
code('''
# Example for idea 1: hold names until they leave the top / bottom 30
banded = sa.long_short(pred["ensemble"], panel["fwd"], k=10, exit_k=30)
display(pd.DataFrame({"daily rebalance": ens.stats(), "hold until outside top 30": banded.stats()}).T
        .style.format("{:.4f}"))
plots.equity({"daily rebalance": ens.net, "hold until outside top 30": banded.net}, "After costs");
''')
code('''
# Your turn: idea 2, the volatility filter. Pick a threshold and only trade on days above it.
THRESHOLD = 0.25                                          # <- change this (annualised market volatility)
yesterday_vol = sa.market_volatility(prices).shift(1).reindex(banded.net.index)
filtered = banded.net.where(yesterday_vol > THRESHOLD, 0.0)    # flat (no positions, no costs) on calm days
print(f"trading on {(yesterday_vol > THRESHOLD).mean():.0%} of days")
display(pd.DataFrame({"banded": sa.perf_stats(banded.net), "banded + vol filter": sa.perf_stats(filtered)}).T.round(4))
''')
md("""
The filter above is a simplification: going flat on a calm day and back in later costs turnover that this quick
version does not charge. Fixing that is a good exercise.
""")

# ------------------------------------------------------------------------------------------- 13 live
md("""
## 13. Going live: the paper-trading dashboard

To trade tomorrow, the paper would train on the latest 750 days and use that model for the next 250. Same here:
train the forest on the most recent 750 days, score every stock on the last close, and size the book.

Live sizing is more conservative than the backtest: **\\$0.50 long and \\$0.50 short per \\$1 of equity**
(`gross_exposure=1.0`), so it fits in a paper account without heavy margin.
""")
code('''
strategy = sa.train_live(panel, model_name="forest", k=10, gross_exposure=1.0)
print(f"trained on {strategy.trained_from} to {strategy.trained_to}, {len(strategy.symbols)} stocks")
today = strategy.scores(prices)
print("decision date:", today.name.date())
display(pd.DataFrame({"P(beats median)": today, "weight": strategy.weights(today)}).dropna().loc[
    lambda d: d["weight"] != 0].sort_values("weight", ascending=False).style.format({"P(beats median)": "{:.3f}", "weight": "{:+.3f}"}))
''')
code('''
path = sa.save_strategy(strategy, ROOT / "workspace" / "ws3" / "strategy.joblib")
print("saved", path)
''')
md("""
**Offline preview.** The same plan the dashboard builds, against a pretend \\$100,000 account that already holds
yesterday's book. Nothing leaves this computer.
""")
code('''
import live
from quantsoc_ml.paper import ReadOnlyBroker

broker = live.demo_broker(strategy, live.snapshot_bars_for_broker(long))
plan = live.build_live_plan(ReadOnlyBroker(broker), strategy)
print("decision date:", plan.decision_date.date())
display(plan.orders.head(12))
''')
md("""
**The dashboard** (presenter's laptop; Streamlit does not run inside Colab):

```bash
cp .env.example .env          # then paste your Alpaca PAPER keys into .env
pip install streamlit alpaca-py python-dotenv
streamlit run workshop3/dashboard.py
```

It shows the account, today's signal, target weights against current positions, and the proposed orders. It only
sends orders to the **paper** endpoint, and only after you tick a box and type `SUBMIT`. Without keys it runs the
same offline demo as the cell above. Details: `workshop3/README.md`.

To run it yourself after the workshop: make a free account at [alpaca.markets](https://alpaca.markets), switch to
Paper Trading, create API keys, and put them in `.env`. Never paste keys into a notebook.
""")
meme("going live, e.g. 'deploying to prod on a Friday'")
md("""
## Recap

* The paper's recipe works mechanically: the models find short-term reversal, and more than a one-line rule does.
* On today's S&P 100 the edge before costs is real but small, and costs eat it outside crisis years.
* Honest research means walk-forward testing, purged CV when labels overlap, and counting every trial.

Next session: bring an improvement, test it on 2023 onwards once, and report its deflated Sharpe ratio.
""")


def build():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}, "colab": {"provenance": []}}
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print("wrote", build(), len(CELLS), "cells")
