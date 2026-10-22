"""Generate workshop4/workshop4.ipynb from the cells below (outputs cleared).

Usage:  python workshop4/build_notebook.py
Edit the text here rather than in the notebook, so the two never drift apart.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "workshop4.ipynb"
CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# ------------------------------------------------------------------------------------------- intro
md("""
# Statistical arbitrage with stochastic processes
### QuantSoc · Workshop 4 · [quant-soc.com](https://quant-soc.com)

Last time a machine learning model learned short-term reversal from 31 past returns. Today we build the same kind
of trade from a **model** instead: strip out what the market and the sector explain, treat what is left as a
mean-reverting stochastic process, and trade when it strays too far. Then we ask which works better, and whether
ML can help the model.

> **Reference.** Avellaneda, M. and Lee, J.-H. (2010). *Statistical arbitrage in the US equities market.*
> Quantitative Finance 10(7), 761-782. [Free working paper (2008)](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb071008.pdf) ·
> [published version](https://doi.org/10.1080/14697680903124632). Section numbers below refer to the free version.

| Part | What we do |
|---|---|
| 0 | Setup |
| 1 | The paper in six lines |
| 2 | Data: the S&P 100, sector ETFs, sectors |
| 3 | Stochastic processes: the Ornstein-Uhlenbeck model |
| 4 | Factors with PCA: unsupervised learning finds the market's structure |
| 5 | Factors with sector ETFs |
| 6 | From residuals to an s-score, step by step |
| 7 | Trading rules, on one stock |
| 8 | The full backtest: four versions |
| 9 | When did it work? Periods and August 2007 |
| 10 | Model against machine: Workshop 3's ML strategy |
| 11 | Can ML pick the good trades? Meta-labelling |
| 12 | How many backtests did we run? |
| 13 | Your turn |
| 14 | Going live |

> **Disclosure.** Prices are real (Yahoo Finance via `yfinance`) but the universe is **today's** S&P 100, so it is
> survivorship-biased: companies that failed or left the index are missing, which flatters any strategy that
> buys stocks after they fall. The paper did not have this problem. Teaching only, not investment advice.
""")

# ------------------------------------------------------------------------------------------- setup
md("""
## 0. Setup

Run the next two cells. The first downloads the workshop files (Colab) or finds them on your computer.
""")
code('''
# @title Setup (run once; safe to run again)
OWNER, REPO, REVISION = "michaelbolebrukh", "quantsoc-ml-for-quant-finance", "main"

import io, shutil, sys, urllib.request, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
FILES = ["quantsoc_ml", "workshop3", "workshop4", "requirements.txt", ".env.example"]


def fetch(root: Path):
    """Download the branch every time: a branch moves, so files from an earlier run may be out of date."""
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
    print("Done.")


if IN_COLAB:
    ROOT = Path("/content/workshop")
    fetch(ROOT)
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in [here, *here.parents]
                 if (p / "quantsoc_ml" / "avlee.py").exists() and (p / "workshop4").is_dir()), None)
    if ROOT is None:
        raise SystemExit("Cannot find the workshop folder (it holds quantsoc_ml/ and workshop4/).")
for p in (ROOT, ROOT / "workshop4"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
print("Workshop folder:", ROOT)
''')
code('''
%matplotlib inline
import sys, time, warnings
for name in [m for m in sys.modules if m.split(".")[0] in ("quantsoc_ml", "live")]:
    del sys.modules[name]                       # re-running this cell picks up freshly downloaded code
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from quantsoc_ml import avlee as al
from quantsoc_ml import research as rs
from quantsoc_ml import ws3_plots as plots3
from quantsoc_ml import ws4_plots as plots
from quantsoc_ml.statarb import perf_stats, yearly

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)
print("Ready.")
''')

# ------------------------------------------------------------------------------------------- 1 paper
md("""
## 1. The paper in six lines

1. **Factors.** Explain each stock's daily return by a few common factors: either its **sector ETF**, or
   **eigenportfolios** found by PCA on the correlation matrix of all stocks.
2. **Residual.** What the factors do not explain is the stock's own, idiosyncratic move.
3. **Model.** The running total of that residual is modelled as an **Ornstein-Uhlenbeck (OU) process**: a random
   walk with a spring pulling it back to an equilibrium level.
4. **Signal.** The **s-score** says how many standard deviations the stock is from that equilibrium.
5. **Trade.** Buy stocks far below equilibrium and short those far above, each hedged with its factors, so the
   book does not care where the market goes. Close when the stock is back near equilibrium.
6. **Universe and costs.** Every US stock above \\$1bn, 5 bps per trade, re-estimated every day.

**Their result (after costs):** PCA strategies earned a Sharpe ratio of **1.44** over 1997 to 2007, but only
**0.9** over 2003 to 2007. ETF strategies earned **1.1** over 1997 to 2007, with a similar decline after 2002.
Adjusting the signal for trading volume ("trading time") lifted the ETF strategy to **1.51** over 2003 to 2007.

*Discuss:* this is Workshop 3's reversal idea again, but every step is a stated model. What do we gain from a model
over a black box? What could we lose?
""")

# ------------------------------------------------------------------------------------------- 2 data
md("""
## 2. Data

The stocks are Workshop 3's snapshot (today's S&P 100, 2000 to 15 October 2026). Added for today: SPY and the
eleven SPDR sector ETFs, and each stock's sector (Yahoo's current classification, mapped to its SPDR fund).
""")
code('''
prices = al.load_prices()
etfs = al.load_etfs()
sectors = al.load_sectors()
R, etf_R = al.returns(prices), al.returns(etfs)
print(f"{prices.shape[1]} stocks, {etfs.shape[1]} ETFs, {prices.index[0].date()} to {prices.index[-1].date()}")
display(sectors.value_counts().rename("stocks per sector fund").to_frame().T)
first = etfs.apply(lambda c: c.first_valid_index().date())
print("ETFs that started late:", {k: str(v) for k, v in first.items() if v.year > 1999})
''')
md("""
XLRE (real estate, 2015) and XLC (communication services, 2018) did not exist earlier, so before they have 60
days of history their stocks are hedged with SPY instead.
""")

# ------------------------------------------------------------------------------------------- 3 OU
md("""
## 3. Stochastic processes: the Ornstein-Uhlenbeck model

A **random walk** $dX = \\sigma\\, dW$ wanders off and never comes back: there is nothing to trade. The
**Ornstein-Uhlenbeck process** adds a pull towards an equilibrium level $m$:

$$dX_t = \\kappa\\,(m - X_t)\\,dt + \\sigma\\, dW_t$$

* $\\kappa$ is the **speed of mean reversion**: the pull is $\\kappa$ times the distance from $m$.
  $\\tau = 1/\\kappa$ is its time scale, and the **half-life** is $\\ln 2 / \\kappa$.
* In the long run $X$ settles into a normal distribution with mean $m$ and standard deviation
  $\\sigma_{eq} = \\sigma / \\sqrt{2\\kappa}$.

Same random shocks, three speeds of mean reversion (per year; 8.4 is the paper's minimum, explained below):
""")
code('''
plots.ou_gallery(kappas=(0.0, 8.4, 40.0));
''')
md("""
**From the continuous model to a regression.** Observed once a day ($\\Delta t = 1/252$), the OU process is
exactly an AR(1) regression:

$$X_{n+1} = a + b\\, X_n + \\zeta_{n+1}, \\qquad b = e^{-\\kappa \\Delta t},\\quad a = m\\,(1-b),\\quad
\\mathrm{Var}(\\zeta) = \\sigma^2 \\frac{1 - e^{-2\\kappa\\Delta t}}{2\\kappa}$$

So fit $a$ and $b$ by least squares and read the parameters back off: $\\kappa = -\\ln(b) \\times 252$,
$m = a/(1-b)$, $\\sigma_{eq} = \\sqrt{\\mathrm{Var}(\\zeta)/(1-b^2)}$. Let's check it recovers a process whose
parameters we know.
""")
code('''
true_kappa, true_m, true_sigma = 40.0, 0.02, 0.30          # <- change these
for days in (60, 250, 5000):
    x = plots.simulate_ou(true_kappa, true_m, true_sigma, days, seed=1)
    est = al._ar1(np.diff(np.concatenate([[0.0], x]))[:, None])      # its cumulative sum is x again
    print(f"{days:5d} days: kappa {est['kappa'][0]:6.1f} (true {true_kappa}),  m {est['m'][0]:+.3f} (true {true_m}),  "
          f"sigma_eq {est['sigma_eq'][0]:.3f} (true {true_sigma / np.sqrt(2 * true_kappa):.3f})")
''')
md("""
Short samples are not just noisy, they are **biased**: the fitted $b$ comes out too small on average, so $\\kappa$
looks too fast. Repeat the experiment 300 times:
""")
code('''
for days in (60, 250, 1000):
    k = [al._ar1(np.diff(np.concatenate([[0.0], plots.simulate_ou(true_kappa, true_m, true_sigma, days, seed=s)]))[:, None])["kappa"][0]
         for s in range(300)]
    print(f"{days:5d} days: median estimated kappa {np.nanmedian(k):5.1f}  (true {true_kappa})")
''')
md("""
*Discuss:* the paper uses 60 days anyway. Why? (Its answer: parameters drift over time, and 60 days is about one
earnings cycle.) What does the bias do to the $\\kappa > 8.4$ filter?
""")

# ------------------------------------------------------------------------------------------- 4 PCA
md("""
## 4. Factors with PCA: unsupervised learning finds the market's structure

**PCA (principal component analysis)** is unsupervised learning: no labels, just "what are the main directions
in which these 100 stocks move together?". The paper (sec. 2.1):

1. Standardise each stock's last 252 daily returns (subtract the mean, divide by the standard deviation).
2. Form their correlation matrix and find its eigenvalues and eigenvectors.
3. Each eigenvector $v^{(j)}$ defines an **eigenportfolio** holding $v^{(j)}_i / \\sigma_i$ dollars of stock $i$.
   Its returns are the factor $F_j$. Different eigenportfolios are uncorrelated by construction.

Keep enough factors to explain 55% of the variance (the paper also tries a fixed 15). The red line is the largest
eigenvalue that pure noise would produce (Marchenko-Pastur), a classic test of which factors are real.
""")
code('''
DAY = pd.Timestamp("2007-05-01")                     # the paper's own example date
t = R.index.get_loc(DAY)
block = R.iloc[t - 251: t + 1]
block = block.loc[:, block.notna().all()]
Q, vals, m = al.pca_eigenportfolios(block.to_numpy(), var_explained=0.55)
print(f"{block.shape[1]} stocks with a full year of history; {m} factors explain 55% of the variance; "
      f"the first explains {vals[0] / len(vals):.0%}")
plots.spectrum(vals, n_days=len(block), title=f"Eigenvalues of the correlation matrix, one year to {DAY.date()}");
''')
md("""
The first eigenvalue is the **market**: every stock moves with it. Only two more stand above the noise line. Which
stocks load on them? Colour by sector:
""")
code('''
weights = pd.DataFrame(Q * block.std().to_numpy()[:, None], index=block.columns)     # back to eigenvector weights
plots.eigenvector_bars(weights[1], sectors, "Eigenvector 2: which stocks move together, sorted");
plots.eigenvector_bars(weights[2], sectors, "Eigenvector 3");
''')
md("""
Nobody told PCA what a sector is, yet eigenvector 2 puts defensive stocks (utilities, staples) at one end and
cyclicals (industrials, energy) at the other. The paper calls this **coherence**: sort the stocks by their weight in
an eigenvector and neighbours tend to share an industry, so eigenportfolios behave like long-short bets between
industries.

How many factors does 55% take over time? The paper found fewer in crises (Fig. 20): when everything falls together,
one factor explains most of it.
""")
code('''
cached = {v: al.load_cached(v) for v in ["etf", "etf_trading_time", "pca_55", "pca_15"]}
n_factors = cached["pca_55"].n_factors
mkt_vol = etf_R["SPY"].rolling(20).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(n_factors.rolling(20).mean(), color=plots3.QS_NAVY, label="factors needed for 55% (20-day average)")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(mkt_vol.reindex(n_factors.index), color=plots3.QS_GOLD, lw=0.8, alpha=0.8, label="SPY volatility (20 days)")
ax.set_ylabel("factors"); ax2.set_ylabel("annualised volatility")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8); ax.set_title("Fewer factors when markets are stressed")
print(f"correlation between the two: {n_factors.corr(mkt_vol.reindex(n_factors.index)):.2f}")
''')

# ------------------------------------------------------------------------------------------- 5 ETF
md("""
## 5. Factors with sector ETFs

The simpler choice (sec. 2.3): regress each stock on **its own sector ETF** only, e.g. JPMorgan on XLF, the
financials fund. Easy to interpret, and the hedge is one tradeable fund. The cost: ETFs weight big companies
most, and a stock's true peers are not always its official sector.
""")

# ------------------------------------------------------------------------------------------- 6 s-score
md("""
## 6. From residuals to an s-score, step by step

The paper's appendix procedure, applied to JPMorgan against XLF on one date. All of it uses the last 60 trading days only.
""")
code('''
DAY = pd.Timestamp("2007-03-01")                   # <- try other dates
SYMBOL, ETF = "JPM", "XLF"
win = R[[SYMBOL]].join(etf_R[[ETF]]).loc[:DAY].iloc[-60:]
y, x = win[SYMBOL].to_numpy(), win[ETF].to_numpy()
beta, alpha = np.polyfit(x, y, 1)                  # step 1: regress the stock on its ETF
resid = y - (alpha + beta * x)
X = np.cumsum(resid)                               # step 2: the cumulative residual X_1 .. X_60 (X_60 = 0)
b, a = np.polyfit(X[:-1], X[1:], 1)                # step 3: AR(1), X_{n+1} = a + b X_n
var_z = np.var(X[1:] - (a + b * X[:-1]), ddof=1)
kappa = -np.log(b) * 252
m_level = a / (1 - b)
sigma_eq = np.sqrt(var_z / (1 - b ** 2))
s_raw = (X[-1] - m_level) / sigma_eq               # step 4: the s-score (before centring across stocks)
print(f"beta {beta:.2f}, kappa {kappa:.1f} per year (tau = {252 / kappa:.1f} days), m {m_level:+.4f}, "
      f"sigma_eq {sigma_eq:.4f}, s-score {s_raw:+.2f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].plot(win.index, X, color=plots3.QS_NAVY); ax[0].axhline(m_level, color=plots3.QS_GOLD, label="equilibrium m")
ax[0].fill_between(win.index, m_level - sigma_eq, m_level + sigma_eq, color=plots3.QS_GOLD, alpha=0.2, label="m +- sigma_eq")
ax[0].set_title(f"{SYMBOL}'s cumulative residual against {ETF}"); ax[0].legend(fontsize=8)
ax[0].tick_params(axis="x", labelrotation=30)
ax[1].scatter(X[:-1], X[1:], s=10, color=plots3.QS_NAVY)
grid = np.linspace(X.min(), X.max(), 10)
ax[1].plot(grid, a + b * grid, color=plots3.QS_RED, label=f"b = {b:.3f}")
ax[1].plot(grid, grid, color="black", lw=0.6, ls=":", label="b = 1 (random walk)")
ax[1].set_xlabel("X today"); ax[1].set_ylabel("X tomorrow"); ax[1].set_title("The AR(1) fit"); ax[1].legend(fontsize=8)
fig.tight_layout();
''')
md("""
Two more details from the paper:

* **Fast reversion only.** A stock is traded only if $\\kappa > 252/30 = 8.4$: it must revert within half the
  60-day window (equivalently $b < 0.967$), or the estimate cannot be trusted.
* **Centred means.** The level $m$ is replaced by $m$ minus the average $m$ across all stocks that day
  (eq. 18 and 22): in aggregate, stocks are assumed to be fairly priced.

How fast do residuals revert across all stocks? The paper reports (Table 3, year 2007) quartiles of 4.9, 7.5 and 11
days. Ours:
""")
code('''
taus = []
for d in R.loc["2007"].index[::5]:
    fit = al.estimate_day(R, R.index.get_loc(d), al.Config(factors="etf"), etf_R, sectors)
    taus += list(252 / fit.kappa[np.isfinite(fit.kappa)])
taus = pd.Series(taus)
print(f"mean-reversion time tau in 2007, days: 25% {taus.quantile(.25):.1f}, median {taus.median():.1f}, "
      f"75% {taus.quantile(.75):.1f};  share fast enough to trade (under 30 days): {(taus < 30).mean():.0%}")
''')

# ------------------------------------------------------------------------------------------- 7 rules
md("""
## 7. Trading rules, on one stock

Eq. 16, with the paper's thresholds (chosen on 2000 to 2004 data):

| | rule |
|---|---|
| open long | s < -1.25 |
| open short | s > +1.25 |
| close a short | s < +0.75 |
| close a long | s > -0.50 |

A long trade is "buy \\$1 of the stock, sell $\\beta$ dollars of its ETF"; a short is the reverse. The
reproduction of the paper's Figure 11 (JPMorgan against XLF, 2006 to 2007):
""")
code('''
cfg_etf = al.Config(factors="etf")
rows, state = [], 0
for d in R.loc["2006":"2007"].index:
    fit = al.estimate_day(R, R.index.get_loc(d), cfg_etf, etf_R, sectors)
    i = fit.symbols.index("JPM")
    state = al.next_state(state, fit.s[i], bool(fit.eligible[i]), cfg_etf)
    rows.append((d, fit.s[i], state))
jpm = pd.DataFrame(rows, columns=["date", "s", "position"]).set_index("date")
plots.sscore_chart(jpm["s"], cfg_etf, jpm["position"], "JPMorgan against XLF: s-score and the trades the rules take");
print(f"{(jpm['position'].diff() != 0).sum() // 2} round trips in two years; in a trade {(jpm['position'] != 0).mean():.0%} of days")
''')

# ------------------------------------------------------------------------------------------- 8 backtest
md("""
## 8. The full backtest: four versions

Every day, for every stock: estimate on the last 60 days (and 252 for PCA), update the trade states, hedge every
open trade, earn the next day's return, pay 5 bps on every dollar the book changes. Each trade is 5% of equity in
the stock plus its hedge. Four versions:

* **etf**: sector ETF hedges (sec. 5.2)
* **etf_trading_time**: the same, with returns scaled by volume (sec. 6): moves on low volume count more
* **pca_55**: as many eigenportfolios as explain 55% of the variance (sec. 5.4)
* **pca_15**: a fixed 15 eigenportfolios (sec. 5.3)

The full run takes about 3 minutes, so we load results made by the same code (`workshop4/build_cache.py`) and
re-run one year live to check them.
""")
code('''
t0 = time.time()
check = al.run(prices, al.Config(factors="etf"), etfs, sectors, start="2006-10-01", end="2007-12-31")
gap = (check.net.loc["2007"] - cached["etf"].net.loc["2007"]).abs().max()     # a quarter of warm-up, so open trades match
print(f"re-ran 2007 in {time.time() - t0:.0f}s; largest daily difference from the cache: {gap:.1e}",
      "(match)" if gap < 1e-6 else "(DIFFERENT)")
''')
md("""
As in Workshop 3, everything up to part 11 uses **2001 to 2022** only; 2023 onwards is the final test.
""")
code('''
RESEARCH_END = "2022-12-31"
research = {k: v.window(end=RESEARCH_END) for k, v in cached.items()}
table = pd.DataFrame({k: v.stats() for k, v in research.items()}).T
display(table.style.format({c: "{:.3f}" for c in table.columns if c not in ("trades", "days")}))
plots3.equity({k: v.net for k, v in research.items()}, "Avellaneda-Lee on today's S&P 100, after 5 bps costs, 2001 to 2022");
''')
md("""
**How to read this.** All four have a positive Sharpe ratio **before** costs: the residuals really do mean-revert.
After costs only the ETF versions stay positive, and part 9 shows how much of that comes from the early 2000s.
Unlike the paper, **ETF beats PCA** here. Possible reasons to discuss:

* **Universe size.** PCA on 100 mega-caps is not PCA on 1,000 stocks: fewer, noisier factors, and each
  eigenportfolio hedge trades every stock in the universe, every day.
* **Turnover.** Compare the `daily turnover` column: the PCA books change more each day.
* **Period.** The paper's best years (1997 to 2002) are mostly outside our data.
""")

# ------------------------------------------------------------------------------------------- 9 periods
md("""
## 9. When did it work? Periods, and August 2007

The paper split its sample at 2003 and found the strategies much weaker afterwards. Same split here:
""")
code('''
cuts = {"2001-2002": ("2001", "2002-12-31"), "2003-2007": ("2003", "2007-12-31"), "2008-2009 crisis": ("2008", "2009-12-31"),
        "2010-2019": ("2010", "2019-12-31"), "2020-2022": ("2020", "2022-12-31")}
by = pd.DataFrame({name: {k: perf_stats(v.net.loc[a:b])["Sharpe"] for k, v in research.items()}
                   for name, (a, b) in cuts.items()})
display(by.round(2).style.set_caption("Sharpe ratio after costs, by period"))
''')
md("""
The paper's pattern again, sharper: strong in 2001 to 2002, then negative after costs from 2003 to 2007 for
every version except trading time. *Discuss:* why would a published, simple, profitable strategy stop working?
Who else was running it by 2003?
""")
md("""
**August 2007, the "quant quake"** (paper sec. 7). Many funds ran similar mean-reversion books; when some had to
sell quickly, the crowded trades all moved against everyone at once. The paper found a sudden drawdown in early
August followed by a recovery in about ten days, smaller for PCA (5%) than for ETF strategies (10%) at their 2+2
leverage (Khandani and Lo, 2007, first reproduced this drawdown with simpler contrarian strategies). Ours, with
about 34 trades open at 5% each, roughly 0.85+0.85 in stocks: under half the paper's leverage.
""")
code('''
fig, ax = plt.subplots(figsize=(11, 3.6))
for k, v in cached.items():
    r = v.net.loc["2007-07-15":"2007-09-15"]
    ax.plot((1 + r).cumprod() - 1, label=k)
ax.axvspan(pd.Timestamp("2007-08-06"), pd.Timestamp("2007-08-10"), color=plots3.QS_RED, alpha=0.1, label="6 to 10 August")
ax.set_title("Cumulative return, mid-July to mid-September 2007"); ax.legend(fontsize=8);
''')
md("""
Our drawdowns are small, and here PCA was not the more resilient version. *Discuss:* why might a book of 100
mega-caps be hit less by a fund unwinding than the paper's 1,000-stock book?
""")

# ------------------------------------------------------------------------------------------- 10 vs ML
md("""
## 10. Model against machine: Workshop 3's ML strategy

Workshop 3's ensemble (random forest, boosting and a neural net on 31 lagged returns) also bets on reversal, on the
same stocks and dates. Load its cached predictions and backtest them the same way as last week.
""")
code('''
from quantsoc_ml import statarb as sa
ml_pred = sa.load_cached_predictions()
ml_panel = sa.make_panel(sa.wide(sa.load_prices()))
ml = sa.long_short(ml_pred["ensemble"], ml_panel["fwd"], k=10).window(end=RESEARCH_END)
# Workshop 3 labels each return by the day the signal was made; here a return is labelled by the day it is
# earned (the next trading day). Shift Workshop 3's series by one day so the two line up.
ml_net = ml.net.set_axis(R.index[R.index.get_indexer(ml.net.index) + 1]).loc[:RESEARCH_END]
both = pd.DataFrame({"Workshop 3 ML (after costs)": ml_net, "Avellaneda-Lee ETF (after costs)": research["etf"].net}).dropna()
display(pd.DataFrame({c: perf_stats(both[c]) for c in both}).T.round(3))
print(f"correlation of their daily returns: {both.corr().iloc[0, 1]:.2f}")
blend = both.mean(axis=1)
print(f"a 50/50 blend: Sharpe {perf_stats(blend)['Sharpe']:.2f}")
''')
md("""
*Discuss:* two strategies built on the same idea (reversal) by very different routes. Is their correlation high or
low? Before costs, both work. After costs, the ML strategy loses heavily and the ETF version only breaks even over
these dates (they start in December 2003, after its best years). Why the difference? Hint: compare their daily
turnover.
""")

# ------------------------------------------------------------------------------------------- 11 meta
md("""
## 11. Can ML pick the good trades? Meta-labelling

The model decides **when** to trade. Machine learning can help decide **which** of those trades to take. This is
**meta-labelling** (Lopez de Prado 2018, *Advances in Financial Machine Learning*, ch. 3): for every trade the
model opened, label it 1 if it made money and 0 if not, and train a classifier on what was known at entry: the
s-score, the speed of mean reversion, the residual volatility, how well the factor explained the stock, and so on.

Walk forward: train on trades that **closed** before each year, predict the trades that **open** in that year.
""")
code('''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

trades = research["etf"].trades.dropna(subset=["kappa"]).copy()    # trades opened and closed by the end of 2022
trades["abs_s"] = trades["s_entry"].abs()
trades["tau"] = 252 / trades["kappa"]
trades["vol_at_entry"] = mkt_vol.reindex(trades["entry"]).to_numpy()
features = ["abs_s", "tau", "sigma_eq", "r2", "alpha", "side", "vol_at_entry"]
trades["win"] = (trades["pnl"] > 0).astype(int)
print(f"{len(trades):,} trades; {trades['win'].mean():.0%} made money before costs")

rows = []
for year in range(2005, 2023):
    train = trades[trades["exit"] < f"{year}-01-01"]
    test = trades[trades["entry"].dt.year == year]
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0)
    clf.fit(train[features], train["win"])
    p = clf.predict_proba(test[features])[:, 1]
    keep = p >= np.median(p)
    rows.append({"year": year, "AUC": roc_auc_score(test["win"], p),
                 "avg pnl, kept half": test["pnl"][keep].mean(), "avg pnl, dropped half": test["pnl"][~keep].mean()})
meta = pd.DataFrame(rows).set_index("year")
display(meta.round(4))
print(f"average AUC {meta['AUC'].mean():.3f}; kept half beat dropped half in {(meta.iloc[:, 1] > meta.iloc[:, 2]).sum()} of {len(meta)} years")
''')
md("""
`pnl` is each trade's hedged return per \\$1, before costs. An AUC above 0.5 means the classifier ranks winning
trades above losing ones more often than not. *Discuss:* is the edge big enough to matter after costs? What other
information at entry might help (volume, earnings dates, the s-score of the sector)?
""")

# ------------------------------------------------------------------------------------------- 12 DSR
md("""
## 12. How many backtests did we run?

We compared four versions and picked the best. Workshop 3's deflated Sharpe ratio asks whether the winner would
still look good after allowing for that choice (Bailey and Lopez de Prado 2014).
""")
code('''
sharpes = pd.Series({k: rs.sharpe(v.net) for k, v in research.items()})
best = sharpes.idxmax()
display(pd.Series(rs.deflated_sharpe(research[best].net, sharpes.values), name=f"best: {best}").round(4).to_frame())
''')
md("""
Four trials is a small number, but we also inherited the paper's thresholds, which were tuned on 2000 to 2004 data,
partly inside our sample. *Discuss:* how many trials did the paper's authors effectively run?
""")

# ------------------------------------------------------------------------------------------- 13 your turn
md("""
## 13. Your turn

Ideas, roughly from easiest to hardest. Each one is a `Config` change; a run over 2001 to 2022 takes under a
minute.

1. **Thresholds.** Try `s_open=1.5` or `2.0`: fewer, stronger trades and lower costs.
2. **Window.** `window=90`: steadier estimates, slower signals.
3. **Hedge at entry.** `rehedge="entry"` keeps each trade's opening hedge instead of re-hedging daily. (We measured
   it before the workshop: it barely changes turnover. The cost is the trading itself.)
4. **The drift term.** The paper's modified s-score (sec. 4.2) subtracts $\\alpha / (\\kappa\\, \\sigma_{eq})$, with
   $\\alpha$ and $\\kappa$ in the same time units. `fit.alpha` is per day and `fit.kappa` per year, so use
   `252 * fit.alpha / (fit.kappa * fit.sigma_eq)`. Does it help?
5. **Meta-label filter.** Only take trades the classifier in part 11 likes.

Test your final choice on 2023 onwards **once**.
""")
code('''
my_cfg = al.Config(factors="etf", s_open=1.5)                       # <- change this
t0 = time.time()
mine = al.run(prices, my_cfg, etfs, sectors, start=research["etf"].net.index[0], end=RESEARCH_END)
print(f"{time.time() - t0:.0f}s")
display(pd.DataFrame({"paper's settings": research["etf"].stats(), "yours": mine.stats()}).T.round(3))
''')
code('''
# The final test: run once, when you have decided.
final = {"paper's settings (etf)": cached["etf"].window(start="2023-01-01"),
         "yours": al.run(prices, my_cfg, etfs, sectors, start="2023-01-01")}
display(pd.DataFrame({k: v.stats() for k, v in final.items()}).T.round(3))
plots3.equity({k: v.net for k, v in final.items()}, "Final test, 2023 to today, after costs", log=False);
''')

# ------------------------------------------------------------------------------------------- 14 live
md("""
## 14. Going live

The dashboard runs the **ETF version** (hedges are single, tradeable funds) on your Alpaca **paper** account. It is
a stateful strategy: whether a stock is held depends on yesterday, so the dashboard reads your open trades from
the account's stock positions. Use a paper account for this strategy only.

```bash
cp .env.example .env          # then paste your Alpaca PAPER keys into .env
pip install -r requirements.txt streamlit
streamlit run workshop4/dashboard.py
```

Today's s-scores for every stock, from the snapshot (offline, no account needed):
""")
code('''
fit, states, book = al.live_signal(prices.iloc[-80:], held={}, cfg=al.Config(factors="etf"),
                                   etf_prices=etfs.iloc[-80:], sectors=sectors)
today = pd.DataFrame({"s-score": fit.s, "tau (days)": 252 / fit.kappa, "beta": fit.beta[:, 0], "hedge": fit.hedge_etf},
                     index=fit.symbols).sort_values("s-score")
print(f"as of {prices.index[-1].date()}: {sum(v > 0 for v in states.values())} longs and "
      f"{sum(v < 0 for v in states.values())} shorts would open from a flat account")
display(pd.concat([today.head(5), today.tail(5)]).round(2))
''')
md("""
## Recap

* A stock's own move, after removing its factors, behaves like an Ornstein-Uhlenbeck process with a half-life of
  about a week: that reproduces the paper.
* PCA finds sector structure without being told; with 100 stocks, sector ETFs made the better hedge.
* Before costs the signal is real; after costs it is thin, and it has faded since the paper's era.
* The model and Workshop 3's machine learning bet on the same idea in different ways; ML can also sit on top of the
  model and choose which trades to take.
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
