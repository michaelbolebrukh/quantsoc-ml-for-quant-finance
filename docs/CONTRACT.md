# Build contract: Workshop 2, "Machine Learning for Quantitative Finance" (Part 1)

Every agent builds against this file. It fixes the package API, the data shape, the timing rule, the
notebook plan and the slide map so parallel work composes. Change it only through the integrator.

## Audience and format (fixed by the operator)
- 60 minutes, presenter-led from the deck; participants run the notebook in parallel (Colab first, local second).
- Zero background assumed: no ML, no finance, basic Python only. Define every term the first time.
- Every coding task is COPY-AND-CHANGE: a worked example cell sits directly above, the task cell has the same
  structure with one or two things to change, a checker prints `[ok]` or one plain sentence. Never more than
  three lines of participant code. `ex.hint(n)`, `ex.show_solution(n)`, `ex.use_reference(n)` as in Workshop 1.
- The five tasks set the five knobs of `my_strategy`; the final backtest runs the participant's own knobs.
- Ends with: backtest of their strategy, export ZIP, `run_paper.py --preview` demo (Alpaca paper, preview only).
- Part 2 next week covers purged CV, triple-barrier/meta-labelling, feature importance, deflated Sharpe.
- Workshop 1 (read-only reference for conventions): /home/user/michaelbolebrukh/quantsoc-first-quant-strategy

## Data (built, do not change)
`data/prices.csv`, long format, columns `date,symbol,open,high,low,close,adj_close,volume`; 40 US large caps,
2000-01-03 to 2019-12-30, daily, Yahoo snapshot (see fetch_data.py). Use `adj_close` for returns.
Survivorship-biased by construction; that is a teaching point (slide 18), never hidden.

## Timing rule (the one rule the whole workshop teaches)
Features for a decision on day t use data with date <= t. The target is the forward 5-trading-day return
`adj_close[t+5] / adj_close[t] - 1` (per symbol). Rebalance happens at the close of day t for simplicity, and
the notebook SAYS this is a simplification (a real system fills at the next open; Workshop 1 taught that).
Every feature function is pure, vectorised with pandas, and never reads a row with a later date.

## Package `quantsoc_ml` (Python 3.11+, numpy, pandas, scikit-learn, matplotlib, joblib; ipywidgets optional)
```
data.py       load_prices(path=None) -> long DataFrame (parsed dates)
              to_wide(long, column="adj_close") -> DataFrame date x symbol
features.py   FEATURE_NAMES: list[str]  (the default set below)
              build_features(prices_wide, volume_wide) -> DataFrame with MultiIndex (date, symbol), one column per feature
              forward_return(prices_wide, horizon=5) -> DataFrame date x symbol (the target, NaN at the end)
              make_dataset(features, target) -> DataFrame rows=(date,symbol), columns=FEATURE_NAMES + ["target"], NaNs dropped
              # Default features (all computed from adj_close / volume up to and including t):
              # ret_1, ret_5, ret_21, ret_63 (past returns), vol_21 (std of daily returns, 21d),
              # vol_63, volume_z (today's volume vs 63-day mean, in std units), hi52 (adj_close / 252-day max - 1)
              # Helper: leaky_feature(prices_wide) -> a deliberately leaky feature (uses t+1) for the demo, named "LEAKY_ret_next"
splits.py     time_split(dataset, train_end, test_start) -> (train_df, test_df); embargo of >= horizon rows between them
              split_report(train_df, test_df) -> small DataFrame for display
models.py     ZeroModel (predicts 0), fit_linear(X, y) -> sklearn Pipeline(StandardScaler, Ridge(alpha=1.0)),
              fit_forest(X, y, n_estimators=200, max_depth=6, random_state=0) -> RandomForestRegressor (n_jobs=-1)
              xy(df) -> (X ndarray, y ndarray) using FEATURE_NAMES (or df.attrs["features"] when set)
evaluate.py   score(y_true, y_pred) -> dict(mse, r2_vs_zero, ic (Spearman), hit_rate)
              compare_table(dict[name -> y_pred], y_true) -> DataFrame
              decile_table(y_pred, y_true, n=10) -> DataFrame(mean target per decile)
              noise_experiment(train_df, test_df, n_noise=30, seed=0) -> DataFrame(rows: real / real+noise; cols: train_r2, test_r2)
              permutation_importance_table(model, X, y, feature_names, n_repeats=5, seed=0) -> DataFrame
portfolio.py  StrategyConfig(dataclass): momentum_lookback=21, extra_feature="vol_63", train_end="2014-12-31",
              test_start="2015-01-12", n_estimators=200, max_depth=6, n_long=4, n_short=4, rebalance_days=5, cost_bps=5
              weights_from_forecasts(pred: Series indexed by symbol, n_long, n_short) -> Series (long +1/n_long, short -1/n_short, else 0)
backtest.py   run_backtest(pred_panel: DataFrame date x symbol, returns_wide: DataFrame date x symbol of DAILY adj returns,
              config) -> BacktestResult(equity: Series, daily_returns: Series, turnover: Series, weights: DataFrame, stats: dict)
              stats = total_return, annual_return, annual_vol, sharpe, max_drawdown, avg_turnover, n_rebalances
              benchmark_equal_weight(returns_wide, dates) -> Series (same cost model, for comparison)
viz.py        matplotlib helpers, every figure titled with labelled axes and units: price_panel, feature_hist,
              split_timeline, decile_bar, train_vs_test_bars, importance_bar, equity_curve (strategy vs benchmark),
              drawdown_area. PALETTE from Workshop 1 viz.py. Return the Figure.
exercises.py  ExerciseTracker with keys "1".."5" (same design as Workshop 1: attempt/check/hint/show_solution/
              use_reference/require/summary; QSW_AUTO_REFERENCE=1 falls back to references for CI)
solutions.py  reference implementations + REFERENCE_STRATEGY (a StrategyConfig)
export.py     export_bundle(model, config, out_dir) -> writes strategy.py (predict + weights functions), model.joblib,
              config.json, feature_names.json, run_paper.py copy, README_EXPORT.md; zip_bundle(out_dir, zip_path)
paper.py      Alpaca paper runner internals: fetch_daily_bars(symbols, days, client), build_latest_features(...),
              propose_orders(weights, account_equity, positions, prices) -> DataFrame; MockBroker with scripted
              responses (one reject, one partial fill); run(mode="preview"|"submit"); "submit" ONLY with --submit flag,
              paper endpoint only, truthful mode banner before anything (Workshop 1 notebook_support.mode_banner style).
              Credentials: Colab secrets -> env -> .env; names APCA_API_KEY_ID / APCA_API_SECRET_KEY; never printed.
notebook_support.py  is_colab/is_kaggle, setup_ipython (friendly ExerciseIncomplete), environment_report, runtime_loss_notice,
              export_checkpoint (Colab download), reading_box(citation) -> HTML callout for the "Reading" boxes
```

## The five tasks (knobs of `my_strategy`)
| Task | Worked example above it | Participant changes | Knob set |
|---|---|---|---|
| 1 | `ret_5 = past_return(prices, 5)` | `5` -> `21`, name `ret_21` | momentum_lookback |
| 2 | `vol_21 = rolling_vol(returns, 21)` | `21` -> `63`, name `vol_63` | extra_feature |
| 3 | `train, test = time_split(ds, "2012-12-31", "2013-01-10")` | the two dates | train_end/test_start |
| 4 | `fit_forest(X, y, n_estimators=100, max_depth=4)` | the two numbers | n_estimators/max_depth |
| 5 | `weights_from_forecasts(pred, n_long=4, n_short=4)` | the two counts | n_long/n_short |
Checkers accept ANY sensible value (not one right answer): e.g. Task 1 passes if the column equals
past_return with the lookback the participant chose and lookback != 5.

## Notebook sections (headings must match slide titles exactly)
0 Setup · 1 The data · 2 Features, the target and the no-peeking rule · 3 A baseline and a fair test ·
4 A real ML model · 5 Overfitting, live · 6 From forecasts to a portfolio · 7 Your strategy's backtest ·
8 Export and paper trading
Each section opens with a "Reading" box (see slide map). Every code cell runs in < 10 s on Colab CPU; the
full notebook runs in < 4 min from a clean kernel with QSW_AUTO_REFERENCE=1.

## Slide map (20 slides) and readings (verified 2026-10-01; cite exactly as written)
1 Title · 2 What we're building · 3 Why most ML in finance fails [López de Prado, "The 10 Reasons Most
Machine Learning Funds Fail", J. Portfolio Management 44(6), 2018, 120-133] · 4 The data · 5 Where it came
from (data card, survivorship flagged) · 6 Past vs future: the target · 7 Features from the literature
[Jegadeesh & Titman, J. Finance 48(1), 1993, 65-91; Jegadeesh, J. Finance 45(3), 1990, 881-898] · 8 Leakage,
live [Kaufman, Rosset, Perlich, Stitelman, ACM TKDD 6(4), 2012, art. 15] · 9 Baseline: predict zero ·
10 Split by time, never at random [López de Prado, Advances in Financial Machine Learning, Wiley 2018, ch. 7
§7.2] · 11 What a tiny edge looks like [Gu, Kelly, Xiu, Review of Financial Studies 33(5), 2020, 2223-2273] ·
12 A decision tree in one picture · 13 A forest [Breiman, Machine Learning 45(1), 2001, 5-32] · 14 Reading the
decile plot [Krauss, Do, Huck, European J. Operational Research 259(2), 2017, 689-702] · 15 Overfitting, live ·
16 Trying many things is the problem [Bailey, Borwein, López de Prado, Zhu, Notices of the AMS 61(5), 2014,
458-471] · 17 Forecast to portfolio, costs [AFML ch. 11] · 18 Survivorship [Shumway, J. Finance 52(1), 1997,
327-340] · 19 Your backtest, read honestly [Arnott, Harvey, Markowitz, J. Financial Data Science 1(1), 2019,
64-74] · 20 Export, paper trading, Part 2, QR codes (Instagram, LinkedIn: PLACEHOLDER URLs until supplied;
website https://quant-soc.com)
Run markers on slides 4,6,8,9,10,13,15,17,19,20: "▶ Notebook: run section N".

## Brand (from the QuantSoc design source of truth)
Brand blue `#3b82f6` (Tailwind blue-500, hsl 217 91% 60%); darker `#1d4ed8`-ish (brand-700 hsl 224 76% 44%);
near-black ink; white ground. Wordmark text "QuantSoc"; society name "Quantitative Finance Society";
site https://quant-soc.com. No sponsor logos in this deck.

## Style rules for all prose (notebook, slides, docs)
British English. Short sentences. No em-dashes or en-dashes (use commas, full stops, or colons). Define
every term on first use. Label everything truthfully (synthetic vs real, preview vs submit, reference vs own).

## Full citations (use these verbatim in notebook Reference boxes and deck Reference footers; verified 2026-10-01)
- López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." The Journal of Portfolio Management, 44(6), 120-133. https://doi.org/10.3905/jpm.2018.44.6.120 (free copy: SSRN 3104816)
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 2 "Financial Data Structures"; Chapter 7 "Cross-Validation in Finance"; Chapter 11 "The Dangers of Backtesting".
- Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." The Journal of Finance, 48(1), 65-91.
- Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security Returns." The Journal of Finance, 45(3), 881-898.
- Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). "Leakage in Data Mining: Formulation, Detection, and Avoidance." ACM Transactions on Knowledge Discovery from Data, 6(4), Article 15.
- Gu, S., Kelly, B. and Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." The Review of Financial Studies, 33(5), 2223-2273.
- Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
- Krauss, C., Do, X. A. and Huck, N. (2017). "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500." European Journal of Operational Research, 259(2), 689-702.
- Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." Notices of the AMS, 61(5), 458-471. (open access at ams.org)
- Shumway, T. (1997). "The Delisting Bias in CRSP Data." The Journal of Finance, 52(1), 327-340.
- Arnott, R., Harvey, C. R. and Markowitz, H. (2019). "A Backtesting Protocol in the Era of Machine Learning." The Journal of Financial Data Science, 1(1), 64-74.
Part 2 (mention only): Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." The Journal of Portfolio Management, 40(5), 94-107; AFML ch. 3, 7 (§7.4), 8, 14; Harvey, Liu & Zhu (2016), Review of Financial Studies 29(1), 5-68.

## Decision D7 (integrator, 2026-10-01 22:05): final research design and the honest numbers
Pre-specified exploration (scratchpad/edge_explore.py, grid fixed before results: features raw vs per-date rank,
target raw vs demeaned, periods 2008-10 / 2011-14 / 2015-19, linear vs forest) and a pre-specified cost check
(scratchpad/edge_backtest.py) decided:
- Features are RANK-NORMALISED per date: `features.rank_normalise(feats)` = groupby(date).rank(pct=True) - 0.5,
  applied after build_features and before make_dataset. Reason: on the honest test period the forest's daily IC
  t-stat goes from 0.07 (raw) to about 1.5 (rank); positive-IC days 52%; top-minus-bottom quintile spread about
  0.1% per 5-day horizon. This is what Gu, Kelly and Xiu do, and the notebook says so in one line.
- Target stays the raw forward 5-day return (demeaning does not change rank correlation; one fewer concept).
- Test period stays 2015-01-12 to 2019-12-20 (pre-stated, the last five years). train_end 2014-12-31.
- Portfolio: 4 long / 4 short of 40 (top and bottom decile), costs 5 bps per unit traded.
- Default rebalance_days = 21 (monthly). Task 5 lets participants pick 5, 10 or 21 and see the cost effect.
Measured (forest 200 trees, depth 6, leaf 50, rank features), test 2015-2019:
| rebalance | cost | total | annual | vol | sharpe | maxDD | turnover/rebalance |
| 5  | 5 bps | -31.3% | -7.3% | 15.7% | -0.41 | -36.3% | 3.02 |
| 5  | 0 bps | +0.2%  | +0.05% | 15.7% | 0.08 | -18.3% | 3.02 |
| 10 | 5 bps | -47.2% | -12.1% | 16.1% | -0.72 | -53.3% | 3.21 |
| 21 | 5 bps | +5.2%  | +1.0% | 17.6% | 0.15 | -39.6% | 3.36 |
| 21 | 0 bps | +16.3% | +3.1% | 17.7% | 0.26 | -36.3% | 3.36 |
Equal-weight long-only "the market" over the same window: +85.7%, Sharpe 1.01, maxDD -18%. The notebook must
explain that a dollar-neutral long/short book is not trying to beat the market: its return is what the forecasts
add on their own, and the market line is shown for scale, not as the benchmark to beat.
Narrative the notebook and slides 11, 14, 15, 19 must carry: the edge is real (IC positive on more days than not,
top decile beats bottom decile) and tiny (a few basis points per week); trading costs are the same size, so the
rebalance frequency decides whether anything is left; nobody should expect their own backtest to look good, and a
good-looking one deserves suspicion first. Slide 15 (noise features): at the reference settings the forest is
regularised enough that noise features barely move the test score (real 4.30%/0.66% train/test r2 vs 4.38%/0.66%
with 30 noise features); the live demo therefore uses the UNBRAKED forest (no depth limit, leaf 5, 50 trees, 20k
rows): train 57.8% / test -3.5%, with noise train 66.8% / test -2.2%. That is the overfitting picture.
Hit rate: the forest predicts a positive return on 99.99% of test rows, so hit rate equals the share of up-days
(55.7%), identical to a constant forecast. Report hit rate with that caveat or not at all.
Addendum to D7 (22:20, from the core agent's measured rerun): daily IC for the rank-feature forest is +0.0117 (naive t 1.90, positive on 51.9% of days; the t ignores overlapping 5-day targets and so overstates the evidence). Gross returns across rebalance frequencies are NOISY: 5-day +0.2%, 10-day -35.5%, 21-day +16.3% before costs. So the clean "costs decide" demonstration is the 5-day case (flat gross, -31% net); do NOT present the frequency comparison as a cost story. rank_normalise is (rank - 0.5)/count - 0.5 (per-date mean exactly 0), a constant shift from pct rank that changes nothing downstream.

### Addendum 2 (after the export/paper falsification, 2026-10-01)

`paper.run` and `run_paper.py` refuse a `--symbols` list that differs from the bundle's training
universe unless `--allow-universe-change` is passed (ranked features make every forecast depend on
the table). A held symbol with no forecast on the day is left alone and named in the notes.
`min_notional` applies to a whole move, so a crossing move is never collapsed into one order.
Export refuses a feature window longer than the shipped history, a universe with no spare symbol
beyond longs plus shorts, a key-shaped string anywhere in the bundle (binary files and UTF-16
included), and re-exporting over a folder that holds a `.env`.

### Addendum 3 (after the notebook falsification, 2026-10-01)

Section 7 refits the participant's forest on the same stratified quarter of the training rows that
section 4 uses (speed: 3 s reference, 10 s at 500 trees and depth 12), and rebuilds the feature table
from `my_strategy` first, so a late change to Task 1 or 2 propagates. The deck's F1 and F3 figures are
drawn from that same quarter-sample forest. Reference-knob numbers the notebook therefore prints:
21-day book +17.2% before costs, +6.4% after, Sharpe 0.16, max drawdown -53.8%, turnover 3.22;
5-day book +22.9% before, -15.8% after; 10-day +18.7% before, -2.3% after; market +86.3% (equal
weight, long only, rebalanced on the strategy's days). The full-training-set numbers in the D7
addendum remain the research record in `docs/core-results.md`; "costs decide" still holds, the
gross differences between frequencies remain path noise.

### Addendum 4 (round 5, 2026-10-02)

Main deck: 23 slides, about 71 minutes. Slide 3 is the textbook-vs-finance table; "In finance" chips on later
slides name its rows; slides 7 to 10 cover the feature universe, momentum and reversal, the eight features drawn,
and making features usable (ranking, z-scores, interactions, fractional differencing, point-in-time). Bonus section:
slides 24 to 45, about 25 minutes, on core principles of learning theory, with its own chapter strip and "Seen in
Part 1" chips. Its figures are computed by `deck/make_bonus_figures.py`; every number is in `docs/bonus-results.md`.
Two honest non-textbook results: on our data the best single tree is depth 1 and no depth beats a constant
forecast (no U-curve), and the double-descent second dip does not beat the best small model. New references were
verified at source on 2026-10-02 (22 of 22). QR codes encode https://quant-soc.com,
https://www.instagram.com/quantsoc_edinburgh/ and https://www.linkedin.com/company/quantsociety (decoded).
