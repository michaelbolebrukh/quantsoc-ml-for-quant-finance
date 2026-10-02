# Core results: the reference strategy, measured

Every number in this file is MEASURED: produced on 2026-10-01 by running the package on
`data/prices.csv.gz`, never estimated. Environment: Python 3.11.15, numpy 2.4.6, pandas 3.0.6,
scikit-learn 1.9.1, scipy 1.17.1, matplotlib 3.11.2, 4 CPU cores. The run used a scratch
runner outside the build tree (`run_core_results.py` in the session scratchpad); every call
in it is a public function of `quantsoc_ml`.

Headline, plainly: the forest does NOT show a usable out-of-sample edge. Its pooled IC is
slightly positive, but its daily cross-sectional IC is indistinguishable from zero, it does
not beat a constant forecast on r2, and the long/short backtest loses money after costs.
Nothing was tuned to change that.

Two designs are measured here. The sections up to "Timings" use RAW features and weekly
rebalancing (the first reference config). The last section, "Rank-normalised features (the
design the notebook uses)", is contract decision D7: per-date ranked features and a 21-day
rebalance, which is now the StrategyConfig default.

## Reference config (raw features, first design)

momentum_lookback 21, extra_feature vol_63 (so the features are exactly FEATURE_NAMES:
ret_1, ret_5, ret_21, ret_63, vol_21, vol_63, volume_z, hi52), train_end 2014-12-31,
test_start 2015-01-12, forest 200 trees, max_depth 6, min_samples_leaf 50, 4 long and 4
short, rebalance every 5 trading days, cost 5 bps per unit of weight traded.

## Dataset and split

| item | value |
|---|---|
| panel rows (5,030 dates x 40 symbols) | 201,200 |
| dataset rows after dropping incomplete rows | 190,960 |
| train rows (2000-12-29 to 2014-12-31, 3,522 trading days) | 140,880 |
| test rows (2015-01-12 to 2019-12-20, 1,246 trading days) | 49,840 |
| earliest test_start the 5-day embargo allows | 2015-01-08 |

The first 252 trading days go to the 52-week-high warm-up and the last 5 have no target, so training starts on 2000-12-29.

## Test scores (49,840 rows)

| model | mse | r2_vs_zero | ic (pooled Spearman) | hit_rate |
|---|---|---|---|---|
| zero | 0.000983 | 0.00% | n/a (constant) | n/a (no direction) |
| constant = training mean (+0.245% per 5 days) | 0.000976 | +0.73% | n/a (constant) | 55.7% |
| linear (StandardScaler + Ridge alpha 1) | 0.000980 | +0.31% | +0.031 | 53.2% |
| forest | 0.000977 | +0.66% | +0.023 | 55.7% |

Neither model beats the constant forecast that only knows "shares rose on average", so their positive r2_vs_zero is the market's drift, not stock picking.

The forest forecasts a positive return on 99.986% of test rows (measured), so its 55.7% hit rate is simply the share of up-moves in the test set (55.67% of the rows that moved, 55.54% of all rows, measured), the same figure a constant positive forecast scores to one decimal (55.66% vs 55.67%): hit rate here shows no skill.

Training scores for comparison: linear r2 +1.16%, ic +0.057; forest r2 +4.30%, ic +0.074.

| model | mean daily cross-sectional IC (test) | std | days | naive t-stat | days with IC > 0 |
|---|---|---|---|---|---|
| linear | +0.0035 | 0.221 | 1,246 | 0.56 | 49.5% |
| forest | +0.0019 | 0.224 | 1,246 | 0.30 | 49.4% |

Ranking stocks against each other on the same day, which is what the long/short book trades, the forest is right on 49.4% of days: no measurable edge.

## Forest decile table (test, pooled, 4,984 rows per decile)

| decile | mean forecast | mean realised 5-day return | standard error |
|---|---|---|---|
| 1 (lowest) | 0.087% | 0.315% | 0.048% |
| 2 | 0.106% | 0.303% | 0.043% |
| 3 | 0.113% | 0.013% | 0.041% |
| 4 | 0.114% | 0.291% | 0.038% |
| 5 | 0.123% | 0.146% | 0.041% |
| 6 | 0.175% | 0.169% | 0.040% |
| 7 | 0.248% | 0.328% | 0.042% |
| 8 | 0.276% | 0.189% | 0.042% |
| 9 | 0.339% | 0.340% | 0.047% |
| 10 (highest) | 0.502% | 0.600% | 0.057% |

Only the top decile stands out and the middle is not monotone; pooled deciles mix dates, so this mostly reflects which weeks were good for everyone, not which stocks.

## Noise experiment (30 columns of N(0,1) noise added, same forest settings)

| features | train r2_vs_zero | test r2_vs_zero |
|---|---|---|
| real | +4.30% | +0.66% |
| real + 30 noise | +4.38% | +0.66% |

With max_depth 6 and min_samples_leaf 50 the forest barely uses the noise, so at the reference settings this experiment shows almost nothing; the brakes work.

Extra measurement for the overfitting slide (20,000 randomly drawn training rows, 50 trees, same test set):

| settings | features | train r2_vs_zero | test r2_vs_zero | runtime |
|---|---|---|---|---|
| max_depth 6, leaf 50 | real | +4.73% | +0.56% | 5.8 s for both rows |
| max_depth 6, leaf 50 | real + 30 noise | +5.51% | +0.59% | |
| no depth limit, leaf 5 | real | +57.8% | -3.49% | 19.8 s for both rows |
| no depth limit, leaf 5 | real + 30 noise | +66.8% | -2.15% | |

Unbraked trees fit 58% of the training variance and then do worse than predicting zero on the test, which is the overfitting picture the slide needs; adding noise raises the training fit further.

## Permutation importance (forest, 10,000 random test rows, 5 repeats)

| feature | rise in test MSE when shuffled | std |
|---|---|---|
| ret_5 | +8.6e-07 | 5.9e-07 |
| volume_z | +9.2e-08 | 1.1e-07 |
| vol_21 | -2.8e-08 | 3.5e-09 |
| hi52 | -7.0e-08 | 1.6e-08 |
| vol_63 | -1.5e-07 | 1.4e-08 |
| ret_63 | -2.0e-07 | 9.2e-08 |
| ret_1 | -2.1e-07 | 6.8e-08 |
| ret_21 | -7.5e-07 | 2.8e-07 |

Only ret_5 (short-term reversal) helps on the test set by more than one standard deviation; shuffling most other features makes the test error slightly smaller, so the forest learned patterns there that did not persist.

## Backtest, 2015-01-12 to 2019-12-20 (1,246 trading days, 250 rebalances)

| stat | forest strategy (4 long / 4 short) | linear strategy (same rules) | equal-weight benchmark (long only) |
|---|---|---|---|
| total_return | -29.13% | -28.14% | +85.68% |
| annual_return (252 days) | -6.73% | -6.47% | +13.33% |
| annual_vol | 16.83% | 16.25% | 13.25% |
| sharpe (mean daily / std daily x sqrt(252), 0% risk-free) | -0.33 | -0.33 | 1.01 |
| max_drawdown | -42.61% | -32.96% | -18.01% |
| avg_turnover per rebalance (sum of abs weight changes, maximum 4) | 3.43 | 2.87 | 0.022 |
| n_rebalances | 250 | 250 | 250 |
| blown_up (equity reached 0) | no | no | no |

The forest strategy loses money after costs, and the benchmark is not a fair bar to beat either way: it is long only in a rising, survivorship-biased market (every stock here survived to 2019).

Same forest strategy at 0 bps cost: total return +8.88%, annual +1.73%, Sharpe 0.19, max drawdown -22.75%. Turnover of 3.43 per rebalance at 5 bps costs about 0.17% per rebalance, roughly 8.6% a year, which is more than the strategy earns before costs.

## Random split vs time split (the "random splits lie" slide)

Shuffled 75/25 split with sklearn `train_test_split` (seed 0): 143,220 train rows, 47,740 test rows. Same models and settings.

| test set | model | r2_vs_zero | ic (pooled) | hit_rate | daily cross-sectional IC | decile 10 minus decile 1 |
|---|---|---|---|---|---|---|
| time split (honest) | linear | +0.31% | +0.031 | 53.2% | +0.0035 | |
| random split | linear | +0.73% | +0.050 | 53.2% | | |
| time split (honest) | forest | +0.66% | +0.023 | 55.7% | +0.0019 | +0.29% |
| random split | forest | +1.41% | +0.058 | 54.3% | +0.0190 | +1.18% |

The hit-rate column is not evidence either way: the random-split forest forecasts a positive return on 98.3% of its test rows (measured) and its 54.3% hit rate sits next to an up-move share of 54.1%.

On a shuffled split the same forest looks twice as good on r2, 2.5 times on pooled IC, 10 times on daily IC and 4 times on the decile spread, because neighbouring days share 4 of their 5 target days and land on both sides of the split.

## Timings (4 cores)

| step | time |
|---|---|
| load prices, build features and dataset, split | about 0.9 s |
| linear fit (140,880 rows) | 0.95 s |
| forest fit (140,880 rows, 8 features, 200 trees, n_jobs=-1) | 27.5 s in the main run, 24.3 s on a later rerun; a reviewer measured 41.4 s on the same box under load, so it varies with machine load (all under the 60 s limit) |
| forest fit on the random split (143,220 rows) | 26.6 s |
| noise experiment at reference settings (two forest fits, the second with 38 features) | 158.3 s |
| permutation importance (10,000 test rows, 5 repeats) | 2.9 s |
| total end-to-end run of everything above | 219.0 s |

The forest fit and the full-size noise experiment are both too slow for a 10-second notebook cell on Colab; the notebook needs a subsample or a precomputed model for those cells.

## Rank-normalised features (the design the notebook uses)

Same data, split, models and settings as above, with `features.rank_normalise` applied after
`build_features` and before `make_dataset`. Each feature becomes its rank among the 40 stocks
on that date, scaled to (rank - 0.5) / count - 0.5, so every date is centred on 0. (Contract D7
writes `rank(pct=True) - 0.5`; that differs only by a constant 1/80 per date, which leaves the
forest's splits and the linear model's slopes unchanged, and this form makes each date's mean
exactly 0.) The backtest numbers below match D7's to the stated precision. Run time for this
whole section: 43.3 s.

| model | mse | r2_vs_zero | ic (pooled) | hit_rate |
|---|---|---|---|---|
| zero | 0.000983 | 0.00% | n/a | n/a |
| constant = training mean | 0.000976 | +0.73% | n/a | 55.7% |
| linear | 0.000978 | +0.55% | +0.007 | 54.6% |
| forest | 0.000981 | +0.25% | +0.010 | 55.4% |

On squared error neither model beats the constant forecast, and the forest still forecasts a positive return on 96.9% of rows (linear 94.2%), so hit rate again mostly counts up-moves.

| model | mean daily cross-sectional IC | std | days | naive t-stat | days with IC > 0 |
|---|---|---|---|---|---|
| linear | +0.0085 | 0.240 | 1,246 | 1.25 | 51.7% |
| forest | +0.0117 | 0.216 | 1,246 | 1.90 | 51.9% |

Ranking stocks on the same day, the forest is now right on slightly more days than not; the naive t-stat ignores the overlap of 5-day targets on neighbouring days, so it overstates the evidence.

Forest decile table (test, pooled, 4,984 rows per decile):

| decile | mean forecast | mean realised 5-day return | standard error |
|---|---|---|---|
| 1 (lowest) | -0.009% | 0.275% | 0.045% |
| 2 | 0.121% | 0.211% | 0.041% |
| 3 | 0.139% | 0.172% | 0.038% |
| 4 | 0.151% | 0.208% | 0.039% |
| 5 | 0.168% | 0.268% | 0.041% |
| 6 | 0.196% | 0.287% | 0.043% |
| 7 | 0.238% | 0.241% | 0.043% |
| 8 | 0.297% | 0.296% | 0.045% |
| 9 | 0.395% | 0.380% | 0.049% |
| 10 (highest) | 0.757% | 0.356% | 0.056% |

The top two deciles beat the bottom two by about 0.13% per 5 days, but decile 1 is not the worst and the steps in between are within one or two standard errors: a small edge, not a staircase.

Backtest, forest strategy, 4 long / 4 short, 2015-01-12 to 2019-12-20:

| rebalance (trading days) | cost | total_return | annual_return | annual_vol | sharpe | max_drawdown | turnover per rebalance | rebalances |
|---|---|---|---|---|---|---|---|---|
| 5 | 5 bps | -31.33% | -7.32% | 15.66% | -0.41 | -36.28% | 3.02 | 250 |
| 5 | 0 bps | +0.23% | +0.05% | 15.67% | 0.08 | -18.26% | 3.02 | 250 |
| 10 | 5 bps | -47.24% | -12.13% | 16.06% | -0.72 | -53.34% | 3.21 | 125 |
| 10 | 0 bps | -35.53% | -8.49% | 16.04% | -0.47 | -45.89% | 3.21 | 125 |
| 21 | 5 bps | +5.15% | +1.02% | 17.64% | 0.15 | -39.59% | 3.36 | 60 |
| 21 | 0 bps | +16.32% | +3.10% | 17.65% | 0.26 | -36.34% | 3.36 | 60 |
| equal weight, long only, rebalance 5 | 5 bps | +85.68% | +13.33% | 13.25% | 1.01 | -18.01% | 0.022 | 250 |
| equal weight, long only, rebalance 21 | 5 bps | +86.32% | +13.41% | 13.24% | 1.02 | -17.89% | 0.055 | 60 |

Sharpe is mean daily / std daily x sqrt(252) with a 0% risk-free rate; no run blew up. Costs take about 0.15% to 0.17% per rebalance, so the 21-day book keeps most of its small gross return and the 5-day book loses all of it; but even before costs the three frequencies differ by far more than any cost (10-day is -35.5% at 0 bps), so much of the spread between rows is which days the book happened to trade, not a reliable law about rebalancing.

Timing: the forest fit on rank features took 13.5 s here against 26.4 s on raw features in the same run, with the defaults unchanged (200 trees, depth 6, leaf 50, n_jobs=-1). Rank features have about 40 to 73 distinct values per column instead of about 140,000, so finding each split is much cheaper.

## Numbers the notebook prints (rank features, quarter-sample forest)

Measured on 2026-10-01 by running `notebooks/workshop2_student.ipynb` from a clean kernel with nbclient and
`QSW_AUTO_REFERENCE=1`, so every task uses the reference knobs (Python 3.11.15, numpy 2.4.6, pandas 3.0.6,
scikit-learn 1.9.1, 4 CPU cores). Sections 4 to 7 fit the forest on `quarter_sample(train)`: a quarter of the
training rows, the same share from every date (seed 0). Section 7 rebuilds the features, dataset and split from
the knobs and fits on that same quarter, so with the reference knobs it is the same forest as section 4. That is
why the backtest numbers below differ from D7's table above, which fitted on all the training rows.

Data and dataset: 201,200 price rows (40 stocks x 5,030 trading days, 2000-01-03 to 2019-12-30); a typical day
moves a stock by about 0.85% (median absolute daily return); 190,960 dataset rows over 4,774 days; MSE of
predicting zero 0.001723, so a typical 5-day return is about 4.2% from zero; average 5-day return +0.25%.

Leakage demo (mean daily correlation with the target, share of days positive): the leaky feature `LEAKY_ret_next`
+0.396 on 96% of days; the honest features from -0.021 (`ret_5`) to +0.011 (`volume_z`).

Random split vs time split (100 trees, depth 6, a quarter of each training set):

| test set | r2 vs zero | IC, all rows pooled | IC per day (mean) | days with IC > 0 | top minus bottom decile (5 days) |
|---|---|---|---|---|---|
| time split (honest) | +0.11% | +0.013 | +0.0123 | 53.1% | +0.10% |
| random split (leaky) | +0.56% | +0.024 | +0.0201 | 52.3% | +0.61% |

Section 4, reference forest (200 trees, depth 6) against the baselines, test 2015-01-12 to 2019-12-20:

| forecast | r2 vs zero | IC, all rows pooled | IC per day (mean) | days with IC > 0 |
|---|---|---|---|---|
| predict zero | +0.00% | n/a | n/a | n/a |
| constant (training average) | +0.73% | n/a | n/a | n/a |
| linear model | +0.45% | +0.008 | +0.0100 | 51.6% |
| forest | +0.09% | +0.011 | +0.0110 | 52.5% |

Decile averages (mean realised 5-day return by forecast decile, 4,984 rows each, standard error in brackets):
1: 0.195% (0.047%), 2: 0.157% (0.040%), 3: 0.206% (0.039%), 4: 0.302% (0.040%), 5: 0.255% (0.040%),
6: 0.340% (0.042%), 7: 0.354% (0.044%), 8: 0.302% (0.045%), 9: 0.304% (0.047%), 10: 0.281% (0.056%).
Bottom two about 0.18%, top two about 0.29%; deciles 6 and 7 are the highest.

Noise experiment (unbraked forest: no depth limit, leaf 5, 50 trees, 20,000 training rows): real features
+48.35% train, -7.46% test; real plus 30 noise columns +59.82% train, -4.14% test.

Costs per year at a turnover of 3.2 and 5 bps: 8.1% (every 5 days), 4.0% (10), 1.9% (21).

Backtests, test 2015-01-12 to 2019-12-20, 4 long and 4 short:

| run | total return | return per year | volatility per year | Sharpe | max drawdown | turnover per rebalance | rebalances |
|---|---|---|---|---|---|---|---|
| every 21 days, after 5 bps | +6.4% | +1.3% | 16.5% | +0.16 | -53.8% | 3.22 | 60 |
| every 21 days, before costs | +17.2% | +3.3% | 16.5% | +0.28 | -51.2% | 3.22 | 60 |
| every 5 days, before costs | +22.9% | +4.3% | 14.7% | +0.36 | -17.9% | 3.02 | 250 |
| every 5 days, after 5 bps | -15.8% | -3.4% | 14.8% | -0.16 | -29.6% | 3.02 | 250 |
| the market, rebalanced every 21 days | +86.3% | +13.4% | 13.2% | +1.02 | -17.9% | 0.05 | 60 |

The market line is rebalanced on the strategy's own days (21 here), which is why it shows +86.3% rather than the
+85.7% of a weekly-rebalanced market. Not printed by the notebook, measured with the same quarter-sample forest:
every 10 days +18.7% before costs, -2.3% after. The notebook's remark on rebalance frequencies quotes the 10-day
book fitted on all the training rows (D7 table above): -35.5% before costs, "about 35%".

Section 8 preview with the reference bundle (decision date 2019-12-30): long DIS, GILD, HD, ORCL; short C, JPM,
QCOM, WFC; eight orders proposed, none submitted; runner exit code 0. The checklist counts 4 backtests.

Cell times on 4 cores (whole notebook 30 to 32 s with the reference knobs): noise experiment 11.2 to 11.7 s;
random vs time split 3.0 s; Task 4 2.8 s; section 7 rebuild and refit 2.9 s; export 1.2 s; preview 1.9 s. At 500
trees and depth 12 (whole notebook 50 s): Task 4 9.1 s, section 7 refit 10.0 s, export 5.5 s.
