# Bonus and feature figures: measured numbers

Written by `deck/make_bonus_figures.py` (run `python3 deck/make_bonus_figures.py`). Every value below
was produced by running that script; nothing is estimated. Percentages are given as %, R squared is
r2_vs_zero (1 - MSE / MSE of predicting zero) as in quantsoc_ml.evaluate. Seeds are fixed, so a rerun
reproduces every value.

## How each figure was computed

- **F5**: features.build_features on data/prices.csv.gz, raw values, JPM, 2008-01-01 to 2010-12-31
- **F6**: build_features then rank_normalise; ret_21 cross-section on one date
- **B1**: numpy Polynomial.fit (least squares), seed 1, n = 30, x uniform on [0, 1]
- **B2**: illustration only, seed 2; no measured numbers
- **B3**: rank_normalise -> make_dataset -> time_split(2014-12-31, 2015-01-12); sklearn DecisionTreeRegressor(max_depth=k, random_state=0) on all training rows; evaluate.score r2_vs_zero
- **B4**: seed 4; fixed evenly spaced x (n = 30), noise redrawn 2,000 times; Polynomial.fit; bias^2 and variance at 200 x
- **B5**: diagram; seed 5 for the shuffled fold assignment; 50 time cells, 5 folds, purge 2, embargo 2
- **B6**: seed 6; sklearn Ridge (fit_intercept False) and lasso_path on standardised X
- **B7**: qualitative positions after James et al. (2021) fig. 2.7 (solid); ridge, k-NN, random forests hollow, placed by us; no measured numbers
- **B8**: seed 8; features max(0, x.w), w uniform on the unit sphere in 5 dimensions; coef = lstsq (minimum norm); expected test MSE on 2,000 fresh x
- **B9**: exact computation, no randomness

## Values

| key | value | how |
|---|---|---|
| `F5.ret_1_min` | -0.2073 | min of raw ret_1 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_1_max` | 0.251 | max of raw ret_1 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_5_min` | -0.3713 | min of raw ret_5 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_5_max` | 0.4909 | max of raw ret_5 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_21_min` | -0.4253 | min of raw ret_21 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_21_max` | 0.7734 | max of raw ret_21 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_63_min` | -0.5344 | min of raw ret_63 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.ret_63_max` | 1.2297 | max of raw ret_63 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.vol_21_min` | 0.0071 | min of raw vol_21 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.vol_21_max` | 0.1044 | max of raw vol_21 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.vol_63_min` | 0.0155 | min of raw vol_63 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.vol_63_max` | 0.0881 | max of raw vol_63 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.volume_z_min` | -2.7539 | min of raw volume_z for JPM, 2008-01-01 to 2010-12-31 |
| `F5.volume_z_max` | 6.3085 | max of raw volume_z for JPM, 2008-01-01 to 2010-12-31 |
| `F5.hi52_min` | -0.6772 | min of raw hi52 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.hi52_max` | 0.0 | max of raw hi52 for JPM, 2008-01-01 to 2010-12-31 |
| `F5.symbol` | JPM | the stock shown |
| `F5.n_days` | 757 | trading days shown |
| `F6.date` | 2008-10-10 | the one date shown (the week of the October 2008 crash, picked to make the scale visible) |
| `F6.n_stocks` | 40 | stocks with a ret_21 value on that date |
| `F6.raw_mean_pct` | -24.87 | mean raw 21-day return across the stocks, % |
| `F6.raw_min_pct` | -43.45 | lowest raw 21-day return, % (GS) |
| `F6.raw_max_pct` | 0.75 | highest raw 21-day return, % (JPM) |
| `F6.n_negative_raw` | 39 | stocks with a negative raw 21-day return that day |
| `F6.rank_min` | -0.4875 | lowest rank value |
| `F6.rank_max` | 0.4875 | highest rank value |
| `F6.rank_mean` | 0.0 | mean rank value (0 by construction) |
| `F6.order_kept` | True | rank order equals raw order |
| `B1.train_mse_degree_1` | 0.2291 | mean squared error on the 30 training points |
| `B1.test_mse_degree_1` | 0.3107 | mean squared error on 2,000 fresh points from the same process |
| `B1.train_mse_degree_4` | 0.0625 | mean squared error on the 30 training points |
| `B1.test_mse_degree_4` | 0.1014 | mean squared error on 2,000 fresh points from the same process |
| `B1.train_mse_degree_15` | 0.0249 | mean squared error on the 30 training points |
| `B1.test_mse_degree_15` | 2935.3391 | mean squared error on 2,000 fresh points from the same process |
| `B1.noise_mse` | 0.09 | irreducible error: the noise variance (sd 0.3) |
| `B3.train_rows` | 140880 | all training rows (2000-12-29 to 2014-12-31), rank features, no sampling |
| `B3.test_rows` | 49840 | test rows from 2015-01-12 |
| `B3.best_test_depth` | 1 | depth with the highest test R squared vs predicting zero |
| `B3.best_test_r2_pct` | 0.547 | that highest test R squared, % |
| `B3.test_r2_at_depth_1` | 0.547 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_1` | 0.4 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_2` | 0.418 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_2` | 0.481 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_3` | 0.268 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_3` | 0.543 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_4` | -0.026 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_4` | 0.714 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_6` | -1.117 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_6` | 1.483 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_8` | -4.287 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_8` | 3.432 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_10` | -16.384 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_10` | 7.789 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_15` | -78.837 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_15` | 31.023 | train R squared vs predicting zero, % |
| `B3.test_r2_at_depth_20` | -160.888 | test R squared vs predicting zero, % |
| `B3.train_r2_at_depth_20` | 62.598 | train R squared vs predicting zero, % |
| `B3.const_mean_test_r2_pct` | 0.733 | constant forecast = training mean target, test R squared, % |
| `B3.n_depths_test_above_zero` | 3 | depths (of 20) whose test R squared > 0 |
| `B3.n_depths_test_above_const` | 0 | depths (of 20) whose test R squared beats the constant training-mean forecast |
| `B3.test_falls_every_step` | True | test R squared lower at every depth than at the depth before |
| `B4.n_train` | 30 | points per training set at fixed, evenly spaced x on [0, 1]; y = sin(2 pi x) + N(0, 0.3^2) |
| `B4.n_training_sets` | 2000 | training sets per degree (the noise is redrawn each time) |
| `B4.best_degree` | 3 | degree with the lowest total expected test error |
| `B4.noise` | 0.09 | irreducible error, sd^2 |
| `B4.variance_rises_every_step` | True | variance higher at every degree than the one before |
| `B4.bias2_degree_1` | 0.202 | squared bias averaged over 200 test x in [0, 1] |
| `B4.variance_degree_1` | 0.00575 | variance of the fit across training sets, averaged over x |
| `B4.total_degree_1` | 0.29738 | bias^2 + variance + noise |
| `B4.measured_test_mse_degree_1` | 0.29775 | check: mean squared error against fresh noisy y at the same x (should match total) |
| `B4.bias2_degree_3` | 0.00496 | squared bias averaged over 200 test x in [0, 1] |
| `B4.variance_degree_3` | 0.01096 | variance of the fit across training sets, averaged over x |
| `B4.total_degree_3` | 0.10592 | bias^2 + variance + noise |
| `B4.measured_test_mse_degree_3` | 0.10573 | check: mean squared error against fresh noisy y at the same x (should match total) |
| `B4.bias2_degree_8` | 9.28e-06 | squared bias averaged over 200 test x in [0, 1] |
| `B4.variance_degree_8` | 0.02402 | variance of the fit across training sets, averaged over x |
| `B4.total_degree_8` | 0.11403 | bias^2 + variance + noise |
| `B4.measured_test_mse_degree_8` | 0.11381 | check: mean squared error against fresh noisy y at the same x (should match total) |
| `B4.bias2_degree_12` | 1.25e-05 | squared bias averaged over 200 test x in [0, 1] |
| `B4.variance_degree_12` | 0.03745 | variance of the fit across training sets, averaged over x |
| `B4.total_degree_12` | 0.12746 | bias^2 + variance + noise |
| `B4.measured_test_mse_degree_12` | 0.12747 | check: mean squared error against fresh noisy y at the same x (should match total) |
| `B6.n` | 120 | rows |
| `B6.p` | 20 | features: 3 true signals (+3.0, -2.0, +1.5) among 20, AR(1) correlation 0.3, noise sd 2.0 |
| `B6.ridge_exact_zeros_any_alpha` | 0 | ridge coefficients exactly 0, any penalty |
| `B6.lasso_max_exact_zeros` | 20 | most lasso coefficients exactly 0 at one penalty on the path |
| `B6.lasso_alpha_3_nonzero` | 1.5304 | largest penalty on the path with exactly 3 nonzero lasso coefficients |
| `B6.lasso_3_kept_are_true` | True | those 3 are the true signals (kept [2, 9, 15]) |
| `B6.lasso_nonzero_at_alpha_0.3` | 4 | nonzero lasso coefficients at penalty 0.304 |
| `B6.ols_max_abs_noise_coef` | 0.43 | largest |coefficient| on a noise feature with no penalty (least squares) |
| `B8.n_train` | 100 | training points, x ~ N(0, I) in 5 dimensions, y = tanh(2 x.b) + 0.5 sin(2 x1) + N(0, 0.3^2) |
| `B8.repetitions` | 20 | independent redraws of data and random ReLU features; median plotted |
| `B8.peak_p` | 100 | number of features with the highest median expected test MSE |
| `B8.peak_test_mse` | 147.006 | that median expected test MSE (error vs truth + noise variance) |
| `B8.best_under_p` | 20 | best p below the threshold p = n |
| `B8.best_under_test_mse` | 0.4736 | its median expected test MSE |
| `B8.best_over_p` | 1400 | best p above the threshold |
| `B8.best_over_test_mse` | 0.8451 | its median expected test MSE |
| `B8.test_mse_p_2000` | 0.8464 | median expected test MSE with 2,000 features |
| `B8.over_beats_under` | False | does the best over-parameterised fit beat the best classical one here |
| `B9.avg_uniqueness_5d` | 0.2008 | mean over 1,000 daily labels of mean(1 / concurrency) across each label's 5 days (AFML ch. 4) |
| `B9.effective_n_1000` | 200.8 | sum of uniqueness: about how many independent labels |
| `B9.interior_uniqueness` | 0.2 | uniqueness of a label away from the ends (1/5) |
| `B9.shared_days` | 4 | days two neighbouring 5-day labels share |
| `B9.example_label_day4_uniqueness` | 0.21 | the drawn label made on day 4, in the 8-label picture |
