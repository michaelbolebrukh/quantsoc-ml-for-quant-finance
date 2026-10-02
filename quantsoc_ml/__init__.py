"""quantsoc_ml: shared code for the QuantSoc workshop "Machine Learning for Quantitative Finance (Part 1)".

Each module maps to one stage of the workshop:

- data      : load the daily price snapshot, reshape long <-> wide
- features  : the feature functions and the forward-return target (the timing rule lives here)
- splits    : split by time with an embargo (and the naive random split, for the demo only)
- models    : the zero baseline, a linear model and a random forest
- evaluate  : scores, decile tables, the noise experiment, permutation importance
- portfolio : StrategyConfig (the five knobs) and forecasts -> long/short weights
- backtest  : the daily backtest engine and the equal-weight benchmark
- viz       : plots (imported on demand so the package loads without matplotlib)

Nothing heavy is imported here, so `import quantsoc_ml` is quick on Colab.
"""

__version__ = "0.1.0"
HORIZON = 5  # trading days ahead that the target looks; the embargo uses the same number
