"""Load the committed daily price snapshot and reshape it.

The file is long format (one row per date and symbol) because that is how it is stored and
shared. Almost every calculation is easier on a wide table (dates down, symbols across), so
`to_wide` is the bridge. Nothing here looks at dates: it only reads and reshapes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def default_path() -> Path:
    """The gzipped snapshot if present, else the plain CSV (both are accepted)."""
    for name in ("prices.csv.gz", "prices.csv"):
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no prices.csv.gz or prices.csv in {DATA_DIR}")


def load_prices(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Long price table with parsed dates, sorted by (date, symbol).

    `path` may be .csv or .csv.gz; pandas infers the compression from the suffix.
    """
    path = Path(path) if path is not None else default_path()
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df["symbol"] = df["symbol"].astype(str)
    return df[COLUMNS].sort_values(["date", "symbol"]).reset_index(drop=True)


def to_wide(long: pd.DataFrame, column: str = "adj_close") -> pd.DataFrame:
    """Date x symbol table of one column (default adj_close, the split and dividend adjusted close)."""
    wide = long.pivot(index="date", columns="symbol", values=column).sort_index()
    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide.columns.name = None
    wide.index.name = "date"
    return wide.astype(float)


def daily_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Simple daily return close[t] / close[t-1] - 1 per symbol. The first row is NaN.

    This is the return EARNED on day t by a position held from the close of t-1. The input is
    sorted by date first, so an unsorted table cannot pair the wrong two days.
    """
    return prices_wide.sort_index().pct_change(fill_method=None)
