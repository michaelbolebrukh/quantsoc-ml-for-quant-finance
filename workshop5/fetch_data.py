"""Build the Workshop 5 extra data: each stock's daily open, high and low, as ratios to that day's close.

Usage:  python workshop5/fetch_data.py            # writes workshop5/data/sp100_ohl.csv.gz
        python workshop5/fetch_data.py --check

The closes, adjusted closes and volumes come from Workshop 3's snapshot (workshop3/data/sp100_daily.csv.gz),
so all three workshops use the same stocks and the same days. This file adds what the price charts need on
top: open/close, high/close and low/close for each stock and day. Ratios do not change when a stock splits,
so they stay right even though Yahoo rescales old raw prices after every split. The file stops on the same
last date as Workshop 3's snapshot. Source: Yahoo Finance via yfinance (teaching use only).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STOCKS = ROOT / "workshop3" / "data" / "sp100_daily.csv.gz"
OUT = HERE / "data" / "sp100_ohl.csv.gz"
START = "2000-01-01"


def fetch(symbols, end: str) -> pd.DataFrame:
    import yfinance as yf
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(symbols, start=START, end=end_excl, auto_adjust=False, progress=False,
                      group_by="ticker", threads=True)
    rows = []
    for sym in symbols:
        d = raw[sym] if sym in raw.columns.get_level_values(0) else None
        if d is None or d["Close"].dropna().empty:            # Yahoo drops a ticker now and then: retry it alone
            d = yf.download(sym, start=START, end=end_excl, auto_adjust=False, progress=False,
                            multi_level_index=False)
        if d is None or d["Close"].dropna().empty:
            print("no data for", sym, file=sys.stderr)
            continue
        d = d.dropna(subset=["Close"]).reset_index()
        d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
        c = d["close"].where(d["close"] > 0)
        out = pd.DataFrame({"date": pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"), "symbol": sym,
                            "open_r": d["open"] / c, "high_r": d["high"] / c, "low_r": d["low"] / c})
        rows.append(out)
    out = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    return out.replace([np.inf, -np.inf], np.nan)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    stocks = pd.read_csv(STOCKS)
    if not a.check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df = fetch(sorted(stocks["symbol"].unique()), stocks["date"].max())
        df = df.merge(stocks[["date", "symbol"]], on=["date", "symbol"], how="inner")   # same days as Workshop 3
        df.to_csv(OUT, index=False, compression="gzip", float_format="%.5f")
    df = pd.read_csv(OUT)
    print(f"{OUT.name}: {len(df):,} rows, {df.symbol.nunique()} symbols, {df.date.min()} to {df.date.max()}, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    print("rows missing any of open/high/low:", int(df[["open_r", "high_r", "low_r"]].isna().any(axis=1).sum()))


if __name__ == "__main__":
    main()
