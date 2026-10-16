"""Build the Workshop 3 price snapshot: daily OHLCV for S&P 100 members, 2000 to the latest close.

Usage:  python workshop3/fetch_data.py            # writes workshop3/data/sp100_daily.csv.gz
        python workshop3/fetch_data.py --check    # re-reads the file and prints a summary

Source: Yahoo Finance through the yfinance package (teaching use only). The universe is the
S&P 100 as listed in October 2026, so it is survivorship-biased: firms that left the index or
failed before then are missing. Columns: date, symbol, close, adj_close, volume (the strategy only needs closing prices).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "sp100_daily.csv.gz"
START = "2000-01-01"
SP100 = """AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK BMY BRK-B C CAT CHTR CL
CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR F FDX GD GE GILD GM GOOG GOOGL GS HD HON IBM INTC INTU
ISRG JNJ JPM KO LIN LLY LMT LOW MA MCD MDLZ MDT MET META MMM MO MRK MS MSFT NEE NFLX NKE NOW NVDA ORCL PEP PFE
PG PLTR PM PYPL QCOM RTX SBUX SCHW SO SPG T TGT TMO TMUS TSLA TXN UBER UNH UNP UPS USB V VZ WFC WMT XOM""".split()


# Known errors in Yahoo's adjusted close, repaired by stitching: on the given date the adjusted return
# is replaced by the raw close return, and every earlier adjusted price is rescaled to match.
# DHR 2016-07-05 (Fortive spin-off): Yahoo shows +61% adjusted against +3.6% raw.
REPAIRS = [("DHR", "2016-07-05")]


def repair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for sym, day in REPAIRS:
        rows = df[df["symbol"] == sym].sort_values("date")
        dates = pd.to_datetime(rows["date"])
        i = int((dates == pd.Timestamp(day)).to_numpy().nonzero()[0][0])
        a, c = rows["adj_close"].to_numpy(), rows["close"].to_numpy()
        factor = (a[i] / a[i - 1]) / (c[i] / c[i - 1])
        if abs(factor - 1) < 0.05:          # already fine (Yahoo fixed it): leave alone
            continue
        before = rows.index[:i]
        df.loc[before, "adj_close"] = df.loc[before, "adj_close"] * factor
    return df


def build() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(SP100, start=START, auto_adjust=False, progress=False, group_by="ticker", threads=True)
    rows = []
    for sym in SP100:
        d = raw[sym] if sym in raw.columns.get_level_values(0) else None
        if d is None or d["Close"].dropna().empty:            # Yahoo drops a ticker now and then: retry it alone
            d = yf.download(sym, start=START, auto_adjust=False, progress=False, multi_level_index=False)
        if d is None or d["Close"].dropna().empty:
            print("no data for", sym, file=sys.stderr)
            continue
        d = d.dropna(subset=["Close"]).reset_index()
        d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
        d["symbol"] = sym
        rows.append(d[["date", "symbol", "close", "adj_close", "volume"]])
    out = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return repair(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    p.add_argument("--repair-only", action="store_true", help="apply REPAIRS to the existing file, no download")
    a = p.parse_args(argv)
    if a.repair_only:
        df = repair(pd.read_csv(OUT))
        df.to_csv(OUT, index=False, compression="gzip", float_format="%.6f")
    elif not a.check:
        df = build()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False, compression="gzip", float_format="%.6f")
    df = pd.read_csv(OUT)
    print(f"{OUT.name}: {len(df):,} rows, {df.symbol.nunique()} symbols, {df.date.min()} to {df.date.max()}, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
