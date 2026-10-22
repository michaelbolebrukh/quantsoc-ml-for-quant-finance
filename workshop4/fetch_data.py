"""Build the Workshop 4 extras: daily prices for SPY and the SPDR sector ETFs, and each stock's sector.

Usage:  python workshop4/fetch_data.py            # writes workshop4/data/etf_daily.csv.gz and sectors.csv
        python workshop4/fetch_data.py --check

The stocks themselves come from Workshop 3's snapshot (workshop3/data/sp100_daily.csv.gz), and the
ETF file stops on the same last date so the two line up. Source: Yahoo Finance via yfinance
(teaching use only). Sectors are Yahoo's current classification, mapped to the matching SPDR
fund; they are today's sectors, not the sectors each company had in the past.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STOCKS = ROOT / "workshop3" / "data" / "sp100_daily.csv.gz"
ETF_OUT = HERE / "data" / "etf_daily.csv.gz"
SECTOR_OUT = HERE / "data" / "sectors.csv"
START = "1999-01-01"
ETFS = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
YAHOO_TO_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Energy": "XLE",
                "Industrials": "XLI", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
                "Utilities": "XLU", "Basic Materials": "XLB", "Real Estate": "XLRE",
                "Communication Services": "XLC"}


def fetch_etfs(end: str) -> pd.DataFrame:
    import yfinance as yf
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ETFS, start=START, end=end_excl, auto_adjust=False, progress=False, group_by="ticker")
    rows = []
    for sym in ETFS:
        d = raw[sym].dropna(subset=["Close"]).reset_index()
        d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
        d["symbol"] = sym
        rows.append(d[["date", "symbol", "close", "adj_close", "volume"]])
    out = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def fetch_sectors(symbols) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for sym in symbols:
        try:
            sector = yf.Ticker(sym).info.get("sector", "")
        except Exception as e:  # network or rate limit: record it, never guess
            print("no sector for", sym, e, file=sys.stderr)
            sector = ""
        rows.append({"symbol": sym, "yahoo_sector": sector, "etf": YAHOO_TO_ETF.get(sector, "")})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    stocks = pd.read_csv(STOCKS)
    if not a.check:
        ETF_OUT.parent.mkdir(parents=True, exist_ok=True)
        fetch_etfs(stocks["date"].max()).to_csv(ETF_OUT, index=False, compression="gzip", float_format="%.6f")
        fetch_sectors(sorted(stocks["symbol"].unique())).to_csv(SECTOR_OUT, index=False)
    e, s = pd.read_csv(ETF_OUT), pd.read_csv(SECTOR_OUT)
    print(f"{ETF_OUT.name}: {len(e):,} rows, {e.symbol.nunique()} ETFs, {e.date.min()} to {e.date.max()}")
    print(f"{SECTOR_OUT.name}: {len(s)} stocks, missing sector: {list(s[s.etf.isna() | (s.etf == '')].symbol)}")
    print(s.etf.value_counts().to_string())


if __name__ == "__main__":
    main()
