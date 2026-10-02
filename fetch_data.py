"""Build data/prices.csv: daily OHLCV for a fixed universe of large US stocks, 2000-01-01 to 2019-12-31.

Source: Yahoo Finance via the yfinance package, pulled once and committed as a snapshot so the
workshop never depends on live downloads. `adj_close` is Yahoo's split- and dividend-adjusted close;
`close` is the raw close. Run `python fetch_data.py` to rebuild, `python fetch_data.py --check` to
verify the committed file matches the expected shape.

Universe: 50 candidates from the S&P 100 that traded throughout 2000-2019 and still trade today
(so the paper-trading script can hold them). The 40 kept are the most liquid by median dollar
volume among those with >= 99% of trading days present. This is SURVIVORSHIP-BIASED by construction
(see data/DATA_CARD.md); the workshop says so on a slide.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

CANDIDATES = ["AAPL","MSFT","AMZN","JPM","JNJ","PG","XOM","CVX","KO","PEP","WMT","HD","MCD","IBM","INTC","CSCO",
              "ORCL","PFE","MRK","BA","CAT","MMM","GE","DIS","VZ","T","BAC","C","WFC","GS","AXP","UNH","AMGN","COST",
              "NKE","LOW","TXN","HON","UNP","MO","BMY","ADBE","QCOM","MDT","LLY","ABT","SBUX","GILD","TGT","DHR"]
START, END, KEEP = "2000-01-01", "2019-12-31", 40
OUT = Path(__file__).parent / "data" / "prices.csv.gz"


def fetch() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(CANDIDATES, start=START, end=END, auto_adjust=False, progress=False, threads=False)
    frames = []
    for sym in CANDIDATES:
        df = raw.xs(sym, axis=1, level=1).rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        df = df.dropna(subset=["close"]).copy()
        df["symbol"] = sym
        frames.append(df.reset_index().rename(columns={"Date": "date"}))
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    return long[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]


def select(long: pd.DataFrame) -> pd.DataFrame:
    days = long["date"].nunique()
    cover = long.groupby("symbol")["date"].nunique() / days
    ok = cover[cover >= 0.99].index
    dollar = (long[long.symbol.isin(ok)].assign(dv=lambda d: d.close * d.volume).groupby("symbol")["dv"].median())
    keep = dollar.sort_values(ascending=False).head(KEEP).index.sort_values()
    return long[long.symbol.isin(keep)].sort_values(["date", "symbol"]).reset_index(drop=True)


def check(path: Path = OUT) -> None:
    df = pd.read_csv(path)
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"], df.columns
    assert df.symbol.nunique() == KEEP, df.symbol.nunique()
    assert df.date.min() <= "2000-01-04" and df.date.max() >= "2019-12-30", (df.date.min(), df.date.max())
    assert (df[["open", "high", "low", "close", "adj_close"]] > 0).all().all()
    assert not df.duplicated(["date", "symbol"]).any()
    print(f"OK {path}: {len(df):,} rows, {df.symbol.nunique()} symbols, {df.date.min()} -> {df.date.max()}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        OUT.parent.mkdir(exist_ok=True)
        sel = select(fetch())
        sel.to_csv(OUT, index=False, float_format="%.6f", compression="gzip")
        check()
        print("symbols:", ", ".join(sorted(sel.symbol.unique())))
