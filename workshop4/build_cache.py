"""Rebuild the cached backtests the Workshop 4 notebook loads by default.

Usage:  python workshop4/build_cache.py      # about 4 minutes on a 4-core laptop

Runs the four variants (sector ETFs, ETFs in trading time, PCA at 55% of variance, PCA with 15
factors) from 2001 to the last date of the snapshot and writes their daily returns and trade logs
to workshop4/data/. The notebook re-runs one year live and compares it with this file, so a stale or
edited cache shows up as a mismatch rather than silently.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from quantsoc_ml import avlee as al  # noqa: E402

START = "2001-01-01"
DAILY = ROOT / "workshop4" / "data" / "backtests_daily.csv.gz"
TRADES = ROOT / "workshop4" / "data" / "backtests_trades.csv.gz"
VARIANTS = {
    "etf": dict(cfg=al.Config(factors="etf"), etf=True, volume=False),
    "etf_trading_time": dict(cfg=al.Config(factors="etf"), etf=True, volume=True),
    "pca_55": dict(cfg=al.Config(factors="pca", var_explained=0.55), etf=False, volume=False),
    "pca_15": dict(cfg=al.Config(factors="pca", n_factors=15), etf=False, volume=False),
}


def run_variant(name, P=None, E=None, S=None, V=None, start=START, end=None):
    v = VARIANTS[name]
    P = al.load_prices() if P is None else P
    if v["etf"]:
        E = al.load_etfs() if E is None else E
        S = al.load_sectors() if S is None else S
    if v["volume"]:
        V = al.load_volume() if V is None else V
    return al.run(P, v["cfg"], E if v["etf"] else None, S if v["etf"] else None, start=start, end=end,
                  volume=V if v["volume"] else None)


if __name__ == "__main__":
    P, E, S, V = al.load_prices(), al.load_etfs(), al.load_sectors(), al.load_volume()
    daily, trades = [], []
    for name in VARIANTS:
        bt = run_variant(name, P, E, S, V)
        d = pd.DataFrame({"gross": bt.gross, "net": bt.net, "turnover": bt.turnover, "n_long": bt.n_long,
                          "n_short": bt.n_short, "n_factors": bt.n_factors})
        d["variant"] = name
        daily.append(d.rename_axis("date").reset_index())
        t = bt.trades.copy()
        t["variant"] = name
        trades.append(t)
        print(name, {k: round(x, 3) for k, x in bt.stats().items()}, flush=True)
    pd.concat(daily).to_csv(DAILY, index=False, compression="gzip", float_format="%.8g")
    pd.concat(trades).to_csv(TRADES, index=False, compression="gzip", float_format="%.8g")
    print("wrote", DAILY, TRADES)
