"""The non-UI half of the Workshop 4 paper-trading dashboard: bars in, s-scores, trades and orders out.

The strategy is the Avellaneda and Lee (2010) replication in quantsoc_ml/avlee.py, run with each
stock's sector SPDR fund as its factor (`Config(factors="etf")`). This file only turns the account
and today's prices into a plan; the dashboard (workshop4/dashboard.py) only draws what comes back.

Reused from Workshop 3, not copied: the ticker mapping, the unfinished-day filters, the read-only
broker view, `propose_orders`, and `submit_plan`, which stays the ONLY function that sends an order.
The plan built here has the fields `submit_plan` reads (decision_date, orders with an alpaca_symbol
column, clock, fingerprint), so it is passed to that function unchanged.

How the account maps to the strategy, stated because the strategy is STATEFUL:

* An open trade is a STOCK position in the universe: long shares mean a long trade (+1), short
  shares a short trade (-1). ETF positions are hedges, never trades. So the paper account should
  run this strategy only: a stock bought by hand, or by another strategy, is read as an open trade.
* A trade that stays open keeps its CURRENT share count. The paper sizes a trade when it opens and
  this layer does not resize it as prices move.
* A trade that opens today is `cfg.position` of equity in the stock, in whole shares (rounded
  towards zero). A trade that closes goes to zero shares.
* Each hedge fund's target is the sum, over the trades open after today, of
  -side x beta x (that trade's dollar value at the last close), with today's beta and today's hedge
  fund. It is recomputed on every run, as the paper re-hedges daily.
* A new short that the broker says is not easy to borrow is simply not opened, with a note. Unlike
  Workshop 3 no other stock is put in its place: a threshold strategy has no "next in line".
* A held stock with no prices at all in today's bars is left alone, with a note (no blind trades).
  A held stock with prices but no valid fit is closed, which is the replication's own rule.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `quantsoc_ml` and `workshop3` import when run from anywhere
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import avlee as al  # noqa: E402
from quantsoc_ml.paper import (  # noqa: E402
    BAR_COLUMNS, MARKET_TZ, BrokerError, MockBroker, ReadOnlyBroker, Refused, drop_unfinished_day, propose_orders,
)
from workshop3 import live as ws3  # noqa: E402
from workshop3.live import (  # noqa: E402,F401  (re-exported so the dashboard and tests use one name)
    CONFIRM_WORD, client_order_id, connect_alpaca, drop_today_until_close, equity_history, from_alpaca,
    recent_orders, results_table, submit_plan, to_alpaca,
)

HISTORY_DAYS = 80          # trading days of bars to fetch; the model needs cfg.window + 1 = 61
DEMO_EQUITY = 100_000.0
DEMO_WARMUP_DAYS = 30      # the offline demo replays this many days of signals to reach a plausible book
NOT_BORROWABLE = "not easy to borrow today, so this short was not opened (no other stock is used instead)"


def live_config() -> al.Config:
    """The live configuration: sector ETF factors, everything else at the paper's values."""
    return al.Config(factors="etf")


def hedge_funds(sectors: pd.Series) -> List[str]:
    """Every fund a trade can be hedged with: the sector SPDRs in the map, plus SPY (the fallback)."""
    return sorted(set(map(str, sectors.dropna().unique())) | {"SPY"})


# --------------------------------------------------------------------------- the plan

@dataclass
class LivePlan:
    decision_date: pd.Timestamp
    table: pd.DataFrame                 # one row per stock: s, kappa, beta, hedge fund, states, what happens today
    held: Dict[str, int]                # trades open before today, read from the account (+1 long, -1 short)
    states: Dict[str, int]              # trades open after today
    weights: pd.Series                  # the strategy's own book from avlee.live_signal, fraction of equity
    book: pd.DataFrame                  # target book: stock legs and fund hedges, in shares and dollars
    orders: pd.DataFrame                # propose_orders output plus an alpaca_symbol column
    opened: List[str]
    kept: List[str]
    closed: List[str]
    skipped: Dict[str, str]             # new shorts not opened, with the reason
    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, float]         # Yahoo tickers, negative = short
    last_prices: Dict[str, float]       # Yahoo tickers, last completed close
    open_orders: List[str]              # Yahoo tickers with an order still open
    clock: dict
    cfg: al.Config = field(default_factory=live_config)
    notes: List[str] = field(default_factory=list)

    order_id_prefix = "qsml-ws4"        # Workshop 3 uses qsml-ws3, so the two never share an order id

    @property
    def fingerprint(self) -> str:
        """The same short hash of the orders that Workshop 3 uses, so submit_plan can check it."""
        return ws3.LivePlan.fingerprint.fget(self)


def _wide(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last").sort_index()


def orders_for_targets(targets: Mapping[str, float], positions: Mapping[str, float],
                       last_prices: Mapping[str, float]) -> pd.DataFrame:
    """Orders that move `positions` to target SHARE counts, through Workshop 3's propose_orders.

    propose_orders takes weights of equity; giving it the share counts as weights, an equity of 1
    and a price of 1 for every name makes its target exactly the share count (rounded towards zero),
    while keeping all its conventions: whole shares, exits and reductions first, a long-to-short
    flip split into a close leg and an open leg, and no order for a move under one share. The
    notional column is then put back in dollars at the last close (NaN where there is no price).
    """
    t = pd.Series({str(k): float(v) for k, v in targets.items()}, dtype=float)
    held = {s: float(q) for s, q in positions.items() if s in t.index}
    orders = propose_orders(t, 1.0, held, {s: 1.0 for s in t.index}, min_notional=1.0)
    px = orders["symbol"].map(lambda s: last_prices.get(s, np.nan)).astype(float)
    orders["notional"] = (orders["qty"].astype(float) * px).round(2)
    orders.insert(1, "alpaca_symbol", orders["symbol"].map(to_alpaca))
    return orders


def build_live_plan(view: ReadOnlyBroker, cfg: Optional[al.Config] = None, sectors: Optional[pd.Series] = None,
                    universe: Optional[Sequence[str]] = None, history_days: int = HISTORY_DAYS) -> LivePlan:
    """Bars, s-scores, trades, target book and orders for the last completed trading day. Cannot submit."""
    if hasattr(view, "submit"):
        raise TypeError("build_live_plan must receive a ReadOnlyBroker (no submit method)")
    cfg = cfg or live_config()
    if cfg.factors != "etf":
        raise ValueError("the dashboard hedges with sector ETFs: use Config(factors='etf')")
    sectors = al.load_sectors() if sectors is None else sectors
    stocks = sorted(map(str, universe if universe is not None else sectors.index))
    funds = hedge_funds(sectors)
    notes: List[str] = []

    # ---- prices: completed days only
    clock = view.clock()
    bars = view.daily_bars([to_alpaca(s) for s in stocks + funds], int(history_days))
    bars = drop_unfinished_day(bars, clock)
    bars = drop_today_until_close(bars, clock)
    if len(bars) == 0:
        raise BrokerError("no daily bars came back for the strategy's stocks and funds")
    bars = bars.assign(symbol=bars["symbol"].map(from_alpaca), date=pd.to_datetime(bars["date"]))
    W = _wide(bars)
    P = W[[s for s in stocks if s in W.columns]]
    E = W[[f for f in funds if f in W.columns]].reindex(P.index)
    if len(P) < cfg.window + 1:
        raise BrokerError(f"only {len(P)} trading days of prices came back; the model needs {cfg.window + 1}")
    if "SPY" not in E.columns or E["SPY"].iloc[-cfg.window:].isna().any():
        raise BrokerError("no complete SPY prices came back, and SPY is the hedge of last resort")
    decision_date = pd.Timestamp(P.index[-1])
    if not view.offline and (pd.Timestamp.now().normalize() - decision_date).days > 6:
        notes.append(f"the newest price is from {decision_date.date()}, which looks stale")
    missing = sorted(set(stocks) - set(P.columns))
    if missing:
        notes.append(f"no prices for {', '.join(missing)}: not scored this time")
    last = pd.concat([P.ffill().iloc[-1], E.ffill().iloc[-1]])
    last_prices = {str(s): float(v) for s, v in last.items() if np.isfinite(v) and v > 0}

    # ---- the account
    acct = view.account()
    equity = float(acct.equity)
    positions = {from_alpaca(s): float(q) for s, q in view.positions().items() if float(q) != 0}
    held_all = {s: int(np.sign(q)) for s, q in positions.items() if s in stocks}
    blind = sorted(s for s in held_all if s not in last_prices)
    if blind:
        notes.append(f"held stocks with no prices today are left alone: {', '.join(blind)}")
    held = {s: v for s, v in held_all.items() if s not in blind}
    outside = sorted(s for s in positions if s not in stocks and s not in funds)
    if outside:
        notes.append(f"positions outside the strategy are left alone: {', '.join(outside)}")

    # ---- the signal
    fit, states, weights = al.live_signal(P, held, cfg, E, sectors)
    skipped: Dict[str, str] = {}
    unknown: List[str] = []
    for sym in sorted(s for s, v in states.items() if v < 0 and held.get(s, 0) == 0):
        ok = view.shortable(to_alpaca(sym))
        if ok is False:
            skipped[sym] = NOT_BORROWABLE
            del states[sym]
        elif ok is None:
            unknown.append(sym)
    if unknown:
        notes.append(f"could not check borrow for {', '.join(unknown)}: opened anyway, but the broker may reject the short")
    if skipped:
        weights = al.book_weights(fit, states, cfg)       # the strategy's own book without the skipped shorts

    # ---- target shares: stock legs
    targets: Dict[str, float] = {}
    for sym in P.columns:
        if sym not in last_prices:
            continue
        st, before = states.get(sym, 0), held.get(sym, 0)
        if st == 0:
            targets[sym] = 0.0
        elif st == before:
            targets[sym] = positions[sym]                 # kept: hold the current shares, no resizing
        else:
            targets[sym] = float(np.trunc(st * cfg.position * equity / last_prices[sym]))
    too_dear = sorted(s for s, v in states.items() if held.get(s, 0) == 0 and targets.get(s, 0.0) == 0.0)
    for sym in too_dear:                                   # one share costs more than the trade's size
        del states[sym]
        skipped[sym] = "one share costs more than the trade's size, so it was not opened"
    opened = sorted(s for s, v in states.items() if held.get(s, 0) == 0)
    kept = sorted(s for s, v in states.items() if held.get(s, 0) == v)
    closed = sorted(s for s, v in held.items() if states.get(s, 0) == 0)
    s_today = dict(zip(fit.symbols, fit.s))
    no_fit = [s for s in closed if not np.isfinite(s_today.get(s, np.nan))]
    if no_fit:
        notes.append(f"closed because there is no valid fit today (a gap in the last {cfg.window} days of prices, "
                     f"or no mean reversion): {', '.join(no_fit)}. This is the replication's rule")

    # ---- target shares: fund hedges, -side x beta x dollar value, summed per fund
    idx = {s: i for i, s in enumerate(fit.symbols)}
    hedge_dollars: Dict[str, float] = {f: 0.0 for f in funds}
    hedge_of: Dict[str, str] = {}
    for sym, st in states.items():
        i = idx[sym]
        fund = str(fit.hedge_etf[i])
        value = abs(targets[sym]) * last_prices[sym]
        hedge_dollars[fund] = hedge_dollars.get(fund, 0.0) - st * float(fit.beta[i, 0]) * value
        hedge_of[sym] = fund
    for fund, dollars in hedge_dollars.items():
        if fund in last_prices:
            targets[fund] = float(np.trunc(dollars / last_prices[fund]))
        elif dollars != 0:
            raise BrokerError(f"no price for {fund}, so its hedge cannot be sized")
        elif positions.get(fund, 0.0) != 0:
            notes.append(f"no price for {fund} today: the position in it is left alone")

    orders = orders_for_targets(targets, positions, last_prices)
    for sym in orders.loc[orders["action"].isin(["open short", "add to short"]) & orders["symbol"].isin(funds), "symbol"]:
        if view.shortable(to_alpaca(sym)) is False:
            notes.append(f"{sym} is not easy to borrow today: the broker will probably reject that hedge short")

    table = _signal_table(fit, sectors, held, states, skipped, stocks)
    book = _book_table(targets, positions, last_prices, states, hedge_of, hedge_dollars, equity, funds)
    open_orders = sorted(from_alpaca(s) for s in view.open_orders())
    return LivePlan(decision_date=decision_date, table=table, held=held, states=dict(states), weights=weights,
                    book=book, orders=orders, opened=opened, kept=kept, closed=closed, skipped=skipped,
                    equity=equity, cash=float(acct.cash), buying_power=float(acct.buying_power),
                    positions=positions, last_prices=last_prices, open_orders=open_orders, clock=clock,
                    cfg=cfg, notes=notes)


# --------------------------------------------------------------------------- tables for display

SIDE = {1: "long", -1: "short"}


def _today(before: int, after: int, skipped: bool) -> str:
    if skipped:
        return "not opened"
    if before == 0 and after != 0:
        return f"open {SIDE[after]}"
    if before != 0 and after == before:
        return f"keep {SIDE[before]}"
    if before != 0 and after == 0:
        return f"close {SIDE[before]}"
    return ""


def _signal_table(fit, sectors: pd.Series, held, states, skipped, stocks) -> pd.DataFrame:
    """Every stock: s-score, kappa, beta, hedge fund, eligibility, state before and after, today's action."""
    cols = ["symbol", "s", "kappa", "beta", "sigma_eq", "sector fund", "hedge fund", "eligible",
            "current state", "new state", "today"]
    rows = []
    idx = {s: i for i, s in enumerate(fit.symbols)}
    for sym in stocks:
        i = idx.get(sym)
        before, after = held.get(sym, 0), states.get(sym, 0)
        rows.append({"symbol": sym,
                     "s": float(fit.s[i]) if i is not None else np.nan,
                     "kappa": float(fit.kappa[i]) if i is not None else np.nan,
                     "beta": float(fit.beta[i, 0]) if i is not None else np.nan,
                     "sigma_eq": float(fit.sigma_eq[i]) if i is not None else np.nan,
                     "sector fund": str(sectors.get(sym, "")),
                     "hedge fund": str(fit.hedge_etf[i]) if i is not None else "",
                     "eligible": bool(fit.eligible[i]) if i is not None else False,
                     "current state": int(before), "new state": int(after),
                     "today": _today(before, after, sym in skipped)})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("s", na_position="last", kind="mergesort").reset_index(drop=True)


def _book_table(targets, positions, last_prices, states, hedge_of, hedge_dollars, equity, funds) -> pd.DataFrame:
    """The target book after today: each open stock leg and each fund hedge, in shares and dollars."""
    cols = ["symbol", "leg", "side", "current shares", "target shares", "last close", "target $", "target weight",
            "hedges"]
    rows = []
    for sym in sorted(states):
        q, px = targets[sym], last_prices[sym]
        rows.append({"symbol": sym, "leg": "stock", "side": SIDE[states[sym]], "current shares": positions.get(sym, 0.0),
                     "target shares": q, "last close": px, "target $": q * px,
                     "target weight": q * px / equity if equity else np.nan, "hedges": hedge_of.get(sym, "")})
    for fund in funds:
        q = targets.get(fund, 0.0)
        if q == 0 and positions.get(fund, 0.0) == 0:
            continue
        px = last_prices.get(fund, np.nan)
        covers = sorted(s for s, f in hedge_of.items() if f == fund)
        rows.append({"symbol": fund, "leg": "hedge", "side": "long" if q > 0 else ("short" if q < 0 else "flat"),
                     "current shares": positions.get(fund, 0.0), "target shares": q, "last close": px,
                     "target $": q * px, "target weight": q * px / equity if equity else np.nan,
                     "hedges": ", ".join(covers)})
    return pd.DataFrame(rows, columns=cols)


def score_chart_data(plan: LivePlan) -> pd.DataFrame:
    """Stocks with a valid s-score, for the chart."""
    return plan.table.dropna(subset=["s"])[["symbol", "s", "eligible", "current state", "new state", "today"]]


def bands(cfg: al.Config) -> pd.DataFrame:
    """The eq. 16 thresholds, for the chart's reference lines."""
    return pd.DataFrame({"s": [-cfg.s_open, cfg.s_open, cfg.s_close_short, -cfg.s_close_long],
                         "band": [f"open long below {-cfg.s_open:g}", f"open short above {cfg.s_open:g}",
                                  f"close short below {cfg.s_close_short:g}", f"close long above {-cfg.s_close_long:g}"]})


# --------------------------------------------------------------------------- the offline demo

def snapshot_bars() -> pd.DataFrame:
    """The committed stock and ETF snapshots as one long bar table with Alpaca tickers, like Alpaca sends."""
    parts = [pd.read_csv(f, parse_dates=["date"]) for f in (al.STOCK_FILE, al.ETF_FILE)]
    out = pd.concat(parts, ignore_index=True)
    if "close" not in out:
        out["close"] = out["adj_close"]
    if "volume" not in out:
        out["volume"] = np.nan
    out["symbol"] = out["symbol"].map(to_alpaca)
    return out[BAR_COLUMNS].sort_values(["date", "symbol"]).reset_index(drop=True)


def _demo_panels(bars: pd.DataFrame, sectors: pd.Series):
    b = bars.assign(symbol=bars["symbol"].map(from_alpaca), date=pd.to_datetime(bars["date"]))
    W = _wide(b)
    funds = [f for f in hedge_funds(sectors) if f in W.columns]
    return W[[s for s in W.columns if s in sectors.index]], W[funds]


def _replay(P: pd.DataFrame, E: pd.DataFrame, cfg: al.Config, sectors: pd.Series, first: int, last: int,
            history_days: int = HISTORY_DAYS):
    """Run the state machine from a flat book over rows first..last (inclusive) of P.

    Returns ({date: states after that day's close}, the last day's fit).
    """
    states: Dict[str, int] = {}
    out: Dict[pd.Timestamp, Dict[str, int]] = {}
    fit = None
    for t in range(max(first, cfg.window), last + 1):
        lo = max(0, t + 1 - history_days)
        fit, states, _ = al.live_signal(P.iloc[lo:t + 1], states, cfg, E.iloc[lo:t + 1], sectors)
        out[P.index[t]] = dict(states)
    return out, fit


def demo_decision_date(bars: pd.DataFrame, cfg: Optional[al.Config] = None, sectors: Optional[pd.Series] = None,
                       warmup: int = DEMO_WARMUP_DAYS, search_days: int = 40) -> pd.Timestamp:
    """The latest day in `bars` on which, for a book replayed from flat, trades open, stay open AND close.

    The demo stops its snapshot there, so one page shows all three kinds of trade. On some days
    (the snapshot's last day among them) every stock past the opening band is already held, so
    nothing opens; using such a day would hide half of what the page is for. Falls back to the
    last day when no day in the search window has all three.
    """
    cfg = cfg or live_config()
    sectors = al.load_sectors() if sectors is None else sectors
    P, E = _demo_panels(bars, sectors)
    last = len(P) - 1
    hist, _ = _replay(P, E, cfg, sectors, last - search_days - warmup, last)
    days = sorted(hist)
    for before_day, day in reversed(list(zip(days[:-1], days[1:]))):
        if days.index(before_day) < warmup:
            break
        before, after = hist[before_day], hist[day]
        opens = any(s not in before for s in after)
        keeps = any(before.get(s) == v for s, v in after.items())
        closes = any(s not in after for s in before)
        if opens and keeps and closes:
            return pd.Timestamp(day)
    return pd.Timestamp(P.index[-1])


def demo_broker(bars: pd.DataFrame, cfg: Optional[al.Config] = None, sectors: Optional[pd.Series] = None,
                equity: float = DEMO_EQUITY, warmup: int = DEMO_WARMUP_DAYS, end=None) -> MockBroker:
    """An offline MockBroker that ends on `end` and holds the trades open the day before it.

    `end` defaults to `demo_decision_date(bars)`. The book is the strategy replayed from flat over
    the `warmup` days before `end`; each open trade is `cfg.position` of equity at that day's close
    in whole shares, with its fund hedge summed per fund. Cash is set so that equity is `equity` at
    the close of `end`. The bars are cut to what the plan and the replay need, so the mock is quick.
    """
    cfg = cfg or live_config()
    sectors = al.load_sectors() if sectors is None else sectors
    bars = bars.assign(date=pd.to_datetime(bars["date"]))
    end = demo_decision_date(bars, cfg, sectors, warmup) if end is None else pd.Timestamp(end)
    dates = sorted(d for d in bars["date"].unique() if d <= end)
    recent = bars[bars["date"].isin(dates[-(HISTORY_DAYS + warmup + 1):])].reset_index(drop=True)
    P, E = _demo_panels(recent, sectors)
    hist, fit = _replay(P, E, cfg, sectors, len(P) - 1 - warmup, len(P) - 2)
    states = hist[P.index[-2]]
    px_y = pd.concat([P.iloc[:-1].ffill().iloc[-1], E.iloc[:-1].ffill().iloc[-1]])
    idx = {s: i for i, s in enumerate(fit.symbols)}
    holdings: Dict[str, float] = {}
    hedges: Dict[str, float] = {}
    for sym, st in states.items():
        q = float(np.trunc(st * cfg.position * equity / px_y[sym]))
        if q == 0:
            continue
        holdings[sym] = q
        i = idx[sym]
        fund = str(fit.hedge_etf[i])
        hedges[fund] = hedges.get(fund, 0.0) - st * float(fit.beta[i, 0]) * abs(q) * float(px_y[sym])
    for fund, dollars in hedges.items():
        holdings[fund] = float(np.trunc(dollars / float(px_y[fund])))
    holdings = {to_alpaca(s): q for s, q in holdings.items() if q != 0}
    now = recent[recent["date"] == dates[-1]].set_index("symbol")["adj_close"]
    cash = equity - sum(q * float(now.get(s, 0.0)) for s, q in holdings.items())
    return MockBroker(bars=recent, cash=cash, positions=holdings)


__all__ = [
    "BrokerError", "CONFIRM_WORD", "HISTORY_DAYS", "LivePlan", "MARKET_TZ", "ReadOnlyBroker", "Refused",
    "bands", "build_live_plan", "client_order_id", "connect_alpaca", "demo_broker", "demo_decision_date",
    "drop_today_until_close", "equity_history", "from_alpaca", "hedge_funds", "live_config", "orders_for_targets",
    "recent_orders", "results_table", "score_chart_data", "snapshot_bars", "submit_plan", "to_alpaca",
]
