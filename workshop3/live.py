"""The non-UI half of the paper-trading dashboard: bars in, scores, weights and orders out.

Everything here is a plain function so it can be tested without Streamlit and without the network.
The dashboard (workshop3/dashboard.py) only draws what these functions return.

Safety, by design:

* `build_live_plan` receives a `ReadOnlyBroker`, an object with no `submit` method at all, so the
  preview path cannot send an order even by mistake.
* `submit_plan` is the ONLY function here that calls `broker.submit`. It refuses unless the broker
  says `paper is True`, the user ticked the box AND typed SUBMIT, the orders are the ones the user
  saw, and no open orders exist for any of the symbols.
* Credentials are never stored here. `connect_alpaca` hands them straight to `AlpacaPaperBroker`,
  whose repr never shows its clients.

Symbols: the strategy and the snapshot use Yahoo tickers (`BRK-B`), Alpaca uses `BRK.B`. Everything
that talks to the broker goes through `to_alpaca`, everything that comes back goes through
`from_alpaca`, so the strategy only ever sees Yahoo tickers.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `quantsoc_ml` imports when run from anywhere
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import statarb as S  # noqa: E402
from quantsoc_ml.paper import (  # noqa: E402
    MARKET_TZ, AlpacaPaperBroker, BrokerError, MockBroker, OrderResult, ReadOnlyBroker, Refused,
    drop_unfinished_day, propose_orders,
)

CLIENT_ID_PREFIX = "qsml-ws3"
CONFIRM_WORD = "SUBMIT"
EXTRA_DAYS = 10                         # a little more history than the longest lag needs


# --------------------------------------------------------------------------- symbols

def to_alpaca(symbol: str) -> str:
    """Yahoo ticker to Alpaca ticker: class shares use a dot at Alpaca (BRK-B becomes BRK.B)."""
    return str(symbol).upper().replace("-", ".")


def from_alpaca(symbol: str) -> str:
    """Alpaca ticker back to the Yahoo ticker the strategy was trained on (BRK.B becomes BRK-B)."""
    return str(symbol).upper().replace(".", "-")


# --------------------------------------------------------------------------- brokers

def connect_alpaca(api_key: str, secret_key: str) -> AlpacaPaperBroker:
    """The Alpaca PAPER broker. Kept as one function so tests can swap in a fake."""
    return AlpacaPaperBroker(api_key, secret_key)


def snapshot_bars_for_broker(long: pd.DataFrame) -> pd.DataFrame:
    """The committed snapshot with Alpaca tickers, so the offline mock behaves like Alpaca does."""
    out = long.copy()
    out["symbol"] = out["symbol"].map(to_alpaca)
    return out


def demo_broker(strategy: S.LiveStrategy, bars: pd.DataFrame, equity: float = 100_000.0) -> MockBroker:
    """An offline MockBroker that already holds YESTERDAY's book, so the demo shows a real rebalance.

    `bars` is the snapshot with Alpaca tickers. Yesterday's book is what the same strategy would
    have targeted one trading day earlier, sized on `equity`.
    """
    bars = bars.assign(date=pd.to_datetime(bars["date"]))
    dates = sorted(bars["date"].unique())
    keep = dates[-(strategy.history_days + EXTRA_DAYS + 1):]
    recent = bars[bars["date"].isin(keep)]
    yesterday = recent[recent["date"] < keep[-1]].copy()
    yesterday["symbol"] = yesterday["symbol"].map(from_alpaca)
    P = S.wide(yesterday, "adj_close")
    w = strategy.weights(strategy.scores(P))
    prices_now = recent[recent["date"] == keep[-1]].set_index("symbol")["adj_close"]
    holdings: Dict[str, float] = {}
    for sym, weight in w[w != 0].items():
        px = float(P[sym].ffill().iloc[-1])
        holdings[to_alpaca(sym)] = float(np.trunc(weight * equity / px))
    cash = equity - sum(q * float(prices_now.get(s, 0.0)) for s, q in holdings.items())
    return MockBroker(bars=recent, cash=cash, positions=holdings)


# --------------------------------------------------------------------------- the plan

@dataclass
class LivePlan:
    decision_date: pd.Timestamp
    scores: pd.Series                   # P(beats tomorrow's median), Yahoo tickers, highest first
    weights: pd.Series                  # target weights, fraction of equity, Yahoo tickers
    skipped_shorts: Dict[str, str]      # short candidates passed over, with the reason
    orders: pd.DataFrame                # propose_orders output plus an alpaca_symbol column
    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, float]         # Yahoo tickers, negative = short
    last_prices: Dict[str, float]       # Yahoo tickers, last completed close
    open_orders: List[str]              # Yahoo tickers with an order still open
    clock: dict
    notes: List[str] = field(default_factory=list)

    @property
    def longs(self) -> pd.Series:
        return self.scores[self.weights[self.weights > 0].index].sort_values(ascending=False)

    @property
    def shorts(self) -> pd.Series:
        return self.scores[self.weights[self.weights < 0].index].sort_values()

    @property
    def fingerprint(self) -> str:
        """A short hash of the orders, so a submit can check they are the ones the user looked at."""
        text = f"{self.decision_date:%Y-%m-%d}|" + self.orders[["symbol", "side", "qty"]].to_csv(index=False)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def weights_with_borrow(strategy: S.LiveStrategy, scores: pd.Series,
                        shortable: Callable[[str], Optional[bool]]) -> tuple:
    """The strategy's own weights, but every short must be borrowable.

    Start from `strategy.weights(scores)`. If a short candidate is not shortable (or not easy to
    borrow), drop it from the ranking and ask the strategy again: the next lowest-ranked stock
    moves into the short book. The long book is unchanged, because only the bottom of the ranking
    loses names. Returns (weights, skipped, unknown): skipped maps symbol to a plain-English reason,
    unknown lists shorts whose borrow could not be checked.
    `shortable` returns True, False, or None (could not check: kept, and named in the notes).
    """
    eligible = scores.dropna()
    skipped: Dict[str, str] = {}
    checked: Dict[str, Optional[bool]] = {}
    while True:
        w = strategy.weights(eligible)          # raises if fewer than 2k names are left
        bad = []
        for sym in w[w < 0].index:
            if sym not in checked:
                checked[sym] = shortable(sym)
            if checked[sym] is False:
                bad.append(sym)
        if not bad:
            unknown = sorted(s for s in w[w < 0].index if checked.get(s) is None)
            # Skipped names stay in the book at weight 0: they were scored (and ranked low), so any
            # position already held in them is closed rather than left alone.
            w = pd.concat([w, pd.Series(0.0, index=sorted(skipped))])
            return w, skipped, unknown
        for sym in bad:
            skipped[sym] = "not easy to borrow today, so not shorted (any position held in it is closed)"
        eligible = eligible.drop(index=bad)


def drop_today_until_close(bars: pd.DataFrame, clock: dict) -> pd.DataFrame:
    """Drop any bar dated today (New York) unless it is past 16:00 New York and the market is closed.

    Before the open, Alpaca can return a forming bar for today; deciding on it would use a partial
    day. After 16:00 with the market closed, today's bar is complete and is kept.
    """
    if len(bars) == 0 or clock.get("timestamp") is None:
        return bars
    now = pd.Timestamp(clock["timestamp"])
    now = now.tz_localize(MARKET_TZ) if now.tzinfo is None else now.tz_convert(MARKET_TZ)
    today = now.tz_localize(None).normalize()
    finished = (not clock.get("is_open")) and now.hour >= 16
    if finished:
        return bars
    return bars[pd.to_datetime(bars["date"]) < today]


def build_live_plan(view: ReadOnlyBroker, strategy: S.LiveStrategy, symbols: Optional[Sequence[str]] = None,
                    min_notional: float = 1.0) -> LivePlan:
    """Bars, scores, weights and orders for the last completed trading day. Cannot submit anything."""
    if hasattr(view, "submit"):
        raise TypeError("build_live_plan must receive a ReadOnlyBroker (no submit method)")
    syms = [str(s) for s in (symbols or strategy.symbols)]
    notes: List[str] = []
    clock = view.clock()
    bars = view.daily_bars([to_alpaca(s) for s in syms], strategy.history_days + EXTRA_DAYS)
    bars = drop_unfinished_day(bars, clock)
    bars = drop_today_until_close(bars, clock)
    if len(bars) == 0:
        raise BrokerError("no daily bars came back for the strategy's symbols")
    bars = bars.assign(symbol=bars["symbol"].map(from_alpaca), date=pd.to_datetime(bars["date"]))
    P = S.wide(bars, "adj_close")
    if len(P) < strategy.history_days:
        raise BrokerError(f"only {len(P)} trading days of prices came back; the strategy needs {strategy.history_days}")
    missing = sorted(set(syms) - set(P.columns))
    if missing:
        notes.append(f"no prices for {', '.join(missing)}: not traded this time")
    scores = strategy.scores(P)
    decision_date = pd.Timestamp(scores.name)
    if not view.offline and (pd.Timestamp.now().normalize() - decision_date).days > 6:
        notes.append(f"the newest price is from {decision_date.date()}, which looks stale")
    weights, skipped, unknown = weights_with_borrow(strategy, scores, lambda s: view.shortable(to_alpaca(s)))
    if unknown:
        notes.append(f"could not check borrow for {', '.join(unknown)}: kept, but the broker may reject the short")

    acct = view.account()
    positions = {from_alpaca(s): float(q) for s, q in view.positions().items()}
    last_prices = {s: float(v) for s, v in P.ffill().iloc[-1].items() if np.isfinite(v)}
    outside = sorted(s for s in positions if s not in weights.index)
    if outside:
        notes.append(f"positions the strategy does not score today are left alone: {', '.join(outside)}")
    in_book = {s: q for s, q in positions.items() if s in weights.index}
    orders = propose_orders(weights, float(acct.equity), in_book, last_prices, min_notional=min_notional)
    orders.insert(1, "alpaca_symbol", orders["symbol"].map(to_alpaca))
    open_orders = sorted(from_alpaca(s) for s in view.open_orders())
    return LivePlan(decision_date=decision_date, scores=scores, weights=weights, skipped_shorts=skipped,
                    orders=orders, equity=float(acct.equity), cash=float(acct.cash),
                    buying_power=float(acct.buying_power), positions=positions, last_prices=last_prices,
                    open_orders=open_orders, clock=clock, notes=notes)


# --------------------------------------------------------------------------- tables for display

def signal_table(plan: LivePlan) -> pd.DataFrame:
    """The long and short books with their probabilities, for the bar chart."""
    rows = [{"symbol": s, "probability": float(p), "book": "long"} for s, p in plan.longs.items()]
    rows += [{"symbol": s, "probability": float(p), "book": "short"} for s, p in plan.shorts.items()]
    return pd.DataFrame(rows, columns=["symbol", "probability", "book"])


def targets_table(plan: LivePlan) -> pd.DataFrame:
    """Target weight, dollars and whole shares for every name in the book."""
    w = plan.weights[plan.weights != 0]
    out = pd.DataFrame({"symbol": w.index, "book": np.where(w > 0, "long", "short"),
                        "probability": plan.scores[w.index].to_numpy(), "target weight": w.to_numpy()})
    out["target $"] = out["target weight"] * plan.equity
    out["last close"] = out["symbol"].map(plan.last_prices)
    out["target shares"] = np.trunc(out["target $"] / out["last close"])
    return out.reset_index(drop=True)


def positions_table(plan: LivePlan) -> pd.DataFrame:
    """Every current position and every target, side by side. Market value is estimated at the last close."""
    names = sorted(set(plan.positions) | set(plan.weights[plan.weights != 0].index))
    rows = []
    for s in names:
        qty = plan.positions.get(s, 0.0)
        px = plan.last_prices.get(s, float("nan"))
        target_w = float(plan.weights.get(s, 0.0)) if s in plan.weights.index else float("nan")
        target_q = float(np.trunc(target_w * plan.equity / px)) if np.isfinite(target_w) and px > 0 else float("nan")
        rows.append({"symbol": s, "current shares": qty, "market value": qty * px,
                     "current weight": qty * px / plan.equity if plan.equity else float("nan"),
                     "target weight": target_w, "target shares": target_q,
                     "change": target_q - qty if np.isfinite(target_q) else float("nan"),
                     "in strategy": s in plan.weights.index})
    return pd.DataFrame(rows, columns=["symbol", "current shares", "market value", "current weight",
                                       "target weight", "target shares", "change", "in strategy"])


# --------------------------------------------------------------------------- read-only account views

def _alpaca_client(broker):
    """The alpaca-py TradingClient inside an AlpacaPaperBroker, or None. Used here only, never shown."""
    return getattr(broker, "_trading", None) if isinstance(broker, AlpacaPaperBroker) else None


def equity_history(broker, period: str = "3M") -> Optional[pd.DataFrame]:
    """Daily account equity from Alpaca's portfolio history, or None when there is none (offline demo)."""
    client = _alpaca_client(broker)
    if client is None:
        fn = getattr(broker, "equity_history", None)
        return fn() if callable(fn) else None
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    try:
        h = client.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe="1D"))
    except Exception as e:
        raise BrokerError(broker.scrub(f"could not read portfolio history: {e}")) from e
    ts = getattr(h, "timestamp", None) or []
    eq = getattr(h, "equity", None) or []
    if not ts:
        return None
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(MARKET_TZ).tz_localize(None).normalize(),
                       "equity": pd.to_numeric(pd.Series(eq), errors="coerce")})
    return df.dropna()


def recent_orders(broker, limit: int = 30) -> pd.DataFrame:
    """The most recent orders (any status) as a plain table."""
    cols = ["submitted", "symbol", "side", "qty", "filled", "status", "client_order_id"]
    client = _alpaca_client(broker)
    if client is None:
        orders = getattr(broker, "orders", {}) or {}
        rows = [{"submitted": "", "symbol": from_alpaca(o.symbol), "side": o.side, "qty": o.qty,
                 "filled": o.filled_qty, "status": o.status, "client_order_id": o.client_order_id}
                for o in list(orders.values())[-limit:]][::-1]
        return pd.DataFrame(rows, columns=cols)
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
    except Exception as e:
        raise BrokerError(broker.scrub(f"could not read recent orders: {e}")) from e
    val = lambda x: str(getattr(x, "value", x))
    rows = [{"submitted": pd.Timestamp(o.submitted_at).tz_convert(MARKET_TZ).strftime("%Y-%m-%d %H:%M") if o.submitted_at else "",
             "symbol": from_alpaca(o.symbol), "side": val(o.side), "qty": float(o.qty or 0),
             "filled": float(o.filled_qty or 0), "status": val(o.status), "client_order_id": o.client_order_id}
            for o in orders]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- submit

def client_order_id(decision_date, alpaca_symbol: str, action: str, prefix: str = CLIENT_ID_PREFIX) -> str:
    """One id per decision date, symbol and leg. Alpaca refuses a reused id, so a double click cannot duplicate."""
    leg = str(action).replace(" ", "-")
    return f"{prefix}-{pd.Timestamp(decision_date):%Y%m%d}-{alpaca_symbol}-{leg}"


def submit_plan(broker, plan: LivePlan, confirmed: bool, typed: str,
                expected_fingerprint: Optional[str] = None) -> List[OrderResult]:
    """Send the plan's orders to a PAPER broker. The ONLY function here that calls broker.submit.

    Refuses (raises `Refused`, nothing sent) unless: the broker says `paper is True`; `confirmed`
    is True and `typed` is exactly SUBMIT; the orders are the ones the user looked at (when a
    fingerprint is given); and the broker has no open orders for any symbol in the plan. Orders go
    in the plan's order, which puts exits and reductions before new exposure.
    """
    if getattr(broker, "paper", False) is not True:
        raise Refused("refused: this broker does not say paper=True, and this app only sends PAPER orders")
    if not callable(getattr(broker, "submit", None)):
        raise Refused("refused: this is a read-only broker view")
    if confirmed is not True or str(typed).strip() != CONFIRM_WORD:
        raise Refused(f"refused: tick the box and type {CONFIRM_WORD} first")
    if expected_fingerprint is not None and expected_fingerprint != plan.fingerprint:
        raise Refused("refused: the proposed orders changed since the page was drawn; look at them again, then submit")
    if len(plan.orders) == 0:
        return []
    wanted = set(plan.orders["alpaca_symbol"])
    busy = sorted(wanted & set(broker.open_orders()))
    if busy:
        raise Refused(f"refused: open orders already exist for {', '.join(busy)}. Wait for them to fill, "
                      "or cancel them on the Alpaca website, then try again")
    closed = not plan.clock.get("is_open")
    if hasattr(broker, "wait_seconds") and closed:
        broker.wait_seconds = 0.0       # market closed: orders wait for the open, so do not poll for fills
    crossing = set(plan.orders.loc[plan.orders["symbol"].duplicated(keep=False), "symbol"])
    results: List[OrderResult] = []
    for row in plan.orders.itertuples(index=False):
        coid = client_order_id(plan.decision_date, row.alpaca_symbol, row.action,
                               getattr(plan, "order_id_prefix", CLIENT_ID_PREFIX))   # each workshop's own ids
        if closed and row.symbol in crossing and str(row.action).startswith("open"):
            # Long to short (or back) while closed: the close leg is still pending, so the broker would
            # reject the opening leg. Send it on the next run, after the close leg has filled.
            results.append(OrderResult(row.alpaca_symbol, row.side, float(row.qty), "not sent", 0.0, coid,
                                       "market closed: send this leg after the closing leg fills"))
            continue
        try:
            res = broker.submit(row.alpaca_symbol, row.side, float(row.qty), coid)
        except BrokerError as e:
            res = OrderResult(row.alpaca_symbol, row.side, float(row.qty), "rejected", 0.0, coid, str(e))
        results.append(res)
    return results


def results_table(results: Sequence[OrderResult]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": r.symbol, "side": r.side, "qty": r.qty, "status": r.status,
                          "filled": r.filled_qty, "client_order_id": r.client_order_id, "message": r.message}
                         for r in results],
                        columns=["symbol", "side", "qty", "status", "filled", "client_order_id", "message"])
