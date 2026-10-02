"""Alpaca paper-trading runner internals: daily bars in, proposed orders out, and (only on request) submission.

One run of `run()` is ONE rebalance of the exported strategy:

  1. a truthful mode banner, printed before anything else
  2. daily, split- and dividend-adjusted bars for the strategy's symbols
  3. the bundle's strategy.py turns them into forecasts for the last completed day, then weights
  4. account equity and current positions from the broker
  5. target whole-share positions, and the orders that move today's positions to them
  6. preview: print the table and stop. submit: ask for confirmation, then send to the PAPER endpoint

Safety, by design (the wrapped broker is kept on a private attribute, not re-exposed) rather than by care:

* The preview path only ever receives a `ReadOnlyBroker`, an object that has no `submit` method
  at all. The only function here that calls `submit` is `_submit_path`, and `run` reaches it only
  when `mode == "submit"`, the broker reports `paper is True`, and a confirmation callback
  returned True. tests/test_export_paper.py checks all three.
* `AlpacaPaperBroker` hard-wires `paper=True`: this module cannot reach a live-money account.
* Credentials are read by `get_credentials` and handed straight to the alpaca-py clients. They
  are never printed, logged, stored on the adapter's repr, or written to any file.

alpaca-py is imported lazily inside the functions that need it, so the MockBroker and the
offline demo work on a machine (or Colab runtime) where alpaca-py is not installed.

Simplification stated out loud: forecasts use the close of the last completed trading day,
and market orders fill at the next open, at a price the runner cannot know in advance. The
notional column is an estimate at the last close.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import HORIZON
from .data import to_wide
from .features import HIGH_WINDOW, VOLUME_WINDOW

CRED_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
MODES = ("preview", "submit")
MARKET_TZ = "America/New_York"
CLIENT_ID_PREFIX = "qsml"
BAR_COLUMNS = ["date", "symbol", "close", "adj_close", "volume"]
ORDER_COLUMNS = ["symbol", "side", "qty", "notional", "current_qty", "target_qty", "action"]
LINE = "=" * 78


class BrokerError(Exception):
    """The broker refused or could not perform a request. The message is safe to print."""


class Refused(Exception):
    """The runner refused to go on (wrong mode, not confirmed, not the paper endpoint, open orders)."""


# --------------------------------------------------------------------------- credentials and banner

def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _read_env_file(path) -> Dict[str, str]:
    """Minimal .env reader (KEY=VALUE per line, # comments, optional quotes) for when python-dotenv is absent."""
    out: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def get_credentials(dotenv_paths: Optional[Iterable[Union[str, Path]]] = None) -> Tuple[Optional[str], Optional[str], str]:
    """(key, secret, source) from Colab secrets, then environment variables, then a .env file.

    `source` is a description of WHERE the keys came from ("Colab secrets", "environment
    variables", "<path> file") or "not found". The key and secret are never printed here, and
    callers must not print them either: print `source` only.
    """
    if _is_colab():
        try:
            from google.colab import userdata
            key, secret = userdata.get(CRED_KEYS[0]), userdata.get(CRED_KEYS[1])
            if key and secret:
                return key, secret, "Colab secrets"
        except Exception:
            pass
    key, secret = os.environ.get(CRED_KEYS[0]), os.environ.get(CRED_KEYS[1])
    if key and secret:
        return key, secret, "environment variables"
    paths = [Path(p) for p in (dotenv_paths or [Path.cwd() / ".env"])]
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            from dotenv import dotenv_values
        except ImportError:                 # python-dotenv missing: read KEY=VALUE lines ourselves
            dotenv_values = _read_env_file
        vals = dotenv_values(path)
        key, secret = vals.get(CRED_KEYS[0]), vals.get(CRED_KEYS[1])
        if key and secret:
            return key, secret, f"{path} file"
    return None, None, "not found"


def mode_banner(mode: str, offline: bool = False) -> str:
    """The banner printed before anything else. It says exactly what the run will and will not do."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if offline:
        text = {"preview": "PREVIEW (OFFLINE DEMO): committed price snapshot and a mock broker, NO orders submitted",
                "submit": "MOCK SUBMIT: orders go to an in-memory mock broker with scripted fills, nothing leaves this computer"}[mode]
    else:
        text = {"preview": "PREVIEW: real paper-account data, NO orders submitted",
                "submit": "PAPER SUBMIT: orders go to the Alpaca PAPER endpoint only"}[mode]
    return f"{LINE}\n  {text}\n{LINE}"


# --------------------------------------------------------------------------- data

def rows_needed(feature_names: Sequence[str]) -> int:
    """Trading days of history a feature list needs before its last row is complete."""
    need = 2
    for name in feature_names:
        m = re.match(r"^(ret|vol)_(\d+)$", name)
        if m:
            need = max(need, int(m.group(2)) + 1)
        elif name == "volume_z":
            need = max(need, VOLUME_WINDOW)
        elif name == "hi52":
            need = max(need, HIGH_WINDOW)
    return need


def fetch_daily_bars(symbols: Sequence[str], lookback_days: int, client, feed: str = "iex") -> pd.DataFrame:
    """Daily bars from alpaca-py's StockHistoricalDataClient, as a long table [date, symbol, close, adj_close, volume].

    The request uses TimeFrame.Day and adjustment="all", so the prices are split AND dividend
    adjusted, the same kind of price as the training data's `adj_close` (Yahoo's adjusted close).
    `close` and `adj_close` are therefore the same column here. The free data plan serves the IEX
    feed, whose volume is only IEX's share of trading; `volume_z` compares a stock's volume with
    its own recent average, so the scale cancels, but the feature is noisier than in training.

    `lookback_days` is in TRADING days; the request spans enough calendar days to cover them and
    the last `lookback_days` dates per symbol are kept. Dates are New York calendar dates.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=int(math.ceil(int(lookback_days) * 365 / 252)) + 14)
    req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day, start=start.to_pydatetime(),
                           end=end.to_pydatetime(), adjustment=Adjustment.ALL, feed=DataFeed(feed))
    try:
        barset = client.get_stock_bars(req)
    except Exception as e:  # alpaca APIError, network errors: the message carries no credential
        raise BrokerError(f"could not download daily bars from Alpaca: {e}") from e
    df = barset.df
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    out = pd.DataFrame({"date": ts.dt.tz_localize(None).dt.normalize(), "symbol": df["symbol"].astype(str),
                        "close": df["close"].astype(float), "volume": df["volume"].astype(float)})
    out["adj_close"] = out["close"]
    out = out.sort_values(["date", "symbol"])
    keep = sorted(out["date"].unique())[-int(lookback_days):]
    return out[out["date"].isin(keep)][BAR_COLUMNS].reset_index(drop=True)


def build_panels(bars: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(prices_wide, volume_wide), dates down and symbols across, from a long bar table.

    Accepts the Alpaca table from `fetch_daily_bars` and the committed snapshot alike (both
    carry `adj_close` and `volume`), so live and offline runs go through the same code.
    """
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    return to_wide(bars, "adj_close"), to_wide(bars, "volume")


def build_latest_features(prices_wide: pd.DataFrame, volume_wide: pd.DataFrame, feature_names: Sequence[str],
                          rank_features: bool = True) -> pd.DataFrame:
    """The model's input rows for the LAST date (symbol x feature), exactly as training built them.

    features.build_features, then (when rank_features, the default) features.rank_normalise over
    the whole table, so each value is a rank among the symbols present on that date. For display
    and checks: the bundle's strategy.py does the same inside predict_returns.
    """
    from .features import build_features, rank_normalise
    feats = build_features(prices_wide, volume_wide, names=list(feature_names))
    if rank_features:
        feats = rank_normalise(feats)
    last = feats.index.get_level_values("date").max()
    return feats.xs(last, level="date")[list(feature_names)]


def drop_unfinished_day(bars: pd.DataFrame, clock: Mapping) -> pd.DataFrame:
    """While the market is open, today's daily bar is still forming: drop it so features use completed days only."""
    if not clock.get("is_open") or len(bars) == 0:
        return bars
    today = pd.Timestamp(clock["timestamp"]).tz_convert(MARKET_TZ).tz_localize(None).normalize()
    return bars[pd.to_datetime(bars["date"]) < today]


# --------------------------------------------------------------------------- orders

def _qty_map(positions) -> Dict[str, float]:
    if positions is None:
        return {}
    if isinstance(positions, Mapping):
        return {str(k): float(v) for k, v in positions.items()}
    return {str(p.symbol): float(p.qty) for p in positions}


def _action(before: float, after: float) -> Tuple[int, str]:
    """(phase, label) of one leg. Phase 0 reduces exposure (frees buying power), phase 1 adds it."""
    if before >= 0 and after >= 0:
        if after > before:
            return 1, "open long" if before == 0 else "add to long"
        return 0, "close long" if after == 0 else "reduce long"
    if after > before:
        return 0, "close short" if after == 0 else "reduce short"
    return 1, "open short" if before == 0 else "add to short"


def propose_orders(weights: pd.Series, equity: float, positions, last_prices: Mapping[str, float],
                   min_notional: float = 1.0) -> pd.DataFrame:
    """Orders that move current positions to the target weights, in whole shares.

    target_qty = weight * equity / last price, rounded TOWARDS ZERO (so a position never exceeds
    its target size). Negative weights and quantities are short positions. A move that crosses
    zero (long to short, or short to long) is split into two legs, close then open, because a
    broker treats them as different trades. Symbols not in `weights` are not touched.

    Order is deterministic: legs that reduce exposure first (they free buying power), then legs
    that add it, each group by symbol. A MOVE (both legs together) smaller than `min_notional`
    dollars is skipped; a crossing move is never split by that test, since dropping its close leg
    would turn the two legs back into the single crossing order the split exists to prevent.
    Columns: symbol, side, qty, notional (estimate at the last price), current_qty, target_qty, action.
    """
    w = pd.Series(weights, dtype=float)
    held = _qty_map(positions)
    rows = []
    for sym in sorted(map(str, w.index)):
        weight = float(w[sym]) if np.isfinite(w[sym]) else 0.0
        cur = held.get(sym, 0.0)
        px = last_prices.get(sym) if hasattr(last_prices, "get") else None
        px = float(px) if px is not None and np.isfinite(px) and px > 0 else None
        if px is None:
            if weight != 0:
                raise ValueError(f"no price for {sym}, so its target position cannot be sized")
            target = 0.0
        else:
            target = float(math.trunc(weight * float(equity) / px))
        if target == cur:
            continue
        if px is not None and abs(target - cur) * px < float(min_notional):
            continue
        if cur != 0 and target != 0 and (cur > 0) != (target > 0):
            legs = [(cur, 0.0), (0.0, target)]
        else:
            legs = [(cur, target)]
        for before, after in legs:
            delta = after - before
            qty = round(abs(delta), 6)
            notional = qty * px if px is not None else float("nan")
            if qty == 0:
                continue
            phase, label = _action(before, after)
            rows.append({"_phase": phase, "symbol": sym, "side": "buy" if delta > 0 else "sell", "qty": qty,
                         "notional": round(notional, 2), "current_qty": cur, "target_qty": target, "action": label})
    if not rows:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    out = pd.DataFrame(rows)
    out["_leg"] = range(len(out))                      # keeps close-before-open within a crossing symbol
    out = out.sort_values(["_phase", "symbol", "_leg"], kind="mergesort")
    return out[ORDER_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- brokers

@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    status: str             # filled, partially_filled, accepted (waiting for the market), rejected, ...
    filled_qty: float
    client_order_id: str
    message: str = ""


class ReadOnlyBroker:
    """A view of a broker with every read method and NO submit method. The preview path only gets this."""

    __slots__ = ("_b",)
    READ_METHODS = ("account", "positions", "open_orders", "clock", "daily_bars", "shortable")

    def __init__(self, broker):
        object.__setattr__(self, "_b", broker)

    name = property(lambda self: self._b.name)
    paper = property(lambda self: bool(getattr(self._b, "paper", False)))
    offline = property(lambda self: bool(getattr(self._b, "offline", False)))

    def account(self) -> AccountInfo:
        return self._b.account()

    def positions(self) -> Dict[str, float]:
        return self._b.positions()

    def open_orders(self) -> List[str]:
        return self._b.open_orders()

    def clock(self) -> dict:
        return self._b.clock()

    def daily_bars(self, symbols, lookback_days) -> pd.DataFrame:
        return self._b.daily_bars(symbols, lookback_days)

    def shortable(self, symbol) -> Optional[bool]:
        return self._b.shortable(symbol)

    def __setattr__(self, key, value):
        raise AttributeError("ReadOnlyBroker cannot be changed")


class AlpacaPaperBroker:
    """alpaca-py adapter, PAPER endpoint only (`paper=True` is hard-wired). Same interface as MockBroker.

    Interface: name, paper, offline, account(), positions() -> {symbol: qty, negative = short},
    open_orders() -> [symbol], clock() -> {is_open, timestamp}, daily_bars(symbols, lookback_days),
    shortable(symbol) -> bool | None, submit(symbol, side, qty, client_order_id) -> OrderResult.
    """

    name = "alpaca-paper"
    paper = True
    offline = False

    def __init__(self, api_key: str, secret_key: str, feed: str = "iex", wait_seconds: float = 5.0):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        if not api_key or not secret_key:
            raise BrokerError(f"Alpaca paper keys are required ({CRED_KEYS[0]} and {CRED_KEYS[1]})")
        self._trading = TradingClient(api_key, secret_key, paper=True)
        self._data = StockHistoricalDataClient(api_key, secret_key)
        self.feed = feed
        self.wait_seconds = float(wait_seconds)

    def __repr__(self) -> str:      # never show the clients, which hold the keys
        return f"AlpacaPaperBroker(feed={self.feed!r}, paper=True)"

    @staticmethod
    def _call(fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            raise BrokerError(f"Alpaca request failed: {e}") from e

    def account(self) -> AccountInfo:
        a = self._call(self._trading.get_account)
        return AccountInfo(equity=float(a.equity), cash=float(a.cash), buying_power=float(a.buying_power))

    def positions(self) -> Dict[str, float]:
        out = {}
        for p in self._call(self._trading.get_all_positions):
            q = abs(float(p.qty))
            side = str(getattr(p.side, "value", p.side)).lower()
            out[str(p.symbol)] = -q if side == "short" else q
        return out

    def open_orders(self) -> List[str]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        orders = self._call(self._trading.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return sorted({str(o.symbol) for o in orders})

    def clock(self) -> dict:
        c = self._call(self._trading.get_clock)
        return {"is_open": bool(c.is_open), "timestamp": pd.Timestamp(c.timestamp), "next_open": c.next_open}

    def daily_bars(self, symbols, lookback_days) -> pd.DataFrame:
        return fetch_daily_bars(symbols, lookback_days, self._data, feed=self.feed)

    def shortable(self, symbol) -> Optional[bool]:
        try:
            a = self._trading.get_asset(symbol)
        except Exception:
            return None
        return bool(a.shortable) and bool(a.easy_to_borrow)

    def submit(self, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY, client_order_id=client_order_id)
        try:
            o = self._trading.submit_order(req)
        except Exception as e:
            raise BrokerError(f"order rejected by Alpaca: {e}") from e
        deadline = time.time() + self.wait_seconds
        status = str(getattr(o.status, "value", o.status))
        while status not in ("filled", "rejected", "canceled", "expired") and time.time() < deadline:
            time.sleep(1.0)
            try:
                o = self._trading.get_order_by_id(o.id)
            except Exception:
                break
            status = str(getattr(o.status, "value", o.status))
        msg = "waiting for the market to open" if status in ("accepted", "new", "pending_new") else ""
        return OrderResult(symbol=symbol, side=side, qty=float(qty), status=status,
                           filled_qty=float(o.filled_qty or 0), client_order_id=client_order_id, message=msg)


class MockBroker:
    """In-memory broker with scripted responses, for the offline demo and the tests. No network, ever.

    Fills market orders at once at the last close. Scripts:
      reject_symbols : orders for these symbols are rejected
      partial_fill   : {symbol: fraction} fills only that fraction; the rest stays open
      script="demo"  : rejects the FIRST order that opens or adds to a short (as if the stock could
                       not be borrowed) and fills only half of the FIRST buy, so a demo shows both
    `paper` is True because nothing here can touch real money; `offline` makes the banner say so.
    """

    name = "mock"
    paper = True
    offline = True

    def __init__(self, bars: Optional[pd.DataFrame] = None, cash: float = 100_000.0,
                 positions: Optional[Mapping[str, float]] = None, prices: Optional[Mapping[str, float]] = None,
                 reject_symbols: Iterable[str] = (), partial_fill: Optional[Mapping[str, float]] = None,
                 script: Optional[str] = None, is_open: bool = False, not_shortable: Iterable[str] = ()):
        self.bars = None if bars is None else bars.assign(date=pd.to_datetime(bars["date"]))
        self.cash = float(cash)
        self.holdings: Dict[str, float] = {str(k): float(v) for k, v in (positions or {}).items()}
        if prices is None and self.bars is not None:
            last = self.bars.sort_values("date").groupby("symbol")["adj_close"].last()
            prices = last.to_dict()
        self.prices: Dict[str, float] = {str(k): float(v) for k, v in (prices or {}).items()}
        self.reject_symbols = set(reject_symbols)
        self.partial_fill = dict(partial_fill or {})
        self.not_shortable = set(not_shortable)
        self.script = script
        self.is_open = bool(is_open)
        self.submissions: List[dict] = []
        self.orders: Dict[str, OrderResult] = {}
        self._rejected_a_short = False
        self._partial_a_buy = False

    def account(self) -> AccountInfo:
        eq = self.cash + sum(q * self.prices.get(s, 0.0) for s, q in self.holdings.items())
        return AccountInfo(equity=eq, cash=self.cash, buying_power=2.0 * eq)

    def positions(self) -> Dict[str, float]:
        return {s: q for s, q in sorted(self.holdings.items()) if abs(q) > 1e-12}

    def open_orders(self) -> List[str]:
        return sorted({o.symbol for o in self.orders.values() if o.status in ("accepted", "partially_filled")})

    def clock(self) -> dict:
        ts = self.bars["date"].max() if self.bars is not None and len(self.bars) else pd.Timestamp.now()
        return {"is_open": self.is_open, "timestamp": pd.Timestamp(ts).tz_localize(MARKET_TZ) + pd.Timedelta(hours=17)}

    def daily_bars(self, symbols, lookback_days) -> pd.DataFrame:
        if self.bars is None:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = self.bars[self.bars["symbol"].isin(list(symbols))]
        keep = sorted(df["date"].unique())[-int(lookback_days):]
        df = df[df["date"].isin(keep)].copy()
        if "close" not in df:
            df["close"] = df["adj_close"]
        return df[BAR_COLUMNS].reset_index(drop=True)

    def shortable(self, symbol) -> Optional[bool]:
        return symbol not in self.not_shortable

    def submit(self, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        if client_order_id in self.orders:
            raise BrokerError(f"client_order_id {client_order_id} was already used (duplicate order refused)")
        self.submissions.append({"symbol": symbol, "side": side, "qty": float(qty), "client_order_id": client_order_id})
        cur = self.holdings.get(symbol, 0.0)
        opens_short = side == "sell" and qty > max(cur, 0.0) + 1e-9
        reason = ""
        if symbol in self.reject_symbols:
            reason = "rejected by the broker (scripted)"
        elif opens_short and symbol in self.not_shortable:
            reason = f"{symbol} is not easy to borrow, so it cannot be sold short (scripted)"
        elif self.script == "demo" and opens_short and not self._rejected_a_short:
            self._rejected_a_short = True
            reason = f"{symbol} is not easy to borrow, so it cannot be sold short (scripted demo)"
        if reason:
            self.orders[client_order_id] = OrderResult(symbol, side, float(qty), "rejected", 0.0, client_order_id, reason)
            raise BrokerError(reason)
        fraction = float(self.partial_fill.get(symbol, 1.0))
        if self.script == "demo" and side == "buy" and not self._partial_a_buy:
            self._partial_a_buy = True
            fraction = 0.5
        filled = float(math.floor(qty * fraction)) if fraction < 1 else float(qty)
        px = self.prices[symbol]
        sign = 1.0 if side == "buy" else -1.0
        self.holdings[symbol] = cur + sign * filled
        self.cash -= sign * filled * px
        status = "filled" if filled >= qty else "partially_filled"
        msg = "" if status == "filled" else f"{filled:g} of {qty:g} filled, the rest is still open (scripted)"
        res = OrderResult(symbol, side, float(qty), status, filled, client_order_id, msg)
        self.orders[client_order_id] = res
        return res


# --------------------------------------------------------------------------- the run

@dataclass
class Plan:
    decision_date: pd.Timestamp
    forecasts: pd.Series
    weights: pd.Series
    orders: pd.DataFrame
    equity: float
    book: float
    buying_power: float
    open_orders: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class RunReport:
    mode: str
    plan: Plan
    submitted: List[OrderResult] = field(default_factory=list)


def build_plan(view: ReadOnlyBroker, bundle, symbols: Optional[Sequence[str]] = None, lookback_days: int = 300,
               capital_fraction: float = 0.5, min_notional: float = 1.0,
               allow_universe_change: bool = False) -> Plan:
    """Everything up to the proposed orders. Receives a ReadOnlyBroker, so it cannot submit.

    `symbols` must be the universe the model was trained on (the bundle's list): with
    rank-normalised features every forecast depends on which other stocks are in the table, so a
    different list silently changes every forecast. Pass allow_universe_change=True to do it anyway.
    A held symbol that has no forecast today (missing data) is left alone and named in the notes,
    never sold because its weight came out as 0.
    """
    if not 0 < float(capital_fraction) <= 1:
        raise ValueError("capital_fraction must be above 0 and at most 1")
    syms = [str(s).upper() for s in (symbols or bundle.symbols)]
    trained = [str(s).upper() for s in bundle.symbols]
    if symbols is not None and trained and sorted(set(syms)) != sorted(set(trained)) and not allow_universe_change:
        raise ValueError("these symbols differ from the universe the model was trained on "
                         f"({len(trained)} symbols in the bundle); the features are ranked across the table, "
                         "so a different list changes every forecast. Drop --symbols, or pass "
                         "--allow-universe-change if you understand that")
    need = rows_needed(bundle.feature_names)
    if int(lookback_days) < need:
        raise ValueError(f"the features need {need} trading days of history; lookback_days={lookback_days} is too short")
    notes: List[str] = []
    bars = drop_unfinished_day(view.daily_bars(syms, int(lookback_days)), view.clock())
    if len(bars) == 0:
        raise BrokerError("no daily bars came back for these symbols")
    P, V = build_panels(bars)
    missing = [s for s in syms if s not in P.columns]
    if missing:
        notes.append(f"no data for {', '.join(missing)}: not traded this time")
    if len(P) < need:
        raise BrokerError(f"only {len(P)} trading days of bars came back; the features need {need}")
    decision_date = P.index.max()
    if not view.offline and (pd.Timestamp.now().normalize() - decision_date).days > 6:
        notes.append(f"the newest bar is from {decision_date.date()}, which looks stale")
    forecasts = bundle.module.predict_returns(P, V, bundle.model, bundle.config)
    weights = bundle.module.target_weights(forecasts, bundle.config)
    acct = view.account()
    positions = view.positions()
    outside = sorted(s for s in positions if s not in weights.index)
    if outside:
        notes.append(f"positions outside the strategy are left alone: {', '.join(outside)}")
    no_forecast = sorted(s for s in positions if s in weights.index
                         and not np.isfinite(float(forecasts.get(s, float("nan")))))
    if no_forecast:
        notes.append(f"no forecast today for {', '.join(no_forecast)} (missing data): position left alone")
        weights = weights.drop(index=no_forecast)
    book = float(acct.equity) * float(capital_fraction)
    last_prices = P.ffill().iloc[-1].to_dict()
    orders = propose_orders(weights, book, {s: q for s, q in positions.items() if s in weights.index},
                            last_prices, min_notional=min_notional)
    for sym in orders.loc[orders["action"].isin(["open short", "add to short"]), "symbol"]:
        if view.shortable(sym) is False:
            notes.append(f"{sym} is not easy to borrow today: the broker will probably reject that short")
    return Plan(decision_date=decision_date, forecasts=forecasts, weights=weights, orders=orders, equity=float(acct.equity),
                book=book, buying_power=float(acct.buying_power), open_orders=list(view.open_orders()), notes=notes)


def _fmt_orders(orders: pd.DataFrame) -> str:
    if len(orders) == 0:
        return "  (no orders: the account already matches the targets)"
    t = orders.copy()
    for c in ("qty", "current_qty", "target_qty"):
        t[c] = t[c].map(lambda x: f"{x:g}")
    t["notional"] = t["notional"].map(lambda x: f"${x:,.0f}")
    return t.to_string(index=False)


def print_plan(plan: Plan, bundle, out: Callable[[str], None] = print) -> None:
    """The forecast summary and the order table, for preview and submit alike."""
    w = plan.weights
    longs, shorts = sorted(w[w > 0].index), sorted(w[w < 0].index)
    f = plan.forecasts
    def pct(s):
        return f"{s} {100 * f[s]:+.2f}%"
    out(f"decision date {plan.decision_date.date()} (last completed trading day); forecasts are {HORIZON}-day returns")
    out(f"long  ({len(longs)}): " + ", ".join(pct(s) for s in longs))
    out(f"short ({len(shorts)}): " + ", ".join(pct(s) for s in shorts))
    out(f"equity ${plan.equity:,.2f}; book sized on {100 * plan.book / plan.equity:.0f}% of it "
        f"= ${plan.book:,.0f} long and ${plan.book:,.0f} short (buying power ${plan.buying_power:,.0f})")
    out(f"proposed orders ({len(plan.orders)}), reducing legs first; notional is an estimate at the last close:")
    out(_fmt_orders(plan.orders))
    for n in plan.notes:
        out(f"note: {n}")
    if plan.open_orders:
        out(f"note: open orders already exist for {', '.join(plan.open_orders)}")


def _finish_preview(plan: Plan, bundle, out: Callable[[str], None]) -> None:
    out(f"PREVIEW: {len(plan.orders)} orders proposed, NONE submitted.")
    nxt = (plan.decision_date + pd.offsets.BDay(int(bundle.config.rebalance_days))).date()
    out(f"Your strategy rebalances every {bundle.config.rebalance_days} trading days: next run on or after about {nxt}.")


def _client_order_id(plan: Plan, i: int, row) -> str:
    return f"{CLIENT_ID_PREFIX}-{plan.decision_date:%Y%m%d}-{i:02d}-{row.symbol}-{row.side}"


def _submit_path(broker, plan: Plan, out: Callable[[str], None]) -> List[OrderResult]:
    """The ONLY function in this module that calls broker.submit. Reached from `run` in submit mode only."""
    results: List[OrderResult] = []
    for i, row in enumerate(plan.orders.itertuples(index=False)):
        coid = _client_order_id(plan, i, row)
        try:
            res = broker.submit(row.symbol, row.side, float(row.qty), coid)
        except BrokerError as e:
            res = OrderResult(row.symbol, row.side, float(row.qty), "rejected", 0.0, coid, str(e))
        results.append(res)
        extra = f"  ({res.message})" if res.message else ""
        out(f"  {res.symbol:<5} {res.side:<4} {res.qty:>7g}  -> {res.status:<16} filled {res.filled_qty:g}{extra}")
    counts = pd.Series([r.status for r in results]).value_counts().to_dict() if results else {}
    out("SUBMITTED: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing to send")
        + ". Rejected orders change nothing else; the rest of the book stands.")
    return results


def run(mode: str, broker, bundle_dir, symbols: Optional[Sequence[str]] = None, lookback_days: int = 300,
        capital_fraction: float = 0.5, min_notional: float = 1.0,
        confirm: Optional[Callable[[pd.DataFrame], bool]] = None, out: Callable[[str], None] = print,
        context: Sequence[str] = (), allow_universe_change: bool = False) -> RunReport:
    """One rebalance. mode="preview" proposes and stops; mode="submit" also sends, on three conditions:

    the broker says `paper is True`, `confirm` is given, and `confirm(orders)` returns True.
    Anything else raises `Refused` before a single order is sent. `context` lines are printed
    straight after the banner (where the data and keys came from, never the keys themselves).
    """
    from .export import load_bundle

    if mode not in MODES:
        raise Refused(f"unknown mode {mode!r}: use preview or submit")
    out(mode_banner(mode, offline=bool(getattr(broker, "offline", False))))
    for line in context:
        out(line)
    if mode == "submit":
        if getattr(broker, "paper", False) is not True:
            raise Refused("submit is only allowed against a paper broker; this one does not say paper=True")
        if confirm is None:
            raise Refused("submit needs an explicit confirmation step, and none was given")
    bundle = load_bundle(bundle_dir)
    out(f"bundle {bundle.path.name}: {bundle.source_note}")
    view = ReadOnlyBroker(broker)
    plan = build_plan(view, bundle, symbols=symbols, lookback_days=lookback_days,
                      capital_fraction=capital_fraction, min_notional=min_notional,
                      allow_universe_change=allow_universe_change)
    print_plan(plan, bundle, out)
    if mode == "preview":
        _finish_preview(plan, bundle, out)
        return RunReport(mode=mode, plan=plan)
    if mode == "submit":
        if plan.open_orders:
            raise Refused(f"open orders already exist for {', '.join(plan.open_orders)}: wait for them to fill "
                          "(or cancel them on the Alpaca website), then run again")
        if len(plan.orders) == 0:
            out("Nothing to submit.")
            return RunReport(mode=mode, plan=plan)
        if confirm(plan.orders) is not True:
            raise Refused("not confirmed, so nothing was submitted")
        return RunReport(mode=mode, plan=plan, submitted=_submit_path(broker, plan, out))
    raise Refused(f"unknown mode {mode!r}")
