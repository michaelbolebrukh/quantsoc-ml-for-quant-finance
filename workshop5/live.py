"""The non-UI half of the Workshop 5 paper-trading dashboard: bars in, chart scores, book and orders out.

The strategy is I5R5 from Jiang, Kelly & Xiu (2023), "(Re-)Imag(in)ing Price Trends": each stock's last
5 trading days are drawn as a 32x15 black and white chart, an ensemble of 5 networks (quantsoc_ml/cnn.py)
gives the probability that the stock is UP over the next 5 days, and every 5 trading days the S&P 100
is sorted into deciles of that probability: long the top decile, short the bottom decile, equal
weights, hold 5 days. This file turns the account and the latest completed days into a plan; the
dashboard (workshop5/dashboard.py) only draws what comes back.

Reused, not copied: the ticker mapping, the unfinished-day filters, the read-only broker view, the
borrow rule (`weights_with_borrow`), `propose_orders`, the table builders and `submit_plan` from
Workshop 3, and Workshop 4's hardened `submit_checked` (stale-plan refusal, open orders on a strategy
symbol block, buying power) and `same_account`. `submit_plan` stays the ONLY function that sends an
order. The plan built here has the fields those functions read, so it is passed to them unchanged.

Sizing (Workshop 3's convention): gross exposure 1.0 of equity, so $0.50 per $1 of equity long and
$0.50 short, split equally across the names on each side, whole shares rounded towards zero. A
decile is about 10 stocks, so each name is about 5% of equity. The long and short decile are exactly
the stocks `cnn.live_scores` labels decile 10 and decile 1. A short that is not easy to borrow is
dropped and the next lowest-ranked stock takes its place (`weights_with_borrow`), and the plan says so.

THE WEEKLY RULE (when to rebalance, when to hold). The strategy trades once every 5 trading days, so
the page decides from the account's own history, not from a clock or a counter it could lose:

* `last_rebalance_date` reads the account's recent orders and finds the newest DECISION DATE among
  orders whose client order id starts `qsml-ws5-` and whose FILLED QUANTITY is above zero, whatever
  their status (a partly filled order that was then cancelled did trade, so it counts; an order that
  filled nothing does not). The decision date is in the id (`qsml-ws5-YYYYMMDD-...`); ids with an
  impossible date, or no id at all, are skipped. Decision dates AFTER today's are ignored with a
  warning note (a typo or a wrong clock must not hold the strategy for ever).
* The order history read is finite (`HISTORY_LIMIT`). If that window is full and holds no filled ws5
  order, the page cannot tell "never rebalanced" from "pushed out of the window by other orders", so
  it does NOT assume a first rebalance: it holds, says why, and the override can be ticked.
* `rebalance_status` then counts the trading days AFTER that date, up to and including today's
  decision date, in the bars the page fetched. Five or more means DUE: the plan proposes orders.
  Fewer means HOLD: the plan still shows today's ranking and book, but proposes NO orders, and says how
  many days remain. No ws5 order has ever filled means DUE (the first run).
* An explicit override (`force=True`, the dashboard's tick box) proposes the orders anyway and the plan
  records that it was forced. With neither due nor forced, `submit_checked` refuses even a plan that
  somehow carries orders.

Other safety, inherited: orders only through `submit_plan` (paper broker, tick box AND typing SUBMIT,
exactly the plan shown on screen, no open order on any strategy symbol); the account must be the one the
plan was drawn for; new exposure must fit in the smaller of `buying_power` and `regt_buying_power`.
Credentials never appear here. Bars for today are dropped while the market is open (and until 16:00
New York once closed), so a chart only ever uses completed days.

Simplification, stated: the notebook's backtest rebalances on a fixed 5-day grid from the start of the
test period. Live, the grid starts at your first filled order and the page counts from there.
"""
from __future__ import annotations

import dataclasses
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `quantsoc_ml` and the other workshops import from anywhere
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import cnn  # noqa: E402
from quantsoc_ml.paper import (  # noqa: E402
    BrokerError, MockBroker, OrderResult, ReadOnlyBroker, Refused, propose_orders,
)
from workshop3 import live as ws3  # noqa: E402
from workshop3.live import (  # noqa: E402,F401  (re-exported so the dashboard and tests use one name)
    CONFIRM_WORD, MARKET_TZ, client_order_id, connect_alpaca, equity_history, from_alpaca,
    positions_table, recent_orders, results_table, submit_plan, targets_table, to_alpaca, weights_with_borrow,
)
from workshop4 import live as ws4  # noqa: E402
from workshop4.live import buying_power_message, same_account  # noqa: E402,F401

CLIENT_ID_PREFIX = "qsml-ws5"           # Workshop 3 uses qsml-ws3 and Workshop 4 qsml-ws4: ids never collide
MODEL_NAME = "I5R5"
CHART_DAYS = 5                          # the picture covers 5 trading days
HORIZON = 5                             # P(up over the next 5 trading days)
REBALANCE_DAYS = 5                      # trade once every 5 trading days
HISTORY_LIMIT = 500                     # orders of account history read to find the last rebalance
MIN_FINAL_DAY_SHARE = 0.9               # fewer than 90% of the universe with a bar on the last day: refuse
HISTORY_DAYS = 30                       # trading days of bars to fetch (the chart needs 2 * 5 - 1 = 9)
MIN_STOCKS = 20                         # fewer scored stocks than this and deciles mean nothing
GROSS_EXPOSURE = 1.0                    # $0.50 long and $0.50 short per $1 of equity (Workshop 3's convention)
DEMO_EQUITY = 100_000.0
SNAPSHOT_DAYS = HISTORY_DAYS + REBALANCE_DAYS + 3
_ID_DATE = re.compile(rf"^{re.escape(CLIENT_ID_PREFIX)}-(\d{{8}})-")


# --------------------------------------------------------------------------- models and universe

def load_models(model_dir=None) -> List[cnn.Trained]:
    """The cached I5R5 networks. FileNotFoundError when there are none yet; ValueError if they are not 5-day ones."""
    models = cnn.load_ensemble(MODEL_NAME, model_dir=model_dir)
    if any(m.days != CHART_DAYS for m in models):
        raise ValueError(f"the cached {MODEL_NAME} networks are not {CHART_DAYS}-day networks")
    return models


def snapshot_symbols() -> List[str]:
    """The stocks the snapshot holds on its last day (Yahoo tickers): the strategy's universe."""
    s = pd.read_csv(cnn.STOCK_FILE, usecols=["date", "symbol"])
    return sorted(s.loc[s["date"] == s["date"].max(), "symbol"].astype(str).unique())


# --------------------------------------------------------------------------- bars to a Market

def bars_to_market(bars: pd.DataFrame) -> cnn.Market:
    """Long OHLC bars (date, symbol, open, high, low, close, adj_close, volume) to the library's `cnn.Market`.

    Symbols are mapped to Yahoo tickers. open_r, high_r and low_r are open, high, low divided by the
    same day's close, which is the library's definition; adj_close and volume pass through. With
    Alpaca's adjustment=all the four prices share one scale each day, so the ratios are the true ones.
    """
    b = bars.assign(date=pd.to_datetime(bars["date"]), symbol=bars["symbol"].map(from_alpaca))

    def wide(col: str) -> pd.DataFrame:
        return b.pivot_table(index="date", columns="symbol", values=col, aggfunc="last", dropna=False).sort_index()

    adj, close = wide("adj_close"), wide("close")
    ratio = lambda col: (wide(col) / close).reindex_like(adj)
    return cnn.Market(adj, ratio("open"), ratio("high"), ratio("low"), wide("volume").reindex_like(adj))


def snapshot_bars(days: int = SNAPSHOT_DAYS) -> pd.DataFrame:
    """The committed snapshot as Alpaca-style OHLC bars, Alpaca tickers, for the offline demo.

    Closes come from workshop3/data/sp100_daily.csv.gz (the adjusted close, which is what Alpaca's
    adjusted bars report as `close`); open, high and low are rebuilt from workshop5/data/sp100_ohl.csv.gz
    as ratio times that close.
    """
    s = pd.read_csv(cnn.STOCK_FILE, parse_dates=["date"])
    keep = sorted(s["date"].unique())[-int(days):]
    s = s[s["date"].isin(keep)]
    o = pd.read_csv(cnn.OHL_FILE, parse_dates=["date"])
    df = s.merge(o[o["date"].isin(keep)], on=["date", "symbol"], how="left")
    out = pd.DataFrame({"date": df["date"], "symbol": df["symbol"].map(to_alpaca), "close": df["adj_close"],
                        "adj_close": df["adj_close"], "volume": df["volume"],
                        "open": df["open_r"] * df["adj_close"], "high": df["high_r"] * df["adj_close"],
                        "low": df["low_r"] * df["adj_close"]})
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------- scores and the book

@dataclass(frozen=True)
class DecileRule:
    """Long the top `n_long` stocks, short the bottom `n_short`, equal weights within a side.

    Each side holds half of `gross_exposure`, as in Workshop 3. Has the `.weights(scores)` method that
    Workshop 3's `weights_with_borrow` expects of a strategy.
    """
    n_long: int
    n_short: int
    gross_exposure: float = GROSS_EXPOSURE

    def weights(self, scores: pd.Series) -> pd.Series:
        s = scores.dropna().sort_values(kind="mergesort")
        if len(s) < self.n_long + self.n_short:
            raise ValueError(f"need at least {self.n_long + self.n_short} scored stocks, got {len(s)}")
        w = pd.Series(0.0, index=s.index)
        w[s.index[-self.n_long:]] = self.gross_exposure / 2 / self.n_long
        w[s.index[:self.n_short]] = -self.gross_exposure / 2 / self.n_short
        return w.sort_values(ascending=False)


def score_market(mkt: cnn.Market, models: Sequence[cnn.Trained], date=None) -> Tuple[pd.DataFrame, DecileRule]:
    """(table of symbol, prob_up, decile, highest first; the decile rule) for `date` (default: the last day)."""
    table = cnn.live_scores(mkt, models, date=date)
    if len(table) < MIN_STOCKS:
        raise BrokerError(f"only {len(table)} stocks have a full {CHART_DAYS}-day chart on the latest day; "
                          f"need at least {MIN_STOCKS} to form deciles")
    rule = DecileRule(n_long=int((table["decile"] == 10).sum()), n_short=int((table["decile"] == 1).sum()))
    return table, rule


# --------------------------------------------------------------------------- the weekly rule

def last_rebalance_date(orders: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Newest decision date among `qsml-ws5` orders with filled quantity above zero (any status), or None.

    Ids with an impossible date, and missing ids, are skipped rather than raising.
    """
    if orders is None or len(orders) == 0:
        return None
    dates = []
    filled_col = pd.to_numeric(orders["filled"], errors="coerce").fillna(0.0)
    for coid, filled in zip(orders["client_order_id"], filled_col):
        m = _ID_DATE.match(coid) if isinstance(coid, str) else None
        if not m or not filled > 0:
            continue
        try:
            dates.append(pd.Timestamp(pd.to_datetime(m.group(1), format="%Y%m%d")))
        except (ValueError, TypeError):
            continue
    return max(dates) if dates else None


def rebalance_history(broker, limit: int = HISTORY_LIMIT) -> Tuple[Optional[pd.Timestamp], bool]:
    """(last rebalance date, history_incomplete) for a broker (needs the full broker: it reads the order history).

    history_incomplete is True when no filled ws5 order was found AND the window read was full, so an
    older one may exist beyond it.
    """
    orders = recent_orders(broker, limit=limit)
    last = last_rebalance_date(orders)
    return last, bool(last is None and len(orders) >= int(limit))


def broker_last_rebalance(broker, limit: int = HISTORY_LIMIT) -> Optional[pd.Timestamp]:
    """`last_rebalance_date` for a broker (see `rebalance_history` for the incomplete-window flag)."""
    return rebalance_history(broker, limit)[0]


def rebalance_status(last_rebalance, trading_days: Sequence, decision_date,
                     every: int = REBALANCE_DAYS) -> Tuple[bool, Optional[int], str]:
    """(due, trading days since, one-line reason). See the module docstring for the rule."""
    if last_rebalance is None:
        return True, None, "no filled qsml-ws5 order found on this account, so this is the first rebalance"
    last, now = pd.Timestamp(last_rebalance).normalize(), pd.Timestamp(decision_date).normalize()
    since = int(sum(1 for d in trading_days if last < pd.Timestamp(d) <= now))
    if since >= int(every):
        return True, since, (f"last rebalance filled for the {last:%d %b %Y} decision; {since} trading days "
                             f"have passed (rebalance every {every})")
    return False, since, (f"last rebalance filled for the {last:%d %b %Y} decision; only {since} of {every} trading "
                          f"days have passed, so hold ({int(every) - since} more)")


# --------------------------------------------------------------------------- the plan

@dataclass
class LivePlan:
    decision_date: pd.Timestamp
    table: pd.DataFrame                 # symbol, prob_up, decile (10 = top), highest first, Yahoo tickers
    scores: pd.Series                   # prob_up by symbol, highest first
    weights: pd.Series                  # target weights (fraction of equity); zero for every scored name not picked
    skipped_shorts: Dict[str, str]      # short candidates passed over, with the reason
    orders: pd.DataFrame                # propose_orders output plus an alpaca_symbol column (empty on a hold)
    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, float]         # Yahoo tickers, negative = short
    last_prices: Dict[str, float]       # Yahoo tickers, last completed close
    open_orders: List[str]              # Yahoo tickers with an order still open
    clock: dict
    mkt: cnn.Market                     # the completed days the charts were drawn from
    due: bool                           # five or more trading days since the last filled ws5 rebalance
    forced: bool = False                # the override was ticked
    last_rebalance: Optional[pd.Timestamp] = None
    days_since: Optional[int] = None
    cadence_note: str = ""
    notes: List[str] = field(default_factory=list)
    universe: List[str] = field(default_factory=list)       # every scored symbol
    regt_buying_power: Optional[float] = None
    blocked: str = ""                   # why no orders are proposed whatever the weekly rule says ("" = not blocked)
    built_fingerprint: str = ""         # the orders' fingerprint when the plan was built: a later change is refused

    order_id_prefix = CLIENT_ID_PREFIX      # read by Workshop 3's submit_plan

    fingerprint = property(lambda self: ws3.LivePlan.fingerprint.fget(self))
    busy = property(lambda self: ws4.LivePlan.busy.fget(self))
    opening_notional = property(lambda self: ws4.LivePlan.opening_notional.fget(self))
    buying_power_limit = property(lambda self: ws4.LivePlan.buying_power_limit.fget(self))
    over_buying_power = property(lambda self: ws4.LivePlan.over_buying_power.fget(self))
    exit_orders = property(lambda self: ws4.LivePlan.exit_orders.fget(self))

    @property
    def can_rebalance(self) -> bool:
        return bool((self.due or self.forced) and not self.blocked)

    @property
    def longs(self) -> pd.Series:
        return self.scores[self.weights[self.weights > 0].index].sort_values(ascending=False)

    @property
    def shorts(self) -> pd.Series:
        return self.scores[self.weights[self.weights < 0].index].sort_values()


def completed_days_only(bars: pd.DataFrame, clock: dict) -> pd.DataFrame:
    """Keep only completed trading days: drop bars dated after today (New York), and today's bar unless the
    market is closed and it is 16:00 or later. A clock with no timestamp counts as open (today's bar is
    dropped, by the machine's own New York date); a naive timestamp is read as New York time."""
    if len(bars) == 0:
        return bars
    ts = clock.get("timestamp")
    if ts is None or pd.isna(ts):
        now, is_open = pd.Timestamp.now(tz=MARKET_TZ), True
    else:
        now = pd.Timestamp(ts)
        now = now.tz_localize(MARKET_TZ) if now.tzinfo is None else now.tz_convert(MARKET_TZ)
        is_open = bool(clock.get("is_open"))
    today = now.tz_localize(None).normalize()
    dated = pd.to_datetime(bars["date"])
    if (not is_open) and now.hour >= 16:
        return bars[dated <= today]
    return bars[dated < today]


def build_live_plan(view: ReadOnlyBroker, models: Sequence[cnn.Trained], symbols: Optional[Sequence[str]] = None,
                    last_rebalance: Optional[pd.Timestamp] = None, force: bool = False,
                    min_notional: float = 1.0, history_incomplete: bool = False) -> LivePlan:
    """Charts, deciles, book and (when a rebalance is due or forced) orders, for the last completed day.

    `last_rebalance` is the decision date of the account's last filled qsml-ws5 order (see
    `rebalance_history`, which also says whether the order history read was too short to be sure). Cannot submit anything: it must be given a ReadOnlyBroker.
    """
    if hasattr(view, "submit"):
        raise TypeError("build_live_plan must receive a ReadOnlyBroker (no submit method)")
    syms = [str(s) for s in (symbols or snapshot_symbols())]
    notes: List[str] = []
    clock = view.clock()
    bars = view.daily_ohlc_bars([to_alpaca(s) for s in syms], HISTORY_DAYS)
    bars = completed_days_only(bars, clock)
    if len(bars) == 0:
        raise BrokerError("no daily bars came back for the strategy's symbols")
    mkt = bars_to_market(bars)
    need = 2 * CHART_DAYS - 1
    if len(mkt.adj_close) < need:
        raise BrokerError(f"only {len(mkt.adj_close)} trading days of prices came back; a chart needs {need}")
    missing = sorted(set(syms) - set(mkt.adj_close.columns))
    if missing:
        notes.append(f"no prices for {', '.join(missing)}: not traded this time")
    table, rule = score_market(mkt, models)
    scores = table.set_index("symbol")["prob_up"]
    decision_date = pd.Timestamp(mkt.adj_close.index[-1])
    unscored = sorted(set(mkt.adj_close.columns) - set(scores.index))
    if unscored:
        notes.append(f"{len(unscored)} stocks are not scored today (no price on the last day, or too little price "
                     f"history before it for a chart): "
                     f"{', '.join(unscored[:8])}{' ...' if len(unscored) > 8 else ''}")
    if not view.offline and (pd.Timestamp.now().normalize() - decision_date).days > 6:
        notes.append(f"the newest price is from {decision_date.date()}, which looks stale")
    weights, skipped, unknown = weights_with_borrow(rule, scores, lambda s: view.shortable(to_alpaca(s)))
    if unknown:
        notes.append(f"could not check borrow for {', '.join(unknown)}: kept, but the broker may reject the short")

    if last_rebalance is not None and pd.Timestamp(last_rebalance).normalize() > decision_date.normalize():
        notes.append(f"the newest filled qsml-ws5 order is dated {pd.Timestamp(last_rebalance):%d %b %Y}, after today's "
                     f"decision date {decision_date:%d %b %Y}: ignored (check the order ids and the clock)")
        last_rebalance = None
    due, since, reason = rebalance_status(last_rebalance, list(mkt.adj_close.index), decision_date)
    if last_rebalance is None and history_incomplete:
        due, reason = False, ("no filled qsml-ws5 order in the latest orders read, but that list was full, so an older "
                              "one may exist: hold. If you are sure this is the first rebalance, tick the override")
    acct = view.account()
    blocked = ""
    eq, bp, regt = float(acct.equity), float(acct.buying_power), getattr(acct, "regt_buying_power", None)
    if not np.isfinite(eq) or eq <= 0:
        blocked = f"the account's equity is {eq:g}, so nothing can be sized"
    elif not np.isfinite(bp) or (regt is not None and not np.isfinite(float(regt))):
        blocked = "the broker reported a buying power that is not a number, so new exposure cannot be checked"
    final_day = int(mkt.adj_close.iloc[-1].reindex(syms).notna().sum())
    if not blocked and final_day < MIN_FINAL_DAY_SHARE * len(syms):
        blocked = (f"only {final_day} of {len(syms)} stocks have a price for {decision_date:%d %b %Y}, so that day "
                   "looks incomplete: wait for the data to fill in")
    can = bool((due or force) and not blocked)
    if blocked:
        notes.append("NO ORDERS PROPOSED: " + blocked)
    if force and not due and not blocked:
        notes.append("OVERRIDE ticked: proposing a rebalance although the weekly rule says hold")

    positions = {from_alpaca(s): float(q) for s, q in view.positions().items()}
    last_prices = {s: float(v) for s, v in mkt.adj_close.ffill().iloc[-1].items() if np.isfinite(v)}
    outside = sorted(s for s in positions if s not in weights.index)
    if outside:
        notes.append(f"positions the strategy does not score today are left alone: {', '.join(outside)}")
    if can:
        in_book = {s: q for s, q in positions.items() if s in weights.index}
        orders = propose_orders(weights, float(acct.equity), in_book, last_prices, min_notional=min_notional)
    else:
        orders = pd.DataFrame(columns=["symbol", "side", "qty", "notional", "current_qty", "target_qty", "action"])
    orders.insert(1, "alpaca_symbol", orders["symbol"].map(to_alpaca))
    open_orders = sorted(from_alpaca(s) for s in view.open_orders())
    plan = LivePlan(decision_date=decision_date, table=table, scores=scores, weights=weights, skipped_shorts=skipped,
                    orders=orders, equity=float(acct.equity), cash=float(acct.cash),
                    buying_power=float(acct.buying_power), positions=positions, last_prices=last_prices,
                    open_orders=open_orders, clock=clock, mkt=mkt, due=due, forced=bool(force),
                    last_rebalance=None if last_rebalance is None else pd.Timestamp(last_rebalance),
                    days_since=since, cadence_note=reason, notes=notes,
                    universe=sorted(set(scores.index) | {from_alpaca(x) for x in syms}),
                    regt_buying_power=getattr(acct, "regt_buying_power", None), blocked=blocked)
    plan.built_fingerprint = plan.fingerprint
    return plan


def submit_checked(broker, shown: Optional[LivePlan], fresh: LivePlan, confirmed: bool, typed: str,
                   exits_only: bool = False):
    """Send `shown` (the plan the user looked at) through Workshop 3's `submit_plan`, after more checks.

    Refuses (raises `Refused`, nothing sent) unless: neither plan is blocked or on a hold (the weekly
    rule); `shown` exists, has the decision date of `fresh` (rebuilt on the click) and its orders are
    still the ones it was built with; the account still has the positions and open orders `shown` was
    drawn for; no strategy symbol (the whole universe, not only today's scored stocks) has an open
    order; and the new exposure fits in buying power on BOTH the plan on screen and the fresh one.
    Then `submit_plan` applies its own checks (paper, the tick box and SUBMIT, a fingerprint match).
    `exits_only=True` sends only the exits and reductions, skipping the buying power check.
    The typed word is compared after stripping surrounding spaces (Workshop 3's behaviour).
    """
    for plan in (shown, fresh):
        if plan is not None and plan.blocked:
            raise Refused("refused: no orders can be sent (" + plan.blocked + ")")
        if plan is not None and not plan.can_rebalance:
            raise Refused("refused: the weekly rule says hold (" + plan.cadence_note + "). Tick the override to "
                          "rebalance anyway")
    if shown is None or shown.decision_date != fresh.decision_date:
        raise Refused("refused: the page changed before the click; look at the orders again, then submit")
    if shown.built_fingerprint and shown.fingerprint != shown.built_fingerprint:
        raise Refused("refused: the orders were changed after the page was drawn; look at them again, then submit")
    if not same_account(shown, fresh):
        raise Refused("refused: the account changed since the page was drawn (positions or open orders are "
                      "different now), so these orders may be wrong. Look at the new orders, then submit")
    busy = sorted(set(fresh.busy) | set(shown.busy))
    if busy:
        raise Refused(f"refused: open orders exist for {', '.join(busy)}. Wait for every open order on a strategy "
                      "symbol to fill, or cancel it on the Alpaca website, then reload and submit")
    if exits_only:
        sent = dataclasses.replace(shown, orders=shown.exit_orders)
        return submit_plan(broker, sent, confirmed=confirmed, typed=typed)
    for plan in (shown, fresh):
        if plan.over_buying_power:
            raise Refused("refused: " + buying_power_message(plan))
    return submit_plan(broker, shown, confirmed=confirmed, typed=typed, expected_fingerprint=shown.built_fingerprint or None)


# --------------------------------------------------------------------------- charts for display

def chart_array(plan: LivePlan, symbol: str) -> np.ndarray:
    """The 32x15 uint8 picture the network scored for `symbol` on the decision date (0 black, 255 white)."""
    return cnn.draw_images(days=CHART_DAYS, **cnn.chart_inputs(plan.mkt, CHART_DAYS, [plan.decision_date], [symbol]))[0]


def enlarge(image: np.ndarray, factor: int = 12) -> np.ndarray:
    """Nearest-neighbour upscale (every pixel becomes a factor x factor block), so the chart stays crisp."""
    return np.repeat(np.repeat(np.asarray(image, dtype=np.uint8), factor, axis=0), factor, axis=1)


def chart_picks(plan: LivePlan, n: int = 3) -> Tuple[List[str], List[str]]:
    """(the n highest-ranked, the n lowest-ranked) stocks, for showing their charts."""
    t = plan.table
    return list(t["symbol"].iloc[:n]), list(t["symbol"].iloc[::-1].iloc[:n])


def decile_table(plan: LivePlan) -> pd.DataFrame:
    """Every scored stock: symbol, prob_up, decile and which book it is in (long, short, skipped short or none)."""
    book = pd.Series("", index=plan.table["symbol"])
    book[plan.weights[plan.weights > 0].index] = "long"
    book[plan.weights[plan.weights < 0].index] = "short"
    book[[s for s in plan.skipped_shorts if s in book.index]] = "skipped (borrow)"
    out = plan.table[["symbol", "prob_up", "decile"]].copy()
    out["book"] = book.reindex(out["symbol"]).to_numpy()
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- offline demo

def demo_broker(bars: pd.DataFrame, models: Optional[Sequence[cnn.Trained]] = None,
                equity: float = DEMO_EQUITY) -> MockBroker:
    """An offline MockBroker fed from the snapshot (`snapshot_bars`), as a paper account would be.

    With models, the account already holds the book the strategy would have chosen
    REBALANCE_DAYS trading days before the last day, filled then, with a filled qsml-ws5 order on file,
    so the weekly rule finds a rebalance DUE and the demo shows a real week of turnover. Without
    models, or with too little history, it is an empty account (also due: the first rebalance).
    """
    bars = bars.assign(date=pd.to_datetime(bars["date"]))
    dates = sorted(bars["date"].unique())
    mock = MockBroker(bars=bars, cash=equity)
    if not models or len(dates) < REBALANCE_DAYS + 2 * CHART_DAYS:
        return mock
    then = dates[-1 - REBALANCE_DAYS]
    mkt_then = bars_to_market(bars[bars["date"] <= then])
    table, rule = score_market(mkt_then, models)
    w = rule.weights(table.set_index("symbol")["prob_up"])
    px_then = mkt_then.adj_close.ffill().iloc[-1]
    holdings: Dict[str, float] = {}
    for sym, weight in w[w != 0].items():
        holdings[to_alpaca(sym)] = float(np.trunc(weight * equity / float(px_then[sym])))
    cash = equity - sum(q * float(px_then[from_alpaca(s)]) for s, q in holdings.items())
    mock = MockBroker(bars=bars, cash=cash, positions=holdings)
    for sym, q in holdings.items():
        action = "open long" if q > 0 else "open short"
        coid = client_order_id(then, sym, action, CLIENT_ID_PREFIX)
        mock.orders[coid] = OrderResult(sym, "buy" if q > 0 else "sell", abs(q), "filled", abs(q), coid)
    return mock
