"""Workshop 4 paper-trading dashboard: the plan builder, the submit safeguards and the app itself.

No network anywhere: the broker is the MockBroker from quantsoc_ml.paper, or a subclass of it that
pretends to be Alpaca. Real alpaca-py calls are therefore NOT exercised here; that limit is stated.
Expected values are recomputed independently from quantsoc_ml.avlee, not read back from the plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import avlee as al  # noqa: E402
from quantsoc_ml.paper import MockBroker, ReadOnlyBroker, Refused  # noqa: E402
from workshop4 import live  # noqa: E402

DASHBOARD = ROOT / "workshop4" / "dashboard.py"
FAKE_KEY, FAKE_SECRET = "PKFAKEKEYWS4MUSTNOTRENDER", "fakeWs4SecretMustNotRender0987654321"
EQUITY = 100_000.0


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """The last 120 trading days of the stock and ETF snapshots, with Alpaca tickers (BRK.B)."""
    b = live.snapshot_bars()
    dates = sorted(b["date"].unique())[-120:]
    return b[b["date"].isin(dates)].reset_index(drop=True)


@pytest.fixture(scope="module")
def sectors() -> pd.Series:
    return al.load_sectors()


@pytest.fixture(scope="module")
def prices(bars):
    """(stock prices, ETF prices) for the last HISTORY_DAYS days, Yahoo tickers: what the plan should see."""
    b = bars.assign(symbol=bars["symbol"].map(live.from_alpaca))
    W = b.pivot(index="date", columns="symbol", values="adj_close").sort_index().iloc[-live.HISTORY_DAYS:]
    funds = [f for f in W.columns if f in {"SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
                                           "XLV", "XLY"}]
    return W.drop(columns=funds), W[funds]


@pytest.fixture(scope="module")
def flat(bars):
    """The plan for an empty account: every trade is a new one."""
    return live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY)))


def pick(table: pd.DataFrame, cond, n=1, exclude=()):
    rows = table[cond & ~table["symbol"].isin(list(exclude))]
    assert len(rows) >= n, "the snapshot slice does not have a stock in the needed state"
    return list(rows["symbol"].iloc[:n])


class FakeAlpaca(MockBroker):
    """A MockBroker that says it is the Alpaca paper broker (online). No network."""
    name = "alpaca-paper"
    offline = False

    def __init__(self, *a, busy=(), paper=True, **kw):
        super().__init__(*a, **kw)
        self.busy = list(busy)
        self.paper = paper

    def open_orders(self):
        return sorted(set(super().open_orders()) | set(self.busy))

    def clock(self):
        return {**super().clock(), "next_open": pd.Timestamp("2026-10-16 09:30", tz="America/New_York")}

    def equity_history(self):
        return pd.DataFrame({"date": pd.date_range("2026-10-01", periods=5, freq="B"),
                             "equity": [100_000, 100_250, 99_900, 100_400, 100_100]})


# --------------------------------------------------------------------------- reading the account

def test_held_is_inferred_from_stock_positions_only(bars, monkeypatch):
    seen = {}
    real = al.live_signal

    def spy(prices, held, *a, **kw):
        seen["held"] = dict(held)
        return real(prices, held, *a, **kw)
    monkeypatch.setattr(live.al, "live_signal", spy)
    mock = MockBroker(bars=bars, cash=EQUITY,
                      positions={"AAPL": 10, "XOM": -5, "BRK.B": 3, "XLK": 20, "SPY": -4, "QQQ": 7})
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    assert seen["held"] == {"AAPL": 1, "XOM": -1, "BRK-B": 1}      # ETFs are hedges, QQQ is not ours
    assert plan.held == seen["held"]
    assert any("QQQ" in n and "left alone" in n for n in plan.notes)
    assert "QQQ" not in set(plan.orders["symbol"])


def test_plan_refuses_a_broker_that_can_submit(bars):
    with pytest.raises(TypeError):
        live.build_live_plan(MockBroker(bars=bars))


# --------------------------------------------------------------------------- targets

def test_new_trades_are_sized_at_position_times_equity(flat):
    assert flat.opened and not flat.kept and not flat.closed
    orders = flat.orders.set_index("symbol")
    for sym in flat.opened:
        side = flat.states[sym]
        expected = np.trunc(side * flat.cfg.position * flat.equity / flat.last_prices[sym])
        assert orders.loc[sym, "target_qty"] == expected
        assert orders.loc[sym, "qty"] == abs(expected)
        assert orders.loc[sym, "action"] == ("open long" if side > 0 else "open short")
        assert orders.loc[sym, "side"] == ("buy" if side > 0 else "sell")
        # the dollar value is at most the position size and within one share of it
        dollars = abs(expected) * flat.last_prices[sym]
        size = flat.cfg.position * flat.equity
        assert size - flat.last_prices[sym] < dollars <= size + 1e-6
    assert set(flat.orders["alpaca_symbol"]) == {live.to_alpaca(s) for s in flat.orders["symbol"]}


def test_kept_trades_keep_their_quantity(bars, flat):
    t = flat.table
    longs = pick(t, t["s"] < -0.6, n=1)                 # a held long stays open while s < -0.5
    shorts = pick(t, t["s"] > 0.85, n=1)                # a held short stays open while s > 0.75
    mock = MockBroker(bars=bars, cash=EQUITY, positions={live.to_alpaca(longs[0]): 7, live.to_alpaca(shorts[0]): -3})
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    assert set(longs + shorts) <= set(plan.kept)
    book = plan.book.set_index("symbol")
    assert book.loc[longs[0], "target shares"] == 7 and book.loc[shorts[0], "target shares"] == -3
    assert not set(longs + shorts) & set(plan.orders["symbol"]), "a kept trade must not be resized"


def test_closed_trades_go_to_zero(bars, flat):
    t = flat.table
    long_close = pick(t, (t["s"] > -0.4) & (t["s"] < 1.0), n=1)[0]
    short_close = pick(t, (t["s"] < 0.65) & (t["s"] > -1.0), n=1, exclude=[long_close])[0]
    mock = MockBroker(bars=bars, cash=EQUITY,
                      positions={live.to_alpaca(long_close): 12, live.to_alpaca(short_close): -9})
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    assert {long_close, short_close} <= set(plan.closed)
    o = plan.orders.set_index("symbol")
    assert (o.loc[long_close, "action"], o.loc[long_close, "side"], o.loc[long_close, "qty"]) == ("close long", "sell", 12)
    assert (o.loc[short_close, "action"], o.loc[short_close, "side"], o.loc[short_close, "qty"]) == ("close short", "buy", 9)
    assert (o.loc[[long_close, short_close], "target_qty"] == 0).all()
    assert long_close not in set(plan.book["symbol"])


def test_etf_hedges_are_minus_side_beta_dollar_value_summed_per_fund(bars, flat, prices, sectors):
    t = flat.table
    kept_long = pick(t, t["s"] < -0.6, n=1)[0]
    mock = MockBroker(bars=bars, cash=EQUITY, positions={live.to_alpaca(kept_long): 7, "XLB": 50, "SPY": -11})
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    # independent recomputation straight from avlee
    P, E = prices
    fit, states, _ = al.live_signal(P, {kept_long: 1}, al.Config(factors="etf"), E.reindex(P.index), sectors)
    assert states == plan.states
    last = pd.concat([P.iloc[-1], E.iloc[-1]])
    dollars = {}
    for sym, side in states.items():
        i = fit.symbols.index(sym)
        qty = 7.0 if sym == kept_long else np.trunc(side * plan.cfg.position * plan.equity / last[sym])
        fund = fit.hedge_etf[i]
        dollars[fund] = dollars.get(fund, 0.0) - side * fit.beta[i, 0] * abs(qty) * last[sym]
    funds = live.hedge_funds(sectors)
    expected = {f: np.trunc(dollars.get(f, 0.0) / last[f]) for f in funds}
    assert len([f for f in expected if expected[f] != 0]) >= 3, "the test should cover several funds"
    book = plan.book.set_index("symbol")
    for f in funds:
        got = book.loc[f, "target shares"] if f in book.index else 0.0
        assert got == expected[f], f
    # orders move each fund from its current position to that target
    after = {f: {"XLB": 50.0, "SPY": -11.0}.get(f, 0.0) for f in funds}
    for row in plan.orders[plan.orders["symbol"].isin(funds)].itertuples():
        after[row.symbol] += row.qty if row.side == "buy" else -row.qty
    assert after == pytest.approx(expected)


def test_not_borrowable_new_short_is_skipped_and_noted_not_substituted(bars, flat):
    new_shorts = sorted(s for s in flat.opened if flat.states[s] < 0)
    assert len(new_shorts) >= 2
    blocked = new_shorts[0]
    mock = MockBroker(bars=bars, cash=EQUITY, not_shortable=[live.to_alpaca(blocked)])
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    assert blocked in plan.skipped and "not easy to borrow" in plan.skipped[blocked]
    assert blocked not in plan.opened and blocked not in plan.states
    assert blocked not in set(plan.orders["symbol"])
    assert set(plan.opened) == set(flat.opened) - {blocked}, "no other stock may take its place"
    row = plan.table.set_index("symbol").loc[blocked]
    assert row["today"] == "not opened" and row["new state"] == 0
    assert plan.weights.get(blocked, 0.0) == 0.0
    # its hedge is gone too: the fund it would have used is lighter by its share
    fund = row["hedge fund"]
    b0, b1 = flat.book.set_index("symbol"), plan.book.set_index("symbol")
    before = float(b0.loc[fund, "target $"])
    after = float(b1.loc[fund, "target $"]) if fund in b1.index else 0.0
    assert after != before


def test_ticker_mapping_reaches_class_shares(bars, flat):
    assert "BRK-B" in set(flat.table["symbol"])
    mock = MockBroker(bars=bars, cash=EQUITY, positions={"BRK.B": 2})
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    assert plan.held.get("BRK-B") == 1 and "BRK.B" not in plan.positions


# --------------------------------------------------------------------------- submit safeguards (Workshop 3's submit_plan, unchanged)

@pytest.fixture()
def fresh(bars):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    return mock, live.build_live_plan(ReadOnlyBroker(mock))


def test_submit_refuses_when_not_paper(fresh):
    mock, plan = fresh
    mock.paper = False
    with pytest.raises(Refused):
        live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


@pytest.mark.parametrize("confirmed,typed", [(False, "SUBMIT"), (True, ""), (True, "submit"), (True, "yes")])
def test_submit_refuses_when_not_confirmed(fresh, confirmed, typed):
    mock, plan = fresh
    with pytest.raises(Refused):
        live.submit_plan(mock, plan, confirmed=confirmed, typed=typed)
    assert mock.submissions == []


def test_submit_refuses_when_open_orders_exist(fresh):
    mock, plan = fresh
    mock.busy = [plan.orders["alpaca_symbol"].iloc[-1]]
    with pytest.raises(Refused, match="open orders"):
        live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_submit_refuses_changed_orders_and_sends_the_plan_in_order(fresh):
    mock, plan = fresh
    with pytest.raises(Refused, match="changed"):
        live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT", expected_fingerprint="not-the-same")
    assert mock.submissions == []
    results = live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT", expected_fingerprint=plan.fingerprint)
    assert [(s["symbol"], s["side"], s["qty"]) for s in mock.submissions] == \
        [(r.alpaca_symbol, r.side, float(r.qty)) for r in plan.orders.itertuples()]
    assert all(r.status == "filled" for r in results)


def test_after_submitting_the_next_plan_keeps_everything(bars, fresh):
    """Run twice on the same day: the second run reads the trades from the account and has nothing to do."""
    mock, plan = fresh
    live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    again = live.build_live_plan(ReadOnlyBroker(mock))
    assert set(again.kept) == set(plan.opened) and not again.opened and not again.closed
    stock_orders = again.orders[~again.orders["symbol"].isin(live.hedge_funds(al.load_sectors()))]
    assert len(stock_orders) == 0


# --------------------------------------------------------------------------- the app (streamlit AppTest)

def _rendered(at) -> str:
    """Every element's protobuf as text: what the browser would receive."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.append(str(proto))
        for child in getattr(node, "children", {}).values():
            walk(child)
    walk(at._tree)
    return "\n".join(out)


def _app(mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    at.session_state["mode"] = mode
    return at


def test_app_offline_renders_without_submit(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    at = _app("Offline demo").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("OFFLINE DEMO" in i.value for i in at.info)
    assert any("One strategy per paper account" in w.value for w in at.warning)
    assert not [b for b in at.button if b.key == "submit"]
    text = _rendered(at)
    assert "Decision date" in text and "Opening" in text and "Staying open" in text and "Closing" in text
    # the demo day is chosen so that all three kinds of trade show
    assert not any("**Opening: 0**" in m.value or "**Closing: 0**" in m.value or "**Staying open: 0**" in m.value
                   for m in at.markdown)


def test_app_alpaca_mode_with_a_fake_never_shows_keys_and_submits_only_when_confirmed(bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fakes = []

    def connect(key, secret):
        assert (key, secret) == (FAKE_KEY, FAKE_SECRET)   # the keys reach the broker, and only the broker
        fakes.append(FakeAlpaca(bars=bars, cash=EQUITY))
        return fakes[-1]
    monkeypatch.setattr(live, "connect_alpaca", connect)

    at = _app("Alpaca paper").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("environment variables" in c.value for c in at.sidebar.caption)
    button = [b for b in at.button if b.key == "submit"]
    assert len(button) == 1 and button[0].disabled

    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("submit").run()
    assert [b for b in at.button if b.key == "submit"][0].disabled
    assert all(not f.submissions for f in fakes)

    at.text_input(key="confirm_text").input("SUBMIT").run()
    assert not [b for b in at.button if b.key == "submit"][0].disabled
    [b for b in at.button if b.key == "submit"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    sent = [f for f in fakes if f.submissions]
    assert len(sent) == 1 and len(sent[0].submissions) > 0
    assert all(s["client_order_id"].startswith("qsml-") for s in sent[0].submissions)
    assert any("Last submission" in m.value for m in at.markdown)

    text = _rendered(at)
    assert FAKE_KEY not in text and FAKE_SECRET not in text


def test_app_refuses_while_open_orders_exist(bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    probe = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY)))
    busy = [probe.orders["alpaca_symbol"].iloc[0]]
    fakes = []

    def connect(key, secret):
        fakes.append(FakeAlpaca(bars=bars, cash=EQUITY, busy=busy))
        return fakes[-1]
    monkeypatch.setattr(live, "connect_alpaca", connect)
    at = _app("Alpaca paper").run()
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    assert [b for b in at.button if b.key == "submit"][0].disabled
    assert any("Open orders already exist" in w.value for w in at.warning)
    assert all(not f.submissions for f in fakes)


def test_app_submits_the_plan_on_screen_even_if_equity_moved(bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fakes = []

    def connect(key, secret):        # every rerun sees a different equity, as during market hours
        fakes.append(FakeAlpaca(bars=bars, cash=EQUITY + 7_777.0 * len(fakes)))
        return fakes[-1]
    monkeypatch.setattr(live, "connect_alpaca", connect)
    at = _app("Alpaca paper").run()
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    shown = at.session_state["shown_plan"]
    [b for b in at.button if b.key == "submit"][0].click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    sent = [f for f in fakes if f.submissions]
    assert len(sent) == 1
    assert [(s["symbol"], s["qty"]) for s in sent[0].submissions] == \
        [(r.alpaca_symbol, float(r.qty)) for r in shown.orders.itertuples()]
    # the plan rebuilt on the click run differs (equity moved), so this really tests "the plan on screen"
    assert at.session_state["shown_plan"].fingerprint != shown.fingerprint



def test_workshop4_orders_use_their_own_id_prefix(fresh):
    mock, plan = fresh
    results = live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    assert results and all(r.client_order_id.startswith("qsml-ws4-") for r in results)


# --------------------------------------------------------------------------- regressions from the falsification pass

class SharedAlpaca(FakeAlpaca):
    """One fake account shared by every rerun, so a test can change it between the page and the click."""

    def __init__(self, *a, bp=None, regt=None, **kw):
        super().__init__(*a, **kw)
        self.bp = bp
        self.regt = regt

    def account(self):
        a = super().account()
        if self.bp is not None:
            a.buying_power = float(self.bp)
        if self.regt is not None:
            a.regt_buying_power = float(self.regt)
        return a


def _shared_app(broker, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    monkeypatch.setattr(live, "connect_alpaca", lambda key, secret: broker)
    at = _app("Alpaca paper").run()
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    return at


def _button(at):
    return [b for b in at.button if b.key == "submit"][0]


def test_item1_click_refuses_when_positions_changed_after_the_page_was_drawn(bars, flat, monkeypatch):
    closing = pick(flat.table, (flat.table["s"] > -0.4) & (flat.table["s"] < 1.0), n=3)
    acct = SharedAlpaca(bars=bars, cash=EQUITY, positions={live.to_alpaca(s): 20 for s in closing})
    at = _shared_app(acct, monkeypatch)
    shown = at.session_state["shown_plan"]
    assert set(closing) <= set(shown.closed)
    acct.holdings = {}                                   # flattened elsewhere, e.g. "Close all" on the website
    acct.cash = EQUITY
    _button(at).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert acct.submissions == [], "a plan drawn for a different account must not be sent"
    assert any("account changed" in e.value for e in at.error)


def test_item1_equity_drift_alone_is_allowed():
    a = live.LivePlan.__new__(live.LivePlan)
    b = live.LivePlan.__new__(live.LivePlan)
    a.positions, a.open_orders = {"AAPL": 10.0, "XLK": -3.0}, []
    b.positions, b.open_orders = {"XLK": -3.0000000001, "AAPL": 10.0}, []
    assert live.same_account(a, b)
    b.positions = {"AAPL": 10.0}
    assert not live.same_account(a, b)
    b.positions, b.open_orders = dict(a.positions), ["AAPL"]
    assert not live.same_account(a, b)


def test_item2_any_open_order_on_a_strategy_symbol_blocks_submit(bars):
    first = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY)))
    stock = sorted(s for s in first.opened if first.states[s] > 0)[0]
    mock = FakeAlpaca(bars=bars, cash=EQUITY, partial_fill={live.to_alpaca(stock): 0.5})
    plan = live.build_live_plan(ReadOnlyBroker(mock))
    live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    assert live.to_alpaca(stock) in mock.open_orders()
    again = live.build_live_plan(ReadOnlyBroker(mock))
    assert stock not in set(again.orders["symbol"]) and len(again.orders) > 0
    assert again.busy == [stock]
    n = len(mock.submissions)
    with pytest.raises(Refused, match="open order"):
        live.submit_checked(mock, again, again, confirmed=True, typed="SUBMIT")
    assert len(mock.submissions) == n


def test_item2_app_disables_submit_while_a_strategy_symbol_has_an_open_order(bars, flat, monkeypatch):
    idle = sorted(set(flat.table["symbol"]) - set(flat.orders["symbol"]))[0]
    acct = SharedAlpaca(bars=bars, cash=EQUITY, busy=[live.to_alpaca(idle)])
    at = _shared_app(acct, monkeypatch)
    assert _button(at).disabled
    assert any("open order" in w.value and idle in w.value for w in at.warning)
    assert acct.submissions == []


def test_item3_left_alone_stock_keeps_its_hedge(bars, sectors):
    filled = FakeAlpaca(bars=bars, cash=EQUITY)
    flat_plan = live.build_live_plan(ReadOnlyBroker(filled))
    live.submit_plan(filled, flat_plan, confirmed=True, typed="SUBMIT")
    short = sorted(s for s in flat_plan.opened if flat_plan.states[s] < 0 and s != "BRK-B")[0]
    fund = sectors[short]
    # (a) the stock's bars are missing today
    no_bars = MockBroker(bars=bars[bars["symbol"] != live.to_alpaca(short)], cash=filled.cash,
                         positions=filled.positions(), prices=filled.prices)
    a = live.build_live_plan(ReadOnlyBroker(no_bars))
    assert fund not in set(a.orders["symbol"]), "the left-alone stock's hedge fund must not be traded"
    assert any(short in n and fund in n for n in a.notes)
    # (b) the stock left the universe: gone from the sectors FILE the code reads, not just from an argument
    monkey = pytest.MonkeyPatch()
    monkey.setattr(live.al, "load_sectors", lambda path=None: sectors.drop(index=short))
    try:
        b = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=filled.cash, positions=filled.positions())))
    finally:
        monkey.undo()
    assert short not in b.held and fund not in set(b.orders["symbol"])
    assert any(short in n and "every fund" in n for n in b.notes)


def test_item5a_live_position_is_two_and_a_half_percent(flat):
    assert live.live_config().position == 0.025 and al.Config().position == 0.05
    sym = flat.opened[0]
    assert abs(flat.orders.set_index("symbol").loc[sym, "target_qty"]) == \
        np.trunc(0.025 * flat.equity / flat.last_prices[sym])


def test_item5b_submit_blocked_when_new_exposure_exceeds_buying_power(bars, monkeypatch):
    probe = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY)))
    assert probe.opening_notional > 0
    small = SharedAlpaca(bars=bars, cash=EQUITY, bp=probe.opening_notional / 2)
    plan = live.build_live_plan(ReadOnlyBroker(small))
    with pytest.raises(Refused, match="buying power"):
        live.submit_checked(small, plan, plan, confirmed=True, typed="SUBMIT")
    at = _shared_app(small, monkeypatch)
    assert _button(at).disabled
    assert any("buying power" in w.value for w in at.warning)
    assert small.submissions == []


def test_item5c_fund_legs_go_before_new_stock_opens_and_after_reductions(bars, flat):
    t = flat.table
    closing = pick(t, (t["s"] > -0.4) & (t["s"] < 1.0), n=1)[0]
    plan = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY,
                                                          positions={live.to_alpaca(closing): 10, "XLB": 500})))
    funds = set(live.hedge_funds(al.load_sectors()))
    adds = plan.orders["action"].str.startswith(("open", "add"))
    kind = np.where(~adds, 0, np.where(plan.orders["symbol"].isin(funds), 1, 2))
    assert (np.diff(kind) >= 0).all(), list(zip(plan.orders["symbol"], plan.orders["action"]))
    assert {0, 1, 2} <= set(kind)


def test_item7_page_shows_the_new_state_straight_after_submit(bars, monkeypatch):
    acct = SharedAlpaca(bars=bars, cash=EQUITY)
    at = _shared_app(acct, monkeypatch)
    _button(at).click().run()
    assert acct.submissions
    assert _button(at).disabled
    assert any("already submitted" in i.value for i in at.info)
    assert any("Last submission" in m.value for m in at.markdown)


def test_item8_weights_are_zero_for_trades_too_dear_to_open(bars):
    plan = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=3_000.0)))
    dear = [s for s, why in plan.skipped.items() if "costs more" in why]
    assert dear
    assert all(plan.weights.get(s, 0.0) == 0.0 for s in dear), plan.weights.reindex(dear)


# --------------------------------------------------------------------------- regressions from the fix-verification pass

def _blind_plan(bars, short, extra=None):
    """A plan where `short` is held short but has no bars today, plus `extra` positions."""
    pos = {live.to_alpaca(short): -6.0, "XLV": 30.0, **(extra or {})}
    return live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars[bars["symbol"] != live.to_alpaca(short)],
                                                          cash=EQUITY, positions=pos)))


def test_n2_no_new_trade_opens_into_a_frozen_fund(bars, flat, sectors):
    in_xlv = sorted(s for s in flat.opened if sectors.get(s) == "XLV" and s != "UNH")
    assert in_xlv, "the flat plan should open trades hedged with XLV"
    plan = _blind_plan(bars, "UNH")
    assert not set(in_xlv) & set(plan.opened)
    for sym in in_xlv:
        assert plan.skipped[sym] == "hedge fund XLV is frozen because UNH is left alone"
    assert "XLV" not in set(plan.orders["symbol"])
    others = sorted(s for s in flat.opened if sectors.get(s) != "XLV" and s != "UNH")
    assert set(others) <= set(plan.opened), "trades hedged in other funds still open"
    row = plan.book.set_index("symbol").loc["XLV"]
    assert row["target shares"] == 30 and str(row["hedges"]).startswith("frozen") and "UNH" in row["hedges"]
    assert not any(s in row["hedges"] for s in in_xlv)


def test_n2_a_close_in_a_frozen_fund_goes_ahead_with_a_note(bars, flat, sectors):
    t = flat.table
    closer = pick(t, (t["sector fund"] == "XLV") & (t["s"] > -0.4) & (t["symbol"] != "UNH"), n=1)[0]
    plan = _blind_plan(bars, "UNH", extra={live.to_alpaca(closer): 10.0})
    assert closer in plan.closed
    row = plan.orders.set_index("symbol").loc[closer]
    assert row["action"] == "close long" and row["qty"] == 10
    assert "XLV" not in set(plan.orders["symbol"])
    assert any(closer in n and "XLV" in n and "left unchanged" in n for n in plan.notes)


def test_n3_stock_gone_from_the_sectors_file_freezes_every_fund(bars, sectors, monkeypatch):
    monkeypatch.setattr(live.al, "load_sectors", lambda path=None: sectors.drop(index="UNH"))
    plan = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY,
                                                          positions={"UNH": -6.0, "XLV": 30.0, "XLK": -4.0})))
    funds = set(live.hedge_funds(sectors))
    assert not funds & set(plan.orders["symbol"]), "no fund may be traded while UNH's hedge fund is unknown"
    assert not plan.opened and all("frozen" in why for why in plan.skipped.values())
    assert any("UNH" in n and "every fund" in n for n in plan.notes)


def test_n4a_account_info_has_an_optional_regt_buying_power():
    from types import SimpleNamespace as NS
    from quantsoc_ml.paper import AccountInfo, AlpacaPaperBroker
    assert AccountInfo(1.0, 2.0, 3.0).regt_buying_power is None
    assert MockBroker(cash=10.0).account().regt_buying_power is None

    class FakeTrading:
        def get_account(self):
            return NS(equity="100000", cash="50000", buying_power="400000", regt_buying_power="200000")
    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)        # skip __init__: no keys, no real clients
    broker._trading = FakeTrading()
    broker.scrub = str
    acct = broker.account()
    assert (acct.buying_power, acct.regt_buying_power) == (400_000.0, 200_000.0)


def test_n4b_limit_is_the_overnight_regt_figure_when_present(bars):
    probe = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=bars, cash=EQUITY)))
    acct = SharedAlpaca(bars=bars, cash=EQUITY, bp=probe.opening_notional * 4, regt=probe.opening_notional / 2)
    plan = live.build_live_plan(ReadOnlyBroker(acct))
    assert plan.buying_power_limit == probe.opening_notional / 2 and plan.over_buying_power


def test_n4c_exits_only_submit_sends_only_reductions(bars, flat, monkeypatch):
    t = flat.table
    closer = pick(t, (t["s"] > -0.4) & (t["s"] < 1.0), n=1)[0]
    acct = SharedAlpaca(bars=bars, cash=EQUITY, positions={live.to_alpaca(closer): 20.0, "XLB": 500.0}, bp=1.0)
    plan = live.build_live_plan(ReadOnlyBroker(acct))
    assert plan.over_buying_power
    with pytest.raises(Refused, match="buying power"):
        live.submit_checked(acct, plan, plan, confirmed=True, typed="SUBMIT")
    at = _shared_app(acct, monkeypatch)
    assert _button(at).disabled
    exits = [b for b in at.button if b.key == "submit_exits"]
    assert len(exits) == 1 and not exits[0].disabled
    exits[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    sent = [(s["symbol"], s["client_order_id"].rsplit("-", 2)[-2:]) for s in acct.submissions]
    assert sent and {sym for sym, _ in sent} == {live.to_alpaca(closer), "XLB"}
    assert all(leg[0] in ("close", "reduce") for _, leg in sent), sent


def test_n4c_exits_only_keeps_the_stale_plan_and_busy_checks(bars, flat):
    t = flat.table
    closer = pick(t, (t["s"] > -0.4) & (t["s"] < 1.0), n=1)[0]
    acct = SharedAlpaca(bars=bars, cash=EQUITY, positions={live.to_alpaca(closer): 20.0}, bp=1.0)
    shown = live.build_live_plan(ReadOnlyBroker(acct))
    acct.holdings = {}
    fresh = live.build_live_plan(ReadOnlyBroker(acct))
    with pytest.raises(Refused, match="account changed"):
        live.submit_checked(acct, shown, fresh, confirmed=True, typed="SUBMIT", exits_only=True)
    acct.holdings, acct.busy = {live.to_alpaca(closer): 20.0}, ["XLK"]
    busy = live.build_live_plan(ReadOnlyBroker(acct))
    with pytest.raises(Refused, match="open orders"):
        live.submit_checked(acct, busy, busy, confirmed=True, typed="SUBMIT", exits_only=True)
    with pytest.raises(Refused):
        live.submit_checked(acct, busy, busy, confirmed=False, typed="SUBMIT", exits_only=True)
    assert acct.submissions == []


def test_n5_zero_equity_account_renders(bars, monkeypatch):
    acct = SharedAlpaca(bars=bars, cash=0.0)
    at = _shared_app(acct, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert any("n/a" in c.value for c in at.caption)
