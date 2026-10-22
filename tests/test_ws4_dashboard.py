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
        expected = np.trunc(side * 0.05 * flat.equity / flat.last_prices[sym])
        assert orders.loc[sym, "target_qty"] == expected
        assert orders.loc[sym, "qty"] == abs(expected)
        assert orders.loc[sym, "action"] == ("open long" if side > 0 else "open short")
        assert orders.loc[sym, "side"] == ("buy" if side > 0 else "sell")
        # the dollar value is at most 5% of equity and within one share of it
        dollars = abs(expected) * flat.last_prices[sym]
        assert 0.05 * flat.equity - flat.last_prices[sym] < dollars <= 0.05 * flat.equity + 1e-6
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
        qty = 7.0 if sym == kept_long else np.trunc(side * 0.05 * plan.equity / last[sym])
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
