"""Workshop 3 paper-trading dashboard: the plan builder, the submit safeguards and the app itself.

No network anywhere: the broker is the MockBroker from quantsoc_ml.paper, or a subclass of it that
pretends to be Alpaca. Real alpaca-py calls are therefore NOT exercised here; that limit is stated.
The strategy is trained on a small slice of the snapshot so the whole file runs in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import statarb as S  # noqa: E402
from quantsoc_ml.paper import BrokerError, MockBroker, ReadOnlyBroker, Refused  # noqa: E402
from workshop3 import live  # noqa: E402

DASHBOARD = ROOT / "workshop3" / "dashboard.py"
SYMBOLS = ["AAPL", "AMZN", "BA", "BAC", "BRK-B", "C", "CAT", "CSCO", "CVX", "DIS", "GE", "GS", "HD", "IBM",
           "INTC", "JNJ", "JPM", "KO", "MCD", "MRK", "MSFT", "PFE", "PG", "WMT", "XOM"]
K = 5
FAKE_KEY, FAKE_SECRET = "PKFAKEKEYMUSTNOTRENDER42", "fakeSecretMustNotRender0987654321"


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def snapshot_slice() -> pd.DataFrame:
    long = S.load_prices()
    long = long[long["symbol"].isin(SYMBOLS)]
    dates = sorted(long["date"].unique())[-520:]
    return long[long["date"].isin(dates)].reset_index(drop=True)


@pytest.fixture(scope="module")
def strategy(snapshot_slice) -> S.LiveStrategy:
    panel = S.make_panel(S.wide(snapshot_slice, "adj_close"))
    return S.train_live(panel, model_name="forest", train_days=200, k=K)


@pytest.fixture(scope="module")
def broker_bars(snapshot_slice) -> pd.DataFrame:
    """The slice as Alpaca would send it: BRK.B, not BRK-B."""
    return live.snapshot_bars_for_broker(snapshot_slice)


@pytest.fixture(scope="module")
def saved(strategy, tmp_path_factory) -> Path:
    return S.save_strategy(strategy, tmp_path_factory.mktemp("ws3") / "strategy.joblib")


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


# --------------------------------------------------------------------------- symbols

def test_symbol_mapping_round_trip():
    assert live.to_alpaca("BRK-B") == "BRK.B"
    assert live.from_alpaca("BRK.B") == "BRK-B"
    assert live.to_alpaca("AAPL") == "AAPL" and live.from_alpaca("AAPL") == "AAPL"
    for sym in S.load_prices()["symbol"].unique():
        assert live.from_alpaca(live.to_alpaca(sym)) == sym
        assert "-" not in live.to_alpaca(sym)


# --------------------------------------------------------------------------- the plan

def test_plan_has_k_longs_and_k_shorts_and_opening_orders(strategy, broker_bars):
    mock = MockBroker(bars=broker_bars, cash=100_000.0)
    plan = live.build_live_plan(ReadOnlyBroker(mock), strategy)
    longs, shorts = set(plan.longs.index), set(plan.shorts.index)
    assert len(longs) == K and len(shorts) == K and not longs & shorts
    assert plan.decision_date == pd.Timestamp(broker_bars["date"].max())
    # every long scores at least as high as every short
    assert plan.scores[list(longs)].min() >= plan.scores[list(shorts)].max()
    # weights are the strategy's own: +g/2k and -g/2k
    each = strategy.gross_exposure / (2 * K)
    assert (plan.weights[list(longs)] == each).all() and (plan.weights[list(shorts)] == -each).all()
    # an empty account: one opening order per name, buys for longs and sells for shorts
    o = plan.orders.set_index("symbol")
    assert set(o.index) == longs | shorts
    assert (o.loc[list(longs), "side"] == "buy").all() and (o.loc[list(longs), "action"] == "open long").all()
    assert (o.loc[list(shorts), "side"] == "sell").all() and (o.loc[list(shorts), "action"] == "open short").all()
    assert (o["qty"] > 0).all()
    # each order is close to its target dollars and never above it (shares round towards zero)
    target = each * plan.equity
    assert ((o["notional"] <= target + 0.01) & (o["notional"] > target - o["notional"] / o["qty"] - 0.01)).all()
    # Alpaca tickers for the broker, Yahoo tickers for the strategy
    if "BRK-B" in o.index:
        assert o.loc["BRK-B", "alpaca_symbol"] == "BRK.B"


def test_plan_against_held_book_reduces_before_adding(strategy, broker_bars):
    mock = live.demo_broker(strategy, broker_bars)          # holds yesterday's book
    assert mock.positions(), "the demo broker should start with yesterday's positions"
    plan = live.build_live_plan(ReadOnlyBroker(mock), strategy)
    assert set(plan.positions) == {live.from_alpaca(s) for s in mock.positions()}
    reducing = plan.orders["action"].isin(["close long", "reduce long", "close short", "reduce short"])
    if reducing.any() and (~reducing).any():
        assert reducing.idxmin() > reducing[reducing].index.max(), "exits must come before new exposure"
    # after the orders, every name sits at its target
    after = dict(plan.positions)
    for row in plan.orders.itertuples():
        after[row.symbol] = after.get(row.symbol, 0.0) + (row.qty if row.side == "buy" else -row.qty)
    targets = live.targets_table(plan).set_index("symbol")["target shares"]
    for sym, q in targets.items():
        assert after.get(sym, 0.0) == pytest.approx(q, abs=1) or abs(after.get(sym, 0.0) - q) * plan.last_prices[sym] < 1


def test_unshortable_names_are_skipped_for_the_next_lowest(strategy, broker_bars):
    base = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=broker_bars)), strategy)
    worst_two = list(base.scores.sort_values().index[:2])
    mock = MockBroker(bars=broker_bars, not_shortable=[live.to_alpaca(s) for s in worst_two])
    plan = live.build_live_plan(ReadOnlyBroker(mock), strategy)
    assert set(plan.skipped_shorts) == set(worst_two)
    assert len(plan.shorts) == K and not set(plan.shorts.index) & set(worst_two)
    expected = list(base.scores.drop(index=worst_two).sort_values().index[:K])
    assert set(plan.shorts.index) == set(expected)
    assert set(plan.longs.index) == set(base.longs.index)


def test_borrow_unknown_is_kept_and_noted(strategy, broker_bars):
    scores = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=broker_bars)), strategy).scores
    w, skipped, unknown = live.weights_with_borrow(strategy, scores, lambda s: None)
    assert not skipped and len(unknown) == K and (w < 0).sum() == K


def test_plan_builder_refuses_a_broker_that_can_submit(strategy, broker_bars):
    with pytest.raises(TypeError):
        live.build_live_plan(MockBroker(bars=broker_bars), strategy)


# --------------------------------------------------------------------------- submit safeguards

@pytest.fixture()
def fresh(strategy, broker_bars):
    mock = FakeAlpaca(bars=broker_bars, cash=100_000.0)
    return mock, live.build_live_plan(ReadOnlyBroker(mock), strategy)


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
    mock.busy = [plan.orders["alpaca_symbol"].iloc[0]]
    with pytest.raises(Refused, match="open orders"):
        live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_submit_refuses_a_read_only_view_and_changed_orders(fresh):
    mock, plan = fresh
    with pytest.raises(Refused):
        live.submit_plan(ReadOnlyBroker(mock), plan, confirmed=True, typed="SUBMIT")
    with pytest.raises(Refused, match="changed"):
        live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT", expected_fingerprint="not-the-same")
    assert mock.submissions == []


def test_submit_sends_in_plan_order_with_deterministic_ids_and_no_duplicates(fresh):
    mock, plan = fresh
    results = live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT", expected_fingerprint=plan.fingerprint)
    assert [s["symbol"] for s in mock.submissions] == list(plan.orders["alpaca_symbol"])
    ids = [r.client_order_id for r in results]
    assert len(set(ids)) == len(ids) == len(plan.orders)
    first = plan.orders.iloc[0]
    assert ids[0] == f"qsml-ws3-{plan.decision_date:%Y%m%d}-{first['alpaca_symbol']}-{first['action'].replace(' ', '-')}"
    assert all(r.status == "filled" for r in results)
    held = dict(mock.holdings)
    # a double click: the same ids again, which the broker refuses, so nothing changes
    again = live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    assert all(r.status == "rejected" for r in again)
    assert mock.holdings == held


def test_client_order_id_is_deterministic():
    a = live.client_order_id(pd.Timestamp("2026-10-15"), "BRK.B", "open short")
    assert a == live.client_order_id("2026-10-15", "BRK.B", "open short") == "qsml-ws3-20261015-BRK.B-open-short"
    assert a != live.client_order_id("2026-10-15", "BRK.B", "close long")
    assert len(a) <= 128                       # Alpaca's limit for client_order_id


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


def _app(saved, mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    at.session_state["mode"] = mode
    at.session_state["strategy_path"] = str(saved)
    return at


def test_app_offline_renders_without_submit(saved, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    at = _app(saved, "Offline demo").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("OFFLINE DEMO" in i.value for i in at.info)
    assert not [b for b in at.button if b.key == "submit"]
    text = _rendered(at)
    assert "Loaded your saved strategy" in text and "Decision date" in text


def test_app_alpaca_mode_with_a_fake_never_shows_keys_and_submits_only_when_confirmed(saved, broker_bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fakes = []

    def connect(key, secret):
        assert (key, secret) == (FAKE_KEY, FAKE_SECRET)   # the keys reach the broker, and only the broker
        fakes.append(FakeAlpaca(bars=broker_bars, cash=100_000.0))
        return fakes[-1]
    monkeypatch.setattr(live, "connect_alpaca", connect)

    at = _app(saved, "Alpaca paper").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("environment variables" in c.value for c in at.sidebar.caption)
    button = [b for b in at.button if b.key == "submit"]
    assert len(button) == 1 and button[0].disabled

    # the box alone, or the wrong word, leaves the button disabled; loading and reruns send nothing
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
    assert any("Last submission" in m.value for m in at.markdown)

    text = _rendered(at)
    assert FAKE_KEY not in text and FAKE_SECRET not in text


def test_alpaca_read_adapters_with_a_fake_trading_client():
    """equity_history and recent_orders build valid alpaca-py requests and plain tables. The client is fake."""
    from types import SimpleNamespace as NS
    from quantsoc_ml.paper import AlpacaPaperBroker

    class FakeTrading:
        def get_portfolio_history(self, req):
            self.history_req = req
            return NS(timestamp=[1760486400, 1760572800], equity=[100_000.0, 100_500.0])

        def get_orders(self, req):
            self.orders_req = req
            return [NS(submitted_at=pd.Timestamp("2026-10-15 14:00", tz="UTC"), symbol="BRK.B", side=NS(value="sell"),
                       qty="3", filled_qty="3", status=NS(value="filled"), client_order_id="qsml-ws3-x")]

    broker = AlpacaPaperBroker.__new__(AlpacaPaperBroker)    # skip __init__: no keys, no real clients
    broker._trading = FakeTrading()
    hist = live.equity_history(broker)
    assert list(hist["equity"]) == [100_000.0, 100_500.0] and broker._trading.history_req.timeframe == "1D"
    orders = live.recent_orders(broker)
    assert orders.loc[0, "symbol"] == "BRK-B" and orders.loc[0, "side"] == "sell" and orders.loc[0, "filled"] == 3.0


# --------------------------------------------------------------------------- regressions from the review

def test_credentials_are_stripped_and_inner_whitespace_refused(monkeypatch, tmp_path):
    from quantsoc_ml import paper as Pp
    monkeypatch.setenv("APCA_API_KEY_ID", "  PKFAKE123 ")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fakesecret456\n")
    assert Pp.get_credentials([tmp_path / ".env"]) == ("PKFAKE123", "fakesecret456", "environment variables")
    monkeypatch.setenv("APCA_API_KEY_ID", "PKFAKE 123")
    k, s, src = Pp.get_credentials([tmp_path / ".env"])
    assert k is None and s is None and "space or line break" in src and "PKFAKE" not in src


def test_alpaca_broker_scrubs_keys_from_error_messages():
    from quantsoc_ml.paper import AlpacaPaperBroker, BrokerError
    b = AlpacaPaperBroker("PKFAKEKEY0001", "FAKESECRET0001")          # builds clients only, no network

    def boom():
        raise ValueError("Invalid header value b'FAKESECRET0001\\n' for PKFAKEKEY0001")
    with pytest.raises(BrokerError) as e:
        b._call(boom)
    assert "FAKESECRET0001" not in str(e.value) and "PKFAKEKEY0001" not in str(e.value)
    assert "FAKESECRET" not in repr(b)


def test_held_long_in_an_unshortable_worst_name_is_closed(strategy, broker_bars):
    base = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=broker_bars)), strategy)
    worst = base.scores.sort_values().index[0]
    mock = MockBroker(bars=broker_bars, not_shortable=[live.to_alpaca(worst)], positions={live.to_alpaca(worst): 100})
    plan = live.build_live_plan(ReadOnlyBroker(mock), strategy)
    assert worst in plan.skipped_shorts and plan.weights[worst] == 0.0
    row = plan.orders[plan.orders["symbol"] == worst]
    assert list(row["action"]) == ["close long"] and float(row["qty"].iloc[0]) == 100


def test_flip_while_market_closed_sends_only_the_closing_leg(strategy, broker_bars):
    base = live.build_live_plan(ReadOnlyBroker(MockBroker(bars=broker_bars)), strategy)
    short_name = base.shorts.index[0]
    mock = FakeAlpaca(bars=broker_bars, cash=100_000.0, positions={live.to_alpaca(short_name): 10})
    plan = live.build_live_plan(ReadOnlyBroker(mock), strategy)
    assert not plan.clock.get("is_open")
    results = live.submit_plan(mock, plan, confirmed=True, typed="SUBMIT")
    legs = {r.client_order_id.rsplit(live.to_alpaca(short_name) + "-", 1)[-1]: r.status
            for r in results if r.symbol == live.to_alpaca(short_name)}
    assert legs == {"close-long": "filled", "open-short": "not sent"}
    assert mock.positions().get(live.to_alpaca(short_name), 0) == 0


@pytest.mark.parametrize("hour,is_open,kept", [(8, False, False), (11, True, False), (17, False, True)])
def test_todays_bar_is_used_only_after_the_close(hour, is_open, kept):
    bars = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"]), "symbol": "A",
                         "close": 1.0, "adj_close": 1.0, "volume": 1.0})
    clock = {"is_open": is_open, "timestamp": pd.Timestamp(f"2026-10-16 {hour:02d}:30", tz="America/New_York")}
    assert (len(live.drop_today_until_close(bars, clock)) == 2) == kept


def test_app_submits_the_plan_on_screen_even_if_equity_moved(saved, broker_bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fakes = []

    def connect(key, secret):        # every rerun sees a different equity, as during market hours
        fakes.append(FakeAlpaca(bars=broker_bars, cash=100_000.0 + 7_777.0 * len(fakes)))
        return fakes[-1]
    monkeypatch.setattr(live, "connect_alpaca", connect)
    at = _app(saved, "Alpaca paper").run()
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    shown = at.session_state["shown_plan"]
    [b for b in at.button if b.key == "submit"][0].click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    sent = [f for f in fakes if f.submissions]
    assert len(sent) == 1
    assert [(s["symbol"], s["qty"]) for s in sent[0].submissions] == \
        [(r.alpaca_symbol, float(r.qty)) for r in shown.orders.itertuples()]


# --------------------------------------------------------------------------- regressions from the Workshop 4 falsification pass

def test_app_warns_one_strategy_per_account(saved, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    at = _app(saved, "Offline demo").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("One strategy per paper account" in w.value and "Workshop 4" in w.value for w in at.warning)


def test_app_refuses_when_the_account_changed_after_the_page_was_drawn(saved, broker_bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    acct = FakeAlpaca(bars=broker_bars, cash=100_000.0)            # one account shared by every rerun
    monkeypatch.setattr(live, "connect_alpaca", lambda key, secret: acct)
    at = _app(saved, "Alpaca paper").run()
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    acct.holdings = {"AAPL": 5.0}                                   # traded elsewhere after the page was drawn
    [b for b in at.button if b.key == "submit"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert acct.submissions == []
    assert any("account changed" in e.value for e in at.error)
