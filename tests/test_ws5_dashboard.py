"""Workshop 5 paper-trading dashboard: the plan builder, the weekly rule, the submit safeguards and the app.

No network anywhere: the broker is the MockBroker from quantsoc_ml.paper, or a subclass of it that
pretends to be Alpaca. Real alpaca-py calls are therefore NOT exercised here (apart from building
the request object for a fake data client); that limit is stated. The networks are UNTRAINED (random
weights): these tests check plumbing and safety, never that the scores mean anything.
Expected values are recomputed independently from the library (quantsoc_ml.cnn) and the snapshot.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import cnn  # noqa: E402
from quantsoc_ml import paper  # noqa: E402
from quantsoc_ml.paper import AccountInfo, BrokerError, MockBroker, OrderResult, ReadOnlyBroker, Refused  # noqa: E402
from workshop5 import live  # noqa: E402

DASHBOARD = ROOT / "workshop5" / "dashboard.py"
FAKE_KEY, FAKE_SECRET = "PKFAKEKEYWS5MUSTNOTRENDER", "fakeWs5SecretMustNotRender0987654321"
EQUITY = 100_000.0


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ws5_models")
    for k in range(2):
        cnn.save_trained(cnn.Trained(5, 5, cnn.build_cnn(5, seed=k), 127.0, 100.0), d / f"I5R5_seed{k}.pt")
    return d


@pytest.fixture(scope="module")
def models(model_dir):
    return live.load_models(model_dir)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """The last days of the snapshot as Alpaca-style OHLC bars (Alpaca tickers)."""
    return live.snapshot_bars()


@pytest.fixture(scope="module")
def dates(bars):
    return sorted(pd.to_datetime(bars["date"]).unique())


@pytest.fixture(scope="module")
def symbols():
    return live.snapshot_symbols()


@pytest.fixture(scope="module")
def full_market() -> cnn.Market:
    return cnn.load_market()


class FakeAlpaca(MockBroker):
    """A MockBroker that says it is the Alpaca paper broker (online). No network."""
    name = "alpaca-paper"
    offline = False

    def __init__(self, *a, busy=(), paper=True, account_override=None, **kw):
        super().__init__(*a, **kw)
        self.busy = list(busy)
        self.paper = paper
        self.account_override = account_override

    def open_orders(self):
        return sorted(set(super().open_orders()) | set(self.busy))

    def account(self):
        return self.account_override or super().account()

    def clock(self):
        return {**super().clock(), "next_open": pd.Timestamp("2026-10-16 09:30", tz="America/New_York")}

    def equity_history(self):
        return pd.DataFrame({"date": pd.date_range("2026-10-01", periods=5, freq="B"),
                             "equity": [100_000, 100_250, 99_900, 100_400, 100_100]})


def plan_for(broker, models, symbols, **kw):
    return live.build_live_plan(ReadOnlyBroker(broker), models, symbols, **kw)


@pytest.fixture(scope="module")
def flat(bars, models, symbols):
    """The plan for an empty account that has never rebalanced: due, every position is new."""
    return plan_for(MockBroker(bars=bars, cash=EQUITY), models, symbols)


def filled_ws5_order(decision_date, symbol="AAPL") -> OrderResult:
    coid = f"qsml-ws5-{pd.Timestamp(decision_date):%Y%m%d}-{symbol}-open-long"
    return OrderResult(symbol, "buy", 5.0, "filled", 5.0, coid)


# --------------------------------------------------------------------------- the chart is the library's chart

def test_dashboard_chart_equals_the_library_chart_built_from_the_snapshot(flat, full_market):
    t = flat.decision_date
    for sym in list(flat.table["symbol"].iloc[:3]) + list(flat.table["symbol"].iloc[-3:]) + ["BRK-B"]:
        expected = cnn.draw_images(days=5, **cnn.chart_inputs(full_market, 5, [t], [sym]))[0]
        got = live.chart_array(flat, sym)
        assert got.shape == (32, 15) and got.dtype == np.uint8
        assert np.array_equal(got, expected), f"{sym}: live bars -> Market conversion changed the chart"


def test_dashboard_scores_equal_the_library_scores_on_the_full_snapshot(flat, models, full_market):
    expected = cnn.live_scores(full_market, models, date=flat.decision_date).set_index("symbol")
    got = flat.table.set_index("symbol")
    assert set(got.index) == set(expected.index)
    assert np.allclose(got.loc[expected.index, "prob_up"], expected["prob_up"], atol=1e-6)
    assert (got.loc[expected.index, "decile"] == expected["decile"]).all()


def test_enlarged_chart_is_a_nearest_neighbour_blow_up(flat):
    small = live.chart_array(flat, flat.table["symbol"].iloc[0])
    big = live.enlarge(small, 7)
    assert big.shape == (32 * 7, 15 * 7) and set(np.unique(big)) <= {0, 255}
    assert np.array_equal(big[::7, ::7], small) and np.array_equal(big[6::7, 6::7], small)


# --------------------------------------------------------------------------- unfinished day

def test_todays_unfinished_bar_is_dropped_while_the_market_is_open(bars, models, symbols, dates):
    open_plan = plan_for(MockBroker(bars=bars, cash=EQUITY, is_open=True), models, symbols)
    closed_plan = plan_for(MockBroker(bars=bars, cash=EQUITY, is_open=False), models, symbols)
    assert closed_plan.decision_date == pd.Timestamp(dates[-1])
    assert open_plan.decision_date == pd.Timestamp(dates[-2])
    assert pd.Timestamp(dates[-1]) not in open_plan.mkt.adj_close.index
    # the chart really is drawn from completed days only: its last close is the previous day's
    sym = open_plan.table["symbol"].iloc[0]
    expected_close = bars[(bars["date"] == dates[-2]) & (bars["symbol"] == live.to_alpaca(sym))]["adj_close"].iloc[0]
    assert open_plan.last_prices[sym] == pytest.approx(expected_close)


def test_a_forming_bar_with_wild_values_cannot_reach_the_chart(bars, models, symbols, dates):
    wild = bars.copy()
    last = wild["date"] == dates[-1]
    wild.loc[last, ["open", "high", "low", "close", "adj_close"]] = 1e6
    plan = plan_for(MockBroker(bars=wild, cash=EQUITY, is_open=True), models, symbols)
    assert plan.decision_date == pd.Timestamp(dates[-2])
    assert max(plan.last_prices.values()) < 1e5


# --------------------------------------------------------------------------- book and sizing

def test_book_is_the_top_and_bottom_decile_in_equal_weights_on_half_of_equity_each(flat):
    t = flat.table
    top, bottom = set(t.loc[t["decile"] == 10, "symbol"]), set(t.loc[t["decile"] == 1, "symbol"])
    w = flat.weights
    assert set(w[w > 0].index) == top and set(w[w < 0].index) == bottom
    assert w[w > 0].sum() == pytest.approx(0.5) and w[w < 0].sum() == pytest.approx(-0.5)
    assert w[w > 0].nunique() == 1 and w[w < 0].nunique() == 1 or np.allclose(w[w > 0], w[w > 0].iloc[0])
    orders = flat.orders.set_index("symbol")
    for sym in top | bottom:
        expected = np.trunc(w[sym] * flat.equity / flat.last_prices[sym])
        assert orders.loc[sym, "target_qty"] == expected
    assert set(flat.orders["alpaca_symbol"]) == {live.to_alpaca(s) for s in flat.orders["symbol"]}


def test_plan_refuses_a_broker_that_can_submit(bars, models, symbols):
    with pytest.raises(TypeError):
        live.build_live_plan(MockBroker(bars=bars), models, symbols)


# --------------------------------------------------------------------------- non-shortable

def test_unshortable_name_is_skipped_for_the_next_lowest_and_explained(bars, models, symbols, flat):
    ranked_low_first = list(flat.table.sort_values("prob_up")["symbol"])
    n_short = int((flat.table["decile"] == 1).sum())
    blocked, replacement = ranked_low_first[0], ranked_low_first[n_short]
    mock = MockBroker(bars=bars, cash=EQUITY, not_shortable=[live.to_alpaca(blocked)])
    plan = plan_for(mock, models, symbols)
    assert blocked in plan.skipped_shorts and "borrow" in plan.skipped_shorts[blocked]
    assert plan.weights[blocked] == 0.0
    assert replacement in set(plan.weights[plan.weights < 0].index)
    assert len(plan.shorts) == n_short, "the short book keeps its size"
    assert blocked not in set(plan.orders.loc[plan.orders["side"] == "sell", "symbol"])
    assert set(plan.longs.index) == set(flat.longs.index), "the long book is unchanged"


def test_unknown_borrow_is_kept_and_noted(bars, models, symbols):
    class Unsure(MockBroker):
        def shortable(self, symbol):
            return None
    plan = plan_for(Unsure(bars=bars, cash=EQUITY), models, symbols)
    assert any("could not check borrow" in n for n in plan.notes) and not plan.skipped_shorts


# --------------------------------------------------------------------------- weekly rule

def test_rebalance_status_counts_trading_days_after_the_last_decision():
    days = pd.bdate_range("2026-10-05", periods=12)
    for since, due in ((0, False), (4, False), (5, True), (7, True)):
        last = days[11 - since]
        got_due, got_since, reason = live.rebalance_status(last, list(days), days[11])
        assert (got_due, got_since) == (due, since), reason
    assert live.rebalance_status(None, list(days), days[-1])[0] is True


def test_last_rebalance_date_reads_only_filled_ws5_orders():
    t = pd.DataFrame({"client_order_id": ["qsml-ws5-20261001-AAPL-open-long", "qsml-ws5-20261008-MSFT-open-long",
                                          "qsml-ws5-20261015-GE-open-short", "qsml-ws4-20261015-XOM-open-long",
                                          "qsml-ws3-20261015-XOM-open-long", "manual-20261015"],
                      "filled": [5, 5, 0, 5, 5, 5],      # the 15th ws5 order never filled
                      "status": ["filled", "filled", "rejected", "filled", "filled", "filled"]})
    assert live.last_rebalance_date(t) == pd.Timestamp("2026-10-08")
    assert live.last_rebalance_date(t.iloc[2:]) is None
    assert live.last_rebalance_date(pd.DataFrame(columns=t.columns)) is None


def test_plan_holds_before_five_days_and_rebalances_from_five(bars, models, symbols, dates):
    mock = MockBroker(bars=bars, cash=EQUITY)
    d = pd.Timestamp(dates[-1])
    hold = plan_for(mock, models, symbols, last_rebalance=pd.Timestamp(dates[-4]))
    assert not hold.due and not hold.can_rebalance and hold.days_since == 3
    assert len(hold.orders) == 0 and "hold" in hold.cadence_note
    assert len(hold.table) > 0 and len(hold.weights[hold.weights != 0]) > 0, "the ranking is still shown on a hold"
    due = plan_for(mock, models, symbols, last_rebalance=pd.Timestamp(dates[-6]))
    assert due.due and due.can_rebalance and due.days_since == 5 and len(due.orders) > 0
    assert due.decision_date == d


def test_override_proposes_orders_on_a_hold_and_says_so(bars, models, symbols, dates):
    mock = MockBroker(bars=bars, cash=EQUITY)
    plan = plan_for(mock, models, symbols, last_rebalance=pd.Timestamp(dates[-2]), force=True)
    assert not plan.due and plan.forced and plan.can_rebalance and len(plan.orders) > 0
    assert any("OVERRIDE" in n for n in plan.notes)


def test_submit_refuses_a_plan_that_is_neither_due_nor_forced(bars, models, symbols, dates):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    hold = plan_for(mock, models, symbols, last_rebalance=pd.Timestamp(dates[-2]))
    smuggled = dataclasses.replace(hold, orders=plan_for(mock, models, symbols).orders)   # orders on a hold plan
    with pytest.raises(Refused, match="weekly rule"):
        live.submit_checked(mock, smuggled, smuggled, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


# --------------------------------------------------------------------------- submit safeguards

@pytest.mark.parametrize("confirmed,typed", [(False, "SUBMIT"), (True, "submit"), (True, ""), (False, ""), (True, "SUBMIT now")])
def test_no_submit_without_the_box_and_the_word(bars, models, symbols, confirmed, typed):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    plan = plan_for(mock, models, symbols)
    with pytest.raises(Refused):
        live.submit_checked(mock, plan, plan, confirmed=confirmed, typed=typed)
    assert mock.submissions == []


def test_submit_sends_exactly_the_plan_on_screen_with_ws5_ids(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    plan = plan_for(mock, models, symbols)
    results = live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
    assert len(results) == len(plan.orders) > 0
    assert [(s["symbol"], s["side"], s["qty"]) for s in mock.submissions] == \
        [(r.alpaca_symbol, r.side, float(r.qty)) for r in plan.orders.itertuples()]
    assert all(s["client_order_id"].startswith("qsml-ws5-") for s in mock.submissions)
    assert not any(s["client_order_id"].startswith(("qsml-ws3", "qsml-ws4")) for s in mock.submissions)
    assert plan.order_id_prefix == "qsml-ws5"


def test_a_plan_drawn_for_a_different_account_is_refused(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    shown = plan_for(mock, models, symbols)
    mock.holdings["AAPL"] = mock.holdings.get("AAPL", 0.0) + 7          # the account changed after the page was drawn
    fresh = plan_for(mock, models, symbols)
    with pytest.raises(Refused, match="account changed"):
        live.submit_checked(mock, shown, fresh, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_a_missing_or_older_shown_plan_is_refused(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    fresh = plan_for(mock, models, symbols)
    with pytest.raises(Refused):
        live.submit_checked(mock, None, fresh, confirmed=True, typed="SUBMIT")
    older = dataclasses.replace(fresh, decision_date=fresh.decision_date - pd.Timedelta(days=1))
    with pytest.raises(Refused):
        live.submit_checked(mock, older, fresh, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_equity_drift_alone_does_not_make_a_plan_stale(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    shown = plan_for(mock, models, symbols)
    mock.cash += 1_000.0                                               # equity moves all day; positions do not
    fresh = plan_for(mock, models, symbols)
    assert live.submit_checked(mock, shown, fresh, confirmed=True, typed="SUBMIT")
    assert len(mock.submissions) == len(shown.orders)                  # the plan ON SCREEN went, not the new one


def test_open_order_on_a_strategy_symbol_blocks_submit(bars, models, symbols, flat):
    sym = flat.orders["alpaca_symbol"].iloc[0]
    mock = FakeAlpaca(bars=bars, cash=EQUITY, busy=[sym])
    plan = plan_for(mock, models, symbols)
    assert plan.busy == [live.from_alpaca(sym)]
    with pytest.raises(Refused, match="open orders"):
        live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
    # even a strategy stock this plan would NOT trade blocks it
    idle = next(s for s in symbols if s not in set(flat.orders["symbol"]) and s in set(flat.table["symbol"]))
    mock2 = FakeAlpaca(bars=bars, cash=EQUITY, busy=[live.to_alpaca(idle)])
    plan2 = plan_for(mock2, models, symbols)
    with pytest.raises(Refused, match="open orders"):
        live.submit_checked(mock2, plan2, plan2, confirmed=True, typed="SUBMIT")
    assert mock.submissions == [] and mock2.submissions == []


def test_refuses_a_non_paper_broker(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY, paper=False)
    plan = plan_for(mock, models, symbols)
    with pytest.raises(Refused, match="paper"):
        live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


# --------------------------------------------------------------------------- buying power

@pytest.mark.parametrize("bp,regt,over", [(1e9, 1_000.0, True),       # the overnight figure is the tighter one
                                          (1_000.0, 1e9, True),       # the headline figure is the tighter one
                                          (1e9, 1e9, False), (1e9, None, False), (1_000.0, None, True)])
def test_new_exposure_must_fit_in_the_smaller_of_the_two_buying_powers(bars, models, symbols, bp, regt, over):
    acct = AccountInfo(equity=EQUITY, cash=EQUITY, buying_power=bp, regt_buying_power=regt)
    mock = FakeAlpaca(bars=bars, cash=EQUITY, account_override=acct)
    plan = plan_for(mock, models, symbols)
    assert plan.buying_power_limit == min(bp, regt if regt is not None else bp)
    assert plan.over_buying_power is over
    if over:
        with pytest.raises(Refused, match="buying power"):
            live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
        assert mock.submissions == []
    else:
        assert live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")


def test_over_buying_power_still_lets_the_exits_go(bars, models, symbols, flat):
    held = {live.to_alpaca(s): 3.0 for s in list(flat.table["symbol"].iloc[40:44])}     # stocks the book will not hold
    acct = AccountInfo(equity=EQUITY, cash=EQUITY, buying_power=1.0, regt_buying_power=1.0)
    mock = FakeAlpaca(bars=bars, cash=EQUITY, positions=held, account_override=acct)
    plan = plan_for(mock, models, symbols)
    assert plan.over_buying_power and len(plan.exit_orders) == len(held)
    live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT", exits_only=True)
    assert {s["symbol"] for s in mock.submissions} == set(held) and all(s["side"] == "sell" for s in mock.submissions)


# --------------------------------------------------------------------------- paper.py additions leave the old paths alone

def test_mock_daily_bars_is_unchanged_by_extra_ohlc_columns(bars):
    plain = bars[["date", "symbol", "close", "adj_close", "volume"]]
    a, b = MockBroker(bars=plain), MockBroker(bars=bars)
    syms = ["AAPL", "MSFT", "BRK.B"]
    pd.testing.assert_frame_equal(a.daily_bars(syms, 12), b.daily_bars(syms, 12))
    assert list(b.daily_bars(syms, 12).columns) == paper.BAR_COLUMNS
    assert list(b.daily_ohlc_bars(syms, 12).columns) == paper.OHLC_COLUMNS
    with pytest.raises(BrokerError, match="open"):
        a.daily_ohlc_bars(syms, 12)
    pd.testing.assert_frame_equal(ReadOnlyBroker(b).daily_ohlc_bars(syms, 12), b.daily_ohlc_bars(syms, 12))


class _FakeDataClient:
    """Stands in for alpaca-py's StockHistoricalDataClient: returns a fixed bar table, no network."""
    def __init__(self, df):
        self.df = df
        self.requests = []

    def get_stock_bars(self, req):
        self.requests.append(req)
        return SimpleNamespace(df=self.df)


def _alpaca_frame():
    ts = pd.to_datetime(["2026-10-12 04:00", "2026-10-13 04:00", "2026-10-14 04:00"], utc=True)
    idx = pd.MultiIndex.from_product([["AAPL", "MSFT"], ts], names=["symbol", "timestamp"])
    n = len(idx)
    return pd.DataFrame({"open": np.linspace(10, 20, n), "high": np.linspace(11, 21, n), "low": np.linspace(9, 19, n),
                         "close": np.linspace(10.5, 20.5, n), "volume": np.arange(n) * 100.0}, index=idx)


def test_fetch_daily_bars_is_unchanged_and_the_ohlc_fetch_agrees_with_it():
    pytest.importorskip("alpaca")
    client = _FakeDataClient(_alpaca_frame())
    old = paper.fetch_daily_bars(["AAPL", "MSFT"], 2, client)
    expected = pd.DataFrame({"date": pd.to_datetime(["2026-10-13", "2026-10-13", "2026-10-14", "2026-10-14"]),
                             "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                             "close": [11.5, 17.5, 12.5, 18.5], "adj_close": [11.5, 17.5, 12.5, 18.5],
                             "volume": [100.0, 400.0, 200.0, 500.0]})
    expected["close"] = [client.df.loc[("AAPL", pd.Timestamp("2026-10-13", tz="UTC") + pd.Timedelta(hours=4)), "close"],
                         client.df.loc[("MSFT", pd.Timestamp("2026-10-13", tz="UTC") + pd.Timedelta(hours=4)), "close"],
                         client.df.loc[("AAPL", pd.Timestamp("2026-10-14", tz="UTC") + pd.Timedelta(hours=4)), "close"],
                         client.df.loc[("MSFT", pd.Timestamp("2026-10-14", tz="UTC") + pd.Timedelta(hours=4)), "close"]]
    expected["adj_close"] = expected["close"]
    expected["volume"] = [client.df.loc[(s, pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4)), "volume"]
                          for s, d in (("AAPL", "2026-10-13"), ("MSFT", "2026-10-13"), ("AAPL", "2026-10-14"), ("MSFT", "2026-10-14"))]
    pd.testing.assert_frame_equal(old, expected)
    assert list(old.columns) == paper.BAR_COLUMNS
    new = paper.fetch_daily_ohlc_bars(["AAPL", "MSFT"], 2, client)
    assert list(new.columns) == paper.OHLC_COLUMNS
    pd.testing.assert_frame_equal(new[paper.BAR_COLUMNS], old)
    row = new[(new["symbol"] == "MSFT") & (new["date"] == "2026-10-14")].iloc[0]
    src = client.df.loc[("MSFT", pd.Timestamp("2026-10-14", tz="UTC") + pd.Timedelta(hours=4))]
    assert (row["open"], row["high"], row["low"]) == (src["open"], src["high"], src["low"])


# --------------------------------------------------------------------------- models and the offline demo

def test_load_models_says_plainly_when_there_are_none(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cached networks"):
        live.load_models(tmp_path)


def test_demo_broker_holds_last_weeks_book_so_a_rebalance_is_due(bars, models, symbols):
    mock = live.demo_broker(bars, models)
    assert mock.positions() and mock.offline
    last = live.broker_last_rebalance(mock)
    assert last is not None
    plan = plan_for(mock, models, symbols, last_rebalance=last)
    assert plan.due and plan.days_since == 5 and len(plan.orders) > 0
    assert (live.demo_broker(bars, None).positions() == {})


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


def _app(model_dir, mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(DASHBOARD), default_timeout=180)
    at.session_state["mode"] = mode
    at.session_state["model_dir"] = str(model_dir)
    return at


def _no_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)


def test_app_offline_renders_charts_and_no_submit(model_dir, monkeypatch):
    _no_keys(monkeypatch)
    at = _app(model_dir, "Offline demo").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("OFFLINE DEMO" in i.value for i in at.info)
    assert any("One strategy per paper account" in w.value for w in at.warning)
    assert not [b for b in at.button if b.key == "submit"]
    text = _rendered(at)
    assert "Decision date" in text and "REBALANCE DUE" in text and "What the networks saw" in text
    assert text.count("imgs {") == 8, "four top and four bottom charts"


def test_app_without_networks_says_so_instead_of_crashing(tmp_path, monkeypatch):
    _no_keys(monkeypatch)
    at = _app(tmp_path, "Offline demo").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No trained networks found" in e.value for e in at.error)
    assert not [b for b in at.button if b.key == "submit"]


def test_app_alpaca_mode_never_shows_keys_and_submits_only_when_confirmed(model_dir, bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fakes = []

    def connect(key, secret):
        assert (key, secret) == (FAKE_KEY, FAKE_SECRET)     # the keys reach the broker, and only the broker
        fakes.append(FakeAlpaca(bars=bars, cash=EQUITY))
        return fakes[-1]
    monkeypatch.setattr(live, "connect_alpaca", connect)

    at = _app(model_dir, "Alpaca paper").run()
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
    assert all(s["client_order_id"].startswith("qsml-ws5-") for s in sent[0].submissions)
    assert any("Last submission" in m.value for m in at.markdown)
    text = _rendered(at)
    assert FAKE_KEY not in text and FAKE_SECRET not in text


def test_app_holds_until_the_override_is_ticked(model_dir, bars, dates, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fakes = []

    def connect(key, secret):
        f = FakeAlpaca(bars=bars, cash=EQUITY)
        prior = filled_ws5_order(dates[-3])                   # last rebalance two trading days ago
        f.orders[prior.client_order_id] = prior
        fakes.append(f)
        return f
    monkeypatch.setattr(live, "connect_alpaca", connect)

    at = _app(model_dir, "Alpaca paper").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "HOLD." in _rendered(at)
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    assert [b for b in at.button if b.key == "submit"][0].disabled, "a hold week must not be submittable"
    at.checkbox(key="override").check().run()
    assert "override is ticked" in _rendered(at)
    assert not [b for b in at.button if b.key == "submit"][0].disabled
    assert all(not f.submissions for f in fakes)


def test_app_refuses_when_the_account_changed_after_the_page_was_drawn(model_dir, bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)
    fake = FakeAlpaca(bars=bars, cash=EQUITY)
    monkeypatch.setattr(live, "connect_alpaca", lambda k, s: fake)
    at = _app(model_dir, "Alpaca paper").run()
    at.checkbox(key="confirm_box").check().run()
    at.text_input(key="confirm_text").input("SUBMIT").run()
    fake.holdings["AAPL"] = 11.0                                 # the account moves under the page
    [b for b in at.button if b.key == "submit"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert fake.submissions == []
    assert any("account changed" in e.value for e in at.error)


# --------------------------------------------------------------------------- falsifier round: fixes

def test_fix1_open_order_on_a_strategy_stock_with_no_bars_today_still_blocks(bars, models, symbols):
    no_aapl = bars[bars["symbol"] != "AAPL"]
    mock = FakeAlpaca(bars=no_aapl, cash=EQUITY, busy=["AAPL"])
    plan = plan_for(mock, models, symbols)
    assert "AAPL" not in set(plan.table["symbol"]) and "AAPL" in plan.universe
    assert plan.busy == ["AAPL"]
    with pytest.raises(Refused, match="open orders"):
        live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_fix2_buying_power_is_checked_on_the_fresh_plan_too(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    shown = plan_for(mock, models, symbols)
    mock.account_override = AccountInfo(equity=EQUITY, cash=EQUITY, buying_power=10.0, regt_buying_power=None)
    fresh = plan_for(mock, models, symbols)
    assert not shown.over_buying_power and fresh.over_buying_power
    with pytest.raises(Refused, match="buying power"):
        live.submit_checked(mock, shown, fresh, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


@pytest.mark.parametrize("equity,bp,regt", [(-1000.0, 1e9, None), (0.0, 1e9, None), (EQUITY, float("nan"), None),
                                            (EQUITY, 1e9, float("nan")), (EQUITY, float("inf"), None)])
def test_fix3_bad_money_fails_closed(bars, models, symbols, equity, bp, regt):
    acct = AccountInfo(equity=equity, cash=EQUITY, buying_power=bp, regt_buying_power=regt)
    mock = FakeAlpaca(bars=bars, cash=EQUITY, account_override=acct)
    plan = plan_for(mock, models, symbols)
    assert plan.blocked and len(plan.orders) == 0 and not plan.can_rebalance
    with pytest.raises(Refused):
        live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_fix4_orders_changed_after_display_are_refused(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    shown = plan_for(mock, models, symbols)
    fresh = plan_for(mock, models, symbols)
    shown.orders.loc[0, "qty"] = shown.orders.loc[0, "qty"] * 1000
    with pytest.raises(Refused, match="changed after the page was drawn"):
        live.submit_checked(mock, shown, fresh, confirmed=True, typed="SUBMIT")
    assert mock.submissions == []


def test_fix5a_a_full_order_window_with_no_ws5_order_is_a_hold_not_a_first_rebalance(bars, models, symbols, dates):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    prior = filled_ws5_order(dates[-3])
    mock.orders[prior.client_order_id] = prior
    for i in range(60):                                         # manual orders push the ws5 order out of the window
        mock.orders[f"manual-{i}"] = OrderResult("AAPL", "buy", 1.0, "filled", 1.0, f"manual-{i}")
    last, incomplete = live.rebalance_history(mock, limit=50)
    assert last is None and incomplete
    plan = plan_for(mock, models, symbols, last_rebalance=last, history_incomplete=incomplete)
    assert not plan.due and len(plan.orders) == 0 and "override" in plan.cadence_note
    forced = plan_for(mock, models, symbols, last_rebalance=last, history_incomplete=incomplete, force=True)
    assert len(forced.orders) > 0
    last_ok, incomplete_ok = live.rebalance_history(mock, limit=500)      # window not full: found, trusted
    assert last_ok == pd.Timestamp(dates[-3]) and not incomplete_ok
    empty = FakeAlpaca(bars=bars, cash=EQUITY)
    assert live.rebalance_history(empty, limit=50) == (None, False), "an empty account IS a first rebalance"


def test_fix5b_malformed_and_missing_order_ids_are_skipped():
    t = pd.DataFrame({"client_order_id": ["qsml-ws5-20261340-AAPL-open-long", "qsml-ws5-20260230-AAPL-open-long",
                                          None, np.nan, "qsml-ws5-20261008-MSFT-open-long"],
                      "filled": [5, 5, 5, 5, 5]})
    assert live.last_rebalance_date(t) == pd.Timestamp("2026-10-08")
    assert live.last_rebalance_date(t.iloc[:4]) is None


def test_fix5c_any_filled_quantity_counts_whatever_the_status():
    t = pd.DataFrame({"client_order_id": ["qsml-ws5-20261008-A-open-long", "qsml-ws5-20261015-B-open-long"],
                      "filled": [5, 0], "status": ["canceled", "filled"]})
    assert live.last_rebalance_date(t) == pd.Timestamp("2026-10-08")      # cancelled after a fill counts; "filled" with 0 does not
    assert "WHATEVER" in live.__doc__.upper() or "whatever their status" in live.__doc__


def test_fix5d_a_last_rebalance_after_the_decision_date_is_ignored_with_a_warning(bars, models, symbols, dates):
    mock = MockBroker(bars=bars, cash=EQUITY)
    plan = plan_for(mock, models, symbols, last_rebalance=pd.Timestamp(dates[-1]) + pd.Timedelta(days=400))
    assert plan.last_rebalance is None and plan.due and len(plan.orders) > 0
    assert any("ignored" in n and "after today" in n for n in plan.notes)


def test_fix6_a_broker_error_does_not_carry_the_keys_in_its_traceback():
    import traceback
    pytest.importorskip("alpaca")
    broker = paper.AlpacaPaperBroker(FAKE_KEY, FAKE_SECRET)

    class Boom:
        def get_stock_bars(self, req):
            raise RuntimeError(f"401 for key {FAKE_KEY} secret {FAKE_SECRET}")
    broker._data = Boom()
    with pytest.raises(BrokerError) as info:
        broker.daily_ohlc_bars(["AAPL"], 10)
    text = "".join(traceback.format_exception(info.value))
    assert FAKE_KEY not in text and FAKE_SECRET not in text
    assert info.value.__cause__ is None


def test_fix6_app_shows_only_scrubbed_text_when_a_broker_raises_something_odd(model_dir, bars, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("APCA_API_SECRET_KEY", FAKE_SECRET)

    class Leaky(FakeAlpaca):
        def daily_ohlc_bars(self, symbols, lookback_days):
            raise RuntimeError(f"HTTP 401 key={FAKE_KEY} secret={FAKE_SECRET}")
    monkeypatch.setattr(live, "connect_alpaca", lambda k, s: Leaky(bars=bars, cash=EQUITY))
    at = _app(model_dir, "Alpaca paper").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Could not build today's plan" in e.value for e in at.error)
    text = _rendered(at)
    assert FAKE_KEY not in text and FAKE_SECRET not in text and "[key]" in text


def test_fix7_a_partial_last_day_proposes_no_orders(bars, models, symbols, dates):
    keep = set(symbols[:25])
    partial = bars[(bars["date"] < dates[-1]) | bars["symbol"].map(live.from_alpaca).isin(keep)]
    mock = FakeAlpaca(bars=partial, cash=EQUITY)
    plan = plan_for(mock, models, symbols)
    assert plan.blocked and "incomplete" in plan.blocked
    assert len(plan.orders) == 0 and not plan.can_rebalance
    with pytest.raises(Refused, match="no orders can be sent"):
        live.submit_checked(mock, plan, plan, confirmed=True, typed="SUBMIT")
    forced = plan_for(mock, models, symbols, force=True)
    assert len(forced.orders) == 0, "the override does not unblock a half-empty day"


def _bars_for(day, price=1.0, n=1):
    return pd.DataFrame({"date": [pd.Timestamp(day)] * n, "symbol": ["AAA"] * n, "open": price, "high": price,
                         "low": price, "close": price, "adj_close": price, "volume": 1.0})


def test_fix8_clock_edges_for_the_completed_days_filter():
    b = pd.concat([_bars_for("2026-10-14"), _bars_for("2026-10-15"), _bars_for("2026-10-16")])
    ny = lambda s: pd.Timestamp(s, tz="America/New_York")
    got = lambda clock: list(pd.to_datetime(live.completed_days_only(b, clock)["date"]).dt.strftime("%m-%d"))
    # bars dated after today are never kept, even after the close
    assert got({"is_open": False, "timestamp": ny("2026-10-15 17:00")}) == ["10-14", "10-15"]
    assert got({"is_open": False, "timestamp": ny("2026-10-15 09:00")}) == ["10-14"]
    assert got({"is_open": True, "timestamp": ny("2026-10-15 11:00")}) == ["10-14"]
    # a naive timestamp is read as New York time and does not raise
    assert got({"is_open": True, "timestamp": pd.Timestamp("2026-10-15 11:00")}) == ["10-14"]
    # no timestamp counts as open: today's bar (the machine's own date) is dropped, nothing from the future survives
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    c = pd.concat([_bars_for(today - pd.Timedelta(days=1)), _bars_for(today), _bars_for(today + pd.Timedelta(days=1))])
    kept = pd.to_datetime(live.completed_days_only(c, {"is_open": False})["date"])
    assert list(kept) == [today - pd.Timedelta(days=1)]


def test_fix9_note_names_stocks_without_a_price_on_the_last_day_accurately(bars, models, symbols, dates):
    no_msft_today = bars[~((bars["symbol"] == "MSFT") & (bars["date"] == dates[-1]))]
    plan = plan_for(MockBroker(bars=no_msft_today, cash=EQUITY), models, symbols)
    note = next(n for n in plan.notes if "not scored" in n)
    assert "MSFT" in note and "missing days" not in note and "no price on the last day" in note
    assert "MSFT" not in set(plan.table["symbol"])


def test_the_typed_word_is_compared_after_stripping_spaces_documented_behaviour(bars, models, symbols):
    mock = FakeAlpaca(bars=bars, cash=EQUITY)
    plan = plan_for(mock, models, symbols)
    assert live.submit_checked(mock, plan, plan, confirmed=True, typed="  SUBMIT ")
