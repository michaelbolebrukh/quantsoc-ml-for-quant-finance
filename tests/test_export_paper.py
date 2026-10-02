"""Export bundle and paper runner.

The question these tests answer: does the exported bundle reproduce the research pipeline's
forecasts for the same inputs, and does the runner propose correct orders while being unable to
submit without an explicit, confirmed, paper-only submit? Nothing here touches the network:
alpaca-py objects are replaced by fakes or by the MockBroker, and that limit is stated in the
tests that rely on it.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import data as D
from quantsoc_ml import export as E
from quantsoc_ml import features as F
from quantsoc_ml import models as M
from quantsoc_ml import paper as Pp
from quantsoc_ml.portfolio import StrategyConfig, weights_from_forecasts

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so the repo-root run_paper.py can be imported as a module
    sys.path.insert(0, str(ROOT))
SNAPSHOT = ROOT / "data" / "prices.csv.gz"
SYMBOLS = ["AAPL", "AMZN", "BA", "C", "CSCO", "DIS", "GE", "INTC", "JPM", "KO", "MSFT", "XOM"]
# Non-default knobs on purpose, so the knob -> feature-list path is exercised, not just the default.
CFG = StrategyConfig(momentum_lookback=10, extra_feature="vol_42", n_long=3, n_short=3, n_estimators=20, max_depth=3)


def _fit_and_export(tmp_path_factory, cfg: StrategyConfig, name: str):
    """A small forest fitted on a slice of the snapshot through the RESEARCH pipeline, then exported.

    The research pipeline is build_features, then rank_normalise when cfg.rank_features (design
    decision D7), then make_dataset. `feats` is the model's input table exactly as trained on.
    """
    long = D.load_prices(SNAPSHOT)
    long = long[long["symbol"].isin(SYMBOLS)]
    dates = sorted(long["date"].unique())[-700:]
    long = long[long["date"].isin(dates)]
    P, V = D.to_wide(long), D.to_wide(long, "volume")
    names = cfg.feature_names
    feats = F.build_features(P, V, names=names)
    if cfg.rank_features:
        feats = F.rank_normalise(feats)
    ds = F.make_dataset(feats, F.forward_return(P))
    model = M.fit_forest(*M.xy(ds), n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, n_jobs=1)
    out = tmp_path_factory.mktemp(name) / "bundle"
    E.export_bundle(model, cfg, out, used_reference=False, symbols=SYMBOLS, data_path=SNAPSHOT,
                    trained_on=ds.attrs["features"])
    return types.SimpleNamespace(P=P, V=V, feats=feats, model=model, bundle=out, names=names, cfg=cfg)


@pytest.fixture(scope="module")
def research(tmp_path_factory):
    """Rank features (the default, as the notebook trains), exported once for the whole module."""
    assert CFG.rank_features is True
    return _fit_and_export(tmp_path_factory, CFG, "export_rank")


@pytest.fixture(scope="module")
def research_raw(tmp_path_factory):
    """The same, with rank_features=False: raw feature values."""
    return _fit_and_export(tmp_path_factory, dataclasses.replace(CFG, rank_features=False), "export_raw")


def _import_strategy(bundle: Path):
    spec = importlib.util.spec_from_file_location("bundle_strategy_under_test", bundle / "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- forecasts reproduce the research pipeline

def _expected_last_date(r) -> pd.Series:
    complete = r.feats.dropna()
    complete.attrs["features"] = r.names
    return M.predict_panel(r.model, complete).loc[r.P.index.max()].dropna()


@pytest.mark.parametrize("which", ["research", "research_raw"])
def test_bundle_forecasts_equal_research_pipeline(which, request):
    r = request.getfixturevalue(which)
    strat = _import_strategy(r.bundle)
    model_b = joblib.load(r.bundle / "model.joblib")
    config_b = json.loads((r.bundle / "config.json").read_text())
    assert config_b["rank_features"] is r.cfg.rank_features
    expected = _expected_last_date(r)
    assert len(expected) >= CFG.n_long + CFG.n_short

    got = strat.predict_returns(r.P, r.V, model_b, config_b)
    assert list(got.dropna().index) == list(expected.index)
    np.testing.assert_allclose(got.dropna().to_numpy(), expected.to_numpy(), rtol=0, atol=1e-10)

    # The runner passes a 300-day window, not the full history: the last date must not move.
    got_window = strat.predict_returns(r.P.iloc[-300:], r.V.iloc[-300:], model_b, config_b)
    np.testing.assert_allclose(got_window.dropna().to_numpy(), expected.to_numpy(), rtol=0, atol=1e-10)

    # The runner's own latest-features helper builds the same model inputs as training did.
    rows = Pp.build_latest_features(r.P, r.V, r.names, rank_features=r.cfg.rank_features)
    pd.testing.assert_frame_equal(rows, r.feats.xs(r.P.index.max(), level="date")[r.names])


def test_ranked_bundle_is_not_fed_raw_features(research):
    # The bug this guards against: a strategy.py that skips rank_normalise feeds raw values into a
    # model trained on ranks. Those forecasts must differ from the bundle's, or the test above
    # could not tell the two apart.
    raw = F.build_features(research.P, research.V, names=research.names)
    raw_rows = raw.xs(research.P.index.max(), level="date")[research.names].dropna()
    wrong = pd.Series(research.model.predict(raw_rows.to_numpy(dtype=float)), index=raw_rows.index)
    expected = _expected_last_date(research)
    assert np.max(np.abs(wrong.reindex(expected.index).to_numpy() - expected.to_numpy())) > 1e-6


def test_old_config_without_rank_features_defaults_to_ranked(research):
    strat = _import_strategy(research.bundle)
    old = json.loads((research.bundle / "config.json").read_text())
    old.pop("rank_features")
    got = strat.predict_returns(research.P, research.V, research.model, old)
    np.testing.assert_allclose(got.dropna().to_numpy(), _expected_last_date(research).to_numpy(), rtol=0, atol=1e-10)


def test_bundle_weights_equal_portfolio_rule(research):
    strat = _import_strategy(research.bundle)
    pred = strat.predict_returns(research.P, research.V, research.model, CFG)
    w = strat.target_weights(pred, json.loads((research.bundle / "config.json").read_text()))
    pd.testing.assert_series_equal(w, weights_from_forecasts(pred, n_long=3, n_short=3))
    assert (w > 0).sum() == 3 and (w < 0).sum() == 3
    assert w[w > 0].sum() == pytest.approx(1.0) and w[w < 0].sum() == pytest.approx(-1.0)


def test_bundle_contents_and_manifest(research):
    b = research.bundle
    for name in E.BUNDLE_FILES + ["quantsoc_ml/features.py", "quantsoc_ml/paper.py", E.RECENT_DATA]:
        assert (b / name).is_file(), name
    assert not (b / ".env").exists()
    m = json.loads((b / "manifest.json").read_text())
    import sklearn
    assert m["sklearn_version"] == sklearn.__version__
    assert m["used_reference_knobs"] is False
    assert m["rank_features"] is True
    assert m["symbols"] == sorted(SYMBOLS)
    assert m["feature_names"] == research.names == json.loads((b / "feature_names.json").read_text())
    assert pd.Timestamp(m["created_at"]).tzinfo is not None
    assert f"scikit-learn=={sklearn.__version__}" in (b / "requirements.txt").read_text()
    assert StrategyConfig.from_dict(json.loads((b / "config.json").read_text())) == CFG
    loaded = E.load_bundle(b)
    assert loaded.feature_names == research.names and loaded.symbols == sorted(SYMBOLS)


def test_export_refuses_to_overwrite_a_folder_that_is_not_a_bundle(tmp_path, research):
    (tmp_path / "notes.txt").write_text("my own file")
    with pytest.raises(ValueError, match="not an earlier strategy bundle"):
        E.export_bundle(research.model, CFG, tmp_path, symbols=SYMBOLS, data_path=False)
    assert (tmp_path / "notes.txt").read_text() == "my own file"


def test_export_rejects_a_model_with_the_wrong_number_of_features(tmp_path, research):
    seven = StrategyConfig(momentum_lookback=5)              # ret_5 twice -> 7 features, the model has 8
    with pytest.raises(ValueError, match="fitted on 8 features"):
        E.export_bundle(research.model, seven, tmp_path / "b", symbols=SYMBOLS, data_path=False)
    # Same count, different names: only caught when the caller says what the model was trained on.
    with pytest.raises(ValueError, match="refit"):
        E.export_bundle(research.model, StrategyConfig(), tmp_path / "c", symbols=SYMBOLS, data_path=False,
                        trained_on=research.names)


# --------------------------------------------------------------------------- zip and credential hygiene

def test_zip_excludes_env_and_check_zip_passes(tmp_path, research):
    (research.bundle / ".env").write_text("APCA_API_KEY_ID=PKABCDEFGHIJKLMNOPQR\n")
    try:
        z = E.zip_bundle(research.bundle, tmp_path / "bundle.zip")
    finally:
        (research.bundle / ".env").unlink()
    with zipfile.ZipFile(z) as zf:
        inner = ["/".join(n.split("/")[1:]) for n in zf.namelist()]
    assert ".env" not in inner and ".env.example" in inner
    assert not any("__pycache__" in n for n in inner)
    report = E.check_zip(z)
    assert report["ok"], report
    assert report["missing"] == [] and report["data_included"]


def test_check_zip_flags_env_files_and_key_shaped_text(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("b/.env", "x")
        zf.writestr("b/strategy.py", "KEY = 'PKABCDEFGHIJKLMNOPQRST'\n")
    report = E.check_zip(z)
    assert not report["ok"]
    assert report["forbidden_present"] == [".env"]
    assert report["secret_hits"] and "PKABCDEFGHIJ" not in " ".join(report["secret_hits"])   # truncated


def test_export_scans_for_the_values_of_credentials_in_the_environment(tmp_path, research, monkeypatch):
    secret = "s3cretValueThatMustNeverLeak0123456789"
    monkeypatch.setenv("APCA_API_SECRET_KEY", secret)
    out = E.export_bundle(research.model, CFG, tmp_path / "b", symbols=SYMBOLS, data_path=False)
    assert all(secret.encode() not in p.read_bytes() for p in out.rglob("*") if p.is_file())
    (out / "notes.md").write_text(f"oops {secret}\n")
    assert any("APCA_* environment variable" in p for p in E._scan_dir(out))
    report = E.check_zip(E.zip_bundle(out, tmp_path / "b.zip"))
    assert not report["ok"] and secret not in json.dumps(report)


# --------------------------------------------------------------------------- propose_orders

def test_propose_orders_fresh_account_whole_shares_towards_zero():
    w = pd.Series({"AAA": 0.5, "BBB": 0.5, "CCC": -0.5, "DDD": -0.5, "EEE": 0.0})
    px = {"AAA": 30.0, "BBB": 7.0, "CCC": 45.0, "DDD": 100.0, "EEE": 10.0}
    o = Pp.propose_orders(w, 1000.0, {}, px)
    assert list(o.columns) == Pp.ORDER_COLUMNS
    t = o.set_index("symbol")
    # 500/30 = 16.67 -> 16; 500/7 = 71.4 -> 71; -500/45 = -11.1 -> -11; -500/100 = -5
    assert t.loc["AAA", "target_qty"] == 16 and t.loc["BBB", "target_qty"] == 71
    assert t.loc["CCC", "target_qty"] == -11 and t.loc["DDD", "target_qty"] == -5
    assert t.loc["CCC", "side"] == "sell" and t.loc["CCC", "qty"] == 11 and t.loc["CCC", "action"] == "open short"
    assert t.loc["AAA", "notional"] == pytest.approx(480.0)
    assert "EEE" not in t.index


def test_propose_orders_existing_positions_shorts_and_crossings():
    w = pd.Series({"AAA": 0.5, "BBB": -0.5, "CCC": 0.0, "DDD": -0.5, "EEE": 0.5})
    px = {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0, "DDD": 10.0, "EEE": 10.0}
    held = {"AAA": -20.0, "BBB": 30.0, "CCC": -7.0, "DDD": -80.0, "EEE": 50.0, "ZZZ": 3.0}
    o = Pp.propose_orders(w, 1000.0, held, px)
    legs = list(zip(o["symbol"], o["side"], o["qty"], o["action"]))
    assert legs == [
        ("AAA", "buy", 20.0, "close short"),     # phase 0: everything that frees buying power, by symbol
        ("BBB", "sell", 30.0, "close long"),
        ("CCC", "buy", 7.0, "close short"),
        ("DDD", "buy", 30.0, "reduce short"),    # -80 -> -50
        ("AAA", "buy", 50.0, "open long"),       # phase 1: crossing short -> long, second leg
        ("BBB", "sell", 50.0, "open short"),     # crossing long -> short, second leg
    ]
    assert "ZZZ" not in set(o["symbol"])         # not in the strategy: never touched
    assert "EEE" not in set(o["symbol"])         # already at target
    assert (o.loc[o["symbol"] == "DDD", ["current_qty", "target_qty"]].to_numpy() == [[-80, -50]]).all()


def test_propose_orders_is_deterministic_and_respects_min_notional():
    w = pd.Series({"BBB": -0.5, "AAA": 0.5, "CCC": 0.0})
    px = {"AAA": 10.0, "BBB": 10.0, "CCC": 0.5}
    a = Pp.propose_orders(w, 1000.0, {"CCC": 1.0}, px, min_notional=1.0)
    b = Pp.propose_orders(w.iloc[::-1], 1000.0, {"CCC": 1.0}, dict(reversed(list(px.items()))), min_notional=1.0)
    pd.testing.assert_frame_equal(a, b)
    assert "CCC" not in set(a["symbol"])         # a 50-cent leg is below min_notional
    with pytest.raises(ValueError, match="no price"):
        Pp.propose_orders(pd.Series({"QQQ": 0.5}), 1000.0, {}, {})


# --------------------------------------------------------------------------- the run: preview cannot submit

class ExplodingBroker(Pp.MockBroker):
    def submit(self, *a, **k):
        raise AssertionError("preview called submit")


@pytest.fixture(scope="module")
def recent_bars(research):
    return pd.read_csv(research.bundle / E.RECENT_DATA, parse_dates=["date"])


def test_preview_never_submits(research, recent_bars):
    b = ExplodingBroker(bars=recent_bars, script="demo")
    lines = []
    rep = Pp.run("preview", b, research.bundle, confirm=lambda o: True, out=lines.append)
    assert lines[0].startswith("=") and "NO orders submitted" in lines[0]      # the banner comes first
    assert len(rep.plan.orders) == CFG.n_long + CFG.n_short and rep.submitted == []
    assert any("NONE submitted" in l for l in lines)
    assert not hasattr(Pp.ReadOnlyBroker(b), "submit")
    with pytest.raises(AttributeError):
        Pp.ReadOnlyBroker(b).submit = b.submit


def test_only_the_submit_path_can_call_submit():
    # Reading source on purpose: the claim is structural ("no code path in preview calls submit"),
    # which a behavioural test can only sample. test_preview_never_submits covers the behaviour.
    tree = ast.parse(Path(Pp.__file__).read_text())
    callers, submit_path_callers = set(), set()
    for top in tree.body:
        for node in ast.walk(top):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "submit":
                callers.add(top.name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_submit_path":
                submit_path_callers.add(top.name)
    assert callers == {"_submit_path"}
    assert submit_path_callers == {"run"}
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    guarded = [n for n in ast.walk(run) if isinstance(n, ast.If) and ast.unparse(n.test) == "mode == 'submit'"
               and "_submit_path" in ast.unparse(n)]
    assert guarded, "_submit_path must sit inside an `if mode == 'submit'` block"


def test_submit_refusals_send_nothing(research, recent_bars):
    b = Pp.MockBroker(bars=recent_bars)
    with pytest.raises(Pp.Refused, match="confirmation"):
        Pp.run("submit", b, research.bundle, confirm=None, out=lambda s: None)
    with pytest.raises(Pp.Refused, match="not confirmed"):
        Pp.run("submit", b, research.bundle, confirm=lambda o: "yes", out=lambda s: None)   # only True counts
    live = Pp.MockBroker(bars=recent_bars)
    live.paper = False
    with pytest.raises(Pp.Refused, match="paper"):
        Pp.run("submit", live, research.bundle, confirm=lambda o: True, out=lambda s: None)
    busy = Pp.MockBroker(bars=recent_bars, partial_fill={})
    busy.orders["x"] = Pp.OrderResult("AAPL", "buy", 10, "accepted", 0, "x")
    with pytest.raises(Pp.Refused, match="open orders"):
        Pp.run("submit", busy, research.bundle, confirm=lambda o: True, out=lambda s: None)
    with pytest.raises(Pp.Refused):
        Pp.run("live", b, research.bundle, confirm=lambda o: True, out=lambda s: None)
    assert b.submissions == [] and live.submissions == [] and busy.submissions == []


def test_mock_submit_reports_one_reject_and_one_partial_fill(research, recent_bars):
    b = Pp.MockBroker(bars=recent_bars, script="demo")
    lines = []
    rep = Pp.run("submit", b, research.bundle, confirm=lambda o: True, out=lines.append)
    assert "MOCK SUBMIT" in lines[0]
    statuses = [r.status for r in rep.submitted]
    assert len(b.submissions) == len(rep.plan.orders) == len(statuses)
    assert statuses.count("rejected") == 1 and statuses.count("partially_filled") == 1
    rejected = next(r for r in rep.submitted if r.status == "rejected")
    assert rejected.side == "sell" and "borrow" in rejected.message
    assert b.positions().get(rejected.symbol, 0.0) == 0.0        # the rejected short changed nothing
    assert len({s["client_order_id"] for s in b.submissions}) == len(b.submissions)
    filled = [r for r in rep.submitted if r.status == "filled"]
    for r in filled:
        assert b.positions()[r.symbol] == (r.qty if r.side == "buy" else -r.qty)
    assert any(l.startswith("SUBMITTED:") for l in lines)
    with pytest.raises(Pp.BrokerError, match="already used"):  # same id twice is refused by the broker
        b.submit("AAPL", "buy", 1, b.submissions[0]["client_order_id"])


# --------------------------------------------------------------------------- credentials, banner, data helpers

def test_get_credentials_order_and_source(tmp_path, monkeypatch):
    monkeypatch.setattr(Pp, "_is_colab", lambda: False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    env = tmp_path / ".env"
    assert Pp.get_credentials([env]) == (None, None, "not found")
    env.write_text("APCA_API_KEY_ID=fromfile\nAPCA_API_SECRET_KEY=fromfilesecret\n")
    k, s, src = Pp.get_credentials([env])
    assert (k, s) == ("fromfile", "fromfilesecret") and src.endswith(".env file")
    monkeypatch.setenv("APCA_API_KEY_ID", "fromenv")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fromenvsecret")
    assert Pp.get_credentials([env]) == ("fromenv", "fromenvsecret", "environment variables")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: {"APCA_API_KEY_ID": "c1", "APCA_API_SECRET_KEY": "c2"}[name])
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(Pp, "_is_colab", lambda: True)
    assert Pp.get_credentials([env]) == ("c1", "c2", "Colab secrets")


def test_mode_banner_is_truthful():
    assert "PREVIEW: real paper-account data, NO orders submitted" in Pp.mode_banner("preview")
    assert "PAPER SUBMIT: orders go to the Alpaca PAPER endpoint only" in Pp.mode_banner("submit")
    assert "mock broker" in Pp.mode_banner("preview", offline=True) and "NO orders" in Pp.mode_banner("preview", offline=True)
    assert "nothing leaves this computer" in Pp.mode_banner("submit", offline=True)
    with pytest.raises(ValueError):
        Pp.mode_banner("live")


def test_fetch_daily_bars_with_a_fake_alpaca(monkeypatch):
    # alpaca-py is not installed here, so these fakes stand in for its request classes. This checks
    # OUR conversion (UTC timestamps -> New York dates, the adjustment we ask for), not alpaca-py.
    seen = {}
    enums = types.ModuleType("alpaca.data.enums")
    enums.Adjustment = types.SimpleNamespace(ALL="all")
    enums.DataFeed = lambda v: f"feed:{v}"
    reqs = types.ModuleType("alpaca.data.requests")
    reqs.StockBarsRequest = lambda **kw: seen.update(kw) or kw
    tf = types.ModuleType("alpaca.data.timeframe")
    tf.TimeFrame = types.SimpleNamespace(Day="1Day")
    for name, mod in {"alpaca": types.ModuleType("alpaca"), "alpaca.data": types.ModuleType("alpaca.data"),
                      "alpaca.data.enums": enums, "alpaca.data.requests": reqs, "alpaca.data.timeframe": tf}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    ts = pd.to_datetime(["2026-09-28 04:00", "2026-09-29 04:00", "2026-09-30 04:00"], utc=True)
    idx = pd.MultiIndex.from_product([["AAPL", "KO"], ts], names=["symbol", "timestamp"])
    df = pd.DataFrame({"close": np.arange(6, dtype=float) + 10, "volume": 1000.0}, index=idx)
    client = types.SimpleNamespace(get_stock_bars=lambda req: types.SimpleNamespace(df=df))
    bars = Pp.fetch_daily_bars(["AAPL", "KO"], 2, client)
    assert seen["adjustment"] == "all" and seen["timeframe"] == "1Day" and seen["feed"] == "feed:iex"
    assert sorted(bars["date"].dt.strftime("%Y-%m-%d").unique()) == ["2026-09-29", "2026-09-30"]
    P, V = Pp.build_panels(bars)
    assert list(P.columns) == ["AAPL", "KO"] and P.loc["2026-09-30", "KO"] == 15.0


def test_drop_unfinished_day_only_while_open():
    bars = pd.DataFrame({"date": pd.to_datetime(["2026-09-30", "2026-10-01"]), "symbol": "A",
                         "close": 1.0, "adj_close": 1.0, "volume": 1.0})
    now = pd.Timestamp("2026-10-01 15:00", tz="America/New_York")
    assert len(Pp.drop_unfinished_day(bars, {"is_open": True, "timestamp": now})) == 1
    assert len(Pp.drop_unfinished_day(bars, {"is_open": False, "timestamp": now})) == 2


def test_rows_needed():
    assert Pp.rows_needed(StrategyConfig().feature_names) == 252
    assert Pp.rows_needed(["ret_300", "vol_21"]) == 301


# --------------------------------------------------------------------------- the CLI

def _cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "run_paper.py"), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=120)


def test_cli_dry_preview_runs_standalone_from_the_bundle(research, tmp_path):
    # From a different folder, with the bundle's own run_paper.py and data: no notebook state at all.
    r = subprocess.run([sys.executable, str(research.bundle / "run_paper.py"), "--dry-data", E.RECENT_DATA],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert lines[0].startswith("=") and "PREVIEW (OFFLINE DEMO)" in lines[1]
    assert "NONE submitted" in r.stdout and "proposed orders (6)" in r.stdout


def test_cli_submit_needs_the_typed_word(research, capsys):
    import run_paper  # noqa: E402  (the repo-root script, imported as a module)
    args = ["--submit", "--dry-data", "data/prices.csv.gz", "--bundle", str(research.bundle)]
    assert run_paper.main(args, input_fn=lambda prompt: "submit") == 2          # lower case is not enough
    assert "REFUSED" in capsys.readouterr().out
    def eof(prompt):
        raise EOFError
    assert run_paper.main(args, input_fn=eof) == 2
    capsys.readouterr()
    assert run_paper.main(args, input_fn=lambda prompt: "SUBMIT") == 0
    out = capsys.readouterr().out
    assert "MOCK SUBMIT" in out and "SUBMITTED:" in out and "1 rejected" in out and "1 partially_filled" in out


def test_cli_without_credentials_exits_1_and_names_the_variables(tmp_path, research, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    r = subprocess.run([sys.executable, str(research.bundle / "run_paper.py"), "--preview"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "APCA_API_KEY_ID" in r.stdout and "APCA_API_SECRET_KEY" in r.stdout


def test_cli_live_path_never_prints_the_keys(research, recent_bars, monkeypatch, capsys):
    import run_paper  # noqa: E402
    key, secret = "PKFAKEKEY0123456789X", "fakeSecretValue0123456789abcdefghijklmnop"
    monkeypatch.setenv("APCA_API_KEY_ID", key)
    monkeypatch.setenv("APCA_API_SECRET_KEY", secret)
    got = {}

    class FakeAlpaca(Pp.MockBroker):          # stands in for AlpacaPaperBroker: alpaca-py is not installed
        offline = False

        def __init__(self, k, s, feed="iex"):
            got.update(k=k, s=s, feed=feed)
            super().__init__(bars=recent_bars)

    monkeypatch.setattr(Pp, "AlpacaPaperBroker", FakeAlpaca)
    assert run_paper.main(["--preview", "--bundle", str(research.bundle)]) == 0
    out = capsys.readouterr().out
    assert got == {"k": key, "s": secret, "feed": "iex"}
    assert "PREVIEW: real paper-account data, NO orders submitted" in out
    assert "keys read from environment variables (not shown)" in out
    assert key not in out and secret not in out and key[:8] not in out


def test_cli_reports_missing_alpaca_py(research, monkeypatch, capsys):
    import run_paper  # noqa: E402
    monkeypatch.setenv("APCA_API_KEY_ID", "PKFAKEKEY0123456789X")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fakeSecretValue0123456789abcdefghijklmnop")
    monkeypatch.setitem(sys.modules, "alpaca", None)                 # import alpaca -> ImportError
    monkeypatch.setitem(sys.modules, "alpaca.data", None)
    monkeypatch.setitem(sys.modules, "alpaca.data.historical", None)
    assert run_paper.main(["--preview", "--bundle", str(research.bundle)]) == 3
    out = capsys.readouterr().out
    assert "pip install -r requirements.txt" in out and "fakeSecret" not in out


# --------------------------------------------------------------------------- fixes from the export/paper falsification

def test_propose_orders_min_notional_never_splits_a_crossing_move():
    # Long 1 X at $0.50 with a target of -1000: the 50-cent close leg must NOT be dropped, or the
    # remaining leg is one crossing order (sell 1000 from +1), which the split exists to prevent.
    o = Pp.propose_orders(pd.Series({"X": -0.5}), 1000.0, {"X": 1.0}, {"X": 0.5}, min_notional=1.0)
    assert list(o["action"]) == ["close long", "open short"]
    assert list(o["qty"]) == [1.0, 1000.0]
    # A whole move below min_notional is still skipped.
    o = Pp.propose_orders(pd.Series({"X": 0.0}), 1000.0, {"X": 1.0}, {"X": 0.5}, min_notional=1.0)
    assert len(o) == 0


def test_a_held_symbol_without_a_forecast_is_left_alone(research, recent_bars):
    sym = SYMBOLS[0]
    bars = recent_bars[~((recent_bars["symbol"] == sym) & (recent_bars["date"] == recent_bars["date"].max()))]
    b = Pp.MockBroker(bars=bars, script="demo")
    b.holdings[sym] = 50.0
    rep = Pp.run("preview", b, research.bundle, out=lambda s: None)
    assert sym not in set(rep.plan.orders["symbol"])
    assert any("no forecast today" in n and sym in n for n in rep.plan.notes)


def test_symbols_differing_from_the_training_universe_are_refused(research, recent_bars):
    b = Pp.MockBroker(bars=recent_bars, script="demo")
    with pytest.raises(ValueError, match="universe"):
        Pp.run("preview", b, research.bundle, symbols=SYMBOLS[:8], out=lambda s: None)
    rep = Pp.run("preview", b, research.bundle, symbols=list(reversed(SYMBOLS)), out=lambda s: None)   # same set
    assert len(rep.plan.orders) == CFG.n_long + CFG.n_short
    rep = Pp.run("preview", b, research.bundle, symbols=SYMBOLS[:8], allow_universe_change=True, out=lambda s: None)
    assert len(rep.plan.orders) == CFG.n_long + CFG.n_short


def test_cli_refuses_a_different_universe_without_the_flag(research):
    r = _cli("--preview", "--dry-data", str(SNAPSHOT), "--bundle", str(research.bundle), "--symbols", ",".join(SYMBOLS[:8]))
    assert r.returncode == 3 and "universe" in r.stdout


def test_export_refuses_a_key_shaped_string_inside_the_model_file(tmp_path, research):
    model = joblib.load(research.bundle / "model.joblib")
    model.note = "PK" + "ABCDEFGHIJKLMNOPQR"
    with pytest.raises(ValueError, match="credential"):
        E.export_bundle(model, CFG, tmp_path / "b", symbols=SYMBOLS, data_path=SNAPSHOT)
    assert not (tmp_path / "b").exists()


def test_check_zip_flags_a_key_inside_a_binary_file(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("b/model.joblib", b"\x80\x04junk " + ("PK" + "ABCDEFGHIJKLMNOPQR").encode() + b"\x00")
        zf.writestr("b/notes.bin", (" PK" + "QRSTUVWXYZ0123456789 ").encode("utf-16-le"))   # a key id stored as UTF-16
    rep = E.check_zip(z)
    assert any("model.joblib" in h for h in rep["secret_hits"])
    assert any("utf-16" in h for h in rep["secret_hits"])


def test_export_scans_for_utf16_copies_of_credential_values(tmp_path, research, monkeypatch):
    monkeypatch.setenv("APCA_API_SECRET_KEY", "zqxw" * 10)
    model = joblib.load(research.bundle / "model.joblib")
    model.note = ("zqxw" * 10).encode("utf-16-le")
    with pytest.raises(ValueError, match="credential"):
        E.export_bundle(model, CFG, tmp_path / "b", symbols=SYMBOLS, data_path=SNAPSHOT)


def test_export_refuses_features_longer_than_the_shipped_history(tmp_path, research):
    cfg = dataclasses.replace(CFG, extra_feature="vol_320")
    with pytest.raises(ValueError, match="recent_days"):
        E.export_bundle(research.model, cfg, tmp_path / "b", symbols=SYMBOLS, data_path=SNAPSHOT)


def test_export_needs_a_spare_symbol_beyond_longs_plus_shorts(tmp_path, research):
    cfg = dataclasses.replace(CFG, n_long=6, n_short=6)
    with pytest.raises(ValueError, match="room to spare"):
        E.export_bundle(research.model, cfg, tmp_path / "b", symbols=SYMBOLS, data_path=SNAPSHOT)


def test_reexport_never_deletes_a_participants_env_file(tmp_path, research):
    out = tmp_path / "b"
    E.export_bundle(research.model, CFG, out, symbols=SYMBOLS, data_path=SNAPSHOT)
    (out / ".env").write_text("APCA_API_KEY_ID=x\n")
    with pytest.raises(ValueError, match=r"\.env"):
        E.export_bundle(research.model, CFG, out, symbols=SYMBOLS, data_path=SNAPSHOT)
    assert (out / ".env").read_text() == "APCA_API_KEY_ID=x\n"


def test_bundle_copy_of_the_package_reproduces_the_research_forecasts(research, tmp_path):
    """Runs strategy.py in a separate interpreter with ONLY the bundle on sys.path (a participant's machine)."""
    last = _expected_last_date(research)
    script = f"""
import sys, json; sys.path[:] = [p for p in sys.path if 'quantsoc-ml-for-quant-finance' not in p]
sys.path.insert(0, {str(research.bundle)!r})
import quantsoc_ml, strategy, pandas as pd
assert quantsoc_ml.__file__.startswith({str(research.bundle)!r}), quantsoc_ml.__file__
from quantsoc_ml import data as D
long = D.load_prices({str(SNAPSHOT)!r}); long = long[long['symbol'].isin({SYMBOLS!r})]
dates = sorted(long['date'].unique())[-700:]; long = long[long['date'].isin(dates)]
P, V = D.to_wide(long), D.to_wide(long, 'volume')
import joblib; model = joblib.load({str(research.bundle / 'model.joblib')!r})
cfg = json.load(open({str(research.bundle / 'config.json')!r}))
pred = strategy.predict_returns(P, V, model, cfg)
print(json.dumps(pred.to_dict()))
"""
    r = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    got = pd.Series(json.loads(r.stdout.strip().splitlines()[-1])).sort_index()
    pd.testing.assert_series_equal(got, last.sort_index(), check_names=False, atol=1e-10)


def test_env_file_is_read_without_python_dotenv(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False); monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# keys\nexport APCA_API_KEY_ID='fromfile'\nAPCA_API_SECRET_KEY=\"fromfilesecret\"\n")
    k, s, src = Pp.get_credentials([env])
    assert (k, s) == ("fromfile", "fromfilesecret") and src.endswith(".env file")
