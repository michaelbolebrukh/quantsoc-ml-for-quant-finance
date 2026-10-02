"""The student notebook runs from a clean kernel, and every task checker tells right from wrong.

1. test_notebook_executes: runs notebooks/workshop2_student.ipynb once (a module fixture) with nbclient
   from a clean kernel, with QSW_AUTO_REFERENCE=1 (every task that is not changed falls back to the
   labelled reference), the kernel's working directory set to notebooks/ so the setup cell finds the
   workshop folder, and QSW_WORK_DIR pointing at a temporary folder so the export never writes into the
   repository. It asserts no cell errors and a total wall time under 5 minutes, and reports the slowest
   cells. test_slow_cells_are_labelled: every code cell that took more than 10 s in that run has a time
   label (a number followed by "s" or "second") in the markdown cell directly above it.
2. The stored notebook: outputs cleared, the nine section headings exactly as the contract names them,
   the instructor switch off. No em-dashes or en-dashes in the notebook, README.md, docs/ or
   data/DATA_CARD.md.
3. Every Reference citation in notebook_support.CITATIONS appears word for word in docs/CONTRACT.md.
4. test_checker_*: each of the five checkers rejects a wrong value and accepts a right one; the
   tracker keeps an earlier accepted answer when a re-check fails; the notebook helpers refuse bad
   forest sizes before fitting, show "n/a" for an undefined IC, and count backtests.

Skipped only when nbclient or ipykernel is not installed (an environment precondition).
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from datetime import date

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "workshop2_student.ipynb"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEADINGS = ["0. Setup", "1. The data", "2. Features, the target and the no-peeking rule",
            "3. A baseline and a fair test", "4. A real ML model", "5. Overfitting, live",
            "6. From forecasts to a portfolio", "7. Your strategy's backtest", "8. Export and paper trading"]
TIME_LIMIT_S = 300
SLOW_CELL_S = 10.0
TIME_LABEL = re.compile(r"\d+(?:\.\d+)?\s*(?:s\b|seconds?\b)")
PROSE_FILES = [ROOT / "README.md", ROOT / "data" / "DATA_CARD.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _cell_seconds(cell) -> float:
    t = cell.metadata.get("execution", {})
    try:
        a = datetime.fromisoformat(t["iopub.execute_input"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(t["shell.execute_reply"].replace("Z", "+00:00"))
    except KeyError:
        return 0.0
    return (b - a).total_seconds()


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    """The notebook, run once from a clean kernel: (executed notebook, wall seconds, work folder)."""
    nbformat = pytest.importorskip("nbformat", reason="nbformat is not installed")
    nbclient = pytest.importorskip("nbclient", reason="nbclient is not installed")
    pytest.importorskip("ipykernel", reason="ipykernel is not installed")
    work = tmp_path_factory.mktemp("nbrun") / "work"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QSW_AUTO_REFERENCE", "1")
        mp.setenv("QSW_WORK_DIR", str(work))
        mp.setenv("MPLBACKEND", "Agg")
        nb = nbformat.read(NOTEBOOK, as_version=4)
        client = nbclient.NotebookClient(nb, timeout=TIME_LIMIT_S, kernel_name="python3",
                                         resources={"metadata": {"path": str(NOTEBOOK.parent)}})
        t0 = time.perf_counter()
        client.execute()                               # raises CellExecutionError on the first failing cell
        wall = time.perf_counter() - t0
    return nb, wall, work


def test_notebook_executes(executed):
    nb, wall, work = executed
    code = [c for c in nb.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
    slow = sorted(((_cell_seconds(c), c.source.splitlines()[0][:60]) for c in code), reverse=True)[:5]
    print(f"\nnotebook: {len(code)} code cells in {wall:.1f} s; slowest: "
          + "; ".join(f"{s:.1f} s {src!r}" for s, src in slow))
    assert wall < TIME_LIMIT_S, f"the notebook took {wall:.0f} s (limit {TIME_LIMIT_S} s)"
    assert (work / "exports" / "my_strategy.zip").is_file()
    text = "".join(o.get("text", "") for c in code for o in c.get("outputs", []))
    assert "[runner exit code 0]" in text
    assert "you have run 4 backtests in this session" in "".join(
        str(o.get("data", {}).get("text/html", "")) + str(o.get("data", {}).get("text/plain", ""))
        for c in code for o in c.get("outputs", [])), "the checklist must count the backtests actually run"


def test_slow_cells_are_labelled(executed):
    nb = executed[0]
    unlabelled = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or _cell_seconds(cell) <= SLOW_CELL_S:
            continue
        above = nb.cells[i - 1] if i else None
        if above is None or above.cell_type != "markdown" or not TIME_LABEL.search(above.source):
            unlabelled.append((i, round(_cell_seconds(cell), 1), cell.source.splitlines()[0][:60]))
    assert not unlabelled, f"cells over {SLOW_CELL_S:g} s with no time label in the markdown above: {unlabelled}"


def test_stored_notebook_shape():
    nbformat = pytest.importorskip("nbformat", reason="nbformat is not installed")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in code), "outputs must be cleared"
    found = [c.source.splitlines()[0][3:] for c in nb.cells
             if c.cell_type == "markdown" and c.source.startswith("## ")]
    assert found == HEADINGS
    assert any("SAVE_DECK_FIGURES = False" in c.source for c in code)
    assert not any(re.search(r"SAVE_DECK_FIGURES\s*=\s*True", c.source) for c in code), "instructor switch must be off"
    text = "\n".join(c.source for c in nb.cells)
    assert "\u2014" not in text and "\u2013" not in text, "no em-dashes or en-dashes"
    assert 'OWNER, REPO, REVISION = "michaelbolebrukh", "quantsoc-ml-for-quant-finance", "v1.0.0"' in code[0].source


@pytest.mark.parametrize("path", PROSE_FILES, ids=[str(p.relative_to(ROOT)) for p in PROSE_FILES])
def test_prose_has_no_dashes(path):
    text = path.read_text(encoding="utf-8")
    bad = [n for n, line in enumerate(text.splitlines(), 1) if "\u2014" in line or "\u2013" in line]
    assert not bad, f"{path.name}: em-dash or en-dash on lines {bad}"


def test_citations_match_contract():
    from quantsoc_ml.notebook_support import CITATIONS
    contract = (ROOT / "docs" / "CONTRACT.md").read_text()
    for key, citation in CITATIONS.items():
        assert citation in contract, f"{key} differs from docs/CONTRACT.md"


# --------------------------------------------------------------------------- checkers

@pytest.fixture(scope="module")
def ctx():
    from quantsoc_ml import data as D, features as F, splits as SP
    from quantsoc_ml.models import fit_forest, predict_panel, xy
    from quantsoc_ml.notebook_support import quarter_sample
    long = D.load_prices()
    prices, volume = D.to_wide(long), D.to_wide(long, "volume")
    returns = D.daily_returns(prices)
    ds = F.make_dataset(F.rank_normalise(F.build_features(prices, volume)), F.forward_return(prices))
    train, test = SP.time_split(ds, "2014-12-31", "2015-01-12")
    Xq, yq = xy(quarter_sample(train, frac=0.05))
    small = fit_forest(Xq, yq, n_estimators=20, max_depth=3)
    pred_today = predict_panel(small, test).iloc[-1]
    return dict(prices=prices, volume=volume, returns=returns, ds=ds, X_quarter=Xq, y_quarter=yq,
                pred_today=pred_today)


def _tracker(ctx):
    from quantsoc_ml.exercises import ExerciseTracker
    ex = ExerciseTracker(auto_reference=False)
    ex.set_context(1, prices=ctx["prices"])
    ex.set_context(2, prices=ctx["prices"], returns=ctx["returns"], volume=ctx["volume"])
    ex.set_context(3, ds=ctx["ds"])
    ex.set_context(4, X_quarter=ctx["X_quarter"], y_quarter=ctx["y_quarter"])
    ex.set_context(5, pred_today=ctx["pred_today"])
    return ex


def _args(task, case, c):
    """(value, *knobs) for a task, built exactly as the task cell builds them."""
    from quantsoc_ml.features import past_return, rolling_vol
    from quantsoc_ml.models import fit_forest
    from quantsoc_ml.portfolio import weights_from_forecasts
    from quantsoc_ml.splits import time_split
    if task == 1:
        n, value_n = case
        return (past_return(c["prices"], value_n), n)
    if task == 2:
        w, value_w = case
        return (rolling_vol(c["returns"], value_w), w)
    if task == 3:
        te, ts = case
        try:
            pair = time_split(c["ds"], te, ts)
        except ValueError:
            pair = None
        return (pair, te, ts)
    if task == 4:
        n, d, fit_n, fit_d = case
        model = fit_forest(c["X_quarter"], c["y_quarter"], n_estimators=fit_n, max_depth=fit_d)
        return (model, n, d)
    if task == 5:
        nl, ns, every = case
        return (weights_from_forecasts(c["pred_today"], n_long=nl, n_short=ns), nl, ns, every)
    raise AssertionError(task)


CASES = [
    # task, case, should pass, what it is
    (1, (21, 21), True, "lookback 21"),
    (1, (126, 126), True, "lookback 126"),
    (1, (5, 5), False, "the worked example's 5"),
    (1, (300, 300), False, "lookback above 252"),
    (1, (21, 63), False, "column does not match the declared lookback"),
    (2, (63, 63), True, "window 63"),
    (2, (10, 10), True, "window 10"),
    (2, (21, 21), False, "the worked example's 21"),
    (2, (1000, 1000), False, "window above 252"),
    (2, ("vol_42", 42), True, "another feature given by name"),
    (2, ("ret_5", 42), False, "a name that is already a feature"),
    (2, ("vol_5000", 42), False, "a name with a window above 252"),
    (2, ("ret_300", 42), False, "a momentum name with a window above 252"),
    (3, ("2014-12-31", "2015-01-12"), True, "the reference dates"),
    (3, ("2010-06-30", "2010-07-08"), True, "an early split"),
    (3, ("2012-12-31", "2013-01-10"), False, "the worked example's dates"),
    (3, ("2014-12-31", "2015-01-05"), False, "test_start inside the embargo"),
    (3, ("2006-12-29", "2007-01-10"), False, "train_end before 2008-12-31"),
    (3, ("2016-12-30", "2018-01-05"), False, "test_start in 2018"),
    (3, ("31/12/2014", "12/01/2015"), False, "day/month dates with slashes"),
    (3, ("2014-12-31", "2015-1-12"), False, "a date not written YYYY-MM-DD"),
    (3, (pd.Timestamp("2014-12-31"), date(2015, 1, 12)), True, "a Timestamp and a datetime.date"),
    (4, (30, 3, 30, 3), True, "30 trees of depth 3"),
    (4, (20, 12, 20, 12), True, "20 trees of depth 12"),
    (4, (100, 4, 100, 4), False, "the worked example's 100 and 4"),
    (4, (30, None, 30, None), False, "max_depth=None"),
    (4, (600, 6, 20, 6), False, "more than 500 trees"),
    (4, (30, 5, 30, 3), False, "model not fitted with the declared depth"),
    (5, (4, 4, 21), True, "the reference portfolio"),
    (5, (2, 6, 10), True, "2 long, 6 short, every 10 days"),
    (5, (4, 4, 5), False, "the worked example's 4, 4, 5"),
    (5, (4, 4, 7), False, "rebalance every 7 days"),
    (5, (9, 4, 21), False, "9 longs"),
]


@pytest.mark.parametrize("task,case,ok,label", CASES, ids=[f"task{t}-{l}" for t, _, _, l in CASES])
def test_checker(ctx, task, case, ok, label, capsys):
    ex = _tracker(ctx)
    ex.check(task, *_args(task, case, ctx))
    out = capsys.readouterr().out
    assert ex.passed(task) is ok, out
    assert ("[ok]" in out) is ok and ("[x] Not yet:" in out) is (not ok), out


def test_passed_knobs_reach_my_strategy(ctx, capsys):
    ex = _tracker(ctx)
    ex.check(5, *_args(5, (3, 5, 10), ctx))
    assert (ex.my_strategy.n_long, ex.my_strategy.n_short, ex.my_strategy.rebalance_days) == (3, 5, 10)
    table = ex.knob_table()
    assert table.loc["n_long", "source"] == "yours" and table.loc["max_depth", "source"] == "reference"
    assert ex.any_reference


def test_hints_solutions_and_reference(ctx, capsys):
    from quantsoc_ml import solutions as S
    ex = _tracker(ctx)
    for k in "12345":
        for _ in range(4):
            ex.hint(k)
        ex.show_solution(k)
        assert len(S.SOLUTION_SOURCE[k].splitlines()) <= 3, f"Task {k} answer is longer than three lines"
    out = capsys.readouterr().out
    assert out.count("Hint 3/3") >= 5
    ex.use_reference(3)
    train, test = ex.result(3)
    assert str(train.index.get_level_values("date").max().date()) == "2014-12-31"
    assert ex.summary().loc["3", "status"] == "reference"
    np.testing.assert_equal(ex.my_strategy.test_start, "2015-01-12")


def test_failed_recheck_keeps_earlier_answer(ctx, capsys):
    ex = _tracker(ctx)
    ex.check(1, *_args(1, (63, 63), ctx))
    ex.check(1, *_args(1, (5, 5), ctx))                # re-run with the worked example's value: refused
    out = capsys.readouterr().out
    assert "Your earlier accepted answer stays in use (momentum_lookback = 63" in out, out
    assert ex.my_strategy.momentum_lookback == 63 and ex.knob_table().loc["momentum_lookback", "source"] == "yours"
    assert ex.passed(1)
    fresh = _tracker(ctx)                              # never passed: a failure leaves the reference, labelled
    fresh.check(4, *_args(4, (100, 4, 100, 4), ctx))
    assert fresh.knob_table().loc["n_estimators", "source"] == "reference" and not fresh.passed(4)


def test_reference_is_labelled_where_it_is_used(ctx, capsys):
    ex = _tracker(ctx)
    ex.uses(1, 5)
    out = capsys.readouterr().out
    assert "[ref] Task 1 (Momentum lookback) is not yours yet: momentum_lookback = 21 is" in out
    assert "[ref] Task 5 (Portfolio) is not yours yet: n_long = 4, n_short = 4, rebalance_days = 21 are" in out
    assert ex.knob_label("rebalance_days") == "the reference value"
    ex.check(5, *_args(5, (4, 4, 21), ctx))
    capsys.readouterr()
    ex.uses(5)
    assert capsys.readouterr().out == "" and ex.knob_label("rebalance_days") == "your choice"
    ex.use_reference(3)
    capsys.readouterr()
    ex.result(3)
    assert "[ref] Task 3 (Split dates) uses the workshop's reference answer" in capsys.readouterr().out


def test_make_forest_refuses_before_fitting(ctx, monkeypatch):
    from quantsoc_ml import models, notebook_support as NS
    calls = []
    monkeypatch.setattr(models, "fit_forest", lambda *a, **k: calls.append(1))
    for n, d in [(2000, 6), (10, 6), (200, 13), (200, 1), (200, None), (200.0, 6)]:
        with pytest.raises(ValueError):
            NS.make_forest(ctx["X_quarter"], ctx["y_quarter"], n, d)
    assert not calls, "make_forest fitted before refusing"
    NS.make_forest(ctx["X_quarter"], ctx["y_quarter"], 20, 2)
    assert calls == [1]


def test_score_table_shows_na_for_undefined_ic(ctx, recwarn):
    import warnings
    from quantsoc_ml.notebook_support import score_table
    ds = ctx["ds"]
    test = ds.loc[ds.index.get_level_values("date") >= "2019-11-01"]
    flat = np.full(len(test), 0.9350724237877682)      # equal values whose float std is not exactly 0
    warnings.simplefilter("always")
    table = score_table({"flat": flat}, test)
    cells = " ".join(table.astype(str).to_numpy().ravel())
    assert "nan" not in cells.lower(), table
    assert table.loc["flat", "IC per day (mean)"] == "n/a" and table.loc["flat", "IC, all rows pooled"] == "n/a"
    assert not [w for w in recwarn if "Constant" in w.category.__name__], [str(w.message) for w in recwarn]


def test_value_error_advice_fits_the_message():
    from quantsoc_ml.notebook_support import STALE_MODEL_ADVICE, TASK_ADVICE, value_error_advice
    stale = "the model was fitted on ['ret_1'] but these knobs give ['ret_2']: refit the model after changing a knob"
    assert value_error_advice(stale) == STALE_MODEL_ADVICE
    assert value_error_advice("test_start (2013-01-01) must come after train_end (2013-12-31).") == TASK_ADVICE


def test_backtests_are_counted(ctx):
    from quantsoc_ml import notebook_support as NS
    from quantsoc_ml.models import fit_forest, predict_panel, xy
    from quantsoc_ml.solutions import reference_strategy
    ds = ctx["ds"]
    test = ds.loc[ds.index.get_level_values("date") >= "2019-06-01"]
    panel = predict_panel(fit_forest(ctx["X_quarter"], ctx["y_quarter"], n_estimators=20, max_depth=3), test)
    before = NS.backtests_run()
    NS.run_backtest(panel, ctx["returns"], reference_strategy())
    NS.run_backtest(panel, ctx["returns"], reference_strategy())
    assert NS.backtests_run() == before + 2
    table = NS.backtest_checklist(reference_strategy(), True)
    assert f"you have run {before + 2} backtests" in table.loc["Every try counted and reported", "why"]
