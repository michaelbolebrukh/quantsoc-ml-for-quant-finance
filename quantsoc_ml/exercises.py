"""Exercise tracker for the five copy-and-change tasks. Each passed task sets knobs of `my_strategy`.

Design (the same as Workshop 1, adapted to knobs):
* `ex.check(n, value, *knobs)` never raises and returns None. It prints `[ok] ...` or
  `[x] Not yet: <one sentence>`. On [ok] the knobs are written into `ex.my_strategy`. On [x] the
  task's knobs go back to the last answer that passed (still "yours", and the tracker says so) or,
  if none ever passed, to the reference. An exact copy of the worked example is refused, so every
  knob marked "yours" was changed.
* `ex.uses(n, ...)` prints a [ref] line, at the cell that uses a knob, for every knob that is still
  the reference; `ex.knob_label(knob)` is "your choice" or "the reference value".
* Checkers accept ANY sensible value, not one right answer; the ranges are in CHECK_RULES.
* `ex.hint(n)` shows hints one at a time (three per task), `ex.show_solution(n)` prints the
  reference code, `ex.use_reference(n)` continues with the reference answer.
* Every knob starts at its reference value (solutions.REFERENCE_STRATEGY). A knob the
  participant has not set stays the reference, and `ex.knob_table()` says so for every knob.
  `ex.result(n)` returns the participant's passed answer, or the labelled reference answer
  with a printed [ref] line. Nothing is substituted silently.
* QSW_AUTO_REFERENCE=1 makes a failed check fall back to the reference (used by the test that
  runs the whole notebook from a clean kernel).
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import solutions as S
from .features import FEATURE_NAMES, compute_feature, past_return, rolling_vol
from .portfolio import weights_from_forecasts
from .splits import embargo_start, time_split

CHECK_RULES = {
    "1": "any lookback from 2 to 252 trading days, other than the worked example's 5",
    "2": "any volatility window from 2 to 252 other than the worked example's 21, or another feature name "
         "(ret_N or vol_N with N from 2 to 252, volume_z or hi52) that is not already a feature",
    "3": "train_end from 2008-12-31 to 2016-12-31; test_start at least 5 trading days later and before 2018 "
         "(not both of the worked example's dates)",
    "4": "n_estimators from 20 to 500 and max_depth from 2 to 12 (not both of the worked example's numbers)",
    "5": "n_long and n_short from 1 to 8 each; rebalance_days 5, 10 or 21 (not all three of the worked example's)",
}
# The worked example's values. A task cell starts as a copy of its worked example, so an unchanged
# copy is refused with a plain sentence: the knob must be the participant's own choice.
WORKED_EXAMPLES = {
    "1": (5,),
    "2": (21,),
    "3": ("2012-12-31", "2013-01-10"),
    "4": (100, 4),
    "5": (4, 4, 5),
}
TRAIN_END_RANGE = ("2008-12-31", "2016-12-31")
TEST_START_BEFORE = "2018-01-01"
REBALANCE_CHOICES = (5, 10, 21)
WINDOW_RANGE = (2, 252)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ExerciseIncomplete(Exception):
    """Raised by ExerciseTracker.require when a task has not been passed or replaced by the reference."""


@dataclass
class TaskSpec:
    key: str
    title: str
    hints: List[str]
    reference: Callable          # (context) -> value
    checker: Callable            # (value, knobs, context) -> (ok, message, {knob: value})


@dataclass
class TaskState:
    status: str = "not_started"  # not_started | failed | passed | reference
    message: str = ""
    result: Any = None
    hints_shown: int = 0
    accepted: Any = None         # the last passed (result, message), kept when a later re-check fails


# --------------------------------------------------------------------------- helpers

def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _same_frame(a, b) -> bool:
    if not isinstance(a, pd.DataFrame) or a.shape != b.shape:
        return False
    try:
        a = a.reindex(index=b.index, columns=b.columns)
    except Exception:
        return False
    an, bn = a.isna().to_numpy(), b.isna().to_numpy()
    if not np.array_equal(an, bn):
        return False
    return bool(np.allclose(a.to_numpy(float)[~an], b.to_numpy(float)[~bn], rtol=1e-9, atol=1e-12))


def _need(ctx: dict, *names) -> Optional[str]:
    missing = [n for n in names if n not in ctx]
    if missing:
        return "the tracker has not been given the data yet: run the cells above this one first."
    return None


def _date(x) -> Tuple[Optional[pd.Timestamp], str]:
    """A split date: an ISO "YYYY-MM-DD" string, a pandas Timestamp or a datetime.date, nothing else.

    Day/month strings such as "31/12/2014" are refused rather than guessed: pandas would read
    "12/01/2015" month first, as 1 December, and the participant would never notice.
    """
    hint = 'write dates as "2014-12-31" (year, month, day, in quotes)'
    if isinstance(x, (pd.Timestamp, _dt.date)):
        try:
            return pd.Timestamp(x), ""
        except Exception:
            return None, hint
    if isinstance(x, str):
        text = x.strip()
        if "/" in text:
            return None, f"{text!r} uses slashes, which can be read day first or month first: {hint}"
        if not _ISO_DATE.match(text):
            return None, f"{text!r} is not a date in the form YYYY-MM-DD: {hint}"
        try:
            return pd.Timestamp(text), ""
        except Exception:
            return None, f"{text!r} is not a real calendar date: {hint}"
    return None, f"{x!r} is not a date: {hint}"


# --------------------------------------------------------------------------- checkers

def check_task1(value, knobs, ctx):
    if (msg := _need(ctx, "prices")):
        return False, msg, {}
    if len(knobs) != 1:
        return False, "call it as ex.check(1, my_momentum, lookback).", {}
    lookback = knobs[0]
    if not _is_int(lookback):
        return False, f"lookback must be a whole number of trading days, like 21, not {lookback!r}.", {}
    lookback = int(lookback)
    if lookback == 5:
        return False, "5 is the worked example's lookback. Choose your own, for example 21 (about one month).", {}
    if not 2 <= lookback <= 252:
        return False, f"choose a lookback from 2 to 252 trading days (252 is about one year), not {lookback}.", {}
    if not isinstance(value, pd.DataFrame):
        return False, f"my_momentum should be the table that past_return gives back, not a {type(value).__name__}.", {}
    if not _same_frame(value, past_return(ctx["prices"], lookback)):
        return False, (f"my_momentum is not past_return(prices, {lookback}). Change only the number on the line "
                       "marked '# <- change this' and run the whole cell again."), {}
    note = ""
    if f"ret_{lookback}" in FEATURE_NAMES and lookback != 21:
        note = f" ret_{lookback} is already one of the features, so your model will use 7 features instead of 8."
    return True, f"momentum feature ret_{lookback}: the return over the last {lookback} trading days.{note}", \
        {"momentum_lookback": lookback}


def check_task2(value, knobs, ctx):
    if (msg := _need(ctx, "prices", "returns")):
        return False, msg, {}
    if len(knobs) != 1:
        return False, "call it as ex.check(2, my_extra, window).", {}
    choice = knobs[0]
    if isinstance(choice, str):
        name = choice.strip()
        m = re.match(r"^(ret|vol)_(\d+)$", name)
        if m and not WINDOW_RANGE[0] <= int(m.group(2)) <= WINDOW_RANGE[1]:
            return False, (f"{name} uses a window of {int(m.group(2))} days; choose a window from "
                           f"{WINDOW_RANGE[0]} to {WINDOW_RANGE[1]} trading days, for example vol_63."), {}
        try:
            ref = compute_feature(name, ctx["prices"], ctx.get("volume"))
        except ValueError as e:
            return False, f"{e}.", {}
    elif _is_int(choice):
        if int(choice) == 21:
            return False, "21 is the worked example's window. Choose your own, for example 63 (about three months).", {}
        if not 2 <= int(choice) <= 252:
            return False, f"choose a window from 2 to 252 trading days, not {int(choice)}.", {}
        name = f"vol_{int(choice)}"
        ref = rolling_vol(ctx["returns"], int(choice))
    else:
        return False, f"window must be a whole number of trading days, like 63, not {choice!r}.", {}
    if name == "vol_21":
        return False, "vol_21 is the worked example. Choose a different window or feature.", {}
    strategy = ctx.get("my_strategy")
    momentum = f"ret_{int(strategy.momentum_lookback)}" if strategy is not None else ctx.get("momentum_name", "ret_21")
    taken = [f for f in FEATURE_NAMES if f != "vol_63" and f != "ret_21"] + [momentum]
    if name in taken:
        return False, f"{name} is already one of the features, so it would add nothing. Pick something new, like vol_63.", {}
    if not isinstance(value, pd.DataFrame):
        return False, f"my_extra should be the table that rolling_vol gives back, not a {type(value).__name__}.", {}
    if not _same_frame(value, ref):
        return False, ("my_extra does not match your window. Change only the number on the line marked "
                       "'# <- change this' and run the whole cell again."), {}
    return True, f"extra feature {name} is in your strategy.", {"extra_feature": name}


def check_task3(value, knobs, ctx):
    if (msg := _need(ctx, "ds")):
        return False, msg, {}
    if len(knobs) != 2:
        return False, "call it as ex.check(3, (train, test), train_end, test_start).", {}
    (te, why_te), (ts, why_ts) = _date(knobs[0]), _date(knobs[1])
    if te is None:
        return False, f"train_end {why_te}.", {}
    if ts is None:
        return False, f"test_start {why_ts}.", {}
    lo, hi = pd.Timestamp(TRAIN_END_RANGE[0]), pd.Timestamp(TRAIN_END_RANGE[1])
    if not lo <= te <= hi:
        return False, (f"train_end {te.date()} is outside {lo.date()} to {hi.date()}: the model needs at least "
                       "eight years to learn from, and the test needs room after it (test_start before 2018 "
                       "leaves at least two years)."), {}
    if (str(te.date()), str(ts.date())) == WORKED_EXAMPLES["3"]:
        return False, ("these are the worked example's two dates. Change at least one of them on the line marked "
                       "'# <- change this', for example train_end \"2014-12-31\" and test_start \"2015-01-12\"."), {}
    earliest = embargo_start(ctx["ds"], te)
    if ts < earliest:
        return False, (f"test_start {ts.date()} is too close: the last training target is only known 5 trading days "
                       f"after {te.date()}, so start the test on or after {earliest.date()}."), {}
    if ts >= pd.Timestamp(TEST_START_BEFORE):
        return False, f"start the test before {TEST_START_BEFORE} so it covers at least two years.", {}
    if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, pd.DataFrame) for v in value)):
        return False, "the first argument should be the pair (train, test) that time_split gives back.", {}
    train, test = value
    ref_tr, ref_te = time_split(ctx["ds"], te, ts)
    if len(train) != len(ref_tr) or len(test) != len(ref_te) or not train.index.equals(ref_tr.index):
        return False, ("train and test do not match your two dates. Change only the dates on the line marked "
                       "'# <- change this' and run the whole cell again."), {}
    days = int(test.index.get_level_values("date").nunique())
    return True, (f"train up to {te.date()} ({len(train):,} rows), test from {ts.date()} "
                  f"({days:,} trading days, {len(test):,} rows)."), \
        {"train_end": str(te.date()), "test_start": str(ts.date())}


def check_task4(value, knobs, ctx):
    if (msg := _need(ctx, "X_quarter")):
        return False, msg, {}
    if len(knobs) != 2:
        return False, "call it as ex.check(4, my_forest, n_estimators, max_depth).", {}
    n, depth = knobs
    if depth is None:
        return False, ("max_depth=None lets every tree grow until it memorises single days; section 5 shows exactly "
                       "that, live. Here pick a number from 2 to 12."), {}
    if not _is_int(n) or not _is_int(depth):
        return False, "n_estimators and max_depth must be whole numbers, like 200 and 6.", {}
    n, depth = int(n), int(depth)
    if (n, depth) == WORKED_EXAMPLES["4"]:
        return False, ("100 trees of depth 4 is the worked example. Change at least one of the two numbers, "
                       "for example 200 trees of depth 6."), {}
    if not 20 <= n <= 500:
        return False, f"choose from 20 to 500 trees, not {n} (more than 500 is slow and adds almost nothing).", {}
    if not 2 <= depth <= 12:
        return False, f"choose a max_depth from 2 to 12, not {depth}.", {}
    if not hasattr(value, "predict"):
        return False, "my_forest should be the fitted forest that fit_forest gives back.", {}
    if getattr(value, "n_estimators", None) != n or getattr(value, "max_depth", None) != depth:
        return False, ("my_forest was not fitted with these two numbers. Change them on the line marked "
                       "'# <- change this' and run the whole cell again."), {}
    if getattr(value, "n_features_in_", None) != ctx["X_quarter"].shape[1]:
        return False, "my_forest was fitted on different columns. Fit it on X_quarter, y_quarter as in the example.", {}
    return True, f"a forest of {n} trees, each at most {depth} questions deep.", {"n_estimators": n, "max_depth": depth}


def check_task5(value, knobs, ctx):
    if (msg := _need(ctx, "pred_today")):
        return False, msg, {}
    if len(knobs) != 3:
        return False, "call it as ex.check(5, my_weights, n_long, n_short, rebalance_days).", {}
    n_long, n_short, every = knobs
    if not (_is_int(n_long) and _is_int(n_short)):
        return False, "n_long and n_short must be whole numbers, like 4.", {}
    if not (1 <= int(n_long) <= 8 and 1 <= int(n_short) <= 8):
        return False, f"choose from 1 to 8 stocks on each side, not {n_long} and {n_short}.", {}
    if not _is_int(every) or int(every) not in REBALANCE_CHOICES:
        return False, "rebalance_days must be 5 (weekly), 10 (every two weeks) or 21 (monthly).", {}
    if (int(n_long), int(n_short), int(every)) == WORKED_EXAMPLES["5"]:
        return False, ("4 long, 4 short, every 5 days is the worked example. Change at least one of the three "
                       "numbers, for example rebalance_days = 21 (about monthly)."), {}
    ref = weights_from_forecasts(ctx["pred_today"], n_long=int(n_long), n_short=int(n_short))
    if not isinstance(value, pd.Series):
        return False, "my_weights should be the Series that weights_from_forecasts gives back.", {}
    got = pd.Series(value, dtype=float).reindex(ref.index)
    if got.isna().any() or not np.allclose(got.to_numpy(), ref.to_numpy(), atol=1e-12):
        return False, ("my_weights do not match your two counts. Change only the numbers on the line marked "
                       "'# <- change this' and run the whole cell again."), {}
    return True, (f"long {int(n_long)} stocks at {1 / int(n_long):.1%} each, short {int(n_short)} at "
                  f"{1 / int(n_short):.1%} each, rebalanced every {int(every)} trading days."), \
        {"n_long": int(n_long), "n_short": int(n_short), "rebalance_days": int(every)}


# --------------------------------------------------------------------------- registry

SPECS: Dict[str, TaskSpec] = {
    "1": TaskSpec("1", "Momentum lookback", [
        "Copy the worked example and change only the number 5 on the first line.",
        "21 trading days is about a month, 63 about three months, 126 about six months, 252 about a year.",
        "The finished cell is three lines: lookback = 21, my_momentum = past_return(prices, lookback), "
        "ex.check(1, my_momentum, lookback).",
    ], lambda c: S.task1(c["prices"]), check_task1),
    "2": TaskSpec("2", "Extra feature", [
        "Copy the worked example and change only the number 21 on the first line.",
        "A longer window gives a smoother, slower volatility: 63 is about three months.",
        "The finished cell: window = 63, my_extra = rolling_vol(returns, window), ex.check(2, my_extra, window).",
    ], lambda c: S.task2(c["returns"]), check_task2),
    "3": TaskSpec("3", "Split dates", [
        "Copy the worked example above and change the two dates in quotes on the first line. "
        "Keep the format \"YYYY-MM-DD\", for example \"2014-12-31\".",
        "train_end can be any day from 2008-12-31 to 2016-12-31. test_start must be at least 5 trading days "
        "later (about a week) and before 2018.",
        "The workshop's choice: train_end, test_start = \"2014-12-31\", \"2015-01-12\".",
    ], lambda c: S.task3(c["ds"]), check_task3),
    "4": TaskSpec("4", "Forest size", [
        "Copy the worked example above and change the two numbers on the first line: the number of trees, "
        "then the depth.",
        "More trees make the average steadier but slower. A deeper tree asks more questions and can memorise more.",
        "The workshop's choice: n_estimators, max_depth = 200, 6.",
    ], lambda c: S.task4(c["X_quarter"], c["y_quarter"]), check_task4),
    "5": TaskSpec("5", "Portfolio", [
        "Copy the worked example above and change the three numbers on the first line: stocks to buy, "
        "stocks to short, days between trades.",
        "n_long and n_short can each be 1 to 8. rebalance_days can be 5, 10 or 21. Trading less often costs less.",
        "The workshop's choice: n_long, n_short = 4, 4 and rebalance_days = 21.",
    ], lambda c: S.task5(c["pred_today"]), check_task5),
}


class ExerciseTracker:
    def __init__(self, auto_reference: Optional[bool] = None):
        self.states: Dict[str, TaskState] = {k: TaskState() for k in SPECS}
        self.contexts: Dict[str, dict] = {}
        self.my_strategy = S.reference_strategy()
        self.knob_source: Dict[str, str] = {k: "reference" for ks in S.TASK_KNOBS.values() for k in ks}
        self.auto_reference = (os.environ.get("QSW_AUTO_REFERENCE", "") == "1") if auto_reference is None \
            else bool(auto_reference)

    # -- context (set by the notebook before each task)
    def set_context(self, key, **context) -> None:
        self.contexts.setdefault(str(key), {}).update(context)

    # -- checking
    def check(self, key, value, *knobs) -> None:
        """Check a task's value and knobs. On [ok] the knobs go into my_strategy. Never raises.

        Returns None (so a task cell prints only the verdict); fetch the answer with ex.result(n).
        """
        key = str(key)
        spec, st = SPECS[key], self.states[key]
        print(f"Task {key}: {spec.title}")
        try:
            ctx = {**self.contexts.get(key, {}), "my_strategy": self.my_strategy}
            ok, msg, updates = spec.checker(value, knobs, ctx)
        except Exception as e:                        # a checker must never crash the notebook
            ok, msg, updates = False, f"the check could not run ({type(e).__name__}: {e}).", {}
        if ok:
            self._set_knobs(updates, "yours")
            st.status, st.message, st.result = "passed", msg, value
            st.accepted = (value, msg, {k: getattr(self.my_strategy, k) for k in S.TASK_KNOBS[key]})
            print(f"  [ok] {msg}")
            return None
        print(f"  [x] Not yet: {msg}")
        print(f"      Allowed: {CHECK_RULES[key]}.")
        print(f"      Help: ex.hint({key})   Answer: ex.show_solution({key})   Continue anyway: ex.use_reference({key})")
        if st.accepted is not None:
            # A re-check of a task that passed earlier failed. Keep the earlier accepted answer, so the
            # knob table's "yours" stays true, and say so: nothing changes silently.
            value, earlier, knob_values = st.accepted
            self._set_knobs(knob_values, "yours")
            st.status, st.result = "passed", value
            st.message = f"{earlier} (a later re-check failed; this earlier answer is still in use)"
            shown = ", ".join(f"{k} = {v}" for k, v in knob_values.items())
            print(f"      Your earlier accepted answer stays in use ({shown}, labelled 'yours'). "
                  "Fix this cell and run it again to change it.")
            return None
        st.status, st.message = "failed", msg
        self._set_knobs({k: getattr(S.REFERENCE_STRATEGY, k) for k in S.TASK_KNOBS[key]}, "reference")
        if self.auto_reference:
            print("  QSW_AUTO_REFERENCE=1: continuing with the reference answer (labelled).")
            self.use_reference(key)
        return None

    def attempt(self, key, fn: Callable, *args, knobs: Tuple = (), **kwargs):
        """Run fn(*args, **kwargs) and check what it returns; an error is reported, not raised."""
        try:
            value = fn(*args, **kwargs)
        except Exception as e:
            print(f"Task {key}: {SPECS[str(key)].title}\n  [x] Your code raised {type(e).__name__}: {e}")
            self.states[str(key)].status, self.states[str(key)].message = "failed", f"{type(e).__name__}: {e}"
            if self.auto_reference:
                self.use_reference(key)
            return None
        return self.check(key, value, *knobs)

    def _set_knobs(self, updates: Dict[str, Any], source: str) -> None:
        if updates:
            self.my_strategy = _update(self.my_strategy, updates)
            for k in updates:
                self.knob_source[k] = source

    # -- results
    def passed(self, key) -> bool:
        return self.states[str(key)].status in ("passed", "reference")

    def result(self, key):
        """The participant's passed answer, or the labelled reference answer (with a printed [ref] line)."""
        key = str(key)
        st = self.states[key]
        if st.status == "passed" and st.result is not None:
            return st.result
        if st.status == "reference" and st.result is not None:
            print(f"  [ref] Task {key} ({SPECS[key].title}) uses the workshop's reference answer, so the cells below "
                  f"do too. Finish Task {key} above and run this cell again to use your own.")
            return st.result
        print(f"  [ref] Task {key} is not passed yet, so the cells below use the workshop's reference answer. "
              f"Finish Task {key} above and run this cell again to use your own.")
        self.use_reference(key)
        return self.states[key].result

    def uses(self, *keys) -> None:
        """Print a [ref] line for every knob of these tasks that is still the reference, where it is used.

        Tasks 1, 2 and 5 hand nothing to the cells below except knobs, so this is where a skipped
        task becomes visible: the cell that uses the knob says the value is the reference.
        """
        for key in (str(k) for k in keys):
            ref = [k for k in S.TASK_KNOBS[key] if self.knob_source[k] == "reference"]
            if ref:
                shown = ", ".join(f"{k} = {getattr(self.my_strategy, k)}" for k in ref)
                print(f"  [ref] Task {key} ({SPECS[key].title}) is not yours yet: {shown} "
                      f"{'is' if len(ref) == 1 else 'are'} the workshop's reference. Finish Task {key} above and "
                      "run this cell again to use your own.")

    def knob_label(self, knob: str) -> str:
        """'your choice' or 'the reference value', from the knob table's source column."""
        return "your choice" if self.knob_source.get(knob) == "yours" else "the reference value"

    def require(self, *keys) -> None:
        for key in keys:
            if not self.passed(key):
                st = self.states[str(key)]
                raise ExerciseIncomplete(
                    f"Task {key} ({SPECS[str(key)].title}) is not complete yet ({st.status}). Finish it above, "
                    f"or run ex.use_reference({key}) to continue with the labelled reference answer.")

    # -- help
    def hint(self, key, level: Optional[int] = None) -> None:
        key = str(key)
        spec, st = SPECS[key], self.states[key]
        if level is None:
            st.hints_shown = min(st.hints_shown + 1, len(spec.hints))
            level = st.hints_shown
        level = max(1, min(int(level), len(spec.hints)))
        for i in range(level):
            print(f"Hint {i + 1}/{len(spec.hints)}: {spec.hints[i]}")

    def show_solution(self, key) -> None:
        key = str(key)
        print(f"--- Reference answer for Task {key} ({SPECS[key].title}) ---")
        print(S.SOLUTION_SOURCE[key])

    def use_reference(self, key):
        key = str(key)
        spec, st = SPECS[key], self.states[key]
        ctx = self.contexts.get(key, {})
        try:
            result = spec.reference(ctx)
        except KeyError:
            print(f"  [ref] Task {key}: run the cells above first, then ex.use_reference({key}) again.")
            return None
        self._set_knobs({k: getattr(S.REFERENCE_STRATEGY, k) for k in S.TASK_KNOBS[key]}, "reference")
        st.status, st.result, st.accepted = "reference", result, None
        st.message = "REFERENCE answer in use (not the participant's own choice)"
        print(f"  [ref] Task {key} ({spec.title}): using the workshop REFERENCE answer. "
              "This is recorded, and labelled in your export.")
        return None

    # -- reporting
    def summary(self) -> pd.DataFrame:
        rows = [{"task": k, "title": SPECS[k].title, "status": v.status, "message": v.message}
                for k, v in self.states.items()]
        return pd.DataFrame(rows).set_index("task")

    def knob_table(self) -> pd.DataFrame:
        rows = []
        for task, knobs in S.TASK_KNOBS.items():
            for k in knobs:
                rows.append({"knob": k, "task": task, "value": getattr(self.my_strategy, k),
                             "reference value": getattr(S.REFERENCE_STRATEGY, k), "source": self.knob_source[k]})
        for k in ("cost_bps",):
            rows.append({"knob": k, "task": "-", "value": getattr(self.my_strategy, k),
                         "reference value": getattr(S.REFERENCE_STRATEGY, k), "source": "fixed"})
        return pd.DataFrame(rows).set_index("knob")

    @property
    def any_reference(self) -> bool:
        """True when any of the five tasks' knobs is still the reference (the export records this)."""
        return any(v == "reference" for v in self.knob_source.values())

    def all_done(self) -> bool:
        return all(self.passed(k) for k in SPECS)


def _update(cfg, updates):
    """Set the knobs IN PLACE, so a notebook variable `my_strategy = ex.my_strategy` stays current."""
    for k, v in updates.items():
        setattr(cfg, k, v)
    return cfg
