"""QuantSoc workshop 3: paper-trading dashboard for the Krauss, Do & Huck (2017) stat-arb strategy.

Run from the repository root:

    streamlit run workshop3/dashboard.py

Two modes, chosen in the sidebar:

* Offline demo: the committed price snapshot and an in-memory mock broker. Nothing leaves your computer.
* Alpaca paper: your Alpaca PAPER account (practice money), with keys read from `.env` at the repo root.

The page never sends an order on its own. Orders go to Alpaca only when you are in Alpaca paper
mode, tick the box, type SUBMIT and press the button. All the trading logic lives in
workshop3/live.py; this file only draws it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `quantsoc_ml` and `workshop3` import from the repo root
    sys.path.insert(0, str(ROOT))

from quantsoc_ml import statarb as S  # noqa: E402
from quantsoc_ml.paper import BrokerError, ReadOnlyBroker, Refused, get_credentials  # noqa: E402
from workshop3 import live  # noqa: E402

DEFAULT_STRATEGY = "workspace/ws3/strategy.joblib"
OFFLINE, ALPACA = "Offline demo", "Alpaca paper"


# --------------------------------------------------------------------------- cached loading

@st.cache_data(show_spinner="Loading the price snapshot...")
def snapshot() -> pd.DataFrame:
    return S.load_prices()


@st.cache_resource(show_spinner="Training a random forest on the snapshot (about 10 seconds)...")
def trained_strategy(k: int) -> S.LiveStrategy:
    """Train on the most recent 750 labelled days of the snapshot, as the paper does for each new block."""
    prices = S.wide(snapshot(), "adj_close")
    recent = prices.iloc[-(S.TRAIN_DAYS + 260):]          # enough rows for R240 on every training day
    return S.train_live(S.make_panel(recent), model_name="forest", k=k)


@st.cache_resource(show_spinner="Loading your saved strategy...")
def saved_strategy(path: str, modified: float) -> S.LiveStrategy:
    # `modified` is only part of the cache key, so a re-saved file is loaded again
    return S.load_strategy(path)


def resolve(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else ROOT / p


def money(x: float) -> str:
    return f"${x:,.2f}"


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="QuantSoc stat-arb dashboard", layout="wide")
st.title("Stat-arb paper-trading dashboard")
st.caption("Krauss, Do & Huck (2017): each day, long the k stocks most likely to beat tomorrow's median "
           "return, short the k least likely. QuantSoc workshop 3. Teaching tool, not investment advice.")

# ---- sidebar: mode, keys and strategy source
# The key and secret stay in these two local variables and go straight to the broker. They are
# never shown, logged, or put in session state.
api_key, api_secret, key_source = get_credentials(dotenv_paths=[ROOT / ".env", Path.cwd() / ".env"])
have_keys = bool(api_key and api_secret)


def safe(err) -> str:
    """Error text with the keys removed, in case a library quoted a request header."""
    text = str(err)
    for value, label in ((api_secret, "[secret]"), (api_key, "[key]")):
        if value:
            text = text.replace(value, label)
    return text


# Defaults go into session state first, so the widgets below never get two competing defaults.
st.session_state.setdefault("mode", ALPACA if have_keys else OFFLINE)
st.session_state.setdefault("strategy_path", DEFAULT_STRATEGY)
st.session_state.setdefault("k", 10)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", [OFFLINE, ALPACA], key="mode")
    if have_keys:
        st.caption(f"Alpaca keys: found in {key_source}")
    else:
        st.caption("Alpaca keys: not found. Copy `.env.example` to `.env` in the repo root and paste your PAPER keys."
                   if key_source == "not found" else f"Alpaca keys: found in {key_source}")
    st.divider()
    strategy_text = st.text_input("Saved strategy file", key="strategy_path",
                                  help="Made with statarb.save_strategy. If the file does not exist, "
                                       "a strategy is trained from the snapshot instead.")
    k_default = st.number_input("k (stocks per side, used when training here)", min_value=2, max_value=25,
                                step=1, key="k")

# ---- banner
if mode == OFFLINE:
    st.info("**OFFLINE DEMO.** Committed price snapshot and a mock broker that holds yesterday's book. "
            "Nothing leaves this computer and no orders can be sent from this mode.")
else:
    st.warning("**ALPACA PAPER.** Real data from your Alpaca PAPER account (practice money). This app cannot "
               "reach a live account. Orders are sent only when you confirm and press Submit at the bottom.")
    if not have_keys:
        st.error("No Alpaca keys found. Copy `.env.example` to `.env` in the repo root, paste your PAPER keys "
                 "after `APCA_API_KEY_ID=` and `APCA_API_SECRET_KEY=`, then reload this page.")
        st.stop()

# ---- strategy
strategy_path = resolve(strategy_text)
if strategy_path.is_file():
    try:
        strategy = saved_strategy(str(strategy_path), strategy_path.stat().st_mtime)
    except Exception as e:
        st.error(safe(f"Could not load {strategy_path}: {e}"))
        st.stop()
    source_line = f"Loaded your saved strategy from `{strategy_path}`."
else:
    strategy = trained_strategy(int(k_default))
    source_line = f"No saved strategy at `{strategy_path}`, so one was trained here from the price snapshot."
st.caption(f"{source_line} Model: {strategy.model_name}, trained on {strategy.trained_from} to "
           f"{strategy.trained_to}, k = {strategy.k}, gross exposure {strategy.gross_exposure:.0%} of equity "
           f"({len(strategy.symbols)} stocks).")

# ---- broker and plan (the plan only ever sees a read-only view)
try:
    if mode == OFFLINE:
        broker = live.demo_broker(strategy, live.snapshot_bars_for_broker(snapshot()))
    else:
        broker = live.connect_alpaca(api_key, api_secret)
    with st.spinner("Fetching prices and scoring today's signal..."):
        plan = live.build_live_plan(ReadOnlyBroker(broker), strategy)
except (BrokerError, ValueError) as e:
    st.error(safe(f"Could not build today's plan: {e}"))
    st.stop()

# The plan drawn on the PREVIOUS run is the one the user looked at; Submit sends exactly that one.
shown_plan = st.session_state.get("shown_plan")

# ---- account and clock
st.subheader("Account and market")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", money(plan.equity))
c2.metric("Cash", money(plan.cash))
c3.metric("Buying power", money(plan.buying_power))
is_open = bool(plan.clock.get("is_open"))
nxt = plan.clock.get("next_open")
c4.metric("Market", "open" if is_open else "closed",
          help="US stock market (New York time)")
if nxt is not None:
    c4.caption(f"Next open: {pd.Timestamp(nxt).tz_convert(live.MARKET_TZ):%a %d %b %Y, %H:%M} New York")
elif mode == OFFLINE:
    c4.caption("Next open: not available in the offline demo")

# ---- today's signal
st.subheader("Today's signal")
st.write(f"Decision date **{plan.decision_date:%A %d %B %Y}**: the last completed trading day. "
         "While the market is open, market orders fill straight away at the current price (not at the close "
         "the signal used). While it is closed, they wait for the next open.")
sig = live.signal_table(plan)
left, right = st.columns(2)
with left:
    st.markdown(f"**Long: top {len(plan.longs)}** (most likely to beat the median)")
    st.dataframe(plan.longs.rename("probability").to_frame().style.format("{:.3f}"))
with right:
    st.markdown(f"**Short: bottom {len(plan.shorts)}** (least likely)")
    st.dataframe(plan.shorts.rename("probability").to_frame().style.format("{:.3f}"))
chart = (alt.Chart(sig).mark_bar()
         .encode(x=alt.X("symbol:N", sort=alt.EncodingSortField("probability", order="descending"), title=None),
                 y=alt.Y("probability:Q", scale=alt.Scale(zero=False), title="P(beats tomorrow's median)"),
                 color=alt.Color("book:N", scale=alt.Scale(domain=["long", "short"], range=["#2e7d32", "#c62828"])),
                 tooltip=["symbol", "book", alt.Tooltip("probability:Q", format=".3f")])
         .properties(height=260))
st.altair_chart(chart)
if plan.skipped_shorts:
    st.warning("Skipped as shorts (the next lowest-ranked stock was used instead): "
               + "; ".join(f"{s}: {why}" for s, why in plan.skipped_shorts.items()))

# ---- target weights
st.subheader("Target weights")
st.caption(f"Each long is {strategy.gross_exposure / (2 * strategy.k):.1%} of equity and each short the same, "
           "so the book is dollar neutral. Shares are rounded towards zero.")
st.dataframe(live.targets_table(plan).style.format({"probability": "{:.3f}", "target weight": "{:+.2%}",
                                                    "target $": "${:,.0f}", "last close": "${:,.2f}",
                                                    "target shares": "{:,.0f}"}),
             hide_index=True)

# ---- positions against targets
st.subheader("Current positions against targets")
pos = live.positions_table(plan)
if len(pos) == 0:
    st.write("No positions and no targets.")
else:
    st.caption("Market value is estimated at the last completed close. Positions the strategy does not "
               "score today are shown but never traded.")
    st.dataframe(pos.style.format({"current shares": "{:,.0f}", "market value": "${:,.0f}",
                                   "current weight": "{:+.2%}", "target weight": "{:+.2%}",
                                   "target shares": "{:,.0f}", "change": "{:+,.0f}"}, na_rep=""),
                 hide_index=True)

# ---- proposed orders
st.subheader("Proposed orders")
if len(plan.orders) == 0:
    st.success("No orders needed: the account already matches the targets.")
else:
    st.caption(f"{len(plan.orders)} orders. Exits and reductions come first (they free buying power), then "
               "new exposure. Notional is an estimate at the last close; market orders fill at a price we "
               "cannot know in advance.")
    st.dataframe(plan.orders.style.format({"qty": "{:,.0f}", "notional": "${:,.0f}",
                                           "current_qty": "{:,.0f}", "target_qty": "{:,.0f}"}),
                 hide_index=True)
for note in plan.notes:
    st.caption(f"Note: {note}")
if plan.open_orders:
    st.warning(f"Open orders already exist for {', '.join(plan.open_orders)}.")
busy = sorted(set(plan.open_orders) & set(plan.orders["symbol"]))

# ---- submit (Alpaca paper only)
st.subheader("Submit")
if mode == ALPACA and getattr(broker, "paper", False) is True:
    st.write("This sends the orders above to your Alpaca **PAPER** account. Each order has an id made from the "
             "decision date, the symbol and the leg, so pressing twice cannot send it twice. One submission "
             "per decision date.")
    ok = st.checkbox("I understand this sends PAPER orders to Alpaca", key="confirm_box")
    typed = st.text_input(f"Type {live.CONFIRM_WORD} to enable the button", key="confirm_text")
    sent_dates = st.session_state.setdefault("submitted_dates", [])
    already = f"{plan.decision_date:%Y-%m-%d}" in sent_dates
    if already:
        st.info("Orders for this decision date were already submitted from this page. Come back after the "
                "next trading day closes.")
    ready = ok and typed.strip() == live.CONFIRM_WORD and len(plan.orders) > 0 and not already and not busy
    clicked = st.button("Submit PAPER orders", type="primary", disabled=not ready, key="submit")
    if clicked and ready:  # never on page load: only a click, after the checks above
        try:
            if shown_plan is None or shown_plan.decision_date != plan.decision_date:
                raise Refused("refused: the page changed before the click; look at the orders again, then submit")
            results = live.submit_plan(broker, shown_plan, confirmed=ok, typed=typed)
            sent_dates.append(f"{plan.decision_date:%Y-%m-%d}")
            st.session_state["last_results"] = live.results_table(results)
        except (Refused, BrokerError) as e:
            st.error(safe(e))
    if "last_results" in st.session_state:
        res = st.session_state["last_results"]
        counts = res["status"].value_counts().to_dict() if len(res) else {}
        st.write("Last submission: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing sent")
                 + ". A rejected order changes nothing else. Reload the page to see updated positions.")
        st.dataframe(res, hide_index=True)
else:
    st.caption("Submitting is switched off in the offline demo. Switch to Alpaca paper mode to send PAPER orders.")

# ---- account history and recent orders
st.subheader("Account equity history")
try:
    hist = live.equity_history(broker)
except BrokerError as e:
    hist = None
    st.caption(safe(e))
if hist is None or len(hist) == 0:
    st.caption("No equity history yet" + (" (not available in the offline demo)." if mode == OFFLINE else "."))
else:
    st.line_chart(hist.set_index("date")["equity"], height=240)

st.subheader("Recent orders")
try:
    recent = live.recent_orders(broker)
except BrokerError as e:
    recent = None
    st.caption(safe(e))
if recent is not None:
    if len(recent) == 0:
        st.caption("No orders yet.")
    else:
        st.dataframe(recent, hide_index=True)

# Remember the plan that is now on screen: the next click submits exactly this one.
st.session_state["shown_plan"] = plan
