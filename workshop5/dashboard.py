"""QuantSoc workshop 5: paper-trading dashboard for the Jiang, Kelly & Xiu (2023) chart-reading strategy.

Run from the repository root:

    python -m streamlit run workshop5/dashboard.py

Two modes, chosen in the sidebar:

* Offline demo: the committed price snapshot and an in-memory mock broker. Nothing leaves your computer.
* Alpaca paper: your Alpaca PAPER account (practice money), with keys read from `.env` at the repo root.

The page never sends an order on its own. Orders go to Alpaca only when you are in Alpaca paper
mode, tick the box, type SUBMIT and press the button. The strategy trades once every 5 trading days:
on the other days the page shows today's ranking and says HOLD. All the trading logic lives in
workshop5/live.py (which reuses Workshops 3 and 4); this file only draws it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `quantsoc_ml` and the workshops import from the repo root
    sys.path.insert(0, str(ROOT))

from quantsoc_ml.paper import BrokerError, ReadOnlyBroker, get_credentials  # noqa: E402
from workshop5 import live  # noqa: E402

DEFAULT_MODEL_DIR = "workshop5/models"
OFFLINE, ALPACA = "Offline demo", "Alpaca paper"
CHARTS_SHOWN = 4


# --------------------------------------------------------------------------- cached loading

@st.cache_data(show_spinner="Loading the price snapshot...")
def snapshot() -> pd.DataFrame:
    return live.snapshot_bars()


@st.cache_data(show_spinner="Listing the S&P 100...")
def universe() -> list:
    return live.snapshot_symbols()


@st.cache_resource(show_spinner="Loading the five networks...")
def cached_models(model_dir: str, stamp: tuple):
    # `stamp` is only part of the cache key, so re-trained files are loaded again
    return live.load_models(model_dir)


def resolve(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else ROOT / p


def money(x: float) -> str:
    return f"${x:,.2f}"


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="QuantSoc chart-reading dashboard", layout="wide")
st.title("Chart-reading paper-trading dashboard")
st.caption("Jiang, Kelly & Xiu (2023): each stock's last 5 trading days are drawn as a 32 by 15 black and white "
           "chart, five neural networks look at it and give the probability the stock is up over the next 5 days. "
           "Every 5 trading days: long the top decile, short the bottom decile, equal weights. QuantSoc workshop 5. "
           "Teaching tool, not investment advice.")

# ---- sidebar: mode, keys and networks
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


st.session_state.setdefault("mode", ALPACA if have_keys else OFFLINE)
st.session_state.setdefault("model_dir", DEFAULT_MODEL_DIR)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", [OFFLINE, ALPACA], key="mode")
    if have_keys:
        st.caption(f"Alpaca keys: found in {key_source}")
    else:
        st.caption("Alpaca keys: not found. Copy `.env.example` to `.env` in the repo root and paste your PAPER keys."
                   if key_source == "not found" else f"Alpaca keys: found in {key_source}")
    st.divider()
    model_text = st.text_input("Folder of trained networks", key="model_dir",
                               help="Holds I5R5_seed0.pt and so on, made by workshop5/build_cache.py.")
    override = st.checkbox("Override the weekly rule (rebalance now even if it is too soon)", key="override")

# ---- banner
if mode == OFFLINE:
    st.info("**OFFLINE DEMO.** Committed price snapshot and a mock broker that already holds last week's book. "
            "Nothing leaves this computer and no orders can be sent from this mode.")
else:
    st.warning("**ALPACA PAPER.** Real data from your Alpaca PAPER account (practice money). This app cannot "
               "reach a live account. Orders are sent only when you confirm and press Submit at the bottom.")
    if not have_keys:
        st.error("No Alpaca keys found. Copy `.env.example` to `.env` in the repo root, paste your PAPER keys "
                 "after `APCA_API_KEY_ID=` and `APCA_API_SECRET_KEY=`, then reload this page.")
        st.stop()

st.warning("**One strategy per paper account.** This page moves the account's stock positions to its own book: every "
           "S&P 100 stock it scores but does not pick is sold, or bought back, to zero. If the same paper account also "
           "runs Workshop 3's or Workshop 4's dashboard, this page would close their stock trades. Use a separate "
           "paper account for each strategy. To start again from nothing, close all positions on the Alpaca website "
           "(Paper Trading, Positions, Close all), wait for the orders to fill, then reload this page.")

# ---- the trained networks
model_path = resolve(model_text)
files = sorted(model_path.glob(f"{live.MODEL_NAME}_seed*.pt")) if model_path.is_dir() else []
if not files:
    st.error(f"No trained networks found in `{model_path}`. This page needs the five {live.MODEL_NAME} networks "
             f"(files named {live.MODEL_NAME}_seed0.pt and so on). Train them with `python workshop5/build_cache.py`, "
             "or copy the folder from the workshop materials, then reload. Nothing can be scored until they exist.")
    st.stop()
try:
    models = cached_models(str(model_path), tuple((f.name, f.stat().st_mtime) for f in files))
except Exception as e:
    st.error(safe(f"Could not load the networks in {model_path}: {e}"))
    st.stop()
st.caption(f"Loaded {len(models)} networks from `{model_path}`; their average is the score. "
           f"{len(universe())} stocks in the S&P 100 snapshot.")

# ---- broker and plan (the plan only ever sees a read-only view)
try:
    if mode == OFFLINE:
        broker = live.demo_broker(snapshot(), models)
    else:
        broker = live.connect_alpaca(api_key, api_secret)
    with st.spinner("Fetching prices, drawing the charts and scoring them..."):
        last_rebalance, history_incomplete = live.rebalance_history(broker)
        plan = live.build_live_plan(ReadOnlyBroker(broker), models, universe(), last_rebalance=last_rebalance,
                                    force=bool(override), history_incomplete=history_incomplete)
except Exception as e:     # any failure: show only the scrubbed text, never a traceback that could quote a key
    st.error(safe(f"Could not build today's plan: {e}"))
    st.stop()

# The plan drawn on the PREVIOUS run is the one the user looked at; Submit sends exactly that one.
shown_plan = st.session_state.get("shown_plan")

# ---- account and clock
st.subheader("Account and market")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", money(plan.equity))
c2.metric("Cash", money(plan.cash))
c3.metric("Buying power", money(plan.buying_power_limit),
          help="The smaller of Alpaca's buying power and its overnight (Reg T) figure, where it reports one.")
is_open = bool(plan.clock.get("is_open"))
nxt = plan.clock.get("next_open")
c4.metric("Market", "open" if is_open else "closed", help="US stock market (New York time)")
if nxt is not None:
    c4.caption(f"Next open: {pd.Timestamp(nxt).tz_convert(live.MARKET_TZ):%a %d %b %Y, %H:%M} New York")
elif mode == OFFLINE:
    c4.caption("Next open: not available in the offline demo")

# ---- the weekly rule
st.subheader("Weekly rule")
if plan.blocked:
    st.error(f"**NO ORDERS.** {plan.blocked}.")
elif plan.due:
    st.success(f"**REBALANCE DUE.** {plan.cadence_note}.")
elif plan.forced:
    st.warning(f"**HOLD, but the override is ticked.** {plan.cadence_note}. Orders are proposed anyway.")
else:
    st.info(f"**HOLD.** {plan.cadence_note}. Below you can still see today's ranking and charts, but no orders "
            "are proposed until a rebalance is due.")
st.caption(f"The strategy trades once every {live.REBALANCE_DAYS} trading days. The page reads the account's own "
           f"orders: the newest decision date of a FILLED `{live.CLIENT_ID_PREFIX}` order is the last rebalance, and "
           f"{live.REBALANCE_DAYS} or more completed trading days since then means due. No such order yet means "
           "the first rebalance.")

# ---- today's ranking
st.subheader("Today's ranking")
st.write(f"Decision date **{plan.decision_date:%A %d %B %Y}**: the last completed trading day (today's unfinished "
         "bar is never used). While the market is open, market orders fill straight away at the current price (not "
         "at the close the charts used). While it is closed, they wait for the next open.")
dec = live.decile_table(plan)
left, right = st.columns(2)
with left:
    st.markdown(f"**Long: top decile, {len(plan.longs)} stocks** (most likely to go up)")
    st.dataframe(dec[dec["decile"] == 10].style.format({"prob_up": "{:.3f}"}), hide_index=True)
with right:
    st.markdown(f"**Short: bottom decile, {int((dec['decile'] == 1).sum())} stocks** (least likely)")
    st.dataframe(dec[dec["decile"] == 1].style.format({"prob_up": "{:.3f}"}), hide_index=True)
with st.expander("Every stock: probability of going up and decile"):
    st.dataframe(dec.style.format({"prob_up": "{:.3f}"}), hide_index=True)
if plan.skipped_shorts:
    st.warning("Skipped as shorts (the next lowest-ranked stock was used instead): "
               + "; ".join(f"{s}: {why}" for s, why in plan.skipped_shorts.items()))

# ---- the charts the networks saw
st.subheader("What the networks saw")
st.caption("Each picture is the 32 by 15 chart the networks scored: for each of the last 5 days, a left tick for the "
           "open, a vertical bar from low to high, a right tick for the close, a line for the 5-day moving average, "
           "and volume bars in the bottom strip. Enlarged here; the networks see the small version.")
top, bottom = live.chart_picks(plan, CHARTS_SHOWN)
for label, picks in (("Highest ranked", top), ("Lowest ranked", bottom)):
    st.markdown(f"**{label}**")
    cols = st.columns(len(picks))
    for col, sym in zip(cols, picks):
        with col:
            st.image(live.enlarge(live.chart_array(plan, sym)), clamp=True,
                     caption=f"{sym}: P(up) {plan.scores[sym]:.3f}, decile {int(plan.table.set_index('symbol').loc[sym, 'decile'])}")

# ---- target weights
st.subheader("Target weights")
k_long, k_short = len(plan.longs), len(plan.shorts)
if k_long and k_short:
    st.caption(f"Half of equity is spread equally across the {k_long} longs and half across the {k_short} shorts "
               f"(about {0.5 / k_long:.1%} and {0.5 / k_short:.1%} each), so the book is dollar neutral. "
               "Shares are rounded towards zero.")
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
if not plan.can_rebalance:
    st.info("Hold: no orders proposed this week. The account keeps its current positions.")
elif len(plan.orders) == 0:
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
busy = plan.busy
if busy:
    st.warning(f"Open orders already exist for strategy symbols: {', '.join(busy)}. Submit stays off until every "
               "open order on a strategy stock has filled or been cancelled on the Alpaca website.")
elif plan.open_orders:
    st.warning(f"Open orders already exist for {', '.join(plan.open_orders)} (not strategy symbols).")
if plan.over_buying_power and len(plan.orders) > 0:
    st.warning("The full Submit is off: " + live.buying_power_message(plan) + ".")

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
                "next rebalance is due.")
    ready = (ok and typed.strip() == live.CONFIRM_WORD and len(plan.orders) > 0 and not already and not busy
             and not plan.over_buying_power and plan.can_rebalance)
    clicked = st.button("Submit PAPER orders", type="primary", disabled=not ready, key="submit")
    exits_clicked = exits_ready = False
    if plan.over_buying_power and len(plan.exit_orders) > 0 and plan.can_rebalance:
        # The full plan is blocked, but cutting risk never needs buying power: offer the exits on their own.
        st.write(f"The full plan does not fit in buying power. You can still send its {len(plan.exit_orders)} exits "
                 "and reductions on their own (same confirmation and checks); the new positions stay unsent.")
        exits_ready = ok and typed.strip() == live.CONFIRM_WORD and not already and not busy
        exits_clicked = st.button("Submit exits and reductions only (PAPER)", disabled=not exits_ready,
                                  key="submit_exits")
    if (clicked and ready) or (exits_clicked and exits_ready):  # never on page load: only a click
        sent = False
        exits_only = bool(exits_clicked and not clicked)
        try:
            # submit_checked sends the plan ON SCREEN, and only if the account is still the one it was drawn for
            results = live.submit_checked(broker, shown_plan, plan, confirmed=ok, typed=typed, exits_only=exits_only)
            if not exits_only:
                sent_dates.append(f"{plan.decision_date:%Y-%m-%d}")
            st.session_state["last_results"] = live.results_table(results)
            sent = True
        except Exception as e:      # Refused, BrokerError or anything else: scrubbed text only
            st.error(safe(e))
        if sent:
            st.rerun()             # redraw at once from the account as it now is; the results stay in session state
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
except Exception as e:
    hist = None
    st.caption(safe(e))
if hist is None or len(hist) == 0:
    st.caption("No equity history yet" + (" (not available in the offline demo)." if mode == OFFLINE else "."))
else:
    st.line_chart(hist.set_index("date")["equity"], height=240)

st.subheader("Recent orders")
try:
    recent = live.recent_orders(broker)
except Exception as e:
    recent = None
    st.caption(safe(e))
if recent is not None:
    if len(recent) == 0:
        st.caption("No orders yet.")
    else:
        st.dataframe(recent, hide_index=True)

# Remember the plan that is now on screen: the next click submits exactly this one.
st.session_state["shown_plan"] = plan
