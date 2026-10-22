"""QuantSoc workshop 4: paper-trading dashboard for the Avellaneda and Lee (2010) stat-arb strategy.

Run from the repository root:

    python -m streamlit run workshop4/dashboard.py

Two modes, chosen in the sidebar:

* Offline demo: the committed price snapshots and an in-memory mock broker. Nothing leaves your computer.
* Alpaca paper: your Alpaca PAPER account (practice money), with keys read from `.env` at the repo root.

The page never sends an order on its own. Orders go to Alpaca only when you are in Alpaca paper
mode, tick the box, type SUBMIT and press the button. All the trading logic lives in
workshop4/live.py (which reuses Workshop 3's broker layer); this file only draws it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # so `quantsoc_ml`, `workshop3` and `workshop4` import from the repo root
    sys.path.insert(0, str(ROOT))

from quantsoc_ml.paper import BrokerError, ReadOnlyBroker, Refused, get_credentials  # noqa: E402
from workshop4 import live  # noqa: E402

OFFLINE, ALPACA = "Offline demo", "Alpaca paper"
ACTION_COLOURS = {"open long": "#2e7d32", "keep long": "#81c784", "close long": "#1565c0",
                  "open short": "#c62828", "keep short": "#e57373", "close short": "#6a1b9a",
                  "not opened": "#f9a825", "": "#9e9e9e"}


# --------------------------------------------------------------------------- cached loading

@st.cache_data(show_spinner="Loading the price snapshots...")
def snapshot() -> pd.DataFrame:
    return live.snapshot_bars()


@st.cache_data(show_spinner="Replaying the strategy over the snapshot to build the demo account...")
def demo_end() -> pd.Timestamp:
    return live.demo_decision_date(snapshot())


def money(x: float) -> str:
    return f"${x:,.2f}"


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="QuantSoc Avellaneda and Lee dashboard", layout="wide")
st.title("Avellaneda and Lee paper-trading dashboard")
st.caption("Avellaneda and Lee (2010): each stock's last 60 daily returns are regressed on its sector ETF, the "
           "leftover (residual) is fitted as a mean-reverting process, and the s-score says how far it has "
           "strayed. Buy stocks that are far below equilibrium, sell short those far above, hedge each with its "
           "sector ETF. QuantSoc workshop 4. Teaching tool, not investment advice.")

# ---- sidebar: mode and keys
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

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", [OFFLINE, ALPACA], key="mode")
    if have_keys:
        st.caption(f"Alpaca keys: found in {key_source}")
    else:
        st.caption("Alpaca keys: not found. Copy `.env.example` to `.env` in the repo root and paste your PAPER keys."
                   if key_source == "not found" else f"Alpaca keys: found in {key_source}")
    st.divider()
    cfg = live.live_config()
    st.caption(f"Strategy settings (the paper's): {cfg.window}-day window, sector ETF factors, open beyond "
               f"s = ±{cfg.s_open:g}, close a short below {cfg.s_close_short:g}, close a long above "
               f"{-cfg.s_close_long:g}, each new trade {cfg.position:.1%} of equity (the backtest uses 5%; 2.5% keeps a full book inside Alpaca's buying power), mean reversion faster than "
               f"{252 / cfg.min_kappa:.0f} trading days.")

# ---- banner
if mode == OFFLINE:
    st.info("**OFFLINE DEMO.** Committed price snapshots and a mock broker that already holds the trades the "
            "strategy had open the day before. Nothing leaves this computer and no orders can be sent from this mode.")
else:
    st.warning("**ALPACA PAPER.** Real data from your Alpaca PAPER account (practice money). This app cannot "
               "reach a live account. Orders are sent only when you confirm and press Submit at the bottom.")
    if not have_keys:
        st.error("No Alpaca keys found. Copy `.env.example` to `.env` in the repo root, paste your PAPER keys "
                 "after `APCA_API_KEY_ID=` and `APCA_API_SECRET_KEY=`, then reload this page.")
        st.stop()

st.warning("**One strategy per paper account.** This strategy remembers nothing between runs: it reads its open "
           "trades from the account's STOCK positions (long shares are an open long trade, short shares an open "
           "short trade; ETF positions are its hedges). Any other stock position in the account, from Workshop 3's "
           "dashboard or bought by hand, is read as one of its trades. Use a paper account for this strategy only. "
           "To start again from nothing, close all positions on the Alpaca website (Paper Trading, Positions, "
           "Close all), wait for the orders to fill, then reload this page.")

# ---- broker and plan (the plan only ever sees a read-only view)
try:
    if mode == OFFLINE:
        broker = live.demo_broker(snapshot(), end=demo_end())
    else:
        broker = live.connect_alpaca(api_key, api_secret)
    with st.spinner("Fetching prices and fitting today's s-scores..."):
        plan = live.build_live_plan(ReadOnlyBroker(broker))
except (BrokerError, ValueError) as e:
    st.error(safe(f"Could not build today's plan: {e}"))
    st.stop()

if mode == OFFLINE:
    st.caption(f"The demo stops the snapshot on {plan.decision_date:%A %d %B %Y}, the latest day on which trades "
               "open, stay open and close, so all three show. The account is the strategy replayed from nothing "
               f"over the {live.DEMO_WARMUP_DAYS} trading days before, with {money(live.DEMO_EQUITY)} of equity.")

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
c4.metric("Market", "open" if is_open else "closed", help="US stock market (New York time)")
if nxt is not None:
    c4.caption(f"Next open: {pd.Timestamp(nxt).tz_convert(live.MARKET_TZ):%a %d %b %Y, %H:%M} New York")
elif mode == OFFLINE:
    c4.caption("Next open: not available in the offline demo")

# ---- today's s-scores
st.subheader("Today's s-scores")
st.write(f"Decision date **{plan.decision_date:%A %d %B %Y}**: the last completed trading day. While the market "
         "is open, market orders fill straight away at the current price (not at the close the signal used). "
         "While it is closed, they wait for the next open.")
cfg = plan.cfg
pts = live.score_chart_data(plan)
x_sort = alt.EncodingSortField("s", order="ascending")
bars_chart = (alt.Chart(pts).mark_bar()
              .encode(x=alt.X("symbol:N", sort=x_sort, title=None),
                      y=alt.Y("s:Q", title="s-score"),
                      color=alt.Color("today:N", title="today",
                                      scale=alt.Scale(domain=list(ACTION_COLOURS), range=list(ACTION_COLOURS.values()))),
                      tooltip=["symbol", alt.Tooltip("s:Q", format=".2f"), "today", "eligible"]))
rules = (alt.Chart(live.bands(cfg)).mark_rule(strokeDash=[6, 4])
         .encode(y="s:Q", color=alt.value("#555555"), tooltip=["band"]))
labels = (alt.Chart(live.bands(cfg)).mark_text(align="left", dx=4, dy=-6, fontSize=11)
          .encode(y="s:Q", x=alt.value(0), text="band"))
st.altair_chart((bars_chart + rules + labels).properties(height=320))
st.caption(f"Dashed lines: open a long below {-cfg.s_open:g}, open a short above {cfg.s_open:g}, close a short "
           f"below {cfg.s_close_short:g}, close a long above {-cfg.s_close_long:g}. A stock opens a new trade only if "
           f"its mean reversion is fast enough (eligible); an open trade closes by its s-score whatever kappa does.")
with st.expander("Every stock: s-score, kappa, beta, hedge ETF, eligibility, state"):
    st.dataframe(plan.table.style.format({"s": "{:+.2f}", "kappa": "{:.1f}", "beta": "{:.2f}", "sigma_eq": "{:.4f}"},
                                         na_rep=""),
                 hide_index=True)
    st.caption("State: +1 long trade open, -1 short trade open, 0 flat. kappa is mean reversion per year; "
               "beta is the stock's sensitivity to its hedge ETF.")

# ---- trades today
st.subheader("Trades today")
c1, c2, c3 = st.columns(3)
for col, title, names in ((c1, "Opening", plan.opened), (c2, "Staying open", plan.kept), (c3, "Closing", plan.closed)):
    side = {s: ("long" if plan.states.get(s, plan.held.get(s, 0)) > 0 else "short") for s in names}
    col.markdown(f"**{title}: {len(names)}**")
    col.write(", ".join(f"{s} ({side[s]})" for s in names) or "none")
if plan.skipped:
    st.warning("Not opened: " + "; ".join(f"{s}: {why}" for s, why in plan.skipped.items()))

# ---- target book
st.subheader("Target book")
st.caption(f"New trades are {cfg.position:.1%} of equity in whole shares (rounded towards zero); the backtest uses 5%, "
           "and the page halves it so a full book of trades and hedges fits in Alpaca's buying power. Trades that "
           "stay open keep their current shares, which is what the paper does in practice (it resizes only new "
           "positions, to save trading costs). The notebook's backtest simplifies by holding each trade at a constant "
           "fraction of equity, so live and backtest profit and loss will differ slightly. Each hedge ETF holds minus "
           "side times beta times each trade's dollar value at the last close, summed over the trades it hedges, "
           "recomputed on every run.")
times = f"{plan.gross_exposure / plan.equity:.2f} times equity" if plan.equity > 0 else "n/a times equity"
st.caption(f"Projected gross exposure after today: ${plan.gross_exposure:,.0f} ({times}). New exposure in the "
           f"orders below: ${plan.opening_notional:,.0f}; buying power for it ${plan.buying_power_limit:,.0f} "
           "(Alpaca's overnight figure where it reports one).")
if len(plan.book) == 0:
    st.write("No open trades and no hedges after today.")
else:
    st.dataframe(plan.book.style.format({"current shares": "{:,.0f}", "target shares": "{:,.0f}",
                                         "last close": "${:,.2f}", "target $": "${:,.0f}",
                                         "target weight": "{:+.2%}"}, na_rep=""),
                 hide_index=True)

# ---- proposed orders
st.subheader("Proposed orders")
if len(plan.orders) == 0:
    st.success("No orders needed: the account already matches the targets.")
else:
    st.caption(f"{len(plan.orders)} orders, sent in this order: exits and reductions first (they free buying "
               "power), then the hedge ETF legs that add exposure, then the new stock trades, so a new trade is "
               "never left without its hedge because buying power ran out. Notional is an estimate at the last close; market orders fill at a price we cannot know "
               "in advance.")
    st.dataframe(plan.orders.style.format({"qty": "{:,.0f}", "notional": "${:,.0f}",
                                           "current_qty": "{:,.0f}", "target_qty": "{:,.0f}"}, na_rep=""),
                 hide_index=True)
for note in plan.notes:
    st.caption(f"Note: {note}")
busy = plan.busy
if busy:
    st.warning(f"Open orders already exist for strategy symbols: {', '.join(busy)}. Submit stays off until every "
               "open order on a strategy stock or hedge ETF has filled or been cancelled on the Alpaca website: "
               "a half-filled trade would otherwise be hedged now at half size and re-hedged later.")
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
                "next trading day closes.")
    ready = (ok and typed.strip() == live.CONFIRM_WORD and len(plan.orders) > 0 and not already and not busy
             and not plan.over_buying_power)
    clicked = st.button("Submit PAPER orders", type="primary", disabled=not ready, key="submit")
    exits_clicked = exits_ready = False
    if plan.over_buying_power and len(plan.exit_orders) > 0:
        # The full plan is blocked, but cutting risk never needs buying power: offer the exits on their own.
        st.write(f"The full plan does not fit in buying power. You can still send its {len(plan.exit_orders)} exits "
                 "and reductions on their own (same confirmation and checks); the new trades stay unsent.")
        exits_ready = ok and typed.strip() == live.CONFIRM_WORD and not already and not busy
        exits_clicked = st.button("Submit exits and reductions only (PAPER)", disabled=not exits_ready,
                                  key="submit_exits")
    if (clicked and ready) or (exits_clicked and exits_ready):  # never on page load: only a click
        sent = False
        exits_only = bool(exits_clicked and not clicked)
        try:
            # submit_checked sends the plan ON SCREEN, and only if the account is still the one it was drawn for
            results = live.submit_checked(broker, shown_plan, plan, confirmed=ok, typed=typed, exits_only=exits_only)
            if not exits_only:     # the exits alone leave the rest of the day's plan still to send
                sent_dates.append(f"{plan.decision_date:%Y-%m-%d}")
            st.session_state["last_results"] = live.results_table(results)
            sent = True
        except (Refused, BrokerError) as e:
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
