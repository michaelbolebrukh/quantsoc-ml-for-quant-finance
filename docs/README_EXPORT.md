# Run your strategy on an Alpaca paper account

This folder is the strategy you built in the QuantSoc workshop "Machine Learning for Quantitative
Finance (Part 1)". It holds your fitted model, your five knobs, and a small program,
`run_paper.py`, that turns today's prices into orders for an Alpaca **paper** account.

A paper account is a practice account with pretend money. Nothing here can touch real money:
the program only ever talks to Alpaca's paper endpoint.

**This is a teaching exercise, not investment advice.** The workshop's own measurements showed
that this kind of model has no reliable edge after costs. Treat the results as a lesson in how a
trading system works, never as a reason to trade real money.

## What is in the folder

| File | What it is |
|---|---|
| `strategy.py` | your strategy: forecasts for each stock, then long and short weights |
| `model.joblib` | the fitted random forest |
| `config.json` | your five knobs (and the rest of the strategy's settings) |
| `feature_names.json` | the model's inputs, in order |
| `manifest.json` | when it was made, library versions, and whether any knob came from the reference solution |
| `run_paper.py` | the program you run |
| `quantsoc_ml/` | the workshop's code, so features are computed exactly as in the notebook |
| `data/prices_recent.csv.gz` | recent days of the workshop's price snapshot, for the offline demo |
| `.env.example` | a template for your paper keys |
| `requirements.txt` | the libraries to install, with scikit-learn pinned to the version that fitted your model |

## Step by step

1. **Unzip the folder** somewhere easy to find, for example your Documents folder. Open a
   terminal (on Windows: PowerShell; on a Mac: Terminal) and go into it:

       cd path/to/the/unzipped/folder

2. **Create a virtual environment.** This is a private set of Python libraries for this folder only.
   You need Python 3.11 or newer.

   macOS or Linux:

       python3 -m venv .venv
       source .venv/bin/activate

   Windows (PowerShell):

       py -3.12 -m venv .venv
       .venv\Scripts\Activate.ps1

   Your prompt now starts with `(.venv)`. Do this `activate` step again every time you open a new terminal.
   If PowerShell refuses with "running scripts is disabled on this system", run this once and try again:

       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

3. **Install the libraries:**

       pip install -r requirements.txt

   `scikit-learn` is pinned to the exact version that fitted your model, because a saved model
   may not load in a different version.

4. **Try the offline demo first.** It needs no account and sends nothing anywhere:

       python run_paper.py --preview --dry-data data/prices_recent.csv.gz

   It uses old prices (the end of the workshop's 2019 snapshot) and a pretend broker. If you see a
   table of proposed orders, everything is installed correctly.

5. **Get Alpaca paper keys.** Sign up free at https://alpaca.markets, log in, switch to
   **Paper Trading** (top left), and on the home page find **API Keys** and press **Generate**.
   You get two values: a Key ID and a Secret Key. The secret is shown only once, so keep the page
   open until step 6 is done. Use paper keys only, never live-trading keys.

6. **Put the keys in a `.env` file.** Copy `.env.example` to a new file called `.env` in this
   same folder, open it in a text editor and paste each value straight after its `=` sign, with no spaces or quotes. It should look like this, with your own values in place of the words:

       APCA_API_KEY_ID=paste-your-key-id-here
       APCA_API_SECRET_KEY=paste-your-secret-key-here

   Save it. Never share this file, never paste the keys into code, a notebook or a chat. The
   program never prints them. If you think a key has leaked, delete it on the Alpaca website and
   generate a new one.

7. **Preview with real data:**

       python run_paper.py --preview

   (`python run_paper.py` on its own does the same.) The banner says
   `PREVIEW: real paper-account data, NO orders submitted`. It downloads recent daily prices,
   computes your forecasts and prints the orders it **would** send. Nothing is sent.

8. **Read the table.** Each row is one order:

   * `symbol`: the stock ticker
   * `side`: `buy` or `sell`
   * `qty`: how many whole shares
   * `notional`: roughly how many dollars, at the last closing price (the real fill price will differ)
   * `current_qty` and `target_qty`: shares you hold now, and shares you will hold afterwards;
     a negative number is a **short** position (you have borrowed shares and sold them, and you
     gain if the price falls)
   * `action`: in plain words, for example `open long`, `reduce short`, `close long`

   The lines above the table show which stocks the model ranks highest (bought) and lowest (sold
   short), and their forecast 5-day returns. Orders that reduce positions come first, because
   they free up buying power.

9. **Send the orders to your paper account:**

       python run_paper.py --submit

   The banner now says `PAPER SUBMIT: orders go to the Alpaca PAPER endpoint only`. You see the
   same table, then the program asks you to type `SUBMIT` in capitals. Anything else cancels and
   nothing is sent.

## What to expect

* **Market orders fill at the next opportunity.** If the US market (9:30 to 16:00 New York time,
  14:30 to 21:00 UK time most of the year) is closed, orders show as `accepted` and fill when it opens.
* **Some orders may be rejected.** The commonest reason is a short sale of a stock Alpaca cannot
  borrow that day. A short that cannot be borrowed will be rejected by the broker and the runner
  reports it, nothing else changes: the other orders stand as they are.
* **Partial fills** can happen: part of an order fills and the rest stays open for the day.
* **If orders are still open**, `--submit` refuses to run again until they have filled or you have
  cancelled them on the Alpaca website. This stops the same trade being sent twice.
* **How big the positions are:** by default the program uses half your account equity on each
  side: half in longs, half in shorts. A paper account starts with $100,000, so that is about
  $50,000 long and $50,000 short. Change it with `--capital-fraction 0.3` (smaller) or up to `1.0`
  (the full size the backtest assumed, which uses almost all of a paper account's buying power
  and makes rejections more likely).
* **Which stocks:** always the 40 the model was trained on (the list is in `manifest.json`). The
  features are ranked across that table, so a different list would change every forecast. The
  runner refuses `--symbols` unless you also pass `--allow-universe-change`.
* **How often to run it:** your strategy rebalances every `rebalance_days` trading days (the
  number is in `config.json`). Each run is one rebalance. The program prints the next date to run it. Run
  `--preview` first each time.
* **The data:** prices come from Alpaca's free IEX feed, split and dividend adjusted like the
  training data. IEX volume is only part of all trading, so the `volume_z` feature is noisier
  than in the notebook.

## How to stop

* The program runs once and exits; there is nothing running in the background.
* To stop it part way, press **Ctrl+C**. Orders already sent stay with the broker.
* To undo everything, log in to Alpaca, cancel open orders and close all positions from the
  paper dashboard. You can also reset the paper account there.
* To stop for good, delete your paper API keys on the Alpaca website and delete `.env`.

## If something goes wrong

* `No Alpaca paper keys found`: the `.env` file is missing, misnamed (it must be exactly `.env`,
  not `.env.txt`) or not in this folder. The two names it needs are `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`.
* `alpaca-py is not installed`: activate the virtual environment (step 2) and repeat step 3.
* A warning that the saved model came from a different scikit-learn version: repeat step 3.
* `the features need N trading days of history`: add `--lookback N` (a little above N is fine).
  The offline demo file holds 300 days, so a feature window longer than that only works live.
* `no forecast today for X (missing data): position left alone`: one stock had no bar for the
  day. The runner does not sell it on a missing number; it will be re-ranked at the next run.
* Exit codes, for the curious: 0 fine, 1 no keys, 2 refused (not confirmed, or open orders),
  3 a setup problem or a bad input (a knob or a data check failed), 130 stopped with Ctrl+C.

Again: a paper account is practice money, and this is a teaching exercise, not investment advice.
