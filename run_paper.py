#!/usr/bin/env python
"""Run your exported strategy once against an Alpaca PAPER account (one rebalance).

    python run_paper.py                       # same as --preview: real paper data, NO orders sent
    python run_paper.py --preview
    python run_paper.py --submit              # asks you to type SUBMIT, then sends to the PAPER endpoint
    python run_paper.py --preview --dry-data data/prices_recent.csv.gz   # offline demo, no account needed

--bundle DIR   the exported bundle folder (default: the folder this script is in)
--symbols      comma-separated symbols instead of the bundle's list (refused unless you also pass
               --allow-universe-change: the features are ranked across the table, so a different
               list changes every forecast)
--lookback     trading days of history to fetch (default 300; raise it if the runner says the
               features need more)
--dry-data     a price file to use instead of Alpaca; runs against an in-memory MOCK broker

Keys: APCA_API_KEY_ID and APCA_API_SECRET_KEY, from Colab secrets, environment variables, or a
.env file next to this script (or in the bundle folder, or the current folder). They are never printed.

Exit codes: 0 ok; 1 no credentials; 2 refused (submit not confirmed, open orders, not paper);
3 setup problem (no bundle, alpaca-py missing, broker unreachable); 130 stopped with Ctrl+C.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preview", action="store_true", help="propose orders only (the default)")
    g.add_argument("--submit", action="store_true", help="send the orders to the Alpaca PAPER endpoint (asks first)")
    p.add_argument("--bundle", default=None, help="bundle folder (default: the folder this script is in)")
    p.add_argument("--symbols", default=None, help="comma-separated symbols instead of the bundle's list "
                   "(needs --allow-universe-change)")
    p.add_argument("--allow-universe-change", action="store_true",
                   help="let --symbols differ from the universe the model was trained on")
    p.add_argument("--dry-data", default=None, help="offline demo: price CSV(.gz) to use instead of Alpaca, with a mock broker")
    p.add_argument("--lookback", type=int, default=300, help="trading days of history to fetch (default 300)")
    p.add_argument("--capital-fraction", type=float, default=0.5,
                   help="share of account equity per side of the book (default 0.5: half long, half short)")
    p.add_argument("--feed", default="iex", help="Alpaca data feed (default iex, the free plan)")
    return p.parse_args(argv)


def _find(path: str, bundle: Path) -> Path:
    for base in (Path.cwd(), bundle, HERE):
        c = (base / path) if not Path(path).is_absolute() else Path(path)
        if c.is_file():
            return c
    raise FileNotFoundError(f"cannot find the price file {path}")


def _make_confirm(where: str, input_fn):
    def confirm(orders) -> bool:
        try:
            answer = input_fn(f"Type SUBMIT (in capitals) to send these {len(orders)} orders to {where}; "
                              "anything else cancels: ")
        except EOFError:
            answer = ""
        if not sys.stdin.isatty():
            print()                                # the typed answer did not echo a newline
        return answer.strip() == "SUBMIT"
    return confirm


def main(argv=None, input_fn=input) -> int:
    args = parse_args(argv)
    mode = "submit" if args.submit else "preview"
    bundle = Path(args.bundle).resolve() if args.bundle else HERE
    for p in (HERE, bundle):                       # the bundle's own quantsoc_ml copy wins
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from quantsoc_ml import paper
    except ImportError as e:
        print(f"[runner] cannot import quantsoc_ml ({e}). Run this from the unzipped bundle folder, "
              "after: pip install -r requirements.txt")
        return 3
    if not (bundle / "strategy.py").is_file():
        print(f"[runner] {bundle} has no strategy.py. Export a bundle from the notebook, or pass --bundle FOLDER.")
        return 3
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    if args.dry_data:
        from quantsoc_ml.data import load_prices
        try:
            path = _find(args.dry_data, bundle)
        except FileNotFoundError as e:
            print(f"[runner] {e}")
            return 3
        long = load_prices(path)
        keep = sorted(long["date"].unique())[-int(args.lookback):]
        broker = paper.MockBroker(bars=long[long["date"].isin(keep)], script="demo")
        where = "the in-memory MOCK broker"
        context = [f"data: last {len(keep)} trading days of {path.name} (offline, not live prices); "
                   "mock broker starting with $100,000 and no positions"]
    else:
        key, secret, source = paper.get_credentials([HERE / ".env", bundle / ".env", Path.cwd() / ".env"])
        if not key or not secret:
            print("[runner] No Alpaca paper keys found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in a .env file "
                  "next to run_paper.py (copy .env.example), or as environment variables.\n"
                  "         The offline demo needs no keys: python run_paper.py --dry-data data/prices_recent.csv.gz")
            return 1
        try:
            broker = paper.AlpacaPaperBroker(key, secret, feed=args.feed)
        except ImportError:
            print("[runner] alpaca-py is not installed. Run: pip install -r requirements.txt")
            return 3
        except paper.BrokerError as e:
            print(f"[runner] {e}")
            return 3
        finally:
            key = secret = None
        where = "your Alpaca PAPER account"
        context = [f"keys read from {source} (not shown); data and account from Alpaca"]

    try:
        paper.run(mode, broker, bundle, symbols=symbols, lookback_days=args.lookback,
                  capital_fraction=args.capital_fraction, allow_universe_change=args.allow_universe_change,
                  confirm=_make_confirm(where, input_fn) if mode == "submit" else None, context=context)
    except paper.Refused as e:
        print(f"[runner] REFUSED: {e}")
        return 2
    except (paper.BrokerError, ValueError, FileNotFoundError) as e:
        print(f"[runner] stopped: {e}")
        return 3
    except KeyboardInterrupt:
        print("\n[runner] stopped with Ctrl+C. Orders already sent (if any) stay with the broker.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
