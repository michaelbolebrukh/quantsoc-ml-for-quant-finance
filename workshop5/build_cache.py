"""Train the Workshop 5 networks and store their out-of-sample predictions, so the notebook runs in minutes.

Usage:
    python workshop5/build_cache.py train I5R5 --seeds 0 1 2 3 4     # writes workshop5/models/I5R5_seed*.pt
    python workshop5/build_cache.py train I5R20 --seeds 0 1 2 3 4
    python workshop5/build_cache.py predict I5R5 I5R20                # writes workshop5/data/predictions.csv.gz
    python workshop5/build_cache.py check

Everything here was fixed before any prediction for 2008 onwards was looked at:
* Training window 2000-01-01 to 2007-12-31. A training picture is used only if its label (the return
  over the next h days) is also known inside that window, so no 2008 price leaks into training.
  Inside the window: a random 70% of pictures for training, 30% for validation (paper sec. 3.3).
* Test: every picture from 2008-01-01 on, scored by networks that are never retrained.
* The paper's learning rate is 1e-5 with millions of pictures. We have about 100 stocks, so an epoch
  has far fewer steps. Pilot runs on I5R5, seed 0, judged ONLY by validation loss inside 2000-2007
  (a coin flip scores ln 2 = 0.6931):
    lr 1e-5: 0.7275, 0.7178 after epochs 1-2 (stopped: far too slow; log only, not in the history file)
    lr 1e-4: 0.7057, 0.6939, 0.7030 after epochs 1-3 (recorded in data/training_history.csv, run "lr1e-4")
    lr 1e-3: 0.7709, 0.6928, 0.6918 after epochs 1-3 (stopped there, then re-run in full as final seed 0)
  So LEARNING_RATE is 1e-3 for 5-day pictures (I5R5, and I5R20 by carrying the choice over).
* The 20-day network (I20R20) is NOT cached. On this CPU it could only be trained on one picture every
  5 days (33,418 pictures for 708,866 parameters) and never learned: at lr 1e-3 seed 0's validation loss
  went 1.777, 1.044, 0.703, 0.980 (accuracy down to 42.9%); at lr 1e-4 it went 0.822, 0.788, 0.798, 0.798
  and early stopping kept 0.788, worse than always predicting the 56.9% share of up labels (0.684).
  Training on every day's picture would take most of a day here. So the monthly model is I5R20
  (5-day pictures, 20-day returns, the paper's Table 3 has it), and the 20-day network is left as a GPU
  exercise in the notebook. The I20R20 settings below stay for anyone who wants to try.
* 20-day pictures: a picture every TRAIN_EVERY trading days in training (the CPU this cache was built
  on trains about 85 pictures a second for this network), predictions every 5 days in the test.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from quantsoc_ml import cnn  # noqa: E402

TRAIN_START, TRAIN_END, TEST_START = "2000-01-01", "2007-12-31", "2008-01-01"
MODELS = {"I5R5": dict(days=5, horizon=5), "I5R20": dict(days=5, horizon=20), "I20R20": dict(days=20, horizon=20)}
LEARNING_RATE = {"I5R5": 1e-3, "I5R20": 1e-3, "I20R20": 1e-4}          # chosen by the pilot runs, see the docstring
TRAIN_EVERY = {"I5R5": 1, "I5R20": 1, "I20R20": 5}
PREDICT_EVERY = {"I5R5": 5, "I5R20": 5, "I20R20": 5}


def training_images(name: str, mkt: cnn.Market) -> cnn.ImageSet:
    m = MODELS[name]
    inside = mkt.window(TRAIN_START, TRAIN_END)          # forward returns past 2007 are unknown here
    return cnn.make_images(inside, m["days"], m["horizon"], every=TRAIN_EVERY[name])


def train(name: str, seeds, lr=None, max_epochs=50, tag=None):
    mkt = cnn.load_market()
    data = training_images(name, mkt)
    m = MODELS[name]
    lr = lr or LEARNING_RATE[name]
    print(f"{name}: {len(data):,} training-window pictures, {data.meta.label.mean():.1%} labelled up, lr {lr:g}", flush=True)
    hist_file = cnn.HISTORY_CACHE
    for seed in seeds:
        t0 = time.time()
        t = cnn.train_cnn(data.images, data.meta["label"].to_numpy(), m["days"], m["horizon"], seed=seed,
                          lr=lr, max_epochs=max_epochs)
        if tag is None:
            cnn.save_trained(t, cnn.MODEL_DIR / f"{name}_seed{seed}.pt")
        h = t.history.assign(model=name, seed=seed, lr=lr, run=tag or "final", n_pictures=len(data))
        old = pd.read_csv(hist_file) if hist_file.exists() else pd.DataFrame()
        if len(old):
            old = old[~((old["model"] == name) & (old["seed"] == seed) & (old["run"] == (tag or "final")) & (old["lr"] == lr))]
        pd.concat([old, h], ignore_index=True).to_csv(hist_file, index=False, float_format="%.6f")
        print(f"{name} seed {seed}: best validation loss {h.val_loss.min():.4f} after {len(h)} epochs, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


def predict(names):
    mkt = cnn.load_market()
    frames = []
    old = pd.read_csv(cnn.PRED_CACHE) if cnn.PRED_CACHE.exists() else pd.DataFrame(columns=["model"])
    for name in names:
        m = MODELS[name]
        models = cnn.load_ensemble(name)
        t0 = time.time()
        imgs = cnn.make_images(mkt, m["days"], m["horizon"], start=TEST_START, every=PREDICT_EVERY[name],
                               need_label=False)
        prob = cnn.predict(models, imgs.images)
        f = imgs.meta[["date", "symbol"]].assign(model=name, prob_up=prob)
        f["date"] = f["date"].dt.strftime("%Y-%m-%d")
        frames.append(f)
        old = old[old["model"] != name]
        print(f"{name}: {len(f):,} predictions with {len(models)} networks, {(time.time() - t0) / 60:.1f} min", flush=True)
    out = pd.concat([old] + frames, ignore_index=True).sort_values(["model", "date", "symbol"])
    out.to_csv(cnn.PRED_CACHE, index=False, compression="gzip", float_format="%.5f")


def check():
    p = pd.read_csv(cnn.PRED_CACHE)
    for name, g in p.groupby("model"):
        print(f"{name}: {len(g):,} rows, {g.date.min()} to {g.date.max()}, {g.symbol.nunique()} stocks, "
              f"mean prob {g.prob_up.mean():.3f}")
    print(sorted(x.name for x in cnn.MODEL_DIR.glob("*.pt")))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("name", choices=list(MODELS))
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--lr", type=float)
    t.add_argument("--max-epochs", type=int, default=50)
    t.add_argument("--pilot", help="label for a pilot run: history recorded, networks not saved")
    pr = sub.add_parser("predict")
    pr.add_argument("names", nargs="+", choices=list(MODELS))
    sub.add_parser("check")
    a = p.parse_args(argv)
    if a.cmd == "train":
        train(a.name, a.seeds, a.lr, a.max_epochs, a.pilot)
    elif a.cmd == "predict":
        predict(a.names)
    else:
        check()


if __name__ == "__main__":
    main()
