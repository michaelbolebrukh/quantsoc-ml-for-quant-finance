"""Workshop 5: a replication of Jiang, Kelly & Xiu (2023), "(Re-)Imag(in)ing Price Trends",
Journal of Finance 78(6), 3193-3249. Free working paper: https://ssrn.com/abstract=3756587
(section and table numbers below refer to the August 2021 working paper).

The paper's recipe:

1. Draw charts (sec. 2). Each stock's last 5, 20 or 60 days become a black and white picture:
   one OHLC bar per day, three pixels wide (open tick, high-low bar, close tick), the moving average
   of the close over the same number of days as a line, and volume bars in the bottom fifth.
   The price axis is rescaled so the highest and lowest point in the picture touch the top and
   bottom edges, and volume so the biggest day fills the volume strip. Sizes are Table 1's:
   32x15, 64x60 and 96x180 pixels.
2. Label (sec. 4.1). 1 if the stock's return over the next 5, 20 or 60 days is positive, else 0.
3. Network (sec. 3.1, Figure 7). Blocks of [5x3 convolution, batch norm, leaky ReLU, 2x1 max
   pooling]: 2 blocks for 5-day pictures, 3 for 20-day, 4 for 60-day, with 64, 128, 256, 512
   filters. The first convolution of the 20 and 60-day networks jumps 3 rows at a time and is
   dilated by 2 and 3 rows. Then 50% dropout, one fully connected layer and a softmax: the
   probability that the stock goes up.
4. Training (sec. 3.3). Cross-entropy loss, Adam with learning rate 1e-5, batches of 128,
   Xavier initial weights, pictures standardised with the training set's pixel mean and standard
   deviation, early stopping after 2 epochs without improvement in validation loss. Data: the
   first 8 years, split at random 70% training and 30% validation. Five networks trained
   independently and averaged.
5. Portfolios (sec. 4.3). Every h days sort stocks into deciles of the predicted probability,
   buy the top decile and sell the bottom one in equal weights, hold for h days.

Choices this replication makes where our setting differs, stated:
* Universe: today's S&P 100 (about 100 stocks, so a decile is about 10 stocks) instead of every
  NYSE, AMEX and NASDAQ stock (thousands). The paper's equal-weight portfolios include thousands of
  small stocks; its value-weight results (Table 4) are the closer yardstick for a mega-cap universe.
* Years: train and validate on 2000-2007 (eight years, like the paper's 1993-2000), test on
  2008 onwards with the networks never retrained.
* Prices: Yahoo Finance, adjusted closes for returns (the paper uses CRSP returns) and each day's
  open, high and low as ratios to that day's close, rescaled onto the return-built price path,
  which is the paper's construction (sec. 4.1).
* Costs: 5 basis points per dollar traded, the same as Workshops 3 and 4 (the paper deducts no
  trading costs).
* Drawing details the paper leaves open, and the reading taken here: an empty row between the
  price area and the volume strip (Table 1's rows add up to one less than the height); volume bars
  one pixel wide in the middle column; the moving average counts towards the highest and lowest
  price (sec. 6.2 says the maximum price "sometimes" is the moving average); the first days'
  moving average uses closes from before the chart; a day with no high or low still shows its
  moving average dot and volume bar.
* Network details the paper leaves open: 12 rows of padding in the strided first layer of the
  20 and 60-day networks (it makes the fully connected sizes match the paper's; other values would
  too), and early stopping keeps the best epoch's weights.
* Learning rate and training images: see workshop5/build_cache.py, which records what was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_FILE = ROOT / "workshop3" / "data" / "sp100_daily.csv.gz"
OHL_FILE = ROOT / "workshop5" / "data" / "sp100_ohl.csv.gz"
MODEL_DIR = ROOT / "workshop5" / "models"
PRED_CACHE = ROOT / "workshop5" / "data" / "predictions.csv.gz"
HISTORY_CACHE = ROOT / "workshop5" / "data" / "training_history.csv"
TRADING_DAYS = 252

# Table 1: (height, width, price rows, volume rows). Height = price rows + 1 empty row + volume rows.
SHAPES = {5: (32, 15, 25, 6), 20: (64, 60, 51, 12), 60: (96, 180, 76, 19)}
# Figure 7 / sec. 3.1: blocks, and the first layer's (vertical stride, vertical dilation).
BLOCKS = {5: 2, 20: 3, 60: 4}
FIRST_LAYER = {5: (1, 1), 20: (3, 2), 60: (3, 3)}
PAPER_PARAMS = {5: 155_138, 20: 708_866, 60: 2_952_962}       # sec. 3.1, as printed in the paper


# --------------------------------------------------------------------------- data

@dataclass
class Market:
    """Wide tables, one row per trading day and one column per stock."""
    adj_close: pd.DataFrame       # dividend and split adjusted close (for returns)
    open_r: pd.DataFrame          # open / close on the same day
    high_r: pd.DataFrame          # high / close
    low_r: pd.DataFrame           # low / close
    volume: pd.DataFrame

    @property
    def returns(self) -> pd.DataFrame:
        return self.adj_close.pct_change(fill_method=None)

    def window(self, start=None, end=None) -> "Market":
        cut = lambda d: d.loc[start:end]
        return Market(cut(self.adj_close), cut(self.open_r), cut(self.high_r), cut(self.low_r), cut(self.volume))


def load_market(stock_file=None, ohl_file=None) -> Market:
    s = pd.read_csv(stock_file or STOCK_FILE, parse_dates=["date"])
    o = pd.read_csv(ohl_file or OHL_FILE, parse_dates=["date"])
    df = s.merge(o, on=["date", "symbol"], how="left")
    w = lambda c: df.pivot(index="date", columns="symbol", values=c).sort_index()
    adj = w("adj_close")
    return Market(adj, w("open_r").reindex_like(adj), w("high_r").reindex_like(adj),
                  w("low_r").reindex_like(adj), w("volume").reindex_like(adj))


def resample_market(mkt: Market, every: int = 4) -> Market:
    """Daily bars -> bars of `every` trading days: first open, highest high, lowest low, last close,
    total volume, dated by each block's last day. Blocks are counted back from the last day, and an
    incomplete first block is dropped. This is the paper's time-scale transfer (sec. 5.1): 20 days
    drawn as 5 bars of 4 days, scored by a network trained on 5 daily bars."""
    adj = mkt.adj_close
    n = len(adj.index)
    block = (n - 1 - np.arange(n)) // every            # 0 for the last block, 1 for the one before, ...
    keep_rows = (np.bincount(block) == every)[block]
    key = block[keep_rows]
    ends = pd.Series(adj.index[keep_rows]).groupby(key).last()
    g = lambda d: d[keep_rows].groupby(key)
    close = g(adj).last()
    tables = [close, g(mkt.open_r * adj).first() / close, g(mkt.high_r * adj).max() / close,
              g(mkt.low_r * adj).min() / close, g(mkt.volume).sum(min_count=1)]
    idx = pd.DatetimeIndex(ends.loc[close.index].to_numpy(), name=adj.index.name)
    return Market(*(t.set_axis(idx).sort_index() for t in tables))


def forward_returns(adj_close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from today's close to the close `horizon` trading days later (NaN where unknown)."""
    return adj_close.shift(-horizon) / adj_close - 1


# --------------------------------------------------------------------------- drawing (sec. 2)

def _rows(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_rows: int) -> np.ndarray:
    """Price -> pixel row (0 = top). The highest price in the picture lands on row 0, the lowest on the last row."""
    span = np.where(hi > lo, hi - lo, 1.0)
    frac = (p - lo) / span
    frac = np.where(hi > lo, frac, 0.5)                      # a perfectly flat picture sits in the middle
    return (n_rows - 1) - np.rint(frac * (n_rows - 1))


def draw_images(close: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray, ma: np.ndarray,
                volume: np.ndarray, days: int, volume_bars: bool = True, ma_line: bool = True) -> np.ndarray:
    """Draw N charts at once. Every input is an (N, days) array of price LEVELS on one scale per chart
    (any scale: each chart is rescaled to fill its own picture), volume in shares.

    NaN rules (paper footnote 3): a day with no high or low gets no bar and no ticks, a missing open
    or close drops just that tick, a missing moving average breaks the line, a missing or zero volume
    draws no volume bar. Returns uint8 pictures (N, H, W) with 0 = black and 255 = white.
    """
    H, W, PR, VR = SHAPES[days]
    if not volume_bars:
        PR = H                                               # no volume strip: prices use the full height
    close, open_, high, low, ma, volume = (np.asarray(a, float).reshape(-1, days) for a in
                                           (close, open_, high, low, ma, volume))
    N = close.shape[0]
    img = np.zeros((N, H, W), dtype=bool)
    ma_used = ma if ma_line else np.full_like(ma, np.nan)
    ok_bar = np.isfinite(high) & np.isfinite(low)
    stack = np.concatenate([np.where(ok_bar, high, np.nan), np.where(ok_bar, low, np.nan), ma_used], axis=1)
    with np.errstate(all="ignore"):
        hi = np.nanmax(np.where(np.isfinite(stack), stack, -np.inf), axis=1, keepdims=True)
        lo = np.nanmin(np.where(np.isfinite(stack), stack, np.inf), axis=1, keepdims=True)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    lo = np.where(np.isfinite(lo), lo, 1.0)
    r_hi, r_lo = _rows(high, lo, hi, PR), _rows(low, lo, hi, PR)
    rows = np.arange(PR)[None, :, None]
    bar = ok_bar[:, None, :] & (rows >= r_hi[:, None, :]) & (rows <= r_lo[:, None, :])   # (N, PR, days)
    img[:, :PR, 1::3] |= bar
    n_idx, d_idx = np.nonzero(ok_bar & np.isfinite(open_))
    tick = lambda p: np.clip(_rows(p, lo, hi, PR), 0, PR - 1)    # an open or close outside high-low stays in the picture
    img[n_idx, tick(open_)[n_idx, d_idx].astype(int), 3 * d_idx] = True
    n_idx, d_idx = np.nonzero(ok_bar & np.isfinite(close))
    img[n_idx, tick(close)[n_idx, d_idx].astype(int), 3 * d_idx + 2] = True
    if ma_line:
        y = (PR - 1) - (ma_used - lo) / np.where(hi > lo, hi - lo, 1.0) * (PR - 1)   # unrounded row of each dot
        y = np.where(hi > lo, y, (PR - 1) / 2)
        ok = np.isfinite(y)
        n_idx, d_idx = np.nonzero(ok)
        img[n_idx, np.rint(y[n_idx, d_idx]).astype(int), 3 * d_idx + 1] = True
        # connect dot j (column 3j+1) to dot j+1 (column 3j+4): each column covers the rows the line
        # passes through within half a column either side, so the line has no gaps
        seg = ok[:, :-1] & ok[:, 1:]
        for k in range(4):
            x0, x1 = max(k - 0.5, 0.0) / 3.0, min(k + 0.5, 3.0) / 3.0
            ya = y[:, :-1] + x0 * (y[:, 1:] - y[:, :-1])
            yb = y[:, :-1] + x1 * (y[:, 1:] - y[:, :-1])
            top, bot = np.rint(np.minimum(ya, yb)), np.rint(np.maximum(ya, yb))
            line = seg[:, None, :] & (rows >= top[:, None, :]) & (rows <= bot[:, None, :])
            cols = 3 * np.arange(days - 1) + 1 + k
            img[:, :PR, cols] |= line
    if volume_bars:
        v = np.where(np.isfinite(volume) & (volume > 0), volume, 0.0)
        vmax = v.max(axis=1, keepdims=True)
        height = np.rint(np.where(vmax > 0, v / np.where(vmax > 0, vmax, 1.0), 0.0) * VR)  # 0..VR pixels
        vrows = np.arange(VR)[None, :, None]
        vb = vrows >= (VR - height)[:, None, :]
        img[:, H - VR:, 1::3] |= vb
    return img.astype(np.uint8) * 255


def chart_inputs(mkt: Market, days: int, ends: Sequence, symbols: Sequence[str]) -> Dict[str, np.ndarray]:
    """The (N, days) arrays that draw_images needs, for the charts ending on `ends[i]` for `symbols[i]`.

    Prices follow the paper (sec. 4.1): the close path is rebuilt from adjusted closes and scaled so
    the first close in the chart is 1; each day's open, high and low keep their ratio to that day's
    close; the moving average is the average of the previous `days` closes on the same scale.
    """
    adj = mkt.adj_close
    ma_full = adj.rolling(days, min_periods=days).mean()
    pos = adj.index.get_indexer(pd.DatetimeIndex(ends))
    col = adj.columns.get_indexer(list(symbols))
    if (pos < 0).any() or (col < 0).any():
        raise KeyError("unknown date or symbol")
    win = pos[:, None] + np.arange(-days + 1, 1)[None, :]                     # (N, days) row numbers
    A = adj.to_numpy()[win, col[:, None]]
    base = A[:, :1]
    close = A / base
    take = lambda df: df.to_numpy()[win, col[:, None]]
    return {"close": close, "open_": take(mkt.open_r) * close, "high": take(mkt.high_r) * close,
            "low": take(mkt.low_r) * close, "ma": take(ma_full) / base, "volume": take(mkt.volume)}


def eligible(mkt: Market, days: int) -> pd.DataFrame:
    """True where a chart can be drawn: the stock has a close on the chart's first and last day and a
    full moving average history before the first day (the paper drops charts that span an IPO)."""
    a = mkt.adj_close.notna()
    first_day = a.shift(days - 1, fill_value=False)
    ma_ready = a.shift(2 * days - 2, fill_value=False)     # the first day's moving average needs days-1 earlier closes
    return a & first_day & ma_ready


@dataclass
class ImageSet:
    images: np.ndarray            # (N, H, W) uint8
    meta: pd.DataFrame            # date (last day in the chart), symbol, fwd_return, label

    def __len__(self):
        return len(self.meta)


def make_images(mkt: Market, days: int, horizon: int, start=None, end=None, every: int = 1,
                offset: int = 0, need_label: bool = True, symbols: Optional[Sequence[str]] = None,
                batch: int = 20000) -> ImageSet:
    """Charts for every eligible stock on every `every`-th trading day between start and end.

    The label is 1 if the return over the next `horizon` days is positive (paper: positive vs
    non-positive). need_label=False keeps the last days, whose future is not known yet.
    """
    ok = eligible(mkt, days)
    fwd = forward_returns(mkt.adj_close, horizon)
    dates = mkt.adj_close.loc[start:end].index[offset::every]
    ok = ok.loc[dates]
    if symbols is not None:
        ok = ok[[s for s in symbols if s in ok.columns]]
    pairs = ok.stack()
    pairs = pairs[pairs]
    meta = pd.DataFrame({"date": pairs.index.get_level_values(0), "symbol": pairs.index.get_level_values(1)})
    f = fwd.stack(future_stack=True)
    meta["fwd_return"] = f.reindex(pd.MultiIndex.from_frame(meta[["date", "symbol"]])).to_numpy()
    if need_label:
        meta = meta[meta["fwd_return"].notna()].reset_index(drop=True)
    meta["label"] = (meta["fwd_return"] > 0).astype(np.int8)
    H, W, _, _ = SHAPES[days]
    out = np.zeros((len(meta), H, W), dtype=np.uint8)
    for i in range(0, len(meta), batch):
        m = meta.iloc[i:i + batch]
        out[i:i + len(m)] = draw_images(days=days, **chart_inputs(mkt, days, m["date"], m["symbol"]))
    return ImageSet(out, meta)


# --------------------------------------------------------------------------- the network (sec. 3)

def build_cnn(days: int, filters: int = 64, dropout: float = 0.5, seed: Optional[int] = None):
    """The paper's network for `days`-day pictures (Figure 7). Returns a torch nn.Sequential whose
    output is two numbers (down, up); softmax turns them into probabilities."""
    import torch
    from torch import nn
    if seed is not None:
        torch.manual_seed(seed)
    H, W, _, _ = SHAPES[days]
    stride, dil = FIRST_LAYER[days]
    layers, ch_in = [], 1
    for b in range(BLOCKS[days]):
        ch_out = filters * 2 ** b
        if b == 0 and stride > 1:     # sparse raw picture: jump 3 rows, dilate, pad so 20/60-day sizes match Figure 7
            conv = nn.Conv2d(ch_in, ch_out, (5, 3), stride=(stride, 1), dilation=(dil, 1), padding=(12, 1))
        else:
            conv = nn.Conv2d(ch_in, ch_out, (5, 3), padding=(2, 1))
        layers += [conv, nn.BatchNorm2d(ch_out), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
        ch_in = ch_out
    feats = nn.Sequential(*layers)
    with torch.no_grad():
        n_flat = feats(torch.zeros(1, 1, H, W)).numel()
    model = nn.Sequential(*layers, nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_flat, 2))
    for m in model.modules():                                # Xavier initial weights (sec. 3.3)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def block_shapes(days: int) -> List[Tuple[int, int, int]]:
    """(channels, height, width) after each building block (computed, not taken from the paper)."""
    import torch
    model = build_cnn(days)
    x, out = torch.zeros(1, 1, *SHAPES[days][:2]), []
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if layer.__class__.__name__ == "MaxPool2d":
                out.append(tuple(int(s) for s in x.shape[1:]))
    return out


@dataclass
class Trained:
    """One fitted network plus what is needed to use it on new pictures."""
    days: int
    horizon: int
    model: object
    pixel_mean: float
    pixel_std: float
    history: pd.DataFrame = field(default_factory=pd.DataFrame)   # epoch, train_loss, val_loss, val_acc
    seed: int = 0


def _device(device=None):
    import torch
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(images: np.ndarray, labels: np.ndarray, days: int, horizon: int, seed: int = 0,
              lr: float = 1e-5, batch_size: int = 128, max_epochs: int = 50, patience: int = 2,
              val_frac: float = 0.3, device=None, verbose: bool = True, max_seconds: Optional[float] = None) -> Trained:
    """Train one network the paper's way (sec. 3.3): random 70/30 split, Adam, cross-entropy,
    early stopping after `patience` epochs without a lower validation loss, keeping the best epoch."""
    import time
    import torch
    from torch import nn
    dev = _device(device)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n_val = int(round(val_frac * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    mean = float(images[tr_idx].mean())
    std = float(images[tr_idx].std()) or 1.0
    model = build_cnn(days, seed=seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    def batches(ix, shuffle):
        if shuffle:
            ix = ix[torch.randperm(len(ix), generator=gen).numpy()]
        for i in range(0, len(ix), batch_size):
            j = np.sort(ix[i:i + batch_size])
            x = torch.from_numpy(images[j]).float().unsqueeze(1).sub_(mean).div_(std).to(dev)
            y = torch.from_numpy(labels[j].astype(np.int64)).to(dev)
            yield x, y

    def evaluate(ix):
        model.eval()
        tot, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in batches(ix, False):
                out = model(x)
                tot += float(loss_fn(out, y)) * len(y)
                correct += int((out.argmax(1) == y).sum())
                n += len(y)
        return tot / n, correct / n

    best, best_state, bad, rows, t0 = np.inf, None, 0, [], time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in batches(tr_idx, True):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        val_loss, val_acc = evaluate(val_idx)
        rows.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "val_acc": val_acc,
                     "seconds": time.time() - t0})
        if verbose:
            print(f"  seed {seed} epoch {epoch}: train loss {tot / n:.4f}, validation loss {val_loss:.4f}, "
                  f"validation accuracy {val_acc:.1%} ({time.time() - t0:.0f}s)", flush=True)
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if max_seconds and time.time() - t0 > max_seconds:
            break
    if best_state is None:
        raise RuntimeError("validation loss was never finite; nothing to keep")
    model.load_state_dict(best_state)
    model.eval()
    return Trained(days, horizon, model.cpu(), mean, std, pd.DataFrame(rows), seed)


def predict(models: Sequence[Trained], images: np.ndarray, batch_size: int = 1024, device=None) -> np.ndarray:
    """Probability of 'up' for each picture, averaged over the networks (the paper's ensemble of five)."""
    import torch
    dev = _device(device)
    out = np.zeros(len(images))
    for t in models:
        model = t.model.to(dev).eval()
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                x = torch.from_numpy(images[i:i + batch_size]).float().unsqueeze(1)
                x = ((x - t.pixel_mean) / t.pixel_std).to(dev)     # never modifies the caller's array
                out[i:i + batch_size] += torch.softmax(model(x), 1)[:, 1].cpu().numpy()
        t.model.cpu()
    return out / len(models)


def save_trained(t: Trained, path: Union[str, Path]) -> Path:
    """Weights stored as 16-bit floats (half the size); load_trained turns them back into 32-bit."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in t.model.state_dict().items()}
    torch.save({"days": t.days, "horizon": t.horizon, "pixel_mean": t.pixel_mean, "pixel_std": t.pixel_std,
                "seed": t.seed, "state": state, "history": t.history.to_dict("list")}, path)
    return path


def load_trained(path: Union[str, Path]) -> Trained:
    import torch
    d = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = build_cnn(d["days"])
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in d["state"].items()})
    model.eval()
    return Trained(d["days"], d["horizon"], model, d["pixel_mean"], d["pixel_std"],
                   pd.DataFrame(d["history"]), d["seed"])


def load_ensemble(name: str, model_dir=None) -> List[Trained]:
    """The cached networks for a model name such as 'I5R5' (files I5R5_seed0.pt ... in workshop5/models)."""
    paths = sorted(Path(model_dir or MODEL_DIR).glob(f"{name}_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"no cached networks for {name} in {model_dir or MODEL_DIR}")
    return [load_trained(p) for p in paths]


# --------------------------------------------------------------------------- looking inside (sec. 6.3)

def grad_cam(t: Trained, image: np.ndarray, block: int = -1, target: int = 1) -> np.ndarray:
    """Grad-CAM heatmap (Selvaraju et al. 2017) for one picture at the output of one building block:
    which pixels pushed the network towards `target` (1 = up, 0 = down). Returns an (H, W) array in [0, 1]."""
    import torch
    import torch.nn.functional as F
    model = t.model.eval()
    pools = [i for i, m in enumerate(model) if m.__class__.__name__ == "MaxPool2d"]
    cut = pools[block] - 1                   # the leaky ReLU just before that block's pooling
    x = torch.from_numpy(image[None, None].astype(np.float32)).sub_(t.pixel_mean).div_(t.pixel_std)
    feats = x
    for m in model[:cut + 1]:
        feats = m(feats)
    feats = feats.detach().requires_grad_(True)
    out = feats
    for m in model[cut + 1:]:
        out = m(out)
    out[0, target].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape, mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


# --------------------------------------------------------------------------- benchmark signals (sec. 4.2)

def benchmark_signals(adj_close: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """The paper's three price-trend benchmarks, signed so that a HIGH value means 'expected to go up':
    MOM = return from 12 months to 1 month ago, STR = minus the last month's return (reversal),
    WSTR = minus the last week's return."""
    p = adj_close
    return {"MOM": p.shift(21) / p.shift(252) - 1, "STR": -(p / p.shift(21) - 1), "WSTR": -(p / p.shift(5) - 1)}


# --------------------------------------------------------------------------- portfolios (sec. 4.3)

@dataclass
class DecileBacktest:
    """A decile long/short strategy rebalanced every `horizon` days and marked to market daily."""
    horizon: int
    period: pd.DataFrame          # one row per rebalance: date, long, short, spread, n, turnover, cost, net
    daily_gross: pd.Series        # daily P&L per $1 of capital ($1 long and $1 short), before costs
    daily_net: pd.Series
    deciles: pd.DataFrame         # holding-period return of each decile (1 = lowest score), per rebalance
    bars_per_year: float = TRADING_DAYS   # 52 for weekly bars

    def paper_stats(self) -> Dict[str, float]:
        """The paper's way of reporting: one return per holding period, annualised by periods per year."""
        k = self.bars_per_year / self.horizon
        out = {}
        for name in ("spread", "net"):
            r = self.period[name]
            out[f"annual return ({'before' if name == 'spread' else 'after'} costs)"] = float(r.mean() * k)
            out[f"Sharpe ({'before' if name == 'spread' else 'after'} costs)"] = float(r.mean() / r.std() * np.sqrt(k))
        out["turnover per rebalance"] = float(self.period["turnover"].mean())
        out["rebalances"] = int(len(self.period))
        return out

    def window(self, start=None, end=None) -> "DecileBacktest":
        """The rebalances dated from `start` to `end`, with the daily series cut to exactly the days those
        rebalances held (so the daily P&L still adds up to the periods' spreads)."""
        if self.period.empty:
            return self
        p = self.period[(self.period["date"] >= pd.Timestamp(start or "1900")) &
                        (self.period["date"] <= pd.Timestamp(end or "2100"))].reset_index(drop=True)
        if p.empty:
            empty = pd.Series(dtype=float)
            return DecileBacktest(self.horizon, p, empty, empty, self.deciles.iloc[:0], self.bars_per_year)
        a, z = p["first_day"].iloc[0], p["last_day"].iloc[-1]
        return DecileBacktest(self.horizon, p, self.daily_gross.loc[a:z], self.daily_net.loc[a:z],
                              self.deciles.loc[p["date"]], self.bars_per_year)


def decile_backtest(scores: pd.DataFrame, adj_close: pd.DataFrame, horizon: int, start=None, end=None,
                    n_deciles: int = 10, cost_bps: float = 5.0, bars_per_year: float = TRADING_DAYS) -> DecileBacktest:
    """Every `horizon` trading days from `start`: rank the stocks with a score that day, put the top
    decile long and the bottom decile short with $1 each, equal weights, and hold them for `horizon`
    days (buy and hold, so weights drift). Costs: cost_bps per dollar traded at each rebalance,
    measured from the drifted old book to the new one.

    `scores` is wide (dates x symbols); only its rows on rebalance dates are read, so a score for date
    t may use data up to and including the close of day t, and the trade happens at that close.

    Missing prices: a stock with no price for a day is held at its last price (forward-filled), so
    a gap's move is earned when the price comes back. A date with too few scored stocks is skipped;
    the previous book is then treated as closed with no cost charged (never happens in our data).
    """
    R = adj_close.ffill().pct_change(fill_method=None)
    dates = adj_close.loc[start:end].index
    rebal = dates[::horizon]
    pos = adj_close.index.get_indexer(rebal)
    rows, dec_rows, gross = [], [], {}
    old = pd.Series(dtype=float)
    for t, p in zip(rebal, pos):
        if p + horizon >= len(adj_close.index):
            break                                            # the holding period is not over yet
        s = scores.loc[t].dropna() if t in scores.index else pd.Series(dtype=float)
        s = s[adj_close.loc[t, s.index].notna()]
        if len(s) < n_deciles * 2:
            old = pd.Series(dtype=float)
            continue
        dec = pd.qcut(s.rank(method="first"), n_deciles, labels=False) + 1
        hold = R.iloc[p + 1:p + horizon + 1]                   # the next `horizon` days
        growth = (1 + hold.fillna(0.0)).cumprod()              # a stock with no price for a day is held flat
        longs, shorts = dec.index[dec == n_deciles], dec.index[dec == 1]
        wl = pd.Series(1.0 / len(longs), longs)
        ws = pd.Series(-1.0 / len(shorts), shorts)
        new = pd.concat([wl, ws])
        turnover = float(new.sub(old, fill_value=0.0).abs().sum())
        L = growth[longs].mean(axis=1)                         # value of $1 spread equally across the longs
        S = growth[shorts].mean(axis=1)
        pnl = (L - S).diff()
        pnl.iloc[0] = (L.iloc[0] - 1) - (S.iloc[0] - 1)
        for d, v in pnl.items():
            gross[d] = gross.get(d, 0.0) + v
        cost = turnover * cost_bps / 1e4
        spread = float(L.iloc[-1] - S.iloc[-1])
        rows.append({"date": t, "long": float(L.iloc[-1] - 1), "short": float(S.iloc[-1] - 1), "spread": spread,
                     "n": int(len(s)), "turnover": turnover, "cost": cost, "net": spread - cost,
                     "first_day": hold.index[0], "last_day": hold.index[-1]})
        h_ret = growth.iloc[-1] - 1
        dec_rows.append(h_ret[dec.index].groupby(dec).mean().rename(t))
        # what the book holds at the end of the period, per $1 of capital, before the next rebalance
        old = pd.concat([wl * growth[longs].iloc[-1], ws * growth[shorts].iloc[-1]])
    period = pd.DataFrame(rows)
    daily_gross = pd.Series(gross).sort_index()
    costs = pd.Series(period.set_index("date")["cost"]) if len(period) else pd.Series(dtype=float)
    # charge each rebalance's cost on the first day of its holding period
    first_days = {t: adj_close.index[adj_close.index.get_loc(t) + 1] for t in costs.index}
    daily_net = daily_gross.copy()
    for t, c in costs.items():
        daily_net.loc[first_days[t]] -= c
    deciles = pd.DataFrame(dec_rows)
    return DecileBacktest(horizon, period, daily_gross, daily_net, deciles, bars_per_year)


def period_accuracy(scores: pd.Series, fwd: pd.Series, threshold: Optional[float] = 0.5) -> pd.Series:
    """Share of stocks whose direction was called right, one number per date (Table 2's 'Acc.').
    threshold=None calls 'up' when the score is above that date's cross-sectional median, which is
    how the paper scores MOM, STR and WSTR."""
    df = pd.DataFrame({"s": scores, "r": fwd}).dropna()
    if threshold is None:
        up = df["s"] > df.groupby(level="date")["s"].transform("median")
    else:
        up = df["s"] > threshold
    hit = (up == (df["r"] > 0)).astype(float)
    return hit.groupby(level="date").mean()


def period_corr(scores: pd.Series, fwd: pd.Series, method: str = "pearson") -> pd.Series:
    """Correlation of score and realised return across stocks, one number per date. Table 2's 'Corr.'
    is Pearson; Table 18 reports both Pearson and Spearman (rank)."""
    if method == "spearman":
        return period_rank_corr(scores, fwd)
    df = pd.DataFrame({"s": scores, "r": fwd}).dropna()
    g = df.groupby(level="date")
    return g.apply(lambda d: d["s"].corr(d["r"]) if len(d) >= 10 else np.nan).dropna()


def period_rank_corr(scores: pd.Series, fwd: pd.Series) -> pd.Series:
    """Spearman rank correlation of score and realised return across stocks, one number per date."""
    df = pd.DataFrame({"s": scores, "r": fwd}).dropna()
    g = df.groupby(level="date")
    return g.apply(lambda d: d["s"].rank().corr(d["r"].rank()) if len(d) >= 10 else np.nan).dropna()


# --------------------------------------------------------------------------- cached results

def load_predictions(name: str, path=None) -> pd.Series:
    """Out-of-sample probabilities from workshop5/build_cache.py, indexed (date, symbol)."""
    d = pd.read_csv(path or PRED_CACHE, parse_dates=["date"])
    d = d[d["model"] == name]
    if d.empty:
        raise ValueError(f"no cached predictions for {name!r}")
    return d.set_index(["date", "symbol"])["prob_up"].sort_index()


def live_scores(mkt: Market, models: Sequence[Trained], date=None) -> pd.DataFrame:
    """Probability of 'up' for every stock with a full chart on `date` (default: the last day in mkt)."""
    days = models[0].days
    date = pd.Timestamp(date) if date is not None else mkt.adj_close.index[-1]
    ok = eligible(mkt, days).loc[date]
    syms = list(ok.index[ok])
    imgs = draw_images(days=days, **chart_inputs(mkt, days, [date] * len(syms), syms))
    prob = predict(models, imgs)
    out = pd.DataFrame({"symbol": syms, "prob_up": prob}).sort_values("prob_up", ascending=False)
    out["decile"] = pd.qcut(out["prob_up"].rank(method="first"), 10, labels=False) + 1
    return out.reset_index(drop=True)
