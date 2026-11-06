"""Generate workshop5/workshop5.ipynb from the cells below (outputs cleared).

Usage:  python workshop5/build_notebook.py
Edit the text here rather than in the notebook, so the two never drift apart.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "workshop5.ipynb"
CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# ------------------------------------------------------------------------------------------- intro
md("""
# Can a neural network read a stock chart?
### QuantSoc · Workshop 5 · [quant-soc.com](https://quant-soc.com)

Two workshops, one question: does a stock's recent price path say anything about where it goes next?
Workshop 3 fed 31 past returns to machine learning models. Workshop 4 modelled what is left after the market as a
mean-reverting stochastic process. Today we do what a chart-reading trader does: **draw the chart and look at it**.
A convolutional neural network (CNN), the kind of model that recognises faces and road signs, gets the picture
and nothing else.

> **Reference.** Jiang, J., Kelly, B. and Xiu, D. (2023). *(Re-)Imag(in)ing Price Trends.* Journal of Finance
> 78(6), 3193-3249. [Published version (free to read)](https://doi.org/10.1111/jofi.13268) ·
> [working paper on SSRN](https://ssrn.com/abstract=3756587). Section and table numbers below refer to the
> August 2021 working paper; the published tables may differ slightly.

| Part | What we do |
|---|---|
| 0 | Setup |
| 1 | The paper in six lines |
| 2 | Data: adding the open, high and low |
| 3 | Drawing a chart, pixel by pixel |
| 4 | The question we ask: will it go up? (and a game: can you tell?) |
| 5 | How a CNN sees: convolution, activation, pooling |
| 6 | The paper's network |
| 7 | Training |
| 8 | How often is it right? |
| 9 | Portfolios |
| 10 | What is it looking at? |
| 11 | Three workshops, one question |
| 12 | Why is it so much weaker than the paper? |
| 13 | Your turn |
| 14 | Going live |

> **Disclosure.** Prices are real (Yahoo Finance via `yfinance`) but the universe is **today's** S&P 100, so it is
> survivorship-biased: companies that failed or left the index are missing. The paper used every US stock in CRSP.
> Teaching only, not investment advice.
""")

# ------------------------------------------------------------------------------------------- setup
md("""
## 0. Setup

Run the next two cells. The first downloads the workshop files (Colab) or finds them on your computer.
Everything runs on a normal CPU. Only the optional training in part 7 is faster with a GPU
(in Colab: *Runtime, Change runtime type, T4 GPU*).
""")
code('''
# @title Setup (run once; safe to run again)
OWNER, REPO, REVISION = "michaelbolebrukh", "quantsoc-ml-for-quant-finance", "main"

import io, shutil, sys, urllib.request, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
FILES = ["quantsoc_ml", "workshop3", "workshop4", "workshop5", "requirements.txt", ".env.example"]


def fetch(root: Path):
    """Download the branch every time: a branch moves, so files from an earlier run may be out of date."""
    url = f"https://github.com/{OWNER}/{REPO}/archive/{REVISION}.zip"
    print("Downloading", url)
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())) as zf:
        tmp = root.parent / "_workshop_src"
        shutil.rmtree(tmp, ignore_errors=True)
        zf.extractall(tmp)
    src = tmp / zf.namelist()[0].split("/")[0]
    root.mkdir(parents=True, exist_ok=True)
    for item in FILES:
        s, d = src / item, root / item
        if s.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(s, d)
        elif s.exists():
            shutil.copy(s, d)
    print("Done.")


if IN_COLAB:
    ROOT = Path("/content/workshop")
    fetch(ROOT)
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in [here, *here.parents]
                 if (p / "quantsoc_ml" / "cnn.py").exists() and (p / "workshop5").is_dir()), None)
    if ROOT is None:
        raise SystemExit("Cannot find the workshop folder (it holds quantsoc_ml/ and workshop5/).")
for p in (ROOT, ROOT / "workshop5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
print("Workshop folder:", ROOT)
''')
code('''
%matplotlib inline
import sys, time, warnings
for name in [m for m in sys.modules if m.split(".")[0] in ("quantsoc_ml", "live", "workshop3", "workshop4", "workshop5")]:
    del sys.modules[name]                       # re-running this cell picks up freshly downloaded code
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
try:
    import torch
except ImportError:
    raise SystemExit("This workshop needs PyTorch: pip install torch (Colab already has it).")

from quantsoc_ml import cnn
from quantsoc_ml import ws3_plots as plots3
from quantsoc_ml import ws5_plots as plots
from quantsoc_ml.statarb import perf_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)
print(f"Ready. PyTorch {torch.__version__}, running on {DEVICE}.")
''')

# ------------------------------------------------------------------------------------------- 1 paper
md("""
## 1. The paper in six lines

1. **Pictures** (sec. 2). The last 5, 20 or 60 trading days of a stock become a small black and white chart:
   one bar per day for the open, high, low and close, a moving average line, and volume bars underneath.
2. **Question** (sec. 4.1). Will the stock's return over the next 5, 20 or 60 days be positive? A yes/no label.
3. **Model** (sec. 3). A CNN: a stack of small pattern detectors that slide over the picture, then one final
   layer that turns what they found into a probability of "up".
4. **Training** (sec. 3.3). Once, on 1993 to 2000: 70% of the pictures to learn from, 30% to decide when to stop.
   Five networks, averaged. Never retrained afterwards.
5. **Test** (sec. 4.1 and 4.3). 2001 to 2019. Sort stocks by their probability of going up, buy the top 10%, sell the
   bottom 10%, hold for the forecast horizon.
6. **Result** (Tables 2 to 6). Only 52.5% to 53.6% of directions called right, yet the 20-day picture strategy has
   an annualised Sharpe ratio of **2.16** (equal weights, monthly, no trading costs deducted), against 0.25 for
   momentum and 1.23 for one-week reversal. With **value** weights, which put most money in big companies, it
   falls to **0.49**.

The value-weight number is the better yardstick for us, because our 100 stocks are all very big companies.
Equal weights give a tiny company as much money as Apple, and thousands of small stocks sit in the paper's
equal-weight portfolios. (One more hint, to read with care: when the paper *trained* only on the 2,000 largest
stocks, still testing on all of them, the equal-weight Sharpe ratio fell from 2.16 to 0.44, while value-weight
barely moved, 0.49 to 0.42; for the 5-day pictures we use, 2.35 to 1.16 and 0.45 to 0.64. The paper itself
concludes there is no benefit to restricting training to large stocks; Table 17.)
""")

# ------------------------------------------------------------------------------------------- 2 data
md("""
## 2. Data: adding the open, high and low

Same stocks and same days as Workshops 3 and 4, plus each day's open, high and low, stored as ratios to that
day's close (ratios survive stock splits).
""")
code('''
mkt = cnn.load_market()
print(f"{mkt.adj_close.shape[1]} stocks, {mkt.adj_close.index[0].date()} to {mkt.adj_close.index[-1].date()}")
ok = mkt.adj_close.notna()
bad = ((mkt.high_r < np.maximum(mkt.open_r, 1) - 1e-4) | (mkt.low_r > np.minimum(mkt.open_r, 1) + 1e-4))[ok]
flat = (mkt.high_r - mkt.low_r).abs()[ok] < 1e-9
print(f"bars where the high is below the open or close, or the low above them: {int(bad.sum().sum())}")
print(f"days with no price range at all (high = low): {int(flat.sum().sum())}, mostly", flat.sum().idxmax())
print(f"stock-days with a close but no open, high or low: {int((mkt.high_r.isna() & ok).sum().sum())}")
''')
md("""
**The paper's price path** (sec. 4.1). Inside each picture the first close is set to 1 and every later close is
built from returns, $p_{t+1} = (1 + r_{t+1})\\,p_t$, so dividends and splits do not show up as jumps. Each day's
open, high and low keep their ratio to that day's close. We do exactly that, with adjusted closes standing in for
CRSP returns.
""")

# ------------------------------------------------------------------------------------------- 3 drawing
md("""
## 3. Drawing a chart, pixel by pixel

The rules (sec. 2.1 and 2.2):

* One day is **three pixels wide**: a tick on the left for the open, a vertical bar from low to high in the
  middle, a tick on the right for the close.
* The price axis is stretched so the highest and lowest points in the picture touch the top and bottom edges.
  Every picture of the same length has the same size, whatever the stock's price.
* The **moving average** of the close (over as many days as the picture shows) is a one-pixel line through the
  middle columns.
* **Volume** bars fill the bottom fifth, scaled so the busiest day reaches the top of that strip.
* Black background, white marks. No colour: the open and close ticks already say whether the day went up.

First, the usual chart of Apple's last 20 days, then the same 20 days as the network sees them.
""")
code('''
SYMBOL, END = "AAPL", mkt.adj_close.index[-1]
x = cnn.chart_inputs(mkt, 20, [END], [SYMBOL])
dates = mkt.adj_close.loc[:END].index[-20:]
plots.ohlc_chart(*(x[k][0] for k in ("close", "open_", "high", "low", "ma", "volume")), dates=dates,
                 title=f"{SYMBOL}, 20 days to {END.date()} (price scaled so the first close is 1)");
pics = {d: cnn.draw_images(days=d, **cnn.chart_inputs(mkt, d, [END], [SYMBOL]))[0] for d in (5, 20, 60)}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [15, 60, 180]})
for ax, (d, img) in zip(axes, pics.items()):
    plots.show(img, ax, f"{d} days: {img.shape[0]} x {img.shape[1]} pixels")
''')
md("""
These are Table 1's sizes: 32x15, 64x60 and 96x180. A picture really is just a grid of numbers. Here is the
5-day one, with `#` for white (255) and `.` for black (0):
""")
code('''
print(plots.matrix(pics[5]))
print(f"shape {pics[5].shape}, {int((pics[5] > 0).sum())} white pixels out of {pics[5].size}")
''')
md("""
The paper also tried leaving out the volume bars and the moving average (Figure 3, Table 17). Here is what each
version looks like:
""")
code('''
x20 = cnn.chart_inputs(mkt, 20, [END], [SYMBOL])
variants = {"no volume, no MA": dict(volume_bars=False, ma_line=False), "no volume, MA": dict(volume_bars=False),
            "volume, no MA": dict(ma_line=False), "volume and MA (the paper's choice)": {}}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (name, kw) in zip(axes, variants.items()):
    plots.show(cnn.draw_images(days=20, **x20, **kw)[0], ax, name)
''')

# ------------------------------------------------------------------------------------------- 4 labels
md("""
## 4. The question we ask: will it go up?

A label is 1 if the stock's return over the next $h$ days is positive, 0 otherwise. Two models are cached:

* **I5R5**: 5-day pictures, is the next **week** up? (the paper's strongest result, Table 6)
* **I5R20**: the same 5-day pictures, is the next **month** (20 trading days) up? (Tables 3 and 4: the
  paper's best equal-weight monthly Sharpe ratio, 2.35; value weight 0.45)

The paper's headline 20-day pictures (I20R20) are drawn and built in parts 3 and 6, but not cached: see part 7.

Training uses pictures from 2000 to 2007 only, and only those whose label is also known by the end of 2007, so
no price from the test years leaks in.
""")
code('''
train5 = cnn.make_images(mkt.window("2000-01-01", "2007-12-31"), days=5, horizon=5)
print(f"{len(train5):,} five-day pictures from 2000 to 2007; {train5.meta.label.mean():.1%} of them are followed by an up week")
display(train5.meta.head())
''')
md("""
More than half the labels are "up". Part of that is the stock market rising on average, and part is
**survivorship**: every one of these companies made it to 2026, so their past weeks were better than a typical
stock's. A model can score above 50% just by saying "up" a lot, which is why part 8 also looks at a score that
only cares about ranking stocks against each other.
""")
md("""
**A game: can you tell?** Below are 16 real 20-day charts from the training years. Half were followed by a rising
month, half by a falling one. Write down your guesses (top row first), then run the next cell for the answers.
""")
code('''
rng = np.random.default_rng(5)
train20 = cnn.make_images(mkt.window("2000-01-01", "2007-12-31"), days=20, horizon=20, every=10)
pick = np.concatenate([rng.choice(np.flatnonzero(train20.meta.label == v), 8, replace=False) for v in (1, 0)])
pick = rng.permutation(pick)
plots.gallery(train20.images[pick], [f"#{i + 1}" for i in range(16)], suptitle="Up or down next month?");
''')
code('''
m = train20.meta.iloc[pick]
print("answers:", " ".join(f"#{i + 1}:{'UP' if l else 'down'}" for i, l in enumerate(m.label)))
display(m.assign(month_return=(m.fwd_return * 100).round(1)).drop(columns=["fwd_return"]).reset_index(drop=True))
''')
md("""
If the room scored about 50%, that is the honest baseline. The paper's networks get 52.5% to 53.6% (Table 2). That
sounds tiny. But a long/short book does not bet on every stock, only on the 10% the model rates highest and the
10% it rates lowest, so what matters is whether the extremes are ranked well (part 9). The paper's own appendix
makes a related point with a market-timing example: every extra 0.5% of accuracy adds about 0.1 to an annual
Sharpe ratio.
""")

# ------------------------------------------------------------------------------------------- 5 CNN
md("""
## 5. How a CNN sees: convolution, activation, pooling

A CNN is built from one repeated **building block** of three steps (sec. 3.1).

**Convolution.** A small grid of weights, a *filter*, slides over the picture. At each position it multiplies
the pixels underneath by its weights and adds them up. A large result means "my pattern is here". The weights
are not chosen by hand: training finds them. This is the paper's own example (Figure 5): a 6x6 picture, white in
the top-left 4x4, and two 3x3 filters, one for vertical edges and one for horizontal edges.
""")
code('''
image = np.zeros((6, 6)); image[:4, :4] = 255
filters = {"filter 1 (vertical edges)": np.array([[1, 0, -1]] * 3, float),
           "filter 2 (horizontal edges)": np.array([[1, 0, -1]] * 3, float).T}
x = torch.tensor(image, dtype=torch.float32)[None, None]
outputs, pooled = {}, {}
for name, f in filters.items():
    out = torch.nn.functional.conv2d(x, torch.tensor(f, dtype=torch.float32)[None, None])
    outputs[name] = out[0, 0].numpy()
    pooled[name] = torch.nn.functional.max_pool2d(out, 2)[0, 0].numpy()
plots.conv_demo(image, filters, outputs, pooled);
''')
md("""
Filter 1 lights up (765 = 3 x 255) where white meets black on the right; filter 2 where it meets black below.
After **max pooling** (keep only the largest number in each 2x2 square) the paper's numbers appear: 765 in the
top-right and 510 in the bottom-right for filter 1. Pooling shrinks the picture and keeps "was the pattern
anywhere near here?", which makes the network tolerant to a pattern moving by a pixel.

**Activation.** Between the two, every number goes through a *leaky ReLU*: positive numbers pass, negative ones
are multiplied by 0.01. Without a bend like this, stacking layers would still only give a linear model.

**Batch normalisation** (between convolution and activation) rescales each filter's output to a stable range
during training, which usually makes training faster and steadier. It is not essential here: the paper's Table 18
finds results barely change without it.

Now a real chart through the **first layer of a trained network**: 64 filters, each producing its own map of
where its pattern appears. Eight of them:
""")
code('''
nets5 = cnn.load_ensemble("I5R5")
net = nets5[0].model.eval()
img = pics[5]
xin = torch.tensor(img, dtype=torch.float32)[None, None].sub(nets5[0].pixel_mean).div(nets5[0].pixel_std)
with torch.no_grad():
    maps = net[2](net[1](net[0](xin)))[0].numpy()        # convolution, batch norm, leaky ReLU
fig, ax = plt.subplots(figsize=(2, 3)); plots.show(img, ax, "input")
plots.feature_maps(maps, 8, "First-layer feature maps: each filter's response at every pixel");
''')

# ------------------------------------------------------------------------------------------- 6 network
md("""
## 6. The paper's network

Choices from sec. 3.1 and Figure 7:

* Filters are **5x3** and pooling is **2x1**, "since our images are largely sparse in the vertical dimension"
  (sec. 3.1). Pooling halves the height and never the width, so the network keeps every day's column.
* **2, 3 or 4 blocks** for 5, 20 or 60-day pictures, with **64, 128, 256, 512** filters.
* In the 20 and 60-day networks the first layer jumps **3 rows** at a time and is **dilated** (it looks at every
  2nd or 3rd row), because the raw picture is mostly black.
* Then **50% dropout** (randomly switch off half the inputs while training, so no single feature is relied on),
  one fully connected layer, and a softmax that turns two numbers into P(up).

Built from those rules, the parameter counts land exactly on the paper's printed totals. Two small honesty notes:
the paper does not say how much padding its first strided layer uses (we pick 12 rows, which makes the final
height match the paper's fully connected sizes, 15,360, 46,080 and 184,320 inputs; other choices also would), and
its totals only add up if batch normalisation's 2 numbers per filter are counted, although footnote 9 says they
are ignored.
""")
code('''
rows = []
for d in (5, 20, 60):
    model = cnn.build_cnn(d)
    rows.append({"picture": f"{d} days", "size": "x".join(map(str, cnn.SHAPES[d][:2])),
                 "after each block (channels x height x width)": "  ".join("x".join(map(str, s)) for s in cnn.block_shapes(d)),
                 "parameters (ours)": f"{cnn.count_params(model):,}", "parameters (paper)": f"{cnn.PAPER_PARAMS[d]:,}"})
display(pd.DataFrame(rows))
print(cnn.build_cnn(20))
''')
md("""
*Discuss:* the 5-day network has about 155,000 parameters and the 20-day one about 700,000. Our training set
has about 170,000 five-day pictures, and the paper's had millions. What stops a model with more knobs than examples from just memorising them? (Hint:
the weight sharing of convolution, dropout, and stopping early. The paper says the *effective* number of
parameters is much smaller than the count.)
""")

# ------------------------------------------------------------------------------------------- 7 training
md("""
## 7. Training

The recipe (sec. 3.3): cross-entropy loss, the Adam optimiser, batches of 128 pictures, Xavier starting weights,
pictures standardised by the training set's pixel mean and spread, and **early stopping**: after each pass over
the data (an *epoch*), check the loss on the 30% validation pictures and stop once it has not improved for two
epochs in a row (we keep the best epoch's weights; the paper does not say). Five networks from five random
starts, averaged.

**What did not work: the 20-day network.** Its pictures are 16 times bigger, and on the laptop CPU this cache was
built on it trains at well under 100 pictures a second, so we tried it on one picture every 5 days per stock (33,418
pictures for 708,866 parameters). It never learned: its validation loss stayed above that of simply always
predicting the training years' 57% share of "up", at either learning rate we tried. With a fifth of the data it
had too little to learn from, so the monthly model here is I5R20. Exercise E is to do better on a GPU.

**I5R20 is only a tie, and you should know that before part 8.** Its best validation loss matches that same
"always say 57% up" line (0.683 to 0.685 against 0.683 to 0.685, seed by seed), and at its best epoch it calls
almost every validation picture "up". So its accuracy tells you nothing. It is kept for one reason: a long/short
book only needs the *ranking*, which stock is more likely to rise than another, and a network can rank a little
even while every probability sits above 50%. Whether it ranks usefully is a question for the test years, in parts 8
and 9. I5R5 does slightly better on validation: every one of its five networks beats its own "always say 53%"
line, by 0.001 to 0.002.

Training all of that took a while on a laptop CPU, so the networks are cached (`workshop5/build_cache.py`
trained them with the code you see here). Their training curves:
""")
code('''
hist = pd.read_csv(cnn.HISTORY_CACHE)
train5m = cnn.make_images(mkt.window("2000-01-01", "2007-12-31"), days=5, horizon=20, every=5)
up_share = {"I5R5": train5.meta.label.mean(), "I5R20": train5m.meta.label.mean()}
for name in ("I5R5", "I5R20"):
    h = hist[(hist.model == name) & (hist.run == "final")]
    plots.training_curves(h, f"{name}: five networks, {int(h.n_pictures.iloc[0]):,} pictures from 2000 to 2007, "
                             f"learning rate {h.lr.iloc[0]:g}", up_share[name])
''')
md("""
Two things to notice. The validation loss barely gets below the red lines for I5R5, and only ties the dotted
line for I5R20 (the line is computed from every 5th day's pictures, so it is a close approximation to each
network's own validation split): **ln 2 = 0.693** is the loss of a
model that always says 50/50, and the dotted line is a model that always says the training years' share of "up"
(57% for months, 53% for weeks), which knows nothing about any one stock. Accuracy sits close to that share too. Stock returns are mostly noise;
the paper's own validation loss for its 20-day model was 0.687 (Table 18). A little signal in a lot of noise is the whole game.

**One change from the paper: the learning rate.** The paper uses 1e-5 (how big a step the optimiser takes).
We have about 100 stocks instead of thousands, so far fewer steps per epoch, and at 1e-5 our networks were
still worse than a coin flip after two epochs. Pilot runs compared 1e-5, 1e-4 and 1e-3 on the 2000 to 2007
**validation** pictures only, and 1e-3 did best for I5R5, so I5R20 uses it too. One honest caveat, shared with the paper:
validation pictures come from the same days and stocks as training pictures, so they are not fully independent,
and early stopping on them is a little optimistic. The 2008 onwards test is untouched either way.

**Train one yourself.** This trains a fresh 5-day network on a sample of the training pictures. Expect a minute
or two per epoch on Colab's CPU and seconds on a GPU. Do not expect much. The cached networks also started worse
than a coin flip, or barely better (first validation losses between 0.70 and 0.77), and needed several epochs over 120,000 pictures to settle;
with 20,000 pictures and 3 epochs, yours will probably still be jumping about, with accuracy near 50%. That is the
point: watch the training loss fall while the validation loss refuses to follow, then try `N = 100_000` on a GPU.
""")
code('''
N, EPOCHS = 20_000, 3
sub = np.random.default_rng(0).choice(len(train5), N, replace=False)
mine = cnn.train_cnn(train5.images[sub], train5.meta.label.to_numpy()[sub], days=5, horizon=5, seed=42,
                     lr=1e-3, max_epochs=EPOCHS, device=DEVICE)
display(mine.history.round(4))
''')

# ------------------------------------------------------------------------------------------- 8 accuracy
md("""
## 8. How often is it right?

From here on everything is **out of sample**: 2008 to October 2026, years the networks never saw. Table 2 of the
paper scores each model by its **accuracy** (share of stocks whose direction it called right, with "up" meaning a
probability above 50%) and by the **correlation** between its forecasts and the returns that followed, both
averaged over dates. Table 2's correlation is the ordinary (Pearson) one; we also show the rank (Spearman)
correlation, which a single wild return cannot dominate. For the benchmarks, "up" means above that day's median, as in the paper:

* **MOM**: momentum, the return from 12 months ago to 1 month ago
* **STR**: short-term reversal, minus the last month's return
* **WSTR**: weekly short-term reversal, minus the last week's return
""")
code('''
TEST_START = "2008-01-01"
probs = {name: cnn.load_predictions(name) for name in ("I5R5", "I5R20")}
bench = cnn.benchmark_signals(mkt.adj_close)
rows = {}
for name, h in (("I5R5", 5), ("I5R20", 20)):
    p = probs[name]
    dates = mkt.adj_close.loc[TEST_START:].index[::h]                  # one forecast per holding period
    fwd = cnn.forward_returns(mkt.adj_close, h).loc[dates].stack()
    sig = {f"CNN {name}": p[p.index.get_level_values("date").isin(dates)]}
    sig.update({f"{k} (next {h} days)": v.loc[dates].stack() for k, v in bench.items()})
    for k, s in sig.items():
        acc = cnn.period_accuracy(s, fwd, threshold=0.5 if k.startswith("CNN") else None)
        corr = cnn.period_corr(s, fwd)
        rank = cnn.period_rank_corr(s, fwd)
        rows[k] = {"accuracy": acc.mean(), "accuracy t-stat": (acc.mean() - 0.5) / acc.std() * np.sqrt(len(acc)),
                   "correlation": corr.mean(), "rank correlation": rank.mean(),
                   "rank corr. t-stat": rank.mean() / rank.std() * np.sqrt(len(rank)), "periods": len(acc)}
table = pd.DataFrame(rows).T
display(table.style.format({"accuracy": "{:.1%}", "correlation": "{:.3f}", "rank correlation": "{:.3f}",
                            "accuracy t-stat": "{:.1f}", "rank corr. t-stat": "{:.1f}", "periods": "{:.0f}"}))
''')
md("""
**How to read it.** A t-statistic above about 2 means the average is unlikely to be luck. Compare with the paper
(Table 2): 5-day pictures for the 20-day horizon had 52.5% accuracy and a 3.1% correlation; WSTR 51.0% and 2.8%;
MOM 52.1% and 1.8%. The paper has no accuracy table for 5-day horizons. Accuracy is
also blown about by the market: in a rising month most stocks go up, so a model that says "up" a lot looks clever.
The correlation, which only cares about ranking stocks against each other, is the fairer score for a long/short
book. And remember Workshop 3's lesson on multiple testing: this table holds eight signals, so one t-statistic
near 2 on its own proves little.

**What we found** (2008 to October 2026). The CNN accuracies look impressive, 54.3% and 57.6% with big
t-statistics, and they are an illusion. I5R5 says "up" for 91% of pictures and I5R20 for all of them, while on an
average test date 54.9% of stocks rose over the next week and 57.6% over the next month. Always saying "up" would
score the same or better. The t-statistic only tests against 50%, and 50% is the wrong yardstick here.

The rank correlation is the honest score. I5R5 ranks stocks about as well as weekly reversal (0.018 for both,
t-statistics 3.4 and 2.8). I5R20 manages 0.011, with a t-statistic of 1.3, no better than luck.
""")

# ------------------------------------------------------------------------------------------- 9 portfolios
md("""
## 9. Portfolios

Every $h$ days: rank the stocks by the signal, put the top decile long and the bottom decile short with $1 each,
equal weights, hold $h$ days (sec. 4.3). With about 100 stocks a decile is about 10 stocks. Costs: 5 basis points
per dollar traded at each rebalance, as in Workshops 3 and 4. Sharpe ratios are computed in the spirit of the
paper: one return per holding period, annualised (here by 252 / h periods a year; the paper's tables use calendar
months and weeks, a difference of a few per cent). MOM here compounds the return from 12 months to 1 month ago;
the paper averages it, which ranks stocks almost the same way.
""")
code('''
def wide(s):
    return s.unstack("symbol")

strategies = {"CNN I5R5 (weekly)": (wide(probs["I5R5"]), 5), "CNN I5R20 (monthly)": (wide(probs["I5R20"]), 20),
              "WSTR (weekly)": (bench["WSTR"], 5), "WSTR (monthly)": (bench["WSTR"], 20),
              "STR (monthly)": (bench["STR"], 20), "MOM (monthly)": (bench["MOM"], 20)}
bts = {k: cnn.decile_backtest(s, mkt.adj_close, h, start=TEST_START) for k, (s, h) in strategies.items()}
summary = pd.DataFrame({k: b.paper_stats() for k, b in bts.items()}).T
display(summary.style.format({c: "{:.2f}" for c in summary.columns if c != "rebalances"}))
plots3.equity({k: b.daily_net for k, b in bts.items() if k.startswith("CNN") or k == "WSTR (weekly)"},
              "Decile long/short after 5 bps costs, 2008 to 2026 (growth of $1)");
''')
md("""
Now the paper's Figure 9 picture: the average return of each decile, holding period by holding period. If the
network has learned something, the bars should rise from decile 1 to decile 10.
""")
code('''
dec = pd.DataFrame({k: bts[k].deciles.mean() * 252 / bts[k].horizon * 100
                    for k in ("CNN I5R5 (weekly)", "CNN I5R20 (monthly)", "WSTR (weekly)")})
plots.deciles(dec, "Average return of each decile, annualised, 2008 to 2026 (before costs)");
''')
md("""
And by period, because a single Sharpe ratio over 19 years hides a lot. The cut points are the same ones used
in Workshops 3 and 4, chosen to describe, not to test: short periods give noisy Sharpe ratios.
""")
code('''
cuts = {"2008-2009 crisis": ("2008", "2009-12-31"), "2010-2019": ("2010", "2019-12-31"),
        "2020-2022": ("2020", "2022-12-31"), "2023 onwards": ("2023", None)}
by = pd.DataFrame({name: {k: b.window(a, z).paper_stats()["Sharpe (after costs)"] for k, b in bts.items()}
                   for name, (a, z) in cuts.items()})
display(by.round(2).style.set_caption("Sharpe ratio after costs, by period"))
''')
md("""
**What we found.** Before costs, the weekly CNN (I5R5) earns a Sharpe ratio of 0.65, the same as plain weekly
reversal (0.64). Both turn over about 3.5 times their capital each week, so 5 bps per dollar traded takes most of
it: 0.12 after costs for the CNN, 0.30 for WSTR. The monthly CNN (I5R20) earns 0.15 before costs and nothing after.
The deciles chart tells the same story: for I5R5, decile 1 earns about 9% a year before costs and deciles 8 to
10 about 20%, but the long leg is no better than WSTR's; I5R20's bars are flat.

By period, most of I5R5's money was made in the 2008 to 2009 crisis (Sharpe 1.47, like WSTR's 1.39), when
reversal was strong. It is near zero from 2010 to 2022 and clearly negative since 2023 (-0.89). That is the
pattern Workshops 3 and 4 found too: price-based signals in big stocks have faded.
""")

# ------------------------------------------------------------------------------------------- 10 Grad-CAM
md("""
## 10. What is it looking at?

**Grad-CAM** (sec. 6.3, Selvaraju et al. 2017) takes one block's output, a stack of feature maps, and asks how
much the score for a class ("up" or "down") would rise if each map were stronger. It weights the maps by that
sensitivity, adds them up and keeps the positive part: a heatmap of the regions that pushed the network
towards that class. The paper drew 10 "up" and 10 "down" pictures at random from 2019 (Figure 13) and saw the
network fire on days with a **wide high-low range**, attend to **open and close ticks**, and light up the **top**
of "up" pictures and the **bottom** of "down" ones. The paper looked at its 20-day network; ours is the 5-day
monthly network (I5R20), with 2 blocks, using the 5 most confident calls each way and the first of the five
networks:
""")
code('''
nets5m = cnn.load_ensemble("I5R20")
p19 = probs["I5R20"].loc["2019"]
up, down = p19.nlargest(5).index, p19.nsmallest(5).index
for title, idx, target in (("Most confident 'up' calls in 2019", up, 1), ("Most confident 'down' calls in 2019", down, 0)):
    imgs = cnn.draw_images(days=5, **cnn.chart_inputs(mkt, 5, [d for d, s in idx], [s for d, s in idx]))
    cams = [[cnn.grad_cam(nets5m[0], im, block=b, target=target) for b in range(2)] for im in imgs]
    plots.cam_grid(imgs, cams, [f"{s} {d:%d %b}" for d, s in idx], title)
''')
md("""
*Discuss:* do you see what the paper saw? Grad-CAM is a rough guide, not proof: one picture, one network, and the
most confident calls are not typical ones (try random picks with `p19.sample(5)`).

A more statistical check (sec. 6.1, Table 12): how much does the network's forecast simply **repeat** signals we
already know? Rank correlation, averaged over dates, between the CNN's probability and some familiar quantities:
""")
code('''
ret = mkt.adj_close.pct_change(fill_method=None)
known = {"WSTR (minus last week's return)": bench["WSTR"], "STR (minus last month's return)": bench["STR"],
         "MOM (12 to 1 month return)": bench["MOM"], "volatility (last 20 days)": ret.rolling(20).std(),
         "dollar volume (last 20 days)": (mkt.adj_close * mkt.volume).rolling(20).mean()}
rows = {}
for name, h in (("I5R5", 5), ("I5R20", 20)):
    p = probs[name]
    dates = mkt.adj_close.loc[TEST_START:].index[::h]
    p = p[p.index.get_level_values("date").isin(dates)]
    rows[name] = {k: cnn.period_rank_corr(p, v.loc[dates].stack()).mean() for k, v in known.items()}
display(pd.DataFrame(rows).round(2).style.set_caption("Average cross-sectional rank correlation with the CNN forecast"))
''')
md("""
In the paper (Table 12), forecasts from the models trained on 5-day returns correlate -26% to -34% with last
week's return, a reversal component. Our WSTR row is *minus* last week's return, so the same pattern shows here as
a **positive** number. The paper's forecasts become more momentum-like and size-like for longer pictures and
horizons. A large positive WSTR correlation would mean the network has largely rediscovered weekly reversal,
which you could compute in one line.

**What we found.** It has. The rank correlation with WSTR is 0.58 for I5R5 and 0.51 for I5R20, more than the
paper's 26% to 34%, and about 0.3 with monthly reversal. Momentum, volatility and size barely register. From five
days of pictures, the network mostly learned "buy what just fell, sell what just rose".
""")

# ------------------------------------------------------------------------------------------- 11 trilogy
md("""
## 11. Three workshops, one question

Same 100 stocks, same years, three answers to "does the recent price path predict the next move?":

* **Workshop 3**: random forest, boosting and a neural net on 31 lagged returns, daily long/short top and bottom 10.
* **Workshop 4**: Avellaneda and Lee, an Ornstein-Uhlenbeck model on each stock's residual against its sector ETF.
* **Workshop 5**: a CNN looking at the chart.

All after their own trading costs (5 bps per dollar traded), as daily returns per dollar of capital, from 2008.
""")
code('''
from quantsoc_ml import statarb as sa, avlee as al
ml_pred = sa.load_cached_predictions()
ml_panel = sa.make_panel(sa.wide(sa.load_prices()))
ml = sa.long_short(ml_pred["ensemble"], ml_panel["fwd"], k=10)
# Workshop 3 labels a return by the day its signal was made; shift it to the day the return is earned
idx = mkt.adj_close.index
ml_net = ml.net.set_axis(idx[idx.get_indexer(ml.net.index) + 1])
ou = al.load_cached("etf")
three = pd.DataFrame({"W3 machine learning (daily)": ml_net, "W4 OU stat arb (daily)": ou.net,
                      "W5 CNN I5R5 (weekly)": bts["CNN I5R5 (weekly)"].daily_net,
                      "W5 CNN I5R20 (monthly)": bts["CNN I5R20 (monthly)"].daily_net}).loc[TEST_START:].dropna()
stats = pd.DataFrame({c: perf_stats(three[c]) for c in three}).T
display(stats[["annual return", "annual vol", "Sharpe", "max drawdown"]].round(3))
display(three.corr().round(2).style.set_caption("Correlation of daily returns"))
plots3.equity({c: three[c] for c in three}, "Three workshops after costs, 2008 to 2026 (growth of $1)");
''')
md("""
**What we found.** After costs, from 2008: Workshop 3's daily machine learning loses heavily (Sharpe -0.76),
Workshop 4's OU stat arb and the weekly CNN are both barely positive (0.13 and 0.12), and the monthly CNN is flat
(-0.02). The daily correlations are low, 0.07 to 0.25, so the three workshops found mostly different things.

*Discuss:* which ideas overlap (look at the correlations), and which is cheapest to trade? Would a blend of the
three be better than any one of them? Try `three.mean(axis=1)` with `perf_stats`.
""")

# ------------------------------------------------------------------------------------------- 12 why weaker
md("""
## 12. Why is it so much weaker than the paper?

Write the reasons down before reading these. They are likely causes, not proven ones; each one is testable.

1. **Big stocks only.** The paper's headline equal-weight results include thousands of small stocks; its
   value-weight Sharpe ratios at the monthly horizon are 0.45, 0.49 and 0.17 (Table 4). Ours are among the 100
   largest companies in America, and only 100 of them.
2. **Far less training data.** About 100 stocks over 8 years, against the whole US market: tens of times fewer
   pictures for a network with hundreds of thousands of parameters. (We also raised the learning rate to make up
   for fewer steps per epoch, a choice made on validation data only.)
3. **Deciles of 10 stocks.** Each leg is 10 names, so one stock's news moves the whole book.
4. **A different era.** The paper tests 2001 to 2019. Workshops 3 and 4 both found price-based signals much weaker
   after the mid-2000s. The working paper has been public since 2020.
5. **Costs.** The paper deducts no trading costs. A weekly strategy turns over its whole book about 50 times a
   year.

Survivorship bias pushes the other way: it flatters any strategy that buys stocks after they fall.
""")

# ------------------------------------------------------------------------------------------- 13 your turn
md("""
## 13. Your turn

Pick one. Each takes a few minutes on a CPU.

**A. Time-scale transfer (sec. 5.1).** One of the paper's most intriguing results: a network trained on *daily*
5-day pictures also works on longer bars, as if chart patterns look alike at any scale. The paper squeezes 20 days
into 5 bars of 4 days each, scores them with the I5R5 network, and forecasts the next 20 days (Table 8). Same here,
without retraining.
""")
code('''
bars4 = cnn.resample_market(mkt, every=4)              # 4-day bars: first open, max high, min low, last close
pics4 = cnn.make_images(bars4, days=5, horizon=5, start=TEST_START, need_label=False)
prob4 = cnn.predict(nets5, pics4.images, device=DEVICE)
scores = pd.Series(prob4, index=pd.MultiIndex.from_frame(pics4.meta[["date", "symbol"]])).unstack("symbol")
transfer = cnn.decile_backtest(scores, bars4.adj_close, horizon=5, start=TEST_START, bars_per_year=252 / 4)
print("transfer, 20 days ahead:", {k: round(v, 2) for k, v in transfer.paper_stats().items()})
print("I5R20 for comparison:   ", {k: round(v, 2) for k, v in bts["CNN I5R20 (monthly)"].paper_stats().items()})
''')
md("""
**B. Is the ensemble worth it?** Score the test pictures with one network instead of five, rebuild the I5R5
strategy, and compare Sharpe ratios. (Hint: `cnn.make_images(mkt, 5, 5, start=TEST_START, every=5,
need_label=False)`, then `cnn.predict(nets5[:1], ...)`.)

**C. Fewer stocks in each leg.** `cnn.decile_backtest(..., n_deciles=5)` uses quintiles (about 20 stocks a leg).
More diversified, less extreme. Better or worse after costs?

**D. Take something away.** Retrain your part 7 network on pictures drawn with `volume_bars=False` or
`ma_line=False` (pass them through `cnn.draw_images`). The paper found the moving average and volume help 20 and
60-day pictures but not 5-day ones (Table 17).

**E. Make the 20-day network learn (GPU).** Part 7 says it failed on a CPU with a fifth of the pictures. In Colab
with a T4 GPU, build 20-day training pictures for every day
(`cnn.make_images(mkt.window("2000-01-01", "2007-12-31"), 20, 20)`, about 170,000), train one network with
`cnn.train_cnn(..., days=20, horizon=20, lr=1e-4, device="cuda")`, and see whether its validation loss beats
always predicting the share of "up". The 60-day network (3 million parameters) is the same exercise, bigger.
""")
code('''
# your code here
''')

# ------------------------------------------------------------------------------------------- 14 live
md("""
## 14. Going live

The dashboard runs **I5R5** on your Alpaca **paper** account: every 5 trading days it draws each stock's last
5 days, scores the charts with the five cached networks, and buys the top decile and sells the bottom decile
($0.50 long and $0.50 short per $1 of equity, equal weights). It shows the charts it is betting on.

**Live data is not training data.** The networks learned from Yahoo's consolidated prices and volume. Alpaca's
free data plan serves the IEX exchange's bars, whose volume is a small slice of all trading and whose high-low
ranges can be narrower. Each chart rescales its own volume, which helps, but live charts can still look a little
different from anything the networks saw.

**The weekly rule.** The page reads your account's filled `qsml-ws5` orders to find the last rebalance. Five or
more trading days later it proposes orders; before that it shows the ranking but proposes nothing (a tick box
overrides this). Use a paper account for this strategy only.

```bash
cp .env.example .env          # then paste your Alpaca PAPER keys into .env
pip install -r requirements.txt streamlit torch
python -m streamlit run workshop5/dashboard.py
```

Today's ranking from the snapshot (offline, no account needed):
""")
code('''
today = cnn.live_scores(mkt, nets5)
print(f"as of {mkt.adj_close.index[-1].date()}: {len(today)} stocks scored")
display(pd.concat([today.head(5), today.tail(5)]).round(3))
top, bottom = list(today.symbol.head(4)), list(today.symbol.tail(4))
imgs = cnn.draw_images(days=5, **cnn.chart_inputs(mkt, 5, [mkt.adj_close.index[-1]] * 8, top + bottom))
plots.gallery(imgs, [f"{s} (buy)" for s in top] + [f"{s} (sell)" for s in bottom], ncols=8,
              suptitle="The charts behind this week's top and bottom calls");
''')
md("""
## Recap

* A chart is a grid of numbers. Drawn the paper's way, it packs trend, volatility and volume into one picture.
* A CNN learns its own pattern detectors from those pictures; built from the paper's rules, ours has exactly the
  paper's parameter counts.
* Accuracy against 50% is a trap: our networks mostly say "up", and so do most stocks. Rank correlation is the
  honest score, and the weekly network does rank stocks (0.018, t 3.4).
* What it learned is mostly weekly reversal (rank correlation 0.58 with WSTR): a Sharpe ratio of 0.65 before costs,
  0.12 after, most of it earned in the 2008 to 2009 crisis. The monthly network learned nothing usable.
* On 100 mega-caps after costs, most of the paper's edge does not survive, as the paper's own large-stock results
  warned. The method is general; the opportunity is where the paper found it, in a wide, small-cap universe.
""")


def build():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}, "colab": {"provenance": []},
                      "accelerator": "GPU"}
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print("wrote", build(), len(CELLS), "cells")
