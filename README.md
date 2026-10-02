# Machine Learning for Quantitative Finance (Part 1)

A hands-on workshop of about 70 minutes, with an optional 25-minute bonus section on learning theory, by the **Quantitative Finance Society** ([quant-soc.com](https://quant-soc.com)).
You build a small machine learning strategy on twenty years of real daily prices for 40 large US companies:
features from published research, an honest baseline, a split by time, a random forest, a live overfitting demo,
a long/short portfolio with trading costs, and a backtest of **your own** five choices. You leave with a ZIP of
your strategy and a preview of the orders it would send to a practice (paper) trading account.

No machine learning or finance background is assumed, only a little Python. Every term is defined the first time
it appears, and every coding task is copy and change: a worked example sits directly above it.

> Teaching project, not investment advice. The prices are real (Yahoo Finance, via `yfinance`), but the universe
> is survivorship-biased (only companies that still trade today), and the strategy has no reliable edge after
> trading costs. The notebook says so, with the measured numbers.

## Start here (browser only, nothing to install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michaelbolebrukh/quantsoc-ml-for-quant-finance/blob/v1.0.0/notebooks/workshop2_student.ipynb)

The badge and the notebook's setup cell point at release `v1.0.0` of this repository. Until the organisers
publish that tag, the link returns a 404 (see `docs/HANDOVER.md`, step 1).

1. Click the badge, then `File → Save a copy in Drive` so your edits are kept.
2. Run the two cells of section **0. Setup**. The first one downloads this repository's pinned release
   (`v1.0.0`) into the Colab runtime; it is safe to run again.
3. Follow the presenter section by section. Five short tasks; `ex.hint(n)`, `ex.show_solution(n)` and
   `ex.use_reference(n)` are there if you get stuck. The whole notebook needs about 30 seconds of compute on a
   4-core laptop and up to about a minute on Colab's free tier; every cell slower than 10 seconds says so above it.
4. **Download your strategy ZIP** in section 8. Colab forgets every file when the runtime disconnects.

Alternative hosted Jupyter (slower to start, no Google account needed):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/michaelbolebrukh/quantsoc-ml-for-quant-finance/v1.0.0?labpath=notebooks%2Fworkshop2_student.ipynb)

## Run it on your own computer

Python 3.11 or 3.12. Download this repository as a ZIP (green **Code** button), unzip it, open a terminal in the
unzipped folder, then:

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab notebooks/workshop2_student.ipynb
```

**macOS or Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/workshop2_student.ipynb
```

The Setup cell finds the workshop folder by itself (it looks for `quantsoc_ml/` and `data/` above the notebook).
Your exported strategy is written to `workspace/exports/`.

After the workshop, the offline paper-trading preview runs from this folder with no account and no keys:
```bash
python run_paper.py --preview --dry-data data/prices.csv.gz --bundle workspace/exports/my_strategy
```
To use a free Alpaca **paper** account with live prices, follow `docs/README_EXPORT.md` (also inside your ZIP).

## What is here

| Path | Purpose |
|---|---|
| `notebooks/workshop2_student.ipynb` | the workshop notebook (stored with outputs cleared) |
| `quantsoc_ml/` | the supporting package: data, features, splits, models, evaluation, portfolio, backtest, export, paper runner, task tracker, plots |
| `run_paper.py` | one rebalance of your exported strategy against a paper account (preview by default) |
| `data/prices.csv.gz` | the price snapshot ([data card](data/DATA_CARD.md)); `fetch_data.py` rebuilds it |
| `docs/` | [prep checklist](docs/student_prep_checklist.md), [troubleshooting](docs/troubleshooting.md), [instructor timing](docs/instructor_timing.md), [running your export](docs/README_EXPORT.md), [measured results](docs/core-results.md) |
| `deck/` | the slide deck and its generator |
| `tests/` | timing rule, models, backtest, export, and a test that runs the whole notebook from a clean kernel |

**Data source.** A snapshot of daily prices from Yahoo Finance, downloaded once with the `yfinance` package and
committed, so the workshop never depends on a live download. It is included for teaching only; see the data card
for the licence note and the survivorship bias statement.

**References.** Each notebook section opens with the paper it draws on: López de Prado (2018, both the JPM paper and
*Advances in Financial Machine Learning*), Jegadeesh and Titman (1993), Jegadeesh (1990), Kaufman et al. (2012),
Gu, Kelly and Xiu (2020), Breiman (2001), Krauss, Do and Huck (2017), Bailey et al. (2014), Shumway (1997) and
Arnott, Harvey and Markowitz (2019). Full citations are in the notebook.

**Tests.** `python3 -m pytest tests -q` (about a minute; the notebook test needs `nbclient` and `ipykernel`).

Release: `michaelbolebrukh/quantsoc-ml-for-quant-finance@v1.0.0`. Part 2 (next week): purged cross-validation,
triple-barrier labels, feature importance, and the deflated Sharpe ratio.
