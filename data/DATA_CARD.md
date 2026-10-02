# Data card: daily prices for 40 large US companies, 2000 to 2019

**File** `data/prices.csv.gz`: 201,200 rows = 40 symbols x 5,030 trading days, from 2000-01-03 to 2019-12-30.
Every symbol has a row on every trading day (no gaps). Long format: one row per (date, symbol).

**Columns**

| Column | Meaning |
|---|---|
| `date` | trading day, `YYYY-MM-DD` (New York exchange calendar) |
| `symbol` | ticker |
| `open`, `high`, `low`, `close` | raw daily prices in US dollars, as Yahoo Finance reports them (not adjusted) |
| `adj_close` | Yahoo's adjusted close: corrected for stock splits and dividends, so returns computed from it include dividends (approximately total returns). The workshop uses only this column for returns |
| `volume` | shares traded that day |

**The 40 symbols** (read from the file): AAPL, AMGN, AMZN, AXP, BA, BAC, BMY, C, CAT, CSCO, CVX, DIS, GE, GILD, GS,
HD, IBM, INTC, JNJ, JPM, KO, LOW, MCD, MMM, MO, MRK, MSFT, ORCL, PEP, PFE, PG, QCOM, T, TGT, TXN, UNH, VZ, WFC, WMT,
XOM.

**Source** Yahoo Finance, downloaded once with the `yfinance` package (`fetch_data.py`, `auto_adjust=False`) and
committed as a snapshot, so the workshop never depends on a live download.

**How the 40 were chosen** (`fetch_data.py`): 50 candidates from the S&P 100 that traded throughout 2000 to 2019
and still trade today; of those with prices on at least 99% of trading days, the 40 with the highest median daily
dollar volume were kept. The list was fixed by this rule, not by looking at any result.

**Survivorship bias, stated plainly.** Every company here survived to 2019 and still trades. Companies that went
bankrupt, were taken over or were removed from the index between 2000 and 2019 are missing. An investor in 2000
could not have known which companies would survive, so any backtest on this file is flattered, and short positions
in particular never meet the collapses they would have profited from. The notebook (sections 1 and 6) and slide 18
say so. See Shumway (1997), "The Delisting Bias in CRSP Data", The Journal of Finance, 52(1), 327-340.

**Other limits.** Daily prices only (no intraday data, no bid and ask prices); Yahoo's adjustments can be revised
and occasionally contain errors, and were not checked against another source.

**Licence note.** The prices come from Yahoo Finance and are subject to Yahoo's terms of use. They are included
here only for this non-commercial teaching workshop. Do not reuse or redistribute them for any other purpose; to
work with the data yourself, download it under your own acceptance of Yahoo's terms (next paragraph).

**Rebuild** `python fetch_data.py` downloads the data again and writes the file (it needs `yfinance` and internet
access; Yahoo may have revised history since, so the result can differ slightly). `python fetch_data.py --check`
verifies the committed file's columns, symbol count, date range and that no row is duplicated.

**Intended use** teaching only. Nothing here is investment advice.
