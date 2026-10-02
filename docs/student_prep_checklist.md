# Student preparation checklist (5 minutes, before the session)

**Everyone (browser route)**
- [ ] A Google account that can open Google Colab (free). Test: open https://colab.research.google.com and create an empty notebook.
- [ ] Open the workshop notebook link sent by the organisers and choose `File → Save a copy in Drive`. You do not need to run anything yet.
- [ ] Basic Python: variables, numbers and strings, calling a function. No pandas, finance or machine learning knowledge is assumed; every term is explained in the notebook.
- [ ] A laptop with a keyboard (tablets struggle with Colab's editor).

**Not required:** a GitHub account, Git, a credit card, a GPU, paid market data, or an Alpaca account.

**Optional: only if you want to run locally instead**
- [ ] Python 3.11 or 3.12 installed (`python3 --version`, or `py -3.12 --version` on Windows).
- [ ] Download the repository ZIP from the release page and unzip it.
- [ ] Follow "Run it on your own computer" in `README.md` up to `jupyter lab ...` and confirm the notebook opens. Installation takes 2 to 5 minutes.
- [ ] Run section 0 once at home: the second cell prints a version table. If it does, you are ready.

**Optional: only if you want to try paper trading at home after the session**
- [ ] Create a free Alpaca account and generate **Paper Trading** API keys (dashboard, Paper Trading, API Keys). Never use live keys.
- [ ] Do not paste keys into a notebook cell. `docs/README_EXPORT.md` explains where they go (a `.env` file next to `run_paper.py`).
- [ ] In the session itself the paper demo runs offline with a mock broker: no account and no keys are needed.
