# Handover: what to do before the workshop

Everything in this folder was built and tested in one session. Three things need a human
before the day, and one is optional. Each takes a few minutes.

## 1. GitHub release (done) and the campus rate limit (needs you)

The repository is public at `michaelbolebrukh/quantsoc-ml-for-quant-finance`. The pinned release the
notebook's setup cell and the README's Colab badge use is `v1.0.0`, published as a **branch** of that
name pointing at the release commit (the session that built this could push branches but not tags;
GitHub resolves a branch name in the same URLs). Do not move that branch during the workshop week. If
you later want a real tag instead, delete the branch first (`git push origin :refs/heads/v1.0.0`), then
`git tag v1.0.0 <commit> && git push origin v1.0.0`, so the name is never ambiguous.

**Do not hand out the GitHub Colab link on campus.** Colab opens a GitHub notebook through GitHub's
API, which allows about 60 anonymous requests per hour per public IP address, and the whole University
of Edinburgh network shares one address. A room of people clicking it gets "API rate limit exceeded".
Open the notebook from Google Drive instead, which has no such limit:

1. Download `notebooks/workshop2_student.ipynb` from the repository.
2. Upload it to the QuantSoc Google Drive, open it once (it opens in Colab), and set sharing to
   "Anyone with the link: Viewer".
3. Copy that Colab address (`https://colab.research.google.com/drive/...`) and point the short link
   **quant-soc.com/links/workshop-2** (shown on slide 1) at it.
4. Test the short link on campus wifi and on a phone hotspot.

Participants then use File, Save a copy in Drive, as before. The setup cell still downloads the code
from GitHub's archive endpoint, which is a plain download and not the rate-limited API.

## 2. Check the two social QR codes once

Slides 23 and 45 carry three QR codes, decoded here to confirm what they encode:
https://quant-soc.com, https://www.instagram.com/quantsoc_edinburgh/ and
https://www.linkedin.com/company/quantsociety. Scan the LinkedIn one with a phone before the day
and check it opens the society's own page. To change a link, edit `deck/make_qr.py`, then from `deck/`:

    python3 make_qr.py && python3 build_deck.py && python3 qa_render.py

## 3. Open the deck once in PowerPoint or Keynote

`deck/quantsoc-ml-workshop-2.pptx` follows the website's design language (near-black canvas,
QuantSoc blue, Inter, glass cards, the logo in every footer) and was rendered here with
LibreOffice Impress and checked slide by slide, but not opened in PowerPoint itself. The
typeface is Inter; the font files are in `deck/assets/fonts/` (OFL licence). Install Inter on
the presenting machine before the day, or PowerPoint substitutes its default font and a few
lines may wrap differently. Speaker notes are in each slide and in `deck/SPEAKER_NOTES.md`.
Every content slide carries the six-step process strip (Prices, Features, Model, Check,
Portfolio, Backtest) with the current step lit, so the audience always knows where they are.

The main deck is 23 slides and about 70 minutes: slide 3 sets textbook ML against finance ML,
the "In finance" tags on later slides point back to its rows, and slides 7 to 10 cover where
features come from. Slides 24 to 45 are a bonus section on core principles of learning theory
(about 25 minutes, three slides marked optional) with its own chapter strip; show it if there is
time, or as a separate talk. `docs/instructor_timing.md` has the minute-by-minute plan for both.

## 4. Alpaca paper keys (optional, for the homework)

Nothing in the room needs an Alpaca account: section 8 runs the paper-trading script offline
against a mock broker. For the homework, participants create a free Alpaca account, generate
PAPER keys, and follow `docs/README_EXPORT.md` (it is copied into every exported bundle). The
script sends nothing unless they pass `--submit` and type SUBMIT.

## What was measured, so you know what to expect

* The notebook runs from a clean kernel in about 32 s here (4 cores); the one labelled slow
  cell (the noise demo) takes 11 to 16 s, and the labels allow twice that for Colab's free tier.
  The whole suite, notebook included, also passes under Colab's library versions (pandas 2.2,
  numpy 2.0, scikit-learn 1.6).
* With the reference knobs the strategy makes +17.2% before costs and +6.4% after over
  2015 to 2019 (21-day rebalance, forest fitted on a quarter of the training rows as in the
  notebook), against +86% for simply holding all 40 stocks. The same book rebalanced every
  5 days makes +22.9% before costs and loses 15.8% after. The edge is thin, and the deck and
  notebook say so; nothing promises a profitable strategy.
* 155 automated tests cover the timing rule, the backtest accounting, the exercise checkers,
  the export bundle and the paper-trading safety rules.

Known limits: the universe is 40 surviving large caps (survivorship bias, stated on slide 18
and in the data card); Yahoo Finance data via yfinance, for teaching only; the deck was not
opened in PowerPoint; Colab itself was not run (a Colab-like set of library versions was).
