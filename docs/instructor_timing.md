# Instructor timing: one page, minute by minute

Deck: `deck/quantsoc-ml-workshop-2.pptx` (notes in `deck/SPEAKER_NOTES.md`). Notebook:
`notebooks/workshop2_student.ipynb`. Run times below were measured on 4 CPU cores (2026-10-01); Colab's free tier
has 2, so allow up to twice as long for the labelled slow cells. The whole notebook runs in about 30 seconds of
compute here with the reference knobs (about 50 seconds with 500 trees of depth 12), so the clock is set by talking
and tasks, not by the machine.

**Golden rule for stalls:** never wait for the whole room. At the end of each task's window say: "If you are not
done, run `ex.use_reference(n)` and carry on; it is labelled, and you can come back to it." The tracker records
it, and the final table and the export say which knobs were the participant's.

| Minutes | Slide | Notebook | What to say if it stalls |
|---|---|---|---|
| 0 to 1 | 1 Title | section 0: both Setup cells | "Run section 0 now, it takes a few seconds." Colab asks to confirm "Run anyway": say yes. |
| 1 to 3 | 2 What we're building | (setup finishing) | Setup error: see `docs/troubleshooting.md`. A 404 means the release tag is not published. |
| 3 to 6 | 3 Textbook ML vs finance ML | section 0 Reference box | Nothing to run. Walk the table row by row; the "In finance" tags on later slides point back to these rows. |
| 6 to 9 | 4 The data | section 1: load cell (about 1 s), price chart | "Each line starts at 100 so you can compare them." |
| 9 to 11 | 5 Where it came from | section 1 text | Name the survivorship bias out loud; it comes back on slide 21. |
| 11 to 14 | 6 Past vs future: the target | section 2: worked example, **Task 1** | "Change only the 5 on the first line, to 21 or 63, and run the cell." Hint: `ex.hint(1)`. |
| 14 to 18 | 7 The feature universe | section 2: "Where features come from" table | Nothing to run. "We use the first three families, because prices and volumes are free." |
| 18 to 21 | 8 Momentum and reversal, from the literature | section 2: **Task 2** | "Change the 21 to 63." |
| 21 to 24 | 9 Our eight features, drawn | section 2: build the features | The build cell takes under 1 s. |
| 24 to 27 | 10 Making features usable | section 2: rank the features | "Ranking keeps the order and removes the scale." |
| 27 to 31 | 11 Leakage, live | section 2: target check, leakage cell (about 1 s) | Point at 0.40 vs 0.02: "too good to be true". The cell deletes the leaky column. |
| 31 to 33 | 12 Baseline: predict zero | section 3: zero baseline cell | "A typical 5-day move is 4%; the average is +0.25%." |
| 33 to 37 | 13 Split by time, never at random | section 3: worked example, **Task 3**, split chart, random vs time (about 3 s) | "Change at least one date; keep the quotes and the YYYY-MM-DD format." Too-close dates print the earliest allowed test_start. |
| 37 to 39 | 14 What a tiny edge looks like | (read the random vs time table) | "Five times better on a shuffled split, and none of it is real." |
| 39 to 41 | 15 A decision tree in one picture | section 4: quarter sample cell | Nothing slow yet. |
| 41 to 45 | 16 A forest | section 4: worked example (about 1 s), **Task 4** (about 3 s at 200 trees, depth 6; 9 s at 500 trees, depth 12), score table | "Two numbers, trees then depth." Values outside 20 to 500 trees or depth 2 to 12 (and depth None) are refused before any fitting starts: section 5 shows why depth matters. |
| 45 to 48 | 17 Reading the decile plot | section 4: decile plot | "Look at the gap between the left and right ends, not the height: every bar is up because the market rose." |
| 48 to 52 | 18 Overfitting, live | section 5: noise experiment (**slow: 11 to 16 s here, up to 30 s on Colab**) | Start it, then talk over it. Point at +48% train, -7.5% test. |
| 52 to 54 | 19 Trying many things is the problem | section 5 text | Nothing to run. |
| 54 to 58 | 20 Forecast to portfolio, costs | section 6: forecasts, worked example, **Task 5**, cost cell | "Change at least one of the three numbers; try 21 days." The worked example (4, 4, 5) is refused unchanged. |
| 58 to 60 | 21 Survivorship | section 6 text | Nothing to run. |
| 60 to 65 | 22 Your backtest, read honestly | section 7: rebuild and refit (about 3 s here at the reference knobs; **up to 10 s at 500 trees, depth 12**), backtest, 5-day cell, checklist | "Nobody should expect a good-looking result. Compare before and after costs, not with the market line." The refit rebuilds the data from the current knobs, so anyone who went back to a task late just runs section 7 again. |
| 65 to 71 | 23 Export, paper trading, Part 2, QR codes | section 8: export (about 1 s; up to 6 s at 500 trees, depth 12), preview (about 2 s) | "Download the ZIP now." The download pop-up may be blocked: use the Files pane on the left. |

**Bonus section (slides 24 to 45, about 25 minutes, no notebook).** Show it only if there is time after slide 23,
or as a separate talk. It has its own chapter strip (Generalisation, Honest testing, Trade-offs, Model families,
Finance), and each slide carries a "Seen in Part 1" tag pointing back to the main slide where the idea came up.
Three slides are marked optional (double descent, explaining a model, two cultures); skipping them saves about
3 minutes. Two of its charts are honest rather than textbook: on our data the best single tree is depth 1 and no
depth beats a constant forecast (slide 33), and the double-descent dip is real but does not beat the best small
model (slide 36). Say so; that is the lesson about noisy data.

**Numbers to have in your head (reference knobs, measured; the full list is in `docs/core-results.md`, "Numbers
the notebook prints"):** leaky feature correlation +0.40 vs honest features 0.02 or less; shuffled vs time split R
squared +0.56% vs +0.11%; reference forest R squared +0.09% vs +0.73% for the constant forecast; unbraked forest
+48% train, -7.5% test (with 30 noise columns +60%, -4.1%); 21-day book +17.2% before costs, +6.4% after, Sharpe
0.16, max drawdown -54%; 5-day book +22.9% before costs, -15.8% after; the market +86% (for scale, not the bar to
beat; it is rebalanced on the same days as the strategy). Section 7 fits on the same quarter of the training rows
as section 4, so these differ from the all-rows numbers in D7 (+16.3%, +5.2%, -31%).

**If you are behind:** at minute 48 skip the noise cell's discussion (one sentence: "more noise, better training
score, no better test score"); at minute 60 make everyone run section 7 with whatever knobs they have, and do the
checklist out loud; section 8 must start by minute 66 so the ZIP is downloaded before people leave.

**Before the session:** run the notebook once from a clean kernel (`python3 -m pytest tests/test_notebook_runs.py -q`),
confirm the release tag in the Setup cell is published, and keep `SAVE_DECK_FIGURES = False` in the copy you share.
