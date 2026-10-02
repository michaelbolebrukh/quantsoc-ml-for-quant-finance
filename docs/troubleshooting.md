# Troubleshooting

Every message below is quoted as the notebook prints it. Words in angle brackets, such as `<folder>`, stand for
the value in your own message.

## Colab and Kaggle

**"Warning: This notebook was not authored by Google"**: choose `Run anyway`. The Setup cell only downloads this
workshop's pinned release from GitHub.

**Setup cell: `HTTP Error 404`**: the release tag in the Setup cell (`REVISION = "v1.0.0"`) is not published on
GitHub. Instructor: push the tag. Participants: ask for the current link, or upload the release ZIP through the
Files pane, unzip it to `/content/workshop` and run Setup again.

**`ModuleNotFoundError: No module named 'quantsoc_ml'`**: the first Setup cell did not finish. Run it again; it
is safe to repeat.

**"Runtime disconnected" or your files are gone**: Colab runtimes are temporary. Run the notebook again from the
top (`Runtime → Run all` takes about a minute of compute on Colab). To skip the tasks you already did, add
`os.environ["QSW_AUTO_REFERENCE"] = "1"` at the top of the second Setup cell: unchanged tasks then use the
labelled reference answers. If you downloaded your ZIP, you still have your strategy.

**A cell takes longer than its label says**: Colab's free tier has 2 CPU cores; the times in the notebook were
measured on 4. The noise experiment (section 5) and the refit (section 7) can take up to twice their label. Wait
for the `[time]` line.

**The ZIP did not download (section 8)**: the browser may block the pop-up. Open the Files pane (folder icon on
the left), find `workspace/exports/my_strategy.zip` under `/content/workshop`, and choose Download. On Kaggle it is
in the Output panel.

## Setup on any machine

**`[!] Cannot save your files in <folder> (<reason>).`** followed by **`For now your files go to a temporary
folder: <temporary folder>`**: the folder named by `QSW_WORK_DIR` (or `workspace/` inside the workshop folder) is
not a folder, or you cannot write to it. The notebook carries on with a temporary folder, which your computer may
empty later. To keep your files, run `os.environ['QSW_WORK_DIR'] = str(Path.home() / 'workshop_files')` (or any
folder you can write to) in a new cell, then run the first Setup cell again.

**`Cannot find the workshop folder (it holds quantsoc_ml/ and data/). Start Jupyter from inside the unzipped
workshop folder.`**: start Jupyter from inside the unzipped workshop folder, or from its `notebooks/` folder.

## The tasks

**`[x] Not yet: <value> is the worked example's ...`**: every task cell starts as a copy of the worked example, and
an unchanged copy is refused on purpose, so that the knob is your choice. Change at least one number on the line
marked `# <- change this`.

**`my_momentum is not past_return(prices, <N>). Change only the number on the line marked '# <- change this' and
run the whole cell again.`** (Task 1) or **`my_extra does not match your window. Change only the number on the line
marked '# <- change this' and run the whole cell again.`** (Task 2): change only the number on the first line and
run the whole cell again, so the second line uses the new number.

**`<name> is already one of the features, so it would add nothing. Pick something new, like vol_63.`** (Task 2): a
21-day volatility, or a name such as `ret_5`, is already in the model. Pick another window.

**`<name> uses a window of <N> days; choose a window from 2 to 252 trading days, for example vol_63.`** (Task 2,
given a name): windows longer than a year are refused, by number or by name.

**`train_end '<text>' uses slashes, which can be read day first or month first: write dates as "2014-12-31" (year,
month, day, in quotes).`** (Task 3): `"31/12/2014"` and `"12/31/2014"` are refused rather than guessed. Write the
year first: `"2014-12-31"`. The same message names `test_start` when that is the date to fix.

**`test_start <date> is too close: the last training target is only known 5 trading days after <train_end>, so
start the test on or after <earliest date>.`** (Task 3): use the date the message prints, or a later one.

**`[x] max_depth=None lets every tree grow until it memorises single days; section 5 shows exactly that, live. Here
pick a max_depth from 2 to 12`** (Task 4): refused on purpose, before any fitting starts. Section 5 shows why.
**`[x] choose from 20 to 500 trees, not <N> (more than 500 is slow and adds almost nothing)`** is the same kind
of refusal for the number of trees.

**`rebalance_days must be 5 (weekly), 10 (every two weeks) or 21 (monthly).`** (Task 5): only those three are
allowed, so that everyone's backtest can be compared.

**`Your earlier accepted answer stays in use (<knob> = <value>, labelled 'yours'). Fix this cell and run it again
to change it.`**: you re-ran a task that had passed, and the new values were refused. Nothing changed: the knob
still holds the answer that passed, and the knob table still calls it yours.

**`[ref] Task <n> is not passed yet, so the cells below use the workshop's reference answer.`**, **`[ref] Task <n>
(<title>) is not yours yet: <knob> = <value> is the workshop's reference.`** or **`[ref] Task <n> (<title>) uses the
workshop's reference answer, so the cells below do too.`**: the cell that uses the knob is telling you it is still
the reference. Nothing is substituted silently: go back, finish the task, and run the cells below it again. The
final table in section 8 shows, knob by knob, which values are yours.

**`[x] <one sentence>`** followed by **`If this is a task cell, fix the value marked '# <- change this' and run the
cell again; otherwise see docs/troubleshooting.md.`**: the package refused a value with a plain sentence (for
example two split dates the wrong way round: `test_start (<date>) must come after train_end (<date>).`). Run `%tb`
in a new cell to see the full details.

## Section 7 and 8

**Section 7 uses a task I changed late**: it should. The refit cell rebuilds the features, the dataset and the
split from your current knobs before fitting, so going back to any task and then running section 7 again uses the
change.

**`[x] the model was fitted on [<features>] but these knobs give [<features>]: refit the model after changing a
knob`** followed by **`A task was changed after section 7 fitted your model. Run section 7 again from its refit
cell, then this cell.`** (the export): you changed Task 1 or 2 after the section 7 refit. Run section 7 again from
the refit cell, then the export cell.

**`[x] <folder> holds .env from an earlier run; move it out of the folder first ...`** (the export): re-exporting
replaces the whole folder, so a key file inside it is refused. Move the `.env` file somewhere else and run the
export cell again.

**`[x] There is no exported strategy in <folder> yet. Run the export cell above until it prints 'contents check:
OK', then run this cell again.`** (the preview): the export cell has not finished. Run it, then the preview.

**`[!] The export cell has not finished in this session, so this previews the strategy exported earlier into
<folder>, not the model above.`**: you restarted the notebook and skipped the export. Run the export cell first.

**The preview prints `[runner exit code 3]`**: the line above it, starting `[runner]`, says what was missing
(usually the price file). Run the cells from the export onwards, without renaming anything.

## On your own computer

**`pip install` fails on Windows with a compiler error**: use Python 3.11 or 3.12 from python.org (ready-made
wheels exist for numpy, pandas and scikit-learn); avoid pre-release Python versions.

**`python run_paper.py` says `[runner] No Alpaca paper keys found.`**: create `.env` from `.env.example` next to
`run_paper.py`, or use the offline demo, which needs no keys:
`python run_paper.py --preview --dry-data data/prices.csv.gz --bundle workspace/exports/my_strategy`.

**Alpaca `forbidden` or 403**: the keys are live keys or wrong; generate **paper** keys. The runner only talks to
the paper endpoint.

## Tests

**`pytest tests/test_notebook_runs.py` is skipped**: install `nbclient` and `ipykernel` in the same environment
(both come with `jupyterlab` from `requirements.txt`).
