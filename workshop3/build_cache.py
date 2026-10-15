"""Rebuild the cached walk-forward predictions the notebook loads by default.

Usage:  python workshop3/build_cache.py      # about 5 minutes on a 4-core laptop

Trains the four "workshop" preset models on every study period (750 days train, 250 days trade)
and writes their out-of-sample probabilities to workshop3/data/predictions_workshop.csv.gz.
The notebook retrains one period live and compares it with this file, so a stale or edited
cache shows up as a mismatch rather than silently.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantsoc_ml import statarb as sa  # noqa: E402

CACHE = ROOT / "workshop3" / "data" / "predictions_workshop.csv.gz"


def save(pred, path=CACHE):
    cols = [c for c in sa.MODEL_NAMES if c in pred.columns]
    out = pred[cols + ["period"]].reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, compression="gzip", float_format="%.5f")
    return path


if __name__ == "__main__":
    panel = sa.make_panel(sa.wide(sa.load_prices()))
    pred = sa.walk_forward(panel, models=sa.MODEL_NAMES)
    print("wrote", save(pred))
