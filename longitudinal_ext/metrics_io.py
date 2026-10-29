"""Metrics + result files in the exact layout the notebook's Sections 9-10 already read:
metrics.csv, confusion_matrices.json ({split: {"confusion": [5 x 3x3]}}), settings.json."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from . import CLASS_NAMES, DISEASES

LABELS = [0, 1, 2]


def evaluate_predictions(y_true, y_pred, split, diseases=DISEASES):
    """y_true, y_pred: int arrays [N, D]. y_true == -100 is masked. Returns (rows, confusions)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows, confs = [], []
    for d, name in enumerate(diseases):
        m = y_true[:, d] != -100
        if not m.any():
            rows.append(dict(disease=name, split=split, n_pairs=0, accuracy=np.nan,
                             balanced_accuracy=np.nan, macro_f1=np.nan))
            confs.append(np.zeros((3, 3), dtype=int).tolist())
            continue
        yt, yp = y_true[m, d], y_pred[m, d]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(dict(
                disease=name, split=split, n_pairs=int(m.sum()),
                accuracy=float(accuracy_score(yt, yp)),
                balanced_accuracy=float(balanced_accuracy_score(yt, yp)),
                macro_f1=float(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)),
            ))
        confs.append(confusion_matrix(yt, yp, labels=LABELS).tolist())
    return rows, confs


def write_results(out_dir, metric_rows, confusions, settings, history=None, predictions=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "confusion_matrices.json").write_text(
        json.dumps({s: {"confusion": c} for s, c in confusions.items()}), encoding="utf-8")
    settings = dict(settings)
    settings.setdefault("diseases", list(DISEASES))
    settings.setdefault("labels", list(CLASS_NAMES))
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    if history is not None:
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if predictions is not None:
        predictions.to_csv(out_dir / "predictions.csv", index=False)
