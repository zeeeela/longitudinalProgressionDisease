"""Tabular-only baseline (no images): per-disease multinomial logistic regression on the MIMIC-IV features.
Tells you how much signal the tabular features carry on their own, before any fusion."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import DISEASES
from .metrics_io import evaluate_predictions, write_results
from .pair_data import KEYS, TabularScaler, _keystr, default_tab_columns


def run_tabular_baseline(manifest_csv, tab_csv, out_dir, c_values=(0.01, 0.1, 1.0, 10.0), diseases=DISEASES,
                         tab_columns=None, class_weight=None, seed=42):
    df = _keystr(pd.read_csv(manifest_csv))
    tab = _keystr(pd.read_csv(tab_csv))
    cols = tab_columns or default_tab_columns(tab)
    df = df.merge(tab[KEYS + cols], on=KEYS, how="left", validate="m:1")
    sc = TabularScaler().fit(df[df["split"] == "train"], cols)
    X = {s: sc.transform(df[df["split"] == s]) for s in ("train", "val", "test")}
    Y = {s: df.loc[df["split"] == s, list(diseases)].to_numpy() for s in X}
    pred = {s: np.zeros_like(Y[s]) for s in X}
    chosen = {}
    for d, name in enumerate(diseases):
        tr = Y["train"][:, d] != -100
        va = Y["val"][:, d] != -100
        if len(np.unique(Y["train"][tr, d])) < 2:
            maj = int(np.bincount(Y["train"][tr, d]).argmax()) if tr.any() else 1
            for s in X:
                pred[s][:, d] = maj
            chosen[name] = "majority"
            continue
        best, best_c = (-2, None), None
        for c in c_values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(C=c, max_iter=3000, class_weight=class_weight, random_state=seed)
                clf.fit(X["train"][tr], Y["train"][tr, d])
            f1 = f1_score(Y["val"][va, d], clf.predict(X["val"][va]), labels=[0, 1, 2], average="macro",
                          zero_division=0) if va.any() else -1
            if f1 > best[0]:
                best, best_c, best_clf = (f1, c), c, clf
        chosen[name] = best_c
        for s in X:
            pred[s][:, d] = best_clf.predict(X[s])
    rows, confs = [], {}
    for s in ("val", "test"):
        r, c = evaluate_predictions(Y[s], pred[s], s, diseases)
        rows += r
        confs[s] = c
    write_results(out_dir, rows, confs, dict(experiment="tabular_only_logreg", chosen_C=chosen, n_tab=len(sc.names),
                                              tab_names=sc.names, seed=seed))
    return dict(out_dir=Path(out_dir), metrics=rows)
