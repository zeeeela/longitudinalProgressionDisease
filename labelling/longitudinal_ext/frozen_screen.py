"""Cheap encoder screening: frozen encoder -> (current - prior) features -> PCA -> per-disease logistic regression.
Same recipe as the notebook's frozen baseline, but works for ANY encoder in encoders.py. Features are cached."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from . import DISEASES
from .metrics_io import evaluate_predictions, write_results


@torch.no_grad()
def extract_pair_features(encoder, loader, device):
    encoder.to(device).eval()
    F_, Y = [], []
    for b in loader:
        f = encoder(torch.cat([b["x_prev"], b["x_cur"]], 0).to(device)).float().cpu()
        f0, f1 = f.chunk(2)
        F_.append((f1 - f0).numpy())
        Y.append(b["y"].numpy())
    return np.concatenate(F_), np.concatenate(Y)


def run_frozen_screen(encoder, data, out_dir, pca_components=50, c_values=(0.01, 0.1, 1.0, 10.0),
                      diseases=DISEASES, device=None, seed=42):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = {s: extract_pair_features(encoder, data["loaders"][s], device) for s in ("train", "val", "test")}
    np.savez_compressed(out_dir / "features.npz", **{f"{s}_X": v[0] for s, v in feats.items()},
                        **{f"{s}_y": v[1] for s, v in feats.items()})
    pred = {s: np.ones_like(feats[s][1]) for s in feats}
    chosen = {}
    for d, name in enumerate(diseases):
        Xtr, ytr = feats["train"][0], feats["train"][1][:, d]
        tr = ytr != -100
        if len(np.unique(ytr[tr])) < 2:
            continue
        sc = StandardScaler().fit(Xtr[tr])
        pca = PCA(n_components=min(pca_components, int(tr.sum()) - 1, Xtr.shape[1]), random_state=seed)
        Ztr = pca.fit_transform(sc.transform(Xtr[tr]))
        proj = lambda s: pca.transform(sc.transform(feats[s][0]))
        va = feats["val"][1][:, d] != -100
        best = (-2, None)
        for c in c_values:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(Ztr, ytr[tr])
            f1 = f1_score(feats["val"][1][va, d], clf.predict(proj("val")[va]), labels=[0, 1, 2], average="macro",
                          zero_division=0) if va.any() else -1
            if f1 > best[0]:
                best, best_clf = (f1, c), clf
        chosen[name] = best[1]
        for s in feats:
            pred[s][:, d] = best_clf.predict(proj(s))
    rows, confs = [], {}
    for s in ("val", "test"):
        r, c = evaluate_predictions(feats[s][1], pred[s], s, diseases)
        rows += r
        confs[s] = c
    write_results(out_dir, rows, confs, dict(experiment="frozen_screen", encoder=encoder.name, chosen_C=chosen,
                                              pca_components=pca_components, seed=seed))
    return dict(out_dir=out_dir, metrics=rows)
