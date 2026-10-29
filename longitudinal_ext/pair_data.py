"""Pair dataset/loaders that read the notebook's PAIR_CSV and optionally merge MIMIC-IV tabular features."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import DISEASES
from .images import ImageStore

KEYS = ["subject_id", "input_study_id", "target_study_id"]  # input = prior study, target = current study
_IN_WORDS = ("input", "prior", "prev", "previous", "first", "baseline")
_TG_WORDS = ("target", "current", "curr", "follow", "second", "later")


def detect_image_columns(df, input_col=None, target_col=None):
    if input_col and target_col:
        return input_col, target_col
    def cands(words):
        out = []
        for c in df.columns:
            lc = c.lower()
            if any(w in lc for w in words) and any(k in lc for k in ("path", "file", "dcm", "dicom", "image", "img")) \
                    and not lc.endswith("_id") and "study" not in lc:
                out.append(c)
        return out
    a, b = cands(_IN_WORDS), cands(_TG_WORDS)
    if len(a) != 1 or len(b) != 1:
        raise ValueError(f"Cannot tell the image-path columns (found input={a}, target={b}). "
                         f"Pass input_col=... and target_col=... Columns: {list(df.columns)}")
    return a[0], b[0]


def default_tab_columns(tab_df, kinds=("cur", "delta", "static", "interval")):
    out = []
    for c in tab_df.columns:
        if c in KEYS or c.endswith("__missing"):
            continue
        if ("cur" in kinds and c.endswith("_cur")) or ("prev" in kinds and c.endswith("_prev")) \
                or ("delta" in kinds and c.endswith("_delta")) \
                or ("static" in kinds and c.startswith("static_")) \
                or ("interval" in kinds and c.startswith("interval_")):
            out.append(c)
    return out


class TabularScaler:
    """Robust (median/IQR) scaling fitted on TRAIN rows only; NaN -> 0 plus missing-indicators."""

    def __init__(self, min_coverage=0.05, clip=5.0):
        self.min_coverage, self.clip = min_coverage, clip

    def fit(self, df, cols):
        self.cols, self.med, self.scale = [], {}, {}
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() < self.min_coverage or s.nunique(dropna=True) < 2:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            sc = (q3 - q1) / 1.349
            if not sc > 1e-8:
                sc = s.std()
            self.cols.append(c)
            self.med[c] = float(s.median())
            self.scale[c] = float(sc) if sc > 1e-8 else 1.0
        self.mask_cols = [c for c in self.cols if not c.endswith("_delta") and pd.to_numeric(df[c], errors="coerce").isna().any()]
        self.names = self.cols + [f"{c}__missing" for c in self.mask_cols]
        return self

    def transform(self, df):
        X = np.zeros((len(df), len(self.names)), dtype=np.float32)
        for j, c in enumerate(self.cols):
            s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            z = np.clip((s - self.med[c]) / self.scale[c], -self.clip, self.clip)
            X[:, j] = np.where(np.isnan(z), 0.0, z)
        for k, c in enumerate(self.mask_cols):
            X[:, len(self.cols) + k] = pd.to_numeric(df[c], errors="coerce").isna().to_numpy(dtype=np.float32)
        return X


class PairDataset(Dataset):
    def __init__(self, df, store, input_col, target_col, diseases, tab=None):
        self.df = df.reset_index(drop=True)
        self.store, self.ic, self.tc = store, input_col, target_col
        self.y = self.df[list(diseases)].to_numpy(dtype=np.int64)
        self.tab = tab
        self.a, self.b = self.df[input_col].astype(str).tolist(), self.df[target_col].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        f = lambda p: torch.from_numpy(self.store.get(p)).unsqueeze(0).float() / 255.0
        t = torch.from_numpy(self.tab[i]) if self.tab is not None else torch.zeros(0)
        return {"x_prev": f(self.a[i]), "x_cur": f(self.b[i]), "tab": t,
                "y": torch.from_numpy(self.y[i]), "idx": torch.tensor(i)}


def _keystr(df):
    out = df.copy()
    for k in KEYS:
        out[k] = out[k].astype(str)
    return out


def make_fusion_loaders(manifest_csv, *, image_size=224, batch_size=8, image_cache=None, keep_aspect=False,
                        tab_csv=None, tab_columns=None, tab_control=None, seed=42, num_workers=0,
                        input_col=None, target_col=None, path_root=None, diseases=DISEASES):
    """Returns dict(loaders, frames, tab_names, scaler, input_col, target_col, n_tab).
    tab_control='shuffle' permutes tabular rows inside each split (negative control)."""
    df = _keystr(pd.read_csv(manifest_csv))
    ic, tc = detect_image_columns(df, input_col, target_col)
    if path_root:
        for c in (ic, tc):
            df[c] = df[c].astype(str).map(lambda p: p if Path(p).is_absolute() else str(Path(path_root) / p))
    df = df[(df[list(diseases)] != -100).any(axis=1)].reset_index(drop=True)
    first = Path(str(df[ic].iloc[0]))
    if not first.exists():
        raise FileNotFoundError(f"First image path not found: {first}. Use path_root=... if paths are relative.")

    use_tab = tab_csv is not None
    if use_tab:
        tab = _keystr(pd.read_csv(tab_csv))
        cols = tab_columns or default_tab_columns(tab)
        df = df.merge(tab[KEYS + [c for c in cols if c not in KEYS]], on=KEYS, how="left", validate="m:1")
    store = ImageStore(image_size, image_cache, keep_aspect)
    scaler = TabularScaler().fit(df[df["split"] == "train"], cols) if use_tab else None

    frames, loaders = {}, {}
    gen = torch.Generator().manual_seed(seed)
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        if part.empty:
            continue
        X = scaler.transform(part) if use_tab else None
        if use_tab and tab_control == "shuffle":
            X = X[np.random.default_rng(seed + 1).permutation(len(X))]
        frames[split] = part
        loaders[split] = DataLoader(
            PairDataset(part, store, ic, tc, diseases, X), batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, generator=gen if split == "train" else None,
            drop_last=False, pin_memory=torch.cuda.is_available())
    return dict(loaders=loaders, frames=frames, scaler=scaler, input_col=ic, target_col=tc,
                tab_names=scaler.names if use_tab else [], n_tab=len(scaler.names) if use_tab else 0)
