"""Reconstruct the original feature selection locally, without training models.

This reproduces the existing extractor, including its documented timing limitations.
It does not claim to recover the missing historical HPC run's exact settings.
"""
import json
from pathlib import Path

import pandas as pd

from longitudinal_ext.mimic_iv_features import build_pair_features, coverage_report
from longitudinal_ext.pair_data import KEYS, TabularScaler, _keystr, default_tab_columns


def audit_features(manifest_csv, tab_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _keystr(pd.read_csv(manifest_csv))
    tab = _keystr(pd.read_csv(tab_csv))
    candidates = default_tab_columns(tab)
    merged = manifest.merge(tab[KEYS + candidates], on=KEYS, how="left", validate="m:1")
    train = merged.loc[merged.split == "train"]
    scaler = TabularScaler().fit(train, candidates)
    rows = []
    for col in candidates:
        values = pd.to_numeric(train[col], errors="coerce")
        coverage = float(values.notna().mean())
        distinct = int(values.nunique())
        rows.append(dict(feature=col, train_coverage=coverage, train_unique=distinct,
                         retained=col in scaler.cols,
                         reason="retained" if col in scaler.cols else
                         "training coverage below 5%" if coverage < .05 else "fewer than two training values",
                         missing_indicator=col in scaler.mask_cols))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "feature_selection_audit.csv", index=False)
    coverage_report(tab).to_csv(out_dir / "feature_coverage.csv")
    result = dict(provenance="local reconstruction using original extractor; not historical HPC settings",
                  manifest=str(manifest_csv), tab_csv=str(tab_csv), n_pairs=len(manifest),
                  n_train=len(train), n_candidates=len(candidates), n_tab=len(scaler.names),
                  retained_features=scaler.cols, missing_indicators=[c + "__missing" for c in scaler.mask_cols],
                  tab_names=scaler.names)
    (out_dir / "retained_features.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return table, result


def main():
    root = Path(__file__).resolve().parent
    out = root / "extensions_outputs" / "local_feature_audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest = root / "labelling/prognosis_labels_hybrid/all_disease_pairs.csv"
    frame = pd.read_csv(manifest)
    for col in ("prior_dicom_path", "current_dicom_path"):
        missing = [p for p in frame[col].unique() if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images in {col}; first: {missing[0]}")
    print(f"All image paths exist for {len(frame)} pairs", flush=True)
    tab = out / "pair_tabular_features.csv"
    if not tab.exists():
        build_pair_features(frame, "prior_dicom_path", "current_dicom_path",
                            root.parent / "MIMIC-IV/3.1", out / "mimic_cache", tab)
    table, result = audit_features(manifest, tab, out)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
