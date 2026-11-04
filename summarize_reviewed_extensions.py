"""Compare the reviewed-data extension rerun with archived notebook results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from longitudinal_ext import DISEASES

ROOT = Path(__file__).resolve().parent
RUN = Path((ROOT / "training_outputs/latest_reviewed_run.txt").read_text().strip())
KEYS = ["subject_id", "input_study_id", "target_study_id"]
# Archived mean-over-disease macro F1, rounded in the original notebook output.
# Source: historical_extension_results.md, original cell 43.
HISTORICAL = {
    "A_base_img": (0.379, 0.316),
    "A2_base_diag_img": (0.428, 0.414),
    "B_base_img_tab": (0.381, 0.378),
    "C_new_img": (0.410, 0.409),
    "D_new_img_tab": (0.453, 0.380),
    "E_new_img_tab_shuffled": (0.405, 0.435),
    "F_tab_nn": (0.312, 0.290),
}


def read(path):
    return pd.read_csv(path, dtype={k: str for k in KEYS})


def means_from_confusion(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(axis=-1) + cm.sum(axis=-2)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0).mean(axis=-1)
    return np.nanmean(np.where(cm.sum(axis=(-1, -2)) > 0, f1, np.nan), axis=-1)


def main():
    old = read(ROOT / "labelling/prognosis_labels_hybrid/all_disease_pairs.csv")
    current = read(RUN / "data/all_disease_pairs_reviewed.csv")
    manifest = current.set_index(KEYS)
    runs, predictions, metrics, rows = {}, {}, [], []
    for name, (old_val, old_test) in HISTORICAL.items():
        completed = sorted(p for p in RUN.glob(f"ext_{name}_s42_*")
                           if (p / "metrics.csv").is_file() and (p / "predictions.csv").is_file())
        if not completed:
            raise RuntimeError(f"Incomplete experiment: {name}")
        folder = completed[-1]
        runs[name] = str(folder)
        history = pd.read_csv(folder / "history.csv")
        if len(history) != 20 or history["epoch"].tolist() != list(range(1, 21)):
            raise ValueError(f"Expected 20 complete epochs: {folder}")
        scores = pd.read_csv(folder / "metrics.csv")
        predictions[name] = pred = read(folder / "predictions.csv")
        if pred.duplicated(KEYS).any():
            raise ValueError(f"Duplicate prediction keys: {name}")
        expected = manifest.loc[pd.MultiIndex.from_frame(pred[KEYS])]
        for d in DISEASES:
            if not np.array_equal(expected[d].to_numpy(), pred[f"true_{d}"].to_numpy()):
                raise ValueError(f"Prediction labels disagree with reviewed manifest: {name}, {d}")
        if not np.array_equal(expected["split"].to_numpy(), pred["split"].to_numpy()):
            raise ValueError(f"Prediction split mismatch: {name}")
        scores.insert(0, "experiment", name)
        metrics.append(scores)
        means = scores.groupby("split")["macro_f1"].mean()
        row = {"experiment": name, "old_val": old_val, "new_val": means["val"],
               "old_test": old_test, "new_test": means["test"]}
        joined = pred.merge(old, on=KEYS, how="inner", suffixes=("", "_old"), validate="1:1")
        for split in ("val", "test"):
            subset = joined[joined["split"] == split]
            shared = []
            for d in DISEASES:
                valid = subset[d] != -100
                if not np.array_equal(subset.loc[valid, d], subset.loc[valid, f"true_{d}"]):
                    raise ValueError("Original evaluation labels changed.")
                shared.append(f1_score(subset.loc[valid, d], subset.loc[valid, f"pred_{d}"],
                                       labels=[0, 1, 2], average="macro", zero_division=0))
            row[f"new_{split}_original_cases"] = float(np.mean(shared))
        rows.append(row)
    comparison = pd.DataFrame(rows).set_index("experiment")
    comparison.to_csv(RUN / "extension_before_after.csv")
    pd.concat(metrics).to_csv(RUN / "extension_metrics.csv", index=False)

    # Resample patients jointly for paired model comparisons; repeated pairs stay together.
    reference = predictions["D_new_img_tab"].query("split == 'test'").sort_values(KEYS).reset_index(drop=True)
    patients = sorted(reference.subject_id.unique())
    patient_index = {p: i for i, p in enumerate(patients)}
    counts = np.random.default_rng(42).multinomial(len(patients), np.ones(len(patients)) / len(patients), size=2000)
    boot, point = {}, {}
    for name in ("A_base_img", "B_base_img_tab", "C_new_img", "D_new_img_tab", "E_new_img_tab_shuffled"):
        data = predictions[name].query("split == 'test'").sort_values(KEYS).reset_index(drop=True)
        if not data[KEYS].equals(reference[KEYS]):
            raise ValueError("Paired comparisons require identical evaluation pairs.")
        cm = np.zeros((len(patients), len(DISEASES), 3, 3), dtype=int)
        subjects = data.subject_id.map(patient_index).to_numpy()
        for i, d in enumerate(DISEASES):
            y, yp = data[f"true_{d}"].to_numpy(), data[f"pred_{d}"].to_numpy()
            valid = y != -100
            np.add.at(cm, (subjects[valid], i, y[valid], yp[valid]), 1)
        point[name] = float(means_from_confusion(cm.sum(axis=0)))
        boot[name] = means_from_confusion(np.tensordot(counts, cm, axes=(1, 0)))
    differences = []
    for title, left, right in (
        ("DenseNet: matched tabular minus image only", "B_base_img_tab", "A_base_img"),
        ("RAD-DINO: matched tabular minus image only", "D_new_img_tab", "C_new_img"),
        ("RAD-DINO: matched tabular minus shuffled", "D_new_img_tab", "E_new_img_tab_shuffled"),
    ):
        low, high = np.quantile(boot[left] - boot[right], [0.025, 0.975])
        differences.append({"comparison": title, "test_difference": point[left] - point[right],
                            "ci_95_low": float(low), "ci_95_high": float(high)})
    diff = pd.DataFrame(differences)
    diff.to_csv(RUN / "extension_paired_comparisons.csv", index=False)
    support = []
    for split in ("train", "val", "test"):
        for d in DISEASES:
            support.append({"split": split, "disease": d,
                            "old_labels": int(((old.split == split) & (old[d] != -100)).sum()),
                            "new_labels": int(((current.split == split) & (current[d] != -100)).sum())})
    pd.DataFrame(support).to_csv(RUN / "label_support_before_after.csv", index=False)
    (RUN / "extension_run_registry.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    lines = ["# Extension rerun with reviewed labels", "",
             "20 epochs per neural model; seed 42; original patient split assignments preserved. "
             "C/D/E retain RAD-DINO for comparison with the historical experiments.", "",
             "915 to 937 pairs; 1,450 to 1,500 disease labels. The 307 accepted manual decisions "
             "confirm 257 existing labels and add 50 labels. Seven conflicting progression reviews "
             "and all reviewed no_comparison decisions were excluded from the manual additions.", "",
             "## Mean macro F1 across five diseases", "",
             "Historical values come from historical_extension_results.md and were rounded to three decimals.", "",
             "| Experiment | Old validation | New validation | Old test | New test | New test, original cases |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in comparison.iterrows():
        lines.append(f"| {name} | {r.old_val:.3f} | {r.new_val:.3f} | {r.old_test:.3f} | {r.new_test:.3f} | {r.new_test_original_cases:.3f} |")
    lines += ["", f"Highest validation mean: **{comparison.new_val.idxmax()}**. "
              f"Highest observed test mean: **{comparison.new_test.idxmax()}**. Model selection uses validation.", "",
              "## Paired test contrasts", "",
              "95% percentile intervals use 2,000 paired bootstrap resamples of test patients. "
              "All pairs for a patient are resampled together. These intervals cover patient sampling "
              "uncertainty, not variability across training seeds.", "",
              "| Comparison | Difference | 95% interval |", "|---|---:|---:|"]
    for r in differences:
        lines.append(f"| {r['comparison']} | {r['test_difference']:+.3f} | [{r['ci_95_low']:+.3f}, {r['ci_95_high']:+.3f}] |")
    lines += ["", "## Interpretation limits", "",
              "This is a one-seed exploratory rerun. New full-cohort scores include additional validation "
              "and test labels; the original-case column holds evaluation cases and labels fixed. "
              "It does not control training randomness, the retrained diagnosis checkpoint for A2, "
              "or any unrecovered historical HPC feature differences. The local tabular pipeline retains "
              "64 inputs; its provenance is a local reconstruction rather than recovered HPC settings. "
              "Differences therefore cannot be attributed solely to the 50 added labels.", "",
              "The manual-review audit preserves comparison notes, including assumed or unavailable "
              "comparison references; review completion does not establish clinical adjudication.", "",
              "The diagnosis backbone uses four report-presence classes. Every extension result above "
              "uses only improving, stable, and worsening as targets.", ""]
    (RUN / "extension_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(comparison.round(4).to_string())
    print(diff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
