"""Merge accepted manual progression decisions into an existing paired manifest."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from data_loader.cxr_prognosis_dataloader import LABEL_TO_INDEX, select_input_dicom
from longitudinal_ext import DISEASES

KEYS = ("subject_id", "input_study_id", "target_study_id")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def prepare_reviewed_manifest(base_csv, reviews_csv, split_csv, reports_root, output_csv, *, verbose=True):
    """Preserve existing labels/splits; reject conflicts and add accepted review labels.

    The supplied review file is the explicitly selected manual-decision set. Its
    comparison notes are preserved in the audit, including assumed comparisons.
    No comparison note is rewritten to claim a verified comparison.
    """
    fields, base = read_csv(base_csv)
    _, reviews = read_csv(reviews_csv)
    _, splits = read_csv(split_csv)
    identity = lambda row: tuple(row[k].strip() for k in KEYS)
    assignments = {r["subject_id"].strip(): r["split"].strip() for r in splits}
    if len(assignments) != len(splits) or set(assignments.values()) - {"train", "val", "test"}:
        raise ValueError("Duplicate or invalid saved patient splits.")
    pairs = {identity(r): dict(r) for r in base}
    if len(pairs) != len(base):
        raise ValueError("Duplicate pairs in the base manifest.")
    for key, row in pairs.items():
        if row["split"] != assignments.get(key[0]):
            raise ValueError(f"Saved patient split disagrees for {key}.")
    audit, seen, paths = [], set(), {}
    for review in reviews:
        key = identity(review)
        disease = review["disease"].strip()
        target = review["reviewed_target"].strip().lower()
        if disease not in DISEASES or target not in LABEL_TO_INDEX:
            raise ValueError(f"Invalid reviewed disease/target: {key}, {disease}, {target}")
        if key + (disease,) in seen:
            raise ValueError(f"Duplicate manual decision: {key}, {disease}")
        seen.add(key + (disease,))
        if key[0] not in assignments or key[1] == key[2]:
            raise ValueError(f"Missing patient split or identical studies: {key}")
        if datetime.fromisoformat(review["input_date"]) >= datetime.fromisoformat(review["target_date"]):
            raise ValueError(f"Invalid study chronology: {key}")
        label = str(LABEL_TO_INDEX[target])
        is_new = key not in pairs
        if is_new:
            pairs[key] = {**dict(zip(KEYS, key)), "split": assignments[key[0]],
                          **{d: "-100" for d in DISEASES},
                          **{d + "_source": "" for d in DISEASES}}
            for prefix, study in (("prior", key[1]), ("current", key[2])):
                image_key = (key[0], study)
                if image_key not in paths:
                    paths[image_key] = str(select_input_dicom(
                        {"subject_id": key[0], "input_study_id": study}, reports_root))
                pairs[key][prefix + "_dicom_path"] = paths[image_key]
        row = pairs[key]
        old = row[disease]
        if old not in ("-100", label):
            raise ValueError(f"Review conflicts with existing label: {key}, {disease}: {old} -> {label}")
        row[disease], row[disease + "_source"] = label, "reviewed"
        audit.append({**dict(zip(KEYS, key)), "disease": disease, "split": row["split"],
                      "previous_label": old, "reviewed_target": target,
                      "comparison_verified": review.get("comparison_verified", ""),
                      "action": "new_pair" if is_new else "new_label" if old == "-100" else "confirmed_label"})
    for key, row in pairs.items():
        for col in ("prior_dicom_path", "current_dicom_path"):
            if not Path(row[col]).is_file():
                raise FileNotFoundError(row[col])
        if any(row[d] not in ("-100", "0", "1", "2") for d in DISEASES):
            raise ValueError(f"Invalid training labels: {key}")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs[k] for k in sorted(pairs))
    with output_csv.with_name("manual_review_merge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = {
        "base_pairs": len(base), "pairs": len(pairs), "added_pairs": len(pairs) - len(base),
        "reviewed_labels": len(audit), "new_labels": sum(r["previous_label"] == "-100" for r in audit),
        "confirmed_labels": sum(r["previous_label"] != "-100" for r in audit),
        "pairs_per_split": dict(Counter(r["split"] for r in pairs.values())),
        "reviewed_labels_per_split": dict(Counter(r["split"] for r in audit)),
        "comparison_notes": dict(Counter(r["comparison_verified"] for r in audit)),
        "source_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (base_csv, reviews_csv, split_csv)},
    }
    output_csv.with_name("manual_review_merge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(summary, indent=2))
    return output_csv, summary
