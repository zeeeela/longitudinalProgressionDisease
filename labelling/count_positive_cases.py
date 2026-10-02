"""Count positive reports and unique positive patients per disease.

Example:
  python count_positive_cases.py --labels cxr_labels_selected_everpositive/study_labels.csv \
    --reviews cxr_labels_selected_everpositive/review_queue.csv \
    --vocab unique_diseases.csv --out disease_positive_counts.csv

Uncertain studies whose review was skipped remain uncertain and are not counted.
Each report/study and each patient is counted once per disease. This counts
report-derived labels, not clinically confirmed patient diagnoses.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


STATES = {"positive", "negative", "uncertain", "not_mentioned",
          "conflicting", "processing_error", "unlabeled"}
REVIEW_STATES = {"positive", "negative", "uncertain", "not_mentioned"}


def normalize_id(value, prefix):
    value = str(value or "").strip()
    match = re.fullmatch(rf"{prefix}?(\d+)(?:\.0)?", value, re.I)
    return match.group(1) if match else value


def row_key(row):
    patient = normalize_id(row.get("subject_id"), "p")
    study = normalize_id(row.get("study_id"), "s")
    disease = (row.get("disease") or "").strip()
    if not disease:
        raise ValueError("A row has no disease name.")
    if patient and study:
        report = ("study", patient, study)
    elif (row.get("report_path") or "").strip():
        report = ("path", row["report_path"].strip())
    else:
        raise ValueError("Each row needs subject_id + study_id, or report_path.")
    return patient, report, disease


def load_reviews(path):
    decisions = {}
    if path is None:
        return decisions
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"disease", "reviewed_state"}.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV needs disease and reviewed_state columns.")
        for row in reader:
            state = (row.get("reviewed_state") or "").strip().lower()
            if not state:
                continue
            if state not in REVIEW_STATES:
                raise ValueError(f"Invalid reviewed_state: {state!r}")
            patient, report, disease = row_key(row)
            key = (report, disease)
            if key in decisions and decisions[key] != state:
                raise ValueError(f"Conflicting manual decisions for {key}")
            decisions[key] = state
    return decisions


def count_cases(labels_path, reviews_path=None, vocab_path=None):
    decisions = load_reviews(reviews_path)
    patients = defaultdict(set)
    reports = defaultdict(set)
    diseases = set()
    # Reject contradictory duplicate final labels instead of silently counting
    # an arbitrary row as a positive study.
    final_by_report = {}
    if vocab_path:
        with vocab_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "disease" not in (reader.fieldnames or []):
                raise ValueError("Vocabulary CSV needs a disease column.")
            diseases.update((r.get("disease") or "").strip() for r in reader)
            diseases.discard("")
    with labels_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "disease" not in (reader.fieldnames or []):
            raise ValueError("Labels CSV needs a disease column.")
        if not {"automatic_state", "final_state"}.intersection(reader.fieldnames or []):
            raise ValueError("Labels CSV needs automatic_state or final_state.")
        for row_number, row in enumerate(reader, 2):
            patient, report, disease = row_key(row)
            diseases.add(disease)
            inline_review = (row.get("reviewed_state") or "").strip().lower()
            if inline_review and inline_review not in REVIEW_STATES:
                raise ValueError(f"Invalid reviewed_state at row {row_number}: {inline_review!r}")
            # A supplied review queue has the latest decisions. Blank review
            # cells never erase a decision already present in the labels file.
            state = (decisions.get((report, disease)) or inline_review
                     or (row.get("final_state") or "").strip().lower()
                     or (row.get("automatic_state") or "").strip().lower())
            if state not in STATES:
                raise ValueError(f"Invalid label at row {row_number}: {state!r}")
            key = (report, disease)
            if key in final_by_report and final_by_report[key] != state:
                raise ValueError(f"Conflicting duplicate labels for {key}; resolve them before counting.")
            final_by_report[key] = state
            if state == "positive":
                if not patient:
                    raise ValueError(f"Positive label at row {row_number} has no subject_id.")
                patients[disease].add(patient)
                reports[disease].add(report)
    return [
        {"disease": disease, "positive_patient_count": len(patients[disease]),
         "positive_report_count": len(reports[disease])}
        for disease in sorted(diseases, key=lambda d: (-len(patients[d]), -len(reports[d]), d))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, help="Review queue with your edited reviewed_state values")
    parser.add_argument("--vocab", type=Path, help="Include vocabulary diseases with zero positive cases")
    parser.add_argument("--out", type=Path, default=Path("disease_positive_counts.csv"))
    args = parser.parse_args()
    output = args.out.resolve()
    if output in {p.resolve() for p in (args.labels, args.reviews, args.vocab) if p}:
        parser.error("Output must differ from the input files.")
    rows = count_cases(args.labels, args.reviews, args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["disease", "positive_patient_count", "positive_report_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'Disease':50} {'Patients':>10} {'Reports':>10}")
    for row in rows:
        print(f"{row['disease']:50} {row['positive_patient_count']:10} {row['positive_report_count']:10}")
    print(f"\nSaved {len(rows)} diseases to: {output}")
    print("Only final positive states are counted. Counts across diseases are not additive.")


if __name__ == "__main__":
    main()
