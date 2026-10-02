"""Conservative report-based progression candidates and next-study targets.

Place beside scan_disease_v2.py. All next-study targets require verification
that the next report actually compares against the input study.
"""
import argparse
import csv
import re
from collections import defaultdict, Counter
from pathlib import Path

try:
    from .scan_disease_v2 import diagnostic_sections, read_patient_ids, report_identifiers, read_dicom_interval, read_study_metadata
except ImportError:
    from scan_disease_v2 import diagnostic_sections, read_patient_ids, report_identifiers, read_dicom_interval, read_study_metadata

DISEASES = {
    "Pleural effusion": r"\b(?:pleural\s+effusions?|effusions?)\b",
    "Cardiomegaly": r"\b(?:cardiomegaly|cardiac\s+silhouette|heart\s+size|cardiac\s+enlargement|enlarged\s+heart)\b",
    "Edema": r"\b(?:o?edema)\b",
    "Pneumonia": r"\b(?:pneumonia|pneumonias)\b",
    "Pulmonary edema": r"\b(?:(?:pulmonary|interstitial|alveolar)\s+o?edema)\b",
}
IMPROVING = re.compile(r"\b(?:improved|improving|improvement|resolved|resolving|resolution|decreased|decreasing|decrease|smaller|reduced|reducing|reduction|clearing|cleared|less\s+(?:prominent|pronounced|extensive))\b", re.I)
WORSENING = re.compile(r"\b(?:worse|worsened|worsening|increased|increasing|increase|progression|progressive|progressing|larger|enlarging|more\s+(?:prominent|extensive|pronounced)|new(?:ly)?(?:\s+(?:seen|identified|develop\w*|appear\w*))?)\b", re.I)
STABLE = re.compile(r"\b(?:unchanged|stable|no\s+(?:(?:significant|interval|appreciable)\s+)*change|similar\s+(?:to|in\s+appearance))\b", re.I)
UNCERTAIN = re.compile(r"\b(?:possible|possibly|probable|probably|questionable|may|might|could|cannot|can't|difficult\s+to\s+exclude|suggestive|suspected)\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|without|neither)\b", re.I)
COMPARISON = re.compile(r"(?im)^\s*comparison\s*:\s*(.*)$")


def classify_trends_strict(report):
    """Assign a trend only when a clause names one target finding family."""
    votes = defaultdict(set)
    evidence = defaultdict(list)
    reasons = defaultdict(set)
    matchers = {name: re.compile(pattern, re.I) for name, pattern in DISEASES.items()}
    for section, text in diagnostic_sections(report):
        # Join wrapped lines; section extraction already preserves boundaries.
        text = " ".join(text.split())
        for clause in re.split(r"(?<=[.!?;])\s+|\b(?:but|however|whereas)\b", text, flags=re.I):
            found = {name for name, pattern in matchers.items() if pattern.search(clause)}
            if not found:
                continue
            cues = []
            for label, pattern in (("stable", STABLE), ("improving", IMPROVING), ("worsening", WORSENING)):
                for cue in pattern.finditer(clause):
                    before = clause[max(0, cue.start() - 45):cue.start()]
                    if NEGATION.search(before):
                        continue
                    cues.append(label)
            # Generic and pulmonary edema intentionally overlap but are flagged.
            families = found - {"Edema"} if "Pulmonary edema" in found else found
            for disease in found:
                evidence[disease].append(f"{section}: {clause.strip()}")
                if "Edema" in found and "Pulmonary edema" in found:
                    reasons[disease].add("overlapping_edema_labels")
                if not cues:
                    continue
                if len(families) > 1:
                    reasons[disease].add("multiple_diseases_in_clause")
                elif UNCERTAIN.search(clause):
                    reasons[disease].add("uncertain_comparison")
                else:
                    votes[disease].update(cues)
    rows = []
    for disease in DISEASES:
        labels = votes[disease]
        trend = next(iter(labels)) if len(labels) == 1 else "mixed" if labels else "no_comparison"
        if len(labels) > 1:
            reasons[disease].add("conflicting_trends")
        if not labels:
            reasons[disease].add("no_unambiguous_comparison")
        rows.append({"disease": disease, "automatic_trend": trend,
                     "evidence": " | ".join(dict.fromkeys(evidence[disease])),
                     "needs_review": bool(reasons[disease]),
                     "review_reason": ";".join(sorted(reasons[disease])), "reviewed_trend": ""})
    return rows


# Disease-specific aliases; do not treat opacity/consolidation as pneumonia.
LENIENT_DISEASES = dict(DISEASES)
LENIENT_DISEASES["Cardiomegaly"] = r"\b(?:cardiomegaly|cardiac\s+(?:silhouette|size|enlargement|contour)|heart\s+size|enlarged\s+heart|cardiomediastinal\s+(?:silhouette|contour))\b"
LENIENT_DISEASES["Pulmonary edema"] = r"\b(?:o?edema)\b"
LENIENT_STABLE = re.compile(r"\b(?:unchanged|stable|no\s+(?:(?:significant|interval|appreciable)\s+)*change|similar|unchanging|remains?\s+(?:enlarged|unchanged|stable))\b", re.I)


def friend_report_trend(report):
    """Report-wide vote method supplied by user, including its original rules."""
    improving = re.compile(r"\b(?:improv(?:ed|ing|ement)|resolv(?:ed|ing)|resolution|decreas(?:ed|ing)|smaller|reduc(?:ed|tion|ing)|clearing|less (?:prominent|pronounced))\b", re.I)
    worsening = re.compile(r"\b(?:worsen(?:ed|ing)|increas(?:ed|ing)|progress(?:ed|ion|ive|ing)|new(?:ly)?\s+(?:seen|identified|develop\w\*|appear\w\*)|enlarg(?:ed|ing|ement)|larger|more (?:prominent|extensive|pronounced)|extensive)\b", re.I)
    stable = re.compile(r"\b(?:unchanged|stable|no (?:significant |interval )?change|similar (?:to|in|appearance))\b", re.I)
    votes = []
    for _, text in diagnostic_sections(report):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            for pattern, label in ((improving, "improving"), (worsening, "worsening"), (stable, "stable")):
                for cue in pattern.finditer(sentence):
                    if not re.search(r"\b(?:no|without|not)\b", sentence[max(0, cue.start()-40):cue.start()], re.I):
                        votes.append(label)
    top = Counter(votes).most_common()
    return "no_comparison" if not top else "mixed" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def classify_trends(report, mode="lenient"):
    strict = classify_trends_strict(report)
    baseline = friend_report_trend(report)
    if mode == "strict":
        for row in strict:
            row.update(strict_trend=row["automatic_trend"], friend_report_trend=baseline,
                       disease_mentioned=bool(row["evidence"]))
        return strict
    votes, evidence, reasons = defaultdict(list), defaultdict(list), defaultdict(set)
    mentioned = set()
    for section, text in diagnostic_sections(report):
        for clause in re.split(r"(?<=[.!?;])\s+|\b(?:but|however|whereas)\b", " ".join(text.split()), flags=re.I):
            hits = [(disease, match) for disease, pattern in LENIENT_DISEASES.items()
                    for match in re.finditer(pattern, clause, re.I)]
            mentioned.update(d for d, _ in hits)
            cues = []
            for label, pattern in (("stable", LENIENT_STABLE), ("improving", IMPROVING), ("worsening", WORSENING)):
                for cue in pattern.finditer(clause):
                    # Scope the cue to the local phrase, rather than all preceding words.
                    before = re.split(r"[,;]|\band\b", clause[:cue.start()], flags=re.I)[-1][-35:]
                    if NEGATION.search(before):
                        continue
                    cues.append((label, cue))
            for disease, mention in hits:
                evidence[disease].append(f"{section}: {clause.strip()}")
                if disease == "Pulmonary edema" and not re.search(DISEASES[disease], clause, re.I):
                    reasons[disease].add("bare_edema_assumed_pulmonary")
                if not cues:
                    continue
                # A single cue can describe a coordinated list: stable edema and effusions.
                # With multiple cues use the nearest one to each disease mention.
                distances = [(max(0, cue.start()-mention.end(), mention.start()-cue.end()), label)
                             for label, cue in cues]
                distance = min(d for d, _ in distances)
                if distance > 80:
                    reasons[disease].add("trend_too_far_from_disease")
                    continue
                nearest = {label for d, label in distances if d == distance}
                if len(nearest) != 1:
                    reasons[disease].add("ambiguous_trend_attachment")
                    continue
                votes[disease].append(next(iter(nearest)))
                if len({d for d, _ in hits} - {"Edema"}) > 1:
                    reasons[disease].add("multi_disease_attachment")
                if UNCERTAIN.search(clause):
                    reasons[disease].add("uncertain_comparison")
    rows = []
    for original in strict:
        disease = original["disease"]
        tally = Counter(votes[disease]).most_common()
        label = "no_comparison" if not tally else "mixed" if len(tally) > 1 and tally[0][1] == tally[1][1] else tally[0][0]
        if len(tally) > 1:
            reasons[disease].add("conflicting_trends_majority_vote")
        if not tally:
            reasons[disease].add("no_comparison" if disease in mentioned else "disease_not_mentioned")
        rows.append({"disease": disease, "automatic_trend": label,
                     "evidence": " | ".join(dict.fromkeys(evidence[disease])),
                     "needs_review": bool(reasons[disease]), "review_reason": ";".join(sorted(reasons[disease])),
                     "reviewed_trend": "", "strict_trend": original["automatic_trend"],
                     "friend_report_trend": baseline, "disease_mentioned": disease in mentioned})
    return rows


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_id(value):
    return re.sub(r"\.0$", "", re.sub(r"^[ps]", "", str(value).strip(), flags=re.I))


def read_presence_labels(path, reviews_path=None):
    """Read explicit states, retaining reviewed decisions and rejecting duplicates."""
    states = {}
    valid = {"positive", "negative", "uncertain", "not_mentioned", "conflicting", "processing_error", "unlabeled"}
    def key(row):
        return (normalize_id(row["subject_id"]), normalize_id(row["study_id"]), row["disease"].strip())
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"subject_id", "study_id", "disease"}.issubset(reader.fieldnames or []):
            raise ValueError("Labels need subject_id, study_id, disease columns.")
        if not {"reviewed_state", "final_state", "automatic_state"}.intersection(reader.fieldnames or []):
            raise ValueError("Labels need reviewed_state, final_state, or automatic_state.")
        for row in reader:
            state = next((row.get(c, "").strip().lower() for c in ("reviewed_state", "final_state", "automatic_state") if row.get(c, "").strip()), "")
            if state not in valid:
                raise ValueError(f"Invalid disease state: {state!r}")
            identity = key(row)
            if identity in states and states[identity] != state:
                raise ValueError(f"Conflicting disease labels: {identity}")
            states[identity] = state
    if reviews_path:
        reviewed = {}
        with reviews_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"subject_id", "study_id", "disease", "reviewed_state"}.issubset(reader.fieldnames or []):
                raise ValueError("Reviews need subject_id, study_id, disease, reviewed_state.")
            for row in reader:
                state = row["reviewed_state"].strip().lower()
                if not state:
                    continue
                if state not in {"positive", "negative", "uncertain", "not_mentioned"}:
                    raise ValueError(f"Invalid manual disease state: {state!r}")
                identity = key(row)
                if identity in reviewed and reviewed[identity] != state:
                    raise ValueError(f"Conflicting review decisions: {identity}")
                reviewed[identity] = state
        states.update(reviewed)
    return states


def combine_transition(text_trend, previous_state, current_state):
    """Add binary transitions without converting unchanged presence into stable."""
    transition = ("improving" if (previous_state, current_state) == ("positive", "negative")
                  else "worsening" if (previous_state, current_state) == ("negative", "positive") else "")
    if not transition:
        return text_trend, "report_trend" if text_trend != "no_comparison" else "no_evidence", "", ""
    if text_trend == "no_comparison":
        return transition, "presence_transition", transition, "presence_transition_requires_validation"
    if text_trend == transition:
        return text_trend, "report_and_presence_transition", transition, "presence_transition_requires_validation"
    return "mixed", "conflicting_evidence", transition, "report_presence_disagreement"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--patients", type=Path)
    parser.add_argument("--study-metadata", type=Path)
    parser.add_argument("--out", type=Path, default=Path("prognosis_labels"))
    parser.add_argument("--mode", choices=["strict", "lenient"], default="lenient")
    parser.add_argument("--labels", type=Path, help="Disease presence study_labels.csv; enables transition candidates")
    parser.add_argument("--reviews", type=Path, help="Optional reviewed disease presence queue (not prognosis reviews)")
    args = parser.parse_args(argv)
    if args.reviews and not args.labels:
        parser.error("--reviews requires --labels.")
    presence = read_presence_labels(args.labels, args.reviews) if args.labels else {}
    selected = read_patient_ids(args.patients) if args.patients else None
    studies = {}
    for root in args.reports:
        if not root.is_dir():
            parser.error(f"Report directory not found: {root}")
        for path in sorted(root.rglob("s*.txt")):
            identifiers = report_identifiers(path)
            subject, study = identifiers["subject_id"], identifiers["study_id"]
            if not subject or not study or (selected is not None and subject not in selected):
                continue
            key = (subject, study)
            if key in studies and studies[key] != path:
                parser.error(f"Duplicate study {key}: select a single report source.")
            studies[key] = path
    if not studies:
        parser.error("No selected reports found.")
    metadata = read_study_metadata(args.study_metadata, set(studies)) if args.study_metadata else None
    if metadata is None:
        try:
            import pydicom
        except ImportError:
            parser.error("Install pydicom in your notebook or supply --study-metadata.")
    grouped = defaultdict(list)
    trends = []
    issues = []
    for index, ((subject, study), path) in enumerate(studies.items(), 1):
        report = path.read_text(encoding="utf-8", errors="replace")
        try:
            interval = metadata.get((subject, study)) if metadata is not None else read_dicom_interval(path)
        except Exception as error:
            interval = None
            issues.append({"subject_id": subject, "study_id": study, "reason": f"timestamp_error: {error}"})
        if interval is None:
            issues.append({"subject_id": subject, "study_id": study, "reason": "missing_timestamp; patient excluded from pairing"})
        comparison = COMPARISON.search(report)
        data = {"subject_id": subject, "study_id": study, "report_path": str(path),
                "study_start": interval[0].isoformat() if interval else "",
                "study_end": interval[1].isoformat() if interval else "",
                "comparison_text": comparison.group(1).strip() if comparison else ""}
        labels = classify_trends(report, args.mode)
        trends.extend({**data, **label} for label in labels)
        grouped[subject].append((interval, data, labels))
        if index % 100 == 0:
            print(f"Read {index}/{len(studies)} reports", flush=True)
    pairs = []
    for subject, entries in grouped.items():
        if any(interval is None for interval, _, _ in entries):
            continue  # Do not silently skip an intervening undated study.
        entries.sort(key=lambda entry: entry[0][0])
        if any(a[0][1] >= b[0][0] for a, b in zip(entries, entries[1:])):
            issues.append({"subject_id": subject, "study_id": "", "reason": "overlapping_timestamps; patient excluded from pairing"})
            continue
        for previous, current in zip(entries, entries[1:]):
            for label in current[2]:
                reasons = set(filter(None, label["review_reason"].split(";")))
                previous_state = presence.get((subject, previous[1]["study_id"], label["disease"]), "unlabeled")
                current_state = presence.get((subject, current[1]["study_id"], label["disease"]), "unlabeled")
                candidate, basis, transition, transition_reason = combine_transition(label["automatic_trend"], previous_state, current_state)
                if transition_reason:
                    reasons.add(transition_reason)
                if label["automatic_trend"] != "no_comparison":
                    reasons.add("verify_comparison_refers_to_input_study")
                if basis == "presence_transition":
                    reasons.discard("no_comparison")
                    reasons.discard("no_unambiguous_comparison")
                    reasons.discard("disease_not_mentioned")
                pairs.append({"subject_id": subject, "input_study_id": previous[1]["study_id"],
                              "target_study_id": current[1]["study_id"],
                              "input_report_path": previous[1]["report_path"],
                              "target_report_path": current[1]["report_path"],
                              "input_date": previous[1]["study_start"], "target_date": current[1]["study_start"],
                              "gap_days": round((current[0][0] - previous[0][0]).total_seconds() / 86400, 4),
                              "disease": label["disease"], "candidate_target": candidate,
                              "text_target": label["automatic_trend"], "presence_transition": transition,
                              "label_basis": basis, "input_disease_state": previous_state,
                              "target_disease_state": current_state,
                              "strict_target": label["strict_trend"], "friend_report_target": label["friend_report_trend"],
                              "target_disease_mentioned": label["disease_mentioned"],
                              "target_evidence": label["evidence"], "comparison_text": current[1]["comparison_text"],
                              "needs_review": True, "review_reason": ";".join(sorted(reasons)),
                              "comparison_verified": "", "reviewed_target": ""})
    args.out.mkdir(parents=True, exist_ok=True)
    trend_fields = list(trends[0])
    pair_fields = ["subject_id", "input_study_id", "target_study_id", "input_report_path", "target_report_path", "input_date", "target_date", "gap_days", "disease", "candidate_target", "text_target", "presence_transition", "label_basis", "input_disease_state", "target_disease_state", "strict_target", "friend_report_target", "target_disease_mentioned", "target_evidence", "comparison_text", "needs_review", "review_reason", "comparison_verified", "reviewed_target"]
    write_csv(args.out / "study_trends.csv", trends, trend_fields)
    write_csv(args.out / "prognosis_pairs.csv", pairs, pair_fields)
    write_csv(args.out / "review_queue.csv", pairs, pair_fields)
    write_csv(args.out / "ordering_issues.csv", issues, ["subject_id", "study_id", "reason"])
    comparison_counts = []
    for disease in DISEASES:
        subset = [p for p in pairs if p["disease"] == disease]
        for method, column in (("selected_mode", "candidate_target"), ("strict", "strict_target"), ("friend_report_wide", "friend_report_target")):
            counts = Counter(p[column] for p in subset)
            comparison_counts.append({"disease": disease, "method": method,
                                      **{label: counts[label] for label in ("improving", "stable", "worsening", "mixed", "no_comparison")}})
    write_csv(args.out / "method_comparison_counts.csv", comparison_counts,
              ["disease", "method", "improving", "stable", "worsening", "mixed", "no_comparison"])
    print(f"Saved {len(trends)} disease-study rows and {len(pairs)} candidate targets to {args.out.resolve()}")
    print("Verify comparison references and fill reviewed_target before training. Split by subject_id.")


if __name__ == "__main__":
    main()
