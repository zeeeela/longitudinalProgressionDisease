"""Provisional report labels using a disease CSV, synonyms, and context rules.

Install FlashText: python -m pip install flashtext
Example (WSL):
  python label_cxr_reports.py --vocab unique_diseases.csv --reports \
    /mnt/c/Users/Zila/Downloads/BN5212/MIMIC-CXR/p10_1/p10 \
    /mnt/c/Users/Zila/Downloads/BN5212/MIMIC-CXR/p10_2/p10

Optional synonyms CSV columns: alias,disease (disease must exist in the vocabulary).
Optional --patients CSV columns: Patient or subject_id. IDs can be p10000001
or 10000001. All reports belonging to those patients are included.
Optional --skip-uncertain-after-positive skips uncertainty review once an earlier
positive establishes patient ever-positive. Study labels remain unchanged.
Use --study-metadata (CSV/CSV.gz) or install pydicom to read local DICOM dates.
Optional --reviewed-labels imports human decisions from an earlier review CSV.
Only recognized Findings/Impression sections are searched. Missing sections
are marked unlabeled. These are heuristic mention labels, not clinical diagnoses.
"""

import argparse
import csv
import re
import gzip
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path

try:
    from flashtext import KeywordProcessor
except ImportError:
    KeywordProcessor = None


# Ignore case only for known headings, not for the generic uppercase-heading rule.
KNOWN_HEADING = re.compile(
    r"^[ \t]*(FINDINGS?(?:\s+AND\s+IMPRESSION)?|IMPRESSION|"
    r"INDICATION|CLINICAL INDICATION|HISTORY|CLINICAL HISTORY|COMPARISON|"
    r"TECHNIQUE|EXAM|EXAMINATION|RECOMMENDATIONS?|CONCLUSION)"
    r"[ \t]*(?::[ \t]*(.*)|[ \t]*)$", re.I
)
OTHER_HEADING = re.compile(r"^[ \t]*([A-Z][A-Z /_-]{2,}):[ \t]*(.*)$")
TARGET_HEADING = re.compile(r"^(?:FINDINGS?(?:\s+AND\s+IMPRESSION)?|IMPRESSION)$", re.I)

NEGATION_BEFORE_FIXED = re.compile(
    r"\b(?:no\b(?!\s+(?:change|interval change|significant change|"
    r"definite change|increase|decrease|improvement|worsening|longer)\b)|"
    r"without|absent|negative for|free of|neither|nor|"
    r"resolution of|resolved|cleared|absence of|ruled out|no longer)\b", re.I
)
NEGATION_AFTER_FIXED = re.compile(
    r"^\s*(?:(?:is|are|has|have|was|were|had)\s+)?"
    r"(?:(?:been|now|completely|fully|largely)\s+)*"
    r"(?:not seen|not identified|not present|absent|resolved|cleared|ruled out)\b",
    re.I,
)
UNCERTAINTY_FIXED = re.compile(
    r"\b(?:possible|possibly|probable|probably|questionable|may represent|"
    r"might represent|could represent|cannot exclude|cannot be excluded|"
    r"can not exclude|can't exclude|not excluded|difficult to exclude|"
    r"suspicious for|suggestive of|concerning for|rule out)\b", re.I
)
UNCERTAINTY_AFTER = re.compile(
    r"^\s*(?:(?:is|are|was|were|remains?)\s+)?"
    r"(?:possible|probable|questionable|suspected|not excluded|"
    r"cannot be excluded|can not be excluded|can't be excluded)\b", re.I
)
CLAUSE_BREAK = re.compile(r";|\b(?:but|however|although|yet|nevertheless)\b", re.I)


def diagnostic_sections(report):
    """Return separate (section name, text) pairs; never scan the whole report."""
    sections = []
    current_name = None
    lines = []
    for line in report.splitlines():
        heading = KNOWN_HEADING.match(line) or OTHER_HEADING.match(line)
        if heading:
            if current_name is not None:
                sections.append((current_name, "\n".join(lines)))
            name = " ".join(heading.group(1).upper().split())
            current_name = name if TARGET_HEADING.fullmatch(name) else None
            lines = [heading.group(2) or ""] if current_name else []
        elif current_name is not None:
            lines.append(line)
    if current_name is not None:
        sections.append((current_name, "\n".join(lines)))
    return sections


def paragraph_joined(text):
    return "\n\n".join(
        " ".join(block.split()) for block in re.split(r"\n\s*\n", text)
    )


def classify_mention(clause, start, end):
    before = clause[max(0, start - 90):start]
    after = clause[end:end + 65]
    # Protect uncertainty phrases such as "not excluded" before negation checks.
    if UNCERTAINTY_AFTER.search(after):
        return "uncertain"
    if NEGATION_AFTER_FIXED.search(after):
        return "negative"
    uncertainty = list(UNCERTAINTY_FIXED.finditer(before))
    if uncertainty:
        # Remove the uncertainty phrase itself before looking for genuine negation.
        # This remains a heuristic: cues can refer to other findings in the clause.
        cleaned = UNCERTAINTY_FIXED.sub("", before)
        if not NEGATION_BEFORE_FIXED.search(cleaned):
            return "uncertain"
    if NEGATION_BEFORE_FIXED.search(before):
        return "negative"
    return "positive"


def read_vocabulary(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "disease" not in reader.fieldnames:
            raise ValueError("Vocabulary CSV must contain a 'disease' column.")
        diseases = sorted({r["disease"].strip() for r in reader if r["disease"].strip()})
    if not diseases:
        raise ValueError("Vocabulary is empty.")
    return diseases


def build_matcher(diseases, synonyms=None):
    aliases = {d.lower(): d for d in diseases}
    if len(aliases) != len(diseases):
        raise ValueError("Vocabulary contains names differing only by case; deduplicate it.")
    if synonyms:
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"alias", "disease"}.issubset(reader.fieldnames or []):
                raise ValueError("Synonyms CSV needs alias,disease columns.")
            for row in reader:
                alias, disease = row["alias"].strip().lower(), row["disease"].strip()
                if disease not in diseases or not alias:
                    raise ValueError(f"Invalid synonym row: {row}")
                if alias in aliases and aliases[alias] != disease:
                    raise ValueError(f"Alias maps to two diseases: {alias}")
                aliases[alias] = disease
    if KeywordProcessor:
        processor = KeywordProcessor(case_sensitive=False)
        for alias, disease in aliases.items():
            processor.add_keyword(alias, disease)
        return lambda text: processor.extract_keywords(text, span_info=True), "FlashText"
    # A slower fallback lets the script run without installing additional packages.
    terms = sorted(aliases, key=lambda t: (-len(t), t))
    patterns = [re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms[i:i+300]))
                           + r")(?!\w)", re.I) for i in range(0, len(terms), 300)]

    def find(text):
        return [(aliases[m.group().lower()], m.start(), m.end())
                for pattern in patterns for m in pattern.finditer(text)]
    return find, "regex fallback (install flashtext for faster matching)"


def label_report(report, find_matches):
    sections = diagnostic_sections(report)
    mentions = []
    for name, text in sections:
        text = paragraph_joined(text)
        for sentence in re.split(r"(?<=[.!?])\s+|\n\n+", text):
            for clause in CLAUSE_BREAK.split(sentence):
                for disease, start, end in find_matches(clause):
                    mentions.append({
                        "disease": disease,
                        "state": classify_mention(clause, start, end),
                        "section": name,
                        "matched_text": clause[start:end],
                        "evidence": sentence.strip(),
                    })
    grouped = defaultdict(list)
    for mention in mentions:
        grouped[mention["disease"]].append(mention)
    labels = []
    for disease, rows in sorted(grouped.items()):
        states = {r["state"] for r in rows}
        conflict = "positive" in states and "negative" in states
        # Contradictory mentions are preserved for review, not silently collapsed.
        state = "conflicting" if conflict else next(
            s for s in ("positive", "uncertain", "negative") if s in states
        )
        labels.append({
            "disease": disease, "automatic_state": state,
            "mention_states": "|".join(sorted(states)),
            "needs_review": conflict or "uncertain" in states,
            "reviewed_state": "",
            "evidence": " | ".join(dict.fromkeys(r["evidence"] for r in rows)),
        })
    status = "unlabeled" if not sections else ("matched" if mentions else "no_matches")
    return mentions, labels, status


def report_identifiers(path):
    subject = next((p[1:] for p in reversed(path.parts) if re.fullmatch(r"p\d{8}", p)), "")
    study_match = re.fullmatch(r"s(\d+)", path.stem)
    study = study_match.group(1) if study_match else ""
    return {"subject_id": subject, "study_id": study, "report_path": str(path)}


def read_patient_ids(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        column = next((c for c in ("Patient", "subject_id")
                       if c in (reader.fieldnames or [])), None)
        if column is None:
            raise ValueError("Patient CSV must contain 'Patient' or 'subject_id'.")
        selected = set()
        for row_number, row in enumerate(reader, 2):
            value = (row.get(column) or "").strip()
            if not value:
                continue
            match = re.fullmatch(r"[pP]?(\d{8})", value)
            if not match:
                raise ValueError(f"Invalid patient ID at CSV row {row_number}: {value!r}")
            selected.add(match.group(1))
    if not selected:
        raise ValueError("Patient CSV contains no patient IDs.")
    return selected


def study_interval(date_value, time_value):
    """Unknown times cover the entire day, preventing guessed same-day order."""
    date_text = str(date_value or "").strip()
    date_text = re.sub(r"\.0$", "", date_text).replace("-", "")
    if not re.fullmatch(r"\d{8}", date_text):
        raise ValueError(f"Invalid StudyDate: {date_value!r}")
    day = datetime.strptime(date_text, "%Y%m%d").date()
    time_text = str(time_value or "").strip()
    if time_text.lower() in ("", "nan", "none"):
        return datetime.combine(day, time.min), datetime.combine(day, time.max)
    whole, dot, fraction = time_text.replace(":", "").partition(".")
    if not whole.isdigit() or len(whole) > 6 or (dot and not fraction.isdigit()):
        raise ValueError(f"Invalid StudyTime: {time_value!r}")
    whole = whole.zfill(6)
    clock = time(int(whole[:2]), int(whole[2:4]), int(whole[4:6]),
                 int((fraction + "000000")[:6]) if dot else 0)
    stamp = datetime.combine(day, clock)
    return stamp, stamp


def normalized_id(value, prefix):
    match = re.fullmatch(rf"{prefix}?(\d+)(?:\.0)?", str(value).strip(), re.I)
    if not match:
        raise ValueError(f"Invalid {prefix} identifier: {value!r}")
    return match.group(1)


def read_study_metadata(path, wanted):
    """Read only selected studies; combine image timestamps into study intervals."""
    intervals = {}
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"subject_id", "study_id", "StudyDate"}.issubset(reader.fieldnames or []):
            raise ValueError("Study metadata needs subject_id, study_id, StudyDate; StudyTime is optional.")
        for row in reader:
            key = (normalized_id(row["subject_id"], "p"), normalized_id(row["study_id"], "s"))
            if key not in wanted:
                continue
            interval = study_interval(row["StudyDate"], row.get("StudyTime", ""))
            if key in intervals:
                previous = intervals[key]
                interval = (min(previous[0], interval[0]), max(previous[1], interval[1]))
            intervals[key] = interval
    return intervals


def read_dicom_interval(report_path):
    import pydicom
    folder = report_path.parent / report_path.stem
    intervals = []
    for image in folder.rglob("*"):
        if not image.is_file() or image.suffix.lower() != ".dcm":
            continue
        ds = pydicom.dcmread(image, stop_before_pixels=True,
                            specific_tags=["StudyDate", "StudyTime"])
        intervals.append(study_interval(ds.get("StudyDate", ""), ds.get("StudyTime", "")))
    if not intervals:
        return None
    return min(i[0] for i in intervals), max(i[1] for i in intervals)


def read_reviewed_labels(path):
    """Use explicit prior human decisions instead of their automatic states."""
    reviewed = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"subject_id", "study_id", "disease", "reviewed_state"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Reviewed CSV needs subject_id, study_id, disease, reviewed_state.")
        for row in reader:
            state = (row["reviewed_state"] or "").strip().lower()
            if not state:
                continue
            if state not in ("positive", "negative", "uncertain", "not_mentioned"):
                raise ValueError(f"Invalid reviewed_state: {state!r}")
            key = (normalized_id(row["subject_id"], "p"),
                   normalized_id(row["study_id"], "s"), row["disease"].strip())
            if key in reviewed and reviewed[key] != state:
                raise ValueError(f"Contradictory manual decisions: {key}")
            reviewed[key] = state
    return reviewed


def apply_prior_positive_review_rule(label, subject_id, study_id, interval, history,
                                     reviewed, enabled):
    """Skip review only; do not turn an uncertain current study into positive."""
    disease = label["disease"]
    decision = reviewed.get((subject_id, study_id, disease), "")
    if decision:
        label["reviewed_state"] = decision
        label["needs_review"] = False
    label.update(review_skipped=False, review_skip_reason="", prior_positive_study_id="",
                 prior_positive_basis="")
    key = (subject_id, disease)
    if (enabled and label["needs_review"] and label["automatic_state"] == "uncertain"
            and interval is not None):
        earlier = [entry for entry in history[key]
                   if entry[0] < interval[0] and entry[1] != study_id]
        if earlier:
            prior = max(earlier, key=lambda entry: entry[0])
            label.update(needs_review=False, review_skipped=True,
                         review_skip_reason="Earlier positive establishes patient ever-positive",
                         prior_positive_study_id=prior[1], prior_positive_basis=prior[2])
    effective = decision or label["automatic_state"]
    if effective == "positive" and interval is not None:
        history[key].append((interval[1], study_id, "reviewed" if decision else "automatic"))
    return label


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--synonyms", type=Path)
    parser.add_argument("--patients", type=Path,
                        help="CSV of patient IDs to include (Patient or subject_id column)")
    parser.add_argument("--out", type=Path, default=Path("cxr_labels"))
    parser.add_argument("--skip-uncertain-after-positive", action="store_true",
                        help="Skip uncertain review for patient ever-positive labeling only")
    parser.add_argument("--study-metadata", type=Path,
                        help="Study dates CSV/CSV.gz; otherwise read dates from local DICOM headers")
    parser.add_argument("--reviewed-labels", type=Path,
                        help="Previous review CSV; explicit manual states override automatic history")
    args = parser.parse_args(argv)
    diseases = read_vocabulary(args.vocab)
    find_matches, engine = build_matcher(diseases, args.synonyms)
    for root in args.reports:
        if not root.is_dir():
            raise FileNotFoundError(f"Report directory does not exist: {root}")
    paths = sorted({p.resolve() for root in args.reports
                    for p in root.rglob("*.txt") if p.is_file()})
    if args.patients:
        selected = read_patient_ids(args.patients)
        paths = [p for p in paths if report_identifiers(p)["subject_id"] in selected]
        found = {report_identifiers(p)["subject_id"] for p in paths}
        missing = sorted(selected - found)
        print(f"Patient filter: {len(selected)} requested, {len(found)} with reports found.")
        if missing:
            print("Requested IDs with no reports found: " + ", ".join(missing))
    if not paths:
        raise ValueError("No .txt reports found matching the provided directories/patient filter.")
    reviewed = read_reviewed_labels(args.reviewed_labels) if args.reviewed_labels else {}
    timeline = {}
    if args.skip_uncertain_after_positive:
        identifiers_by_path = {p: report_identifiers(p) for p in paths}
        if args.study_metadata:
            wanted = {(r["subject_id"], r["study_id"]) for r in identifiers_by_path.values()}
            intervals = read_study_metadata(args.study_metadata, wanted)
            timeline = {p: intervals.get((r["subject_id"], r["study_id"]))
                        for p, r in identifiers_by_path.items()}
        else:
            try:
                import pydicom
            except ImportError:
                parser.error("Supply --study-metadata or install pydicom: python -m pip install pydicom")
            print("Reading dates from DICOM headers (image pixels are not loaded).", flush=True)
            for p in paths:
                try:
                    timeline[p] = read_dicom_interval(p)
                except Exception as exc:
                    timeline[p] = None
                    print(f"Date unavailable for {p}: {exc}")
        paths.sort(key=lambda p: (report_identifiers(p)["subject_id"],
                                 timeline[p][0] if timeline[p] else datetime.max, str(p)))
        missing_dates = sum(timeline[p] is None for p in paths)
        print(f"Reports without dates: {missing_dates}; these remain eligible for review.")
    history = defaultdict(list)
    args.out.mkdir(parents=True, exist_ok=True)
    common = ["subject_id", "study_id", "report_path"]
    columns = {
        "mentions": common + ["disease", "state", "section", "matched_text", "evidence"],
        "study_labels": common + ["disease", "automatic_state", "mention_states",
                                   "needs_review", "reviewed_state", "review_skipped",
                                   "review_skip_reason", "prior_positive_study_id",
                                   "prior_positive_basis", "study_start", "study_end", "evidence"],
        "review_queue": common + ["disease", "automatic_state", "mention_states",
                                   "needs_review", "reviewed_state", "review_skipped",
                                   "review_skip_reason", "prior_positive_study_id",
                                   "prior_positive_basis", "study_start", "study_end", "evidence"],
        "report_summary": common + ["status", "mention_count", "matched_disease_count",
                                    "skipped_uncertain_count"],
    }
    print(f"Matcher: {engine}; vocabulary: {len(diseases)}; reports: {len(paths)}")
    # Stream outputs instead of keeping every patient's results in memory.
    from contextlib import ExitStack
    with ExitStack() as stack:
        writers = {}
        for name, fields in columns.items():
            f = stack.enter_context((args.out / f"{name}.csv").open(
                "w", encoding="utf-8", newline=""))
            writers[name] = csv.DictWriter(f, fieldnames=fields)
            writers[name].writeheader()
        skipped_total = 0
        for i, path in enumerate(paths, 1):
            identifiers = report_identifiers(path)
            mentions, labels, status = label_report(
                path.read_text(encoding="utf-8", errors="replace"), find_matches)
            writers["mentions"].writerows({**identifiers, **m} for m in mentions)
            skipped_report = 0
            for label in labels:
                interval = timeline.get(path)
                apply_prior_positive_review_rule(label, identifiers["subject_id"],
                    identifiers["study_id"], interval, history, reviewed,
                    args.skip_uncertain_after_positive)
                label["study_start"] = interval[0].isoformat() if interval else ""
                label["study_end"] = interval[1].isoformat() if interval else ""
                skipped_report += int(label["review_skipped"])
                row = {**identifiers, **label}
                writers["study_labels"].writerow(row)
                if label["needs_review"]:
                    writers["review_queue"].writerow(row)
            writers["report_summary"].writerow({
                **identifiers, "status": status, "mention_count": len(mentions),
                "matched_disease_count": len(labels), "skipped_uncertain_count": skipped_report})
            skipped_total += skipped_report
            if i % 500 == 0 or i == len(paths):
                print(f"Scanned {i}/{len(paths)}", flush=True)
    print(f"Saved results to {args.out.resolve()}")
    print(f"Uncertain reviews skipped after an earlier positive: {skipped_total}")
    print("Labels are sparse: missing rows are NOT negative labels. Check report_summary.")
    print("Review samples of positives, negatives, and no_matches as well as review_queue.")


if __name__ == "__main__":
    main()
