#This script is like Bibit, Roger, and Zang Zheng's code in BN5212_Custom_Disease_Loader.ipynb to scan for diseases in the MIMIC-CXR dataset

import argparse
import csv
import re
from collections import defaultdict
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--synonyms", type=Path)
    parser.add_argument("--patients", type=Path,
                        help="CSV of patient IDs to include (Patient or subject_id column)")
    parser.add_argument("--out", type=Path, default=Path("cxr_labels"))
    args = parser.parse_args()
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
    args.out.mkdir(parents=True, exist_ok=True)
    common = ["subject_id", "study_id", "report_path"]
    columns = {
        "mentions": common + ["disease", "state", "section", "matched_text", "evidence"],
        "study_labels": common + ["disease", "automatic_state", "mention_states",
                                   "needs_review", "reviewed_state", "evidence"],
        "review_queue": common + ["disease", "automatic_state", "mention_states",
                                   "needs_review", "reviewed_state", "evidence"],
        "report_summary": common + ["status", "mention_count", "matched_disease_count"],
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
        for i, path in enumerate(paths, 1):
            identifiers = report_identifiers(path)
            mentions, labels, status = label_report(
                path.read_text(encoding="utf-8", errors="replace"), find_matches)
            writers["mentions"].writerows({**identifiers, **m} for m in mentions)
            for label in labels:
                row = {**identifiers, **label}
                writers["study_labels"].writerow(row)
                if label["needs_review"]:
                    writers["review_queue"].writerow(row)
            writers["report_summary"].writerow({
                **identifiers, "status": status, "mention_count": len(mentions),
                "matched_disease_count": len(labels)})
            if i % 500 == 0 or i == len(paths):
                print(f"Scanned {i}/{len(paths)}", flush=True)
    print(f"Saved results to {args.out.resolve()}")
    print("Labels are sparse: missing rows are NOT negative labels. Check report_summary.")
    print("Review samples of positives, negatives, and no_matches as well as review_queue.")


if __name__ == "__main__":
    main()
