"""Five-finding current-study diagnosis loader for backbone pretraining.

Uses report-derived four-state labels, not future prognosis targets.
Requires an existing patient split CSV shared with prognosis training.
"""
import csv
from collections import Counter
from pathlib import Path

try:
    from .cxr_prognosis_dataloader import select_input_dicom, load_dicom_tensor
    from labelling.build_prognosis_labels import read_presence_labels, normalize_id
except ImportError:
    from cxr_prognosis_dataloader import select_input_dicom, load_dicom_tensor
    from labelling.build_prognosis_labels import read_presence_labels, normalize_id

DISEASES = ["Pleural effusion", "Cardiomegaly", "Edema", "Pneumonia", "Pulmonary edema"]
STATE_TO_INDEX = {"not_mentioned": 0, "uncertain": 1, "negative": 2, "positive": 3}


def prepare_backbone_records(labels_csv, report_summary_csv, split_csv, reviews_csv=None):
    states = read_presence_labels(Path(labels_csv), Path(reviews_csv) if reviews_csv else None)
    with Path(split_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"subject_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV requires subject_id, split.")
        split_rows = list(reader)
    assignments = {normalize_id(r["subject_id"]): r["split"].strip() for r in split_rows}
    if len(assignments) != len(split_rows) or set(assignments.values()) - {"train", "val", "test"}:
        raise ValueError("Invalid or duplicate patient splits.")
    records, seen, excluded = [], set(), Counter()
    with Path(report_summary_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"subject_id", "study_id", "report_path", "status"}.issubset(reader.fieldnames or []):
            raise ValueError("Report summary requires subject_id, study_id, report_path, status.")
        for row in reader:
            subject, study = normalize_id(row["subject_id"]), normalize_id(row["study_id"])
            if subject not in assignments:
                excluded["patient_not_in_shared_split"] += 1
                continue
            if row["status"] not in {"matched", "no_matches"}:
                excluded["unusable_report"] += 1
                continue
            if (subject, study) in seen:
                raise ValueError(f"Duplicate study in report summary: {subject}, {study}")
            seen.add((subject, study))
            disease_states = [states.get((subject, study, disease), "not_mentioned") for disease in DISEASES]
            # Conflicting or failed labels are masked, never turned into absence.
            targets = [STATE_TO_INDEX.get(state, -100) for state in disease_states]
            if all(target == -100 for target in targets):
                excluded["all_labels_invalid"] += 1
                continue
            records.append({"subject_id": subject, "input_study_id": study,
                            "input_report_path": row["report_path"], "split": assignments[subject],
                            "states": disease_states, "targets": targets})
    if not records:
        raise ValueError(f"No eligible studies: {dict(excluded)}")
    return records, dict(excluded)


class CXRBackboneDataset:
    def __init__(self, records, image_size=224, imagenet_normalize=True):
        self.records = records
        self.image_size = image_size
        self.imagenet_normalize = imagenet_normalize

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        return {"image": load_dicom_tensor(row["dicom_path"], self.image_size, self.imagenet_normalize),
                "targets": torch.tensor(row["targets"], dtype=torch.long),
                "subject_id": row["subject_id"], "study_id": row["input_study_id"],
                "dicom_path": row["dicom_path"]}


def make_backbone_dataloaders(labels_csv, report_summary_csv, split_csv, *, reviews_csv=None,
                              reports_root=None, batch_size=16, image_size=224,
                              num_workers=0, seed=42, manifest_path=None, imagenet_normalize=True):
    import torch
    from torch.utils.data import DataLoader
    records, excluded = prepare_backbone_records(labels_csv, report_summary_csv, split_csv, reviews_csv)
    for row in records:
        row["dicom_path"] = str(select_input_dicom(row, reports_root))
    datasets = {name: CXRBackboneDataset([r for r in records if r["split"] == name], image_size, imagenet_normalize)
                for name in ["train", "val", "test"]}
    if any(not len(dataset) for dataset in datasets.values()):
        raise ValueError("One or more backbone splits are empty; check your shared patient split CSV.")
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["subject_id", "study_id", "split", "dicom_path"] + DISEASES
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({"subject_id": row["subject_id"], "study_id": row["input_study_id"],
                                 "split": row["split"], "dicom_path": row["dicom_path"],
                                 **dict(zip(DISEASES, row["states"]))})
    loaders = {name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", num_workers=num_workers,
                                pin_memory=torch.cuda.is_available(),
                                generator=torch.Generator().manual_seed(seed) if name == "train" else None)
               for name, dataset in datasets.items()}
    summary = []
    for name, dataset in datasets.items():
        print(name, "studies:", len(dataset), "patients:", len({r["subject_id"] for r in dataset.records}))
        for i, disease in enumerate(DISEASES):
            counts = Counter(r["states"][i] for r in dataset.records)
            summary.append({"split": name, "disease": disease, **dict(counts)})
    print("Excluded:", excluded)
    return {**loaders, "datasets": datasets, "summary": summary,
            "diseases": DISEASES, "state_to_index": STATE_TO_INDEX}
