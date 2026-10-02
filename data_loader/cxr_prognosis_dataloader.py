"""Image-only next-study prognosis: one input-study DICOM per example.

Dependencies: torch, numpy, pydicom, Pillow. Split assignment uses all patient
IDs in the CSV before filtering by disease, so diseases share the same split.
"""
import csv
import random
from collections import Counter
from pathlib import Path

LABEL_TO_INDEX = {"improving": 0, "stable": 1, "worsening": 2}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "input_study_id", "target_study_id", "disease", "candidate_target"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prognosis CSV requires {sorted(required)}")
        rows = list(reader)
    for row in rows:
        for column in ("subject_id", "input_study_id", "target_study_id"):
            row[column] = row[column].strip().lstrip("ps").removesuffix(".0")
    return rows


def split_patients(rows, seed=42, fractions=(0.70, 0.15, 0.15)):
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or abs(sum(fractions)-1) > 1e-8:
        raise ValueError("Three positive split fractions must sum to one.")
    subjects = sorted({r["subject_id"] for r in rows})
    if len(subjects) < 3:
        raise ValueError("At least three patients are needed for train/val/test.")
    random.Random(seed).shuffle(subjects)
    n_train = max(1, min(len(subjects)-2, int(len(subjects)*fractions[0])))
    n_val = max(1, min(len(subjects)-n_train-1, int(len(subjects)*fractions[1])))
    return {s: "train" if i < n_train else "val" if i < n_train+n_val else "test"
            for i, s in enumerate(subjects)}


def prepare_records(pairs_csv, disease, *, use_candidates=False, positive_input_only=True,
                    seed=42, fractions=(0.70, 0.15, 0.15), split_csv=None):
    rows = read_rows(pairs_csv)
    if disease not in {r["disease"] for r in rows}:
        raise ValueError(f"Disease not found: {disease}")
    assignments = split_patients(rows, seed, fractions)
    if split_csv:
        split_csv = Path(split_csv)
        if split_csv.exists():
            with split_csv.open(encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            assignments = {r["subject_id"]: r["split"] for r in saved}
            if len(assignments) != len(saved) or set(assignments.values()) - {"train", "val", "test"}:
                raise ValueError("Invalid or duplicate patient split assignments.")
            if {r["subject_id"] for r in rows} - assignments.keys():
                raise ValueError("New patients absent from saved split; create a new split CSV explicitly.")
        else:
            split_csv.parent.mkdir(parents=True, exist_ok=True)
            with split_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject_id", "split"])
                writer.writeheader()
                writer.writerows({"subject_id": s, "split": split} for s, split in sorted(assignments.items()))
    records, seen = [], set()
    excluded = Counter()
    for row in rows:
        if row["disease"] != disease:
            continue
        if positive_input_only and "input_disease_state" not in row:
            raise ValueError("Positive-input filtering requires a hybrid CSV with input_disease_state.")
        if positive_input_only and row["input_disease_state"].strip().lower() != "positive":
            excluded["input_not_positive"] += 1
            continue
        reviewed = row.get("reviewed_target", "").strip().lower()
        target = reviewed or (row["candidate_target"].strip().lower() if use_candidates else "")
        if target not in LABEL_TO_INDEX:
            excluded["unreviewed_or_nontraining_label"] += 1
            continue
        if reviewed and row.get("label_basis", "") != "presence_transition":
            if row.get("comparison_verified", "").strip().lower() != "yes":
                excluded["comparison_not_verified"] += 1
                continue
        identity = (row["subject_id"], row["input_study_id"], row["target_study_id"])
        if identity in seen:
            raise ValueError(f"Duplicate disease-study pair: {identity}")
        seen.add(identity)
        records.append({**row, "label": LABEL_TO_INDEX[target], "label_name": target,
                        "label_source": "reviewed" if reviewed else "candidate",
                        "split": assignments[row["subject_id"]]})
    if not records:
        raise ValueError(f"No eligible training labels. Excluded: {dict(excluded)}. Review the queue or explicitly set use_candidates=True for an exploratory baseline.")
    return records, assignments, dict(excluded)


def select_input_dicom(record, reports_root=None):
    """Prefer a frontal image, then lexicographic order; never open target images."""
    import pydicom
    if reports_root is not None:
        directory = Path(reports_root) / ("p" + record["subject_id"]) / ("s" + record["input_study_id"])
    else:
        report = Path(record["input_report_path"])
        directory = report.parent / ("s" + record["input_study_id"])
    images = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm")
    if not images:
        raise FileNotFoundError(f"No input-study DICOM images: {directory}")
    for path in images:
        header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["ViewPosition"])
        if str(header.get("ViewPosition", "")).strip().upper() in {"AP", "PA"}:
            return path
    return images[0]


def load_dicom_tensor(path, image_size=224, imagenet_normalize=True):
    import numpy as np
    import pydicom
    import torch
    from PIL import Image
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f"Expected one grayscale frame, got {pixels.shape}: {path}")
    photo = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f"Unsupported photometric interpretation: {photo}")
    valid = np.isfinite(pixels)
    if "PixelPaddingValue" in ds:
        padding = float(ds.PixelPaddingValue)
        limit = float(ds.get("PixelPaddingRangeLimit", padding))
        valid &= ~((pixels >= min(padding, limit)) & (pixels <= max(padding, limit)))
    array = pixels.astype(np.float32) * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    valid &= np.isfinite(array)
    if not valid.any():
        raise ValueError(f"No valid image pixels: {path}")
    lo, hi = np.percentile(array[valid], [1, 99])
    if hi <= lo:
        lo, hi = array[valid].min(), array[valid].max()
    array = np.clip((array-lo) / max(float(hi-lo), 1e-6), 0, 1)
    if photo == "MONOCHROME1":
        array = 1-array
    array[~valid] = 0
    image = Image.fromarray((array*255).astype(np.uint8)).resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(image, dtype=np.float32)/255).unsqueeze(0).repeat(3, 1, 1)
    if imagenet_normalize:
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return tensor


class CXRPrognosisDataset:
    """Map-style PyTorch dataset; every sample is one study-pair target."""
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
                "label": torch.tensor(row["label"], dtype=torch.long),
                "subject_id": row["subject_id"], "input_study_id": row["input_study_id"],
                "target_study_id": row["target_study_id"], "dicom_path": row["dicom_path"]}


def make_dataloaders(pairs_csv, disease, *, reports_root=None, batch_size=16,
                     image_size=224, num_workers=0, seed=42, use_candidates=False,
                     positive_input_only=True, split_csv=None, manifest_path=None,
                     imagenet_normalize=True):
    import torch
    from torch.utils.data import DataLoader
    records, assignments, excluded = prepare_records(pairs_csv, disease, use_candidates=use_candidates,
                                                     positive_input_only=positive_input_only, seed=seed, split_csv=split_csv)
    for row in records:
        row["dicom_path"] = str(select_input_dicom(row, reports_root))
    datasets = {split: CXRPrognosisDataset([r for r in records if r["split"] == split], image_size, imagenet_normalize)
                for split in ("train", "val", "test")}
    empty = [name for name, dataset in datasets.items() if not len(dataset)]
    if empty:
        raise ValueError(f"Empty splits for {disease}: {empty}. Do not train/evaluate an empty split.")
    if manifest_path:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["subject_id", "input_study_id", "target_study_id", "disease", "label", "label_name", "label_source", "split", "dicom_path"]
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train",
                                num_workers=num_workers, generator=generator if name == "train" else None,
                                pin_memory=torch.cuda.is_available()) for name, dataset in datasets.items()}
    summary = {}
    for name, dataset in datasets.items():
        counts = Counter(r["label_name"] for r in dataset.records)
        summary[name] = {"patients": len({r["subject_id"] for r in dataset.records}),
                         "samples": len(dataset), **{label: counts[label] for label in LABEL_TO_INDEX}}
        print(name, summary[name])
        missing = [label for label in LABEL_TO_INDEX if not counts[label]]
        if missing:
            print(f"Class coverage warning: {name} has no examples of {missing}.")
    print("Excluded:", excluded)
    return {**loaders, "datasets": datasets, "summary": summary, "label_to_index": LABEL_TO_INDEX,
            "patient_splits": assignments}
