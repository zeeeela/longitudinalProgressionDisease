"""Prepare pairs once; load prior/current images and five masked targets.

Save in data_loader/, beside cxr_prognosis_dataloader.py.
"""
import csv
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .cxr_prognosis_dataloader import prepare_records, select_input_dicom, load_dicom_tensor
except ImportError:
    from cxr_prognosis_dataloader import prepare_records, select_input_dicom, load_dicom_tensor

DISEASES = ["Pleural effusion", "Cardiomegaly", "Edema", "Pneumonia", "Pulmonary edema"]


def prepare_pair_csv(pairs_csv, split_csv, reports_root, output_csv, *, use_candidates=False):
    """One-time preparation; no positive-input filter or new patient splits."""
    if not Path(split_csv).is_file():
        raise FileNotFoundError("Use the existing backbone patient split CSV.")
    pairs = {}
    for disease in DISEASES:
        try:
            records, _, _ = prepare_records(pairs_csv, disease, split_csv=split_csv,
                                           positive_input_only=False, use_candidates=use_candidates)
        except ValueError as error:
            if str(error).startswith("No eligible training labels."):
                continue
            raise
        for record in records:
            key = (record['subject_id'], record['input_study_id'], record['target_study_id'])
            if key not in pairs:
                pairs[key] = {**record, **{d: -100 for d in DISEASES},
                              **{d + '_source': '' for d in DISEASES}}
            row = pairs[key]
            for field in ('split', 'input_date', 'target_date'):
                if row.get(field, '') != record.get(field, ''):
                    raise ValueError(f"Inconsistent {field} for pair {key}.")
            row[disease] = record['label']
            row[disease + '_source'] = record['label_source']
    if not pairs:
        raise ValueError("No usable labels. Review decisions or explicitly use_candidates=True.")
    paths = {}
    for row in pairs.values():
        if row['input_study_id'] == row['target_study_id']:
            raise ValueError('Prior and current studies must differ.')
        if row.get('input_date') and row.get('target_date'):
            if datetime.fromisoformat(row['input_date']) >= datetime.fromisoformat(row['target_date']):
                raise ValueError('Prior study must precede current study.')
        for name, study in [('prior', row['input_study_id']), ('current', row['target_study_id'])]:
            key = (row['subject_id'], study)
            if key not in paths:
                paths[key] = str(select_input_dicom({'subject_id': row['subject_id'], 'input_study_id': study}, reports_root))
            row[name + '_dicom_path'] = paths[key]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['subject_id', 'input_study_id', 'target_study_id', 'split', 'prior_dicom_path', 'current_dicom_path'] + DISEASES + [d + '_source' for d in DISEASES]
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pairs[key] for key in sorted(pairs))
    print(f'Saved {len(pairs)} unique pairs to {output_csv}')
    return output_csv


class PairDataset(Dataset):
    def __init__(self, rows):
        self.records = rows

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        targets = torch.tensor([int(row[d]) for d in DISEASES], dtype=torch.long)
        return {'prior_image': load_dicom_tensor(row['prior_dicom_path']),
                'current_image': load_dicom_tensor(row['current_dicom_path']),
                'targets': targets, 'target_mask': targets != -100,
                'subject_id': row['subject_id'], 'prior_study_id': row['input_study_id'],
                'current_study_id': row['target_study_id']}


def make_pair_loaders(pair_csv, batch_size=16):
    """Three loaders from a prepared manifest; workers=0 for Windows notebooks."""
    with Path(pair_csv).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'subject_id', 'input_study_id', 'target_study_id', 'split', 'prior_dicom_path', 'current_dicom_path', *DISEASES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Use the prepared pair CSV, not the raw prognosis CSV.')
        rows = list(reader)
    seen, patient_split = set(), {}
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Invalid split name.')
        subject = row['subject_id']
        if subject in patient_split and patient_split[subject] != row['split']:
            raise ValueError(f'Patient {subject} appears in different splits.')
        patient_split[subject] = row['split']
        key = (subject, row['input_study_id'], row['target_study_id'])
        if key in seen or key[1] == key[2]:
            raise ValueError('Duplicate pair or identical prior/current studies.')
        seen.add(key)
        targets = [int(row[d]) for d in DISEASES]
        if any(t not in {-100, 0, 1, 2} for t in targets) or all(t == -100 for t in targets):
            raise ValueError('Invalid targets or pair without usable labels.')
        prior, current = Path(row['prior_dicom_path']), Path(row['current_dicom_path'])
        if prior.resolve() == current.resolve():
            raise ValueError('Prior/current image paths must differ.')
        for path in (prior, current):
            if not path.is_file():
                raise FileNotFoundError(path)
    loaders = {}
    for name in ('train', 'val', 'test'):
        dataset = PairDataset([r for r in rows if r['split'] == name])
        if not len(dataset):
            raise ValueError(f'Empty {name} split.')
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=name == 'train', num_workers=0,
                                   pin_memory=torch.cuda.is_available(),
                                   generator=torch.Generator().manual_seed(42) if name == 'train' else None)
        print(f'{name}: {len(dataset)} pairs, {len({r["subject_id"] for r in dataset.records})} patients')
    return loaders
