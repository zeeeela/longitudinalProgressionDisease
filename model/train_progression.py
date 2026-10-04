"""Extract paired features once, then train five progression classifiers.

Save in model/ beside progression_model.py and cxr_backbone.py.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from .progression_model import PairFeatureExtractor, fit_disease_classifier, predict_probabilities, DISEASES, LABELS
except ImportError:
    from progression_model import PairFeatureExtractor, fit_disease_classifier, predict_probabilities, DISEASES, LABELS


def validate_splits(loaders):
    subjects = {}
    for name in ('train', 'val', 'test'):
        rows = loaders[name].dataset.records
        if not rows:
            raise ValueError(f'Empty {name} split.')
        subjects[name] = {str(row['subject_id']) for row in rows}
        if any(row['split'] != name for row in rows):
            raise ValueError('Loader records have inconsistent split membership.')
    if subjects['train'] & subjects['val'] or subjects['train'] & subjects['test'] or subjects['val'] & subjects['test']:
        raise ValueError('Patient overlap between training, validation, and test.')


def extract_features(extractor, loader, device, name):
    features, targets, identities = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'{name}: paired features', unit='batch'):
            prior = batch['prior_image'].to(device, non_blocking=True)
            current = batch['current_image'].to(device, non_blocking=True)
            y = batch['targets'].cpu().numpy()
            if y.ndim != 2 or y.shape[1] != len(DISEASES) or not np.isin(y, [-100, 0, 1, 2]).all():
                raise ValueError('Expected five progression targets with values -100/0/1/2.')
            features.append(extractor(prior, current).cpu().numpy())
            targets.append(y)
            identities.extend(zip(batch['subject_id'], batch['prior_study_id'], batch['current_study_id']))
    if len(identities) != len(set(identities)):
        raise ValueError('Repeated image pairs in the loader. Use ordinary loaders without replacement sampling.')
    x = np.concatenate(features)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite image features.')
    return x, np.concatenate(targets), np.asarray(identities, dtype=str)


def evaluate(y, probabilities):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
    predicted = probabilities.argmax(axis=1)
    metrics = {'n_pairs': int(len(y)), 'accuracy': float(accuracy_score(y, predicted)),
               'macro_f1': float(f1_score(y, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
               'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist()}
    for index, label in enumerate(LABELS):
        binary = y == index
        metrics[label + '_support'] = int(binary.sum())
        metrics[label + '_auroc'] = float(roc_auc_score(binary, probabilities[:, index])) if len(np.unique(binary)) == 2 else None
        metrics[label + '_auprc'] = float(average_precision_score(binary, probabilities[:, index])) if len(np.unique(binary)) == 2 else None
    aucs = [metrics[label + '_auroc'] for label in LABELS]
    metrics['macro_auroc'] = float(np.mean(aucs)) if all(v is not None for v in aucs) else None
    return metrics


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def train_progression(checkpoint_path, pair_loaders, out_dir, *, device=None,
                      pca_components=50, c_values=(0.1, 0.5, 1.0, 10.0), seed=42):
    import joblib
    # Import dependency before starting costly DICOM feature extraction.
    import sklearn
    validate_splits(pair_loaders)
    checkpoint_path, out_dir = Path(checkpoint_path), Path(out_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory to preserve previous results.')
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    extractor = PairFeatureExtractor(checkpoint_path).to(device)
    print(f'Device: {device} | Backbone: frozen | Labels: {LABELS}', flush=True)
    data = {}
    for split in ('train', 'val', 'test'):
        x, y, ids = extract_features(extractor, pair_loaders[split], device, split)
        data[split] = (x, y, ids)
        np.savez_compressed(out_dir / f'{split}_features.npz', features=x, targets=y, pair_ids=ids)
    classifiers, metrics, skipped, summaries, predictions = {}, {}, {}, [], []
    for index, disease in enumerate(DISEASES):
        selected = {split: (x[y[:, index] != -100], y[y[:, index] != -100, index], ids[y[:, index] != -100])
                    for split, (x, y, ids) in data.items()}
        train_x, train_y, _ = selected['train']
        val_x, val_y, _ = selected['val']
        if len(train_y) < 3 or len(np.unique(train_y)) < 2 or not len(val_y):
            skipped[disease] = 'Insufficient training classes/pairs or missing validation labels.'
            print(f'Skipped {disease}: {skipped[disease]}')
            continue
        classifier, trials = fit_disease_classifier(train_x, train_y, val_x, val_y,
                                                     components=pca_components, c_values=c_values, seed=seed)
        classifiers[disease] = classifier
        metrics[disease] = {'validation_trials': trials, 'selected_C': float(classifier['classifier'].C),
                            'pca_components': int(classifier['pca'].n_components_),
                            'training_classes': classifier.classes_.astype(int).tolist()}
        for split in ('val', 'test'):
            x, y, ids = selected[split]
            if not len(y):
                metrics[disease][split] = None
                continue
            probabilities = predict_probabilities(classifier, x)
            scores = evaluate(y, probabilities)
            metrics[disease][split] = scores
            summaries.append({'disease': disease, 'split': split, **{k: v for k, v in scores.items() if k != 'confusion_matrix'}})
            for identity, target, probability in zip(ids, y, probabilities):
                predictions.append({'disease': disease, 'split': split, 'subject_id': identity[0],
                                    'prior_study_id': identity[1], 'current_study_id': identity[2],
                                    'actual': LABELS[int(target)], 'predicted': LABELS[int(probability.argmax())],
                                    **{f'p_{label}': float(probability[i]) for i, label in enumerate(LABELS)}})
        print(f"{disease}: {len(train_y)} train pairs | C={metrics[disease]['selected_C']} | test={metrics[disease].get('test')}", flush=True)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    joblib.dump({'classifiers': classifiers, 'diseases': list(DISEASES), 'labels': list(LABELS),
                 'backbone_checkpoint': str(checkpoint_path.resolve()), 'checkpoint_sha256': checkpoint_hash,
                 'feature': 'current_embedding - prior_embedding', 'sklearn_version': sklearn.__version__},
                out_dir / 'progression_classifiers.joblib')
    with (out_dir / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump({'diseases': metrics, 'skipped': skipped, 'seed': seed}, handle, indent=2)
    write_csv(out_dir / 'metrics.csv', summaries)
    write_csv(out_dir / 'predictions.csv', predictions)
    print(f'Saved {len(classifiers)} disease classifiers to {out_dir.resolve()}')
    return {'classifiers': classifiers, 'metrics': metrics, 'skipped': skipped, 'out_dir': out_dir}
