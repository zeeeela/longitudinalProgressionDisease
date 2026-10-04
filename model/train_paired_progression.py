"""End-to-end paired-image training, from scratch or diagnosis checkpoint."""
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
try:
    from .paired_progression_model import PairedProgressionModel, DISEASES, PROGRESSION_STATES
except ImportError:
    from paired_progression_model import PairedProgressionModel, DISEASES, PROGRESSION_STATES


def check_splits(loaders):
    subjects = {}
    for name in ('train', 'val', 'test'):
        records = loaders[name].dataset.records
        if not records:
            raise ValueError(f'Empty {name} split.')
        if any(r['split'] != name for r in records):
            raise ValueError('Split metadata does not match the loader.')
        subjects[name] = {r['subject_id'] for r in records}
    if subjects['train'] & subjects['val'] or subjects['train'] & subjects['test'] or subjects['val'] & subjects['test']:
        raise ValueError('Patient leakage across splits.')


def epoch_pass(model, loader, device, description, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_valid = 0.0, 0
    confusion = np.zeros((len(DISEASES), 3, 3), dtype=np.int64)
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=description, unit='batch', file=sys.stdout, dynamic_ncols=True)
        for batch in progress:
            prior = batch['prior_image'].to(device, non_blocking=True)
            current = batch['current_image'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            if targets.ndim != 2 or targets.shape[1] != len(DISEASES):
                raise ValueError('Targets must have five disease columns.')
            valid = targets != -100
            count = int(valid.sum().item())
            if not count:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(prior, current)
                summed = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction='sum')
                loss = summed/count
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite progression loss.')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(summed.detach().item())
            total_valid += count
            y = targets.detach().cpu().numpy()
            predictions = logits.detach().argmax(-1).cpu().numpy()
            for disease in range(len(DISEASES)):
                keep = y[:, disease] != -100
                np.add.at(confusion[disease], (y[keep, disease], predictions[keep, disease]), 1)
            progress.set_postfix(loss=f'{total_loss/total_valid:.4f}', refresh=False)
    if not total_valid:
        raise ValueError('No valid targets in this split.')
    return total_loss/total_valid, confusion


def scores_from_confusion(confusion, split):
    rows = []
    for disease, matrix in zip(DISEASES, confusion):
        support, predicted = matrix.sum(1), matrix.sum(0)
        true_positive = np.diag(matrix)
        denominator = support + predicted
        f1 = np.divide(2*true_positive, denominator, out=np.zeros(3, dtype=float), where=denominator > 0)
        count = int(support.sum())
        rows.append({'disease': disease, 'split': split, 'n_pairs': count,
                     'accuracy': float(true_positive.sum()/count) if count else None,
                     'macro_f1': float(f1.mean()) if count else None,
                     **{label+'_support': int(support[i]) for i, label in enumerate(PROGRESSION_STATES)}})
    return rows


def make_optimizer_and_scheduler(model, backbone_lr, head_lr, weight_decay, epochs, final_lr_ratio):
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': backbone_lr, 'name': 'backbone'},
        {'params': model.classifier.parameters(), 'lr': head_lr, 'name': 'head'},
    ], weight_decay=weight_decay)
    def factor(step):
        if epochs == 1:
            return 1.0
        progress = min(step, epochs-1) / (epochs-1)
        return final_lr_ratio + (1-final_lr_ratio) * (1+math.cos(math.pi*progress))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
    return optimizer, scheduler


def train_paired_progression(pair_loaders, out_dir, *, diagnosis_checkpoint=None,
                             epochs=20, learning_rate=1e-4, weight_decay=1e-4,
                             seed=42, device=None, backbone_lr=None, head_lr=None,
                             final_lr_ratio=0.1):
    check_splits(pair_loaders)
    if epochs < 1:
        raise ValueError('epochs must be positive.')
    backbone_lr = backbone_lr if backbone_lr is not None else learning_rate * (0.1 if diagnosis_checkpoint else 1.0)
    head_lr = head_lr if head_lr is not None else learning_rate
    if backbone_lr <= 0 or head_lr <= 0 or not 0 < final_lr_ratio <= 1:
        raise ValueError('Learning rates must be positive; final_lr_ratio must be in (0, 1].')
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Choose a new output directory.')
    if diagnosis_checkpoint is not None and not Path(diagnosis_checkpoint).is_file():
        raise FileNotFoundError(diagnosis_checkpoint)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if pair_loaders['train'].generator is not None:
        pair_loaders['train'].generator.manual_seed(seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = PairedProgressionModel(diagnosis_checkpoint).to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model, backbone_lr, head_lr, weight_decay, epochs, final_lr_ratio)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {'initialization': model.initialization, 'diagnosis_checkpoint': str(diagnosis_checkpoint) if diagnosis_checkpoint else None,
                'epochs': epochs, 'backbone_lr': backbone_lr, 'head_lr': head_lr,
                'schedule': 'cosine', 'final_lr_ratio': final_lr_ratio, 'weight_decay': weight_decay, 'seed': seed,
                'diseases': list(DISEASES), 'labels': list(PROGRESSION_STATES), 'backbone_trainable': True}
    (out_dir/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(f'Device: {device} | Initialization: {model.initialization} | All backbone layers trainable', flush=True)
    print(f'Cosine schedule: backbone {backbone_lr:g} -> {backbone_lr*final_lr_ratio:g}; '
          f'head {head_lr:g} -> {head_lr*final_lr_ratio:g}', flush=True)
    best_loss, best_epoch, history = float('inf'), None, []
    fields = ['epoch', 'train_loss', 'val_loss', 'backbone_lr', 'head_lr', 'seconds']
    with (out_dir/'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs+1):
            start = time.perf_counter()
            used_backbone_lr, used_head_lr = [group['lr'] for group in optimizer.param_groups]
            train_loss, _ = epoch_pass(model, pair_loaders['train'], device, f'{epoch}/{epochs} train', optimizer, scaler)
            val_loss, _ = epoch_pass(model, pair_loaders['val'], device, f'{epoch}/{epochs} val')
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       backbone_lr=used_backbone_lr, head_lr=used_head_lr, seconds=round(time.perf_counter()-start, 2))
            history.append(row)
            writer.writerow(row)
            handle.flush()
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save({'model_state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            'epoch': epoch, 'val_loss': val_loss, 'settings': settings}, out_dir/'best_pair_model.pt')
            print(f'Epoch {epoch}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}; '
                  f'backbone_lr={used_backbone_lr:.2e}, head_lr={used_head_lr:.2e}; best={best_epoch}', flush=True)
            if epoch < epochs:
                scheduler.step()
    torch.save({'model_state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                'epoch': epochs, 'settings': settings}, out_dir/'last_pair_model.pt')
    model.load_state_dict(torch.load(out_dir/'best_pair_model.pt', map_location=device, weights_only=True)['model_state_dict'])
    metrics, matrices = [], {}
    for split in ('val', 'test'):
        loss, confusion = epoch_pass(model, pair_loaders[split], device, f'Best model {split}')
        metrics.extend(scores_from_confusion(confusion, split))
        matrices[split] = {'loss': loss, 'confusion': confusion.tolist()}
    with (out_dir/'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (out_dir/'confusion_matrices.json').write_text(json.dumps(matrices, indent=2), encoding='utf-8')
    model.cpu()
    del model, optimizer, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'out_dir': out_dir, 'best_path': out_dir/'best_pair_model.pt', 'best_epoch': best_epoch,
            'history': history, 'metrics': metrics}
