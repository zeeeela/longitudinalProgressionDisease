"""Train the five-finding DenseNet backbone for a fixed number of epochs."""
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

try:
    from .cxr_backbone import DISEASES, STATES
except ImportError:
    from cxr_backbone import DISEASES, STATES


def check_patient_splits(loaders):
    patients = {}
    for name in ("train", "val", "test"):
        loader = loaders[name]
        if not len(loader.dataset):
            raise ValueError(f"Empty {name} dataset.")
        if not hasattr(loader.dataset, "records"):
            raise ValueError("Expected backbone datasets with records for patient split verification.")
        patients[name] = {r["subject_id"] for r in loader.dataset.records}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if patients[first] & patients[second]:
            raise ValueError(f"Patient leakage between {first} and {second}.")


def run_epoch(model, loader, device, optimizer=None, description=None):
    training = optimizer is not None
    model.train(training)
    loss_total = valid_total = correct_total = 0
    confusion = torch.zeros((len(DISEASES), len(STATES), len(STATES)), dtype=torch.long)
    data_seconds = compute_seconds = 0.0
    previous_end = time.perf_counter()
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=description or ("Train" if training else "Evaluate"),
                        unit="batch", leave=True, file=sys.stdout,
                        mininterval=1.0, dynamic_ncols=True)
        progress.refresh()
        for batch in progress:
            batch_start = time.perf_counter()
            data_seconds += batch_start - previous_end
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            valid = targets != -100
            count = int(valid.sum().item())
            if not count:
                previous_end = time.perf_counter()
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss_sum = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="sum")
            loss = loss_sum / count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect the input images and targets.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            predicted = logits.argmax(dim=-1)
            loss_total += float(loss_sum.detach().item())
            valid_total += count
            correct_total += int(((predicted == targets) & valid).sum().item())
            for disease in range(len(DISEASES)):
                keep = valid[:, disease]
                encoded = (targets[keep, disease] * len(STATES) + predicted[keep, disease]).detach().cpu()
                confusion[disease] += torch.bincount(encoded, minlength=len(STATES)**2).reshape(len(STATES), len(STATES))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_seconds += time.perf_counter() - batch_start
            progress.set_postfix(loss=f"{loss_total/valid_total:.4f}",
                                 accuracy=f"{correct_total/valid_total:.1%}",
                                 data_s=f"{data_seconds:.0f}", compute_s=f"{compute_seconds:.0f}", refresh=False)
            previous_end = time.perf_counter()
    if not valid_total:
        raise ValueError("No valid disease targets in this epoch.")
    return {"loss": loss_total/valid_total, "accuracy": correct_total/valid_total,
            "valid_targets": valid_total, "confusion": confusion,
            "data_seconds": data_seconds, "compute_seconds": compute_seconds}


def train_backbone(model, loaders, out_dir, *, epochs=20, learning_rate=1e-4,
                   weight_decay=1e-4, seed=42, device=None):
    """Train exactly epochs; return history and best-checkpoint test results."""
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    check_patient_splits(loaders)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_backbone.pt"
    last_path = out_dir / "last_backbone.pt"
    if best_path.exists() or last_path.exists():
        raise FileExistsError("Training checkpoints already exist; choose a new out_dir to preserve them.")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=weight_decay)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    print(f"Device: {device} | Epochs: {epochs} | Learning rate: {learning_rate}", flush=True)
    fields = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "seconds"]
    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs+1):
            started = time.perf_counter()
            print(f"Starting epoch {epoch}/{epochs}: {len(loaders['train'])} training batches, "
                  f"{len(loaders['val'])} validation batches. Waiting for first batch...", flush=True)
            train = run_epoch(model, loaders["train"], device, optimizer,
                              description=f"Epoch {epoch:02d}/{epochs} train")
            val = run_epoch(model, loaders["val"], device,
                            description=f"Epoch {epoch:02d}/{epochs} validation")
            record = {"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
                      "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"],
                      "seconds": round(time.perf_counter()-started, 2)}
            history.append(record)
            writer.writerow(record)
            handle.flush()
            improved = val["loss"] < best_loss
            if improved:
                best_loss, best_epoch = val["loss"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model_state_dict": best_state, "epoch": epoch, "val_loss": best_loss,
                            "diseases": list(DISEASES), "states": list(STATES)}, best_path)
            print(f"Epoch {epoch:02d}/{epochs} | train loss {train['loss']:.4f} | val loss {val['loss']:.4f} | "
                  f"train accuracy {train['accuracy']:.1%} | val accuracy {val['accuracy']:.1%} | "
                  f"{record['seconds']:.0f}s{' | saved best' if improved else ''}", flush=True)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epochs,
                "diseases": list(DISEASES), "states": list(STATES)}, last_path)
    model.load_state_dict(best_state)
    test = run_epoch(model, loaders["test"], device, description="Best checkpoint test")
    with (out_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["best_epoch", "test_loss", "test_accuracy", "valid_targets"])
        writer.writeheader()
        writer.writerow({"best_epoch": best_epoch, "test_loss": test["loss"],
                         "test_accuracy": test["accuracy"], "valid_targets": test["valid_targets"]})
    with (out_dir / "test_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["disease", "actual", "predicted", "count"])
        writer.writeheader()
        for d, disease in enumerate(DISEASES):
            for actual, actual_name in enumerate(STATES):
                for predicted, predicted_name in enumerate(STATES):
                    writer.writerow({"disease": disease, "actual": actual_name, "predicted": predicted_name,
                                     "count": int(test["confusion"][d, actual, predicted])})
    print(f"Best epoch: {best_epoch} | Test loss: {test['loss']:.4f} | Test accuracy: {test['accuracy']:.1%}", flush=True)
    print(f"Saved checkpoints and metrics to {out_dir.resolve()}")
    return {"history": history, "best_epoch": best_epoch, "best_path": best_path, "test": test}
