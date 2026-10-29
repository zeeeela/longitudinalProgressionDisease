"""Training loop: masked cross-entropy, AdamW, per-epoch cosine LR to final_lr_ratio, best checkpoint by
validation loss (same protocol as the existing train_paired_progression), optional encoder warm-up freeze."""
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import DISEASES
from .metrics_io import evaluate_predictions, write_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce(logits, y, weights=None):
    """logits [B,D,3], y [B,D] (-100 masked). Sum over valid targets / number of valid targets."""
    valid = (y != -100).sum().clamp(min=1)
    total = logits.new_zeros(())
    for d in range(logits.shape[1]):
        w = None if weights is None else weights[d]
        total = total + F.cross_entropy(logits[:, d].float(), y[:, d], weight=w, ignore_index=-100, reduction="sum")
    return total / valid


def class_weights(frame, diseases, device):
    out = []
    for d in diseases:
        c = np.array([(frame[d] == k).sum() for k in range(3)], dtype=float)
        w = np.where(c > 0, c.sum() / (3 * np.maximum(c, 1)), 0.0)
        out.append(torch.tensor(w, dtype=torch.float32, device=device))
    return out


def _batch_to(batch, device, use_img, use_tab):
    kw = {}
    if use_img:
        kw["x_prev"], kw["x_cur"] = batch["x_prev"].to(device, non_blocking=True), batch["x_cur"].to(device, non_blocking=True)
    if use_tab:
        kw["tab"] = batch["tab"].to(device)
    return kw, batch["y"].to(device)


@torch.no_grad()
def predict(model, loader, device, use_img, use_tab):
    model.eval()
    ys, logits = [], []
    for b in loader:
        kw, y = _batch_to(b, device, use_img, use_tab)
        logits.append(model(**kw).float().cpu())
        ys.append(y.cpu())
    return torch.cat(ys), torch.cat(logits)


def train_fusion(model, data, out_dir, *, epochs=20, backbone_lr=1e-5, head_lr=1e-4, final_lr_ratio=0.1,
                 weight_decay=1e-4, freeze_encoder_epochs=0, freeze_bn=False, accumulate=1, use_class_weights=False,
                 patience=None, seed=42, device=None, settings=None, diseases=DISEASES):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model.to(device)
    use_img, use_tab = model.encoder is not None, model.tab is not None
    L = data["loaders"]

    enc_params = list(model.encoder.parameters()) if use_img else []
    groups = [{"params": model.head_parameters(), "lr": head_lr, "base": head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": backbone_lr, "base": backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cw = class_weights(data["frames"]["train"], diseases, device) if use_class_weights else None

    history, best, bad = [], float("inf"), 0
    best_path = out_dir / "best_model.pt"
    for ep in range(epochs):
        frac = 0.5 * (1 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * (final_lr_ratio + (1 - final_lr_ratio) * frac)
        frozen = use_img and ep < freeze_encoder_epochs
        if use_img:
            model.encoder.requires_grad_(not frozen)
        model.train()
        if use_img and (frozen or freeze_bn):
            model.encoder.eval() if frozen else model.encoder.freeze_bn()
        t0, run, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(L["train"]):
            kw, y = _batch_to(b, device, use_img, use_tab)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(**kw)
            loss = masked_ce(logits, y, cw)
            scaler.scale(loss / accumulate).backward()
            if (i + 1) % accumulate == 0 or i + 1 == len(L["train"]):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run += loss.detach().item() * len(y)
            n += len(y)
        yv, lv = predict(model, L["val"], device, use_img, use_tab)
        val_loss = float(masked_ce(lv, yv))
        rows, _ = evaluate_predictions(yv.numpy(), lv.argmax(-1).numpy(), "val", diseases)
        f1 = float(np.nanmean([r["macro_f1"] for r in rows]))
        history.append(dict(epoch=ep + 1, train_loss=run / max(n, 1), val_loss=val_loss, val_mean_macro_f1=f1,
                            lr_head=opt.param_groups[0]["lr"], lr_backbone=opt.param_groups[-1]["lr"] if enc_params else np.nan,
                            encoder_frozen=frozen, seconds=time.time() - t0))
        print(f"epoch {ep+1:02d}/{epochs} train {history[-1]['train_loss']:.4f} val {val_loss:.4f} "
              f"valF1 {f1:.3f}{' (encoder frozen)' if frozen else ''}")
        if val_loss < best:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if patience and bad >= patience:
                print("early stop")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    all_rows, confs, preds = [], {}, []
    for split in ("val", "test"):
        if split not in L:
            continue
        y, lg = predict(model, L[split], device, use_img, use_tab)
        yp = lg.argmax(-1).numpy()
        rows, cm = evaluate_predictions(y.numpy(), yp, split, diseases)
        losses = [float(F.cross_entropy(lg[:, d], y[:, d], ignore_index=-100)) if (y[:, d] != -100).any() else np.nan
                  for d in range(len(diseases))]
        for r, l in zip(rows, losses):
            r["loss"] = l
        all_rows += rows
        confs[split] = cm
        fr = data["frames"][split]
        p = fr[["subject_id", "input_study_id", "target_study_id"]].copy()
        p.insert(0, "split", split)
        for d, name in enumerate(diseases):
            p[f"true_{name}"], p[f"pred_{name}"] = y[:, d].numpy(), yp[:, d]
        preds.append(p)
    write_results(out_dir, all_rows, confs, dict(settings or {}, seed=seed, epochs=epochs, best_val_loss=best,
                                                n_tab=data["n_tab"], tab_names=data["tab_names"]),
                  history=history, predictions=pd.concat(preds))
    return dict(out_dir=out_dir, metrics=all_rows, history=history, best_val_loss=best)
