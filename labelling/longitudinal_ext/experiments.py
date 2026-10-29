"""One-call experiment runner used by the notebook."""
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

from .encoders import build_encoder, load_densenet_checkpoint
from .fusion_model import PairedFusionNet
from .frozen_screen import run_frozen_screen
from .pair_data import make_fusion_loaders
from .train_fusion import set_seed, train_fusion


def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def run_experiment(name, *, manifest_csv, out_root, encoder="densenet121_imagenet", use_images=True, tab_csv=None,
                   tab_control=None, tab_columns=None, pretrained=True, diagnosis_checkpoint=None, img_mode="diff",
                   seed=42, epochs=20, batch_size=8, backbone_lr=1e-5, head_lr=1e-4, final_lr_ratio=0.1,
                   freeze_encoder_epochs=0, freeze_bn=False, accumulate=1, use_class_weights=False, patience=None,
                   image_size=None, image_cache=None, device=None, frozen_only=False, num_workers=0, **loader_kw):
    """tab_csv=None -> image only. use_images=False + tab_csv -> tabular-only network.
    frozen_only=True -> frozen-encoder PCA/LR screen instead of training."""
    set_seed(seed)
    enc = build_encoder(encoder, pretrained=pretrained, image_size=image_size) if use_images else None
    if enc is not None and diagnosis_checkpoint:
        load_densenet_checkpoint(enc, diagnosis_checkpoint)
    data = make_fusion_loaders(manifest_csv, image_size=enc.image_size if enc else 224, batch_size=batch_size,
                               image_cache=image_cache, tab_csv=tab_csv, tab_columns=tab_columns,
                               tab_control=tab_control, seed=seed, num_workers=num_workers, **loader_kw)
    out_dir = Path(out_root) / f"ext_{name}_s{seed}_{_stamp()}"
    if frozen_only:
        return run_frozen_screen(enc, data, out_dir, device=device, seed=seed)
    model = PairedFusionNet(enc, n_tab=data["n_tab"], img_mode=img_mode)
    if enc is not None and freeze_bn:
        enc.freeze_bn()
    settings = dict(experiment=name, encoder=encoder if use_images else None, use_images=use_images,
                    use_tab=tab_csv is not None, tab_control=tab_control, pretrained=pretrained,
                    diagnosis_checkpoint=str(diagnosis_checkpoint) if diagnosis_checkpoint else None, img_mode=img_mode,
                    backbone_lr=backbone_lr, head_lr=head_lr, final_lr_ratio=final_lr_ratio, batch_size=batch_size,
                    freeze_encoder_epochs=freeze_encoder_epochs, use_class_weights=use_class_weights)
    return train_fusion(model, data, out_dir, epochs=epochs, backbone_lr=backbone_lr, head_lr=head_lr,
                        final_lr_ratio=final_lr_ratio, freeze_encoder_epochs=freeze_encoder_epochs,
                        freeze_bn=freeze_bn, accumulate=accumulate, use_class_weights=use_class_weights,
                        patience=patience, seed=seed, device=device, settings=settings)
