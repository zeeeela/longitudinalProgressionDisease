"""Image encoders behind one interface.

Every encoder takes x: [B,1,H,W] float in [0,1] (what the dataset yields) and returns [B, out_dim].
Each wrapper applies its own channel expansion / normalisation, so the data pipeline never changes.

build_encoder(name):
  "densenet121_imagenet"        torchvision DenseNet-121 (ImageNet)  -> the current baseline, re-run through this pipeline
  "densenet121_scratch"         same, random weights
  "xrv:<weights>"               TorchXRayVision DenseNet-121, e.g. xrv:densenet121-res224-chex
                                (-nih / -pc / -chex avoid MIMIC-CXR; -all / -mimic_* were trained on MIMIC-CXR)
  "hf:microsoft/rad-dino"       RAD-DINO ViT-B/14 (CLS + mean patch token); trained with MIMIC-CXR images
  "biomedclip"                  BiomedCLIP ViT-B/16 image tower (open_clip)
  "timm:<name>"                 any timm backbone, e.g. timm:convnext_small.fb_in22k_ft_in1k
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class ImageEncoder(nn.Module):
    def __init__(self, name, backbone, forward_fn, norm, image_size, license_note=""):
        super().__init__()
        self.name, self.backbone, self._fwd = name, backbone, forward_fn
        self.norm, self.image_size, self.note = norm, int(image_size), license_note
        if norm not in ("xrv",):
            mean, std = norm
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.out_dim = int(self.forward(torch.zeros(1, 1, self.image_size, self.image_size)).shape[-1])
        self.train(was_training)

    def forward(self, x):
        if self.norm == "xrv":
            x = (x * 2.0 - 1.0) * 1024.0
        else:
            x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self._fwd(self.backbone, x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)


def _densenet(pretrained):
    import torchvision
    w = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
    return torchvision.models.densenet121(weights=w).features


def _densenet_fwd(m, x):
    return F.adaptive_avg_pool2d(F.relu(m(x)), 1).flatten(1)


def build_encoder(name, pretrained=True, image_size=None):
    if name in ("densenet121_imagenet", "densenet121_scratch"):
        use_pre = pretrained and name == "densenet121_imagenet"
        return ImageEncoder(name, _densenet(use_pre), _densenet_fwd, (IMAGENET_MEAN, IMAGENET_STD), image_size or 224)

    if name.startswith("xrv:"):
        try:
            import torchxrayvision as xrv
        except ImportError as e:
            raise ImportError("pip install torchxrayvision") from e
        weights = name.split(":", 1)[1]
        model = xrv.models.DenseNet(weights=weights if pretrained else None)
        return ImageEncoder(name, model, lambda m, x: m.features2(x), "xrv", 224)

    if name.startswith("hf:"):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise ImportError("pip install transformers") from e
        repo = name.split(":", 1)[1]
        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo) if pretrained else AutoModel.from_config(
            __import__("transformers").AutoConfig.from_pretrained(repo))

        def fwd(m, x):
            h = m(pixel_values=x)
            return torch.cat([h.pooler_output, h.last_hidden_state[:, 1:].mean(1)], dim=1)

        return ImageEncoder(name, model, fwd, (tuple(proc.image_mean), tuple(proc.image_std)), image_size or 224)

    if name == "biomedclip":
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("pip install open_clip_torch") from e
        model, _ = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        return ImageEncoder(name, model.visual, lambda m, x: m(x), (CLIP_MEAN, CLIP_STD), 224)

    if name.startswith("timm:"):
        try:
            import timm
        except ImportError as e:
            raise ImportError("pip install timm") from e
        tname = name.split(":", 1)[1]
        kw = {"img_size": image_size} if image_size else {}
        model = timm.create_model(tname, pretrained=pretrained, num_classes=0, **kw)
        cfg = timm.data.resolve_data_config({}, model=model)
        return ImageEncoder(name, model, lambda m, x: m(x), (tuple(cfg["mean"]), tuple(cfg["std"])),
                            image_size or cfg["input_size"][-1])

    raise ValueError(f"Unknown encoder '{name}'. See the module docstring for valid names.")


def load_densenet_checkpoint(encoder, path, min_fraction=0.9):
    """Load a diagnosis-backbone checkpoint into a torchvision-DenseNet encoder by suffix-matching
    parameter names (handles prefixes such as 'backbone.features.'). Raises if too few tensors match."""
    if encoder.name != "densenet121_imagenet":
        raise ValueError("Diagnosis checkpoints only fit the torchvision DenseNet-121 encoder.")
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]
                break
    target = encoder.backbone.state_dict()
    new, hit = {}, 0
    for k, v in ck.items():
        parts = k.split(".")
        for i in range(len(parts)):
            cand = ".".join(parts[i:])
            if cand in target and tuple(target[cand].shape) == tuple(v.shape):
                new[cand] = v
                hit += 1
                break
    frac = len(new) / max(len(target), 1)
    if frac < min_fraction:
        raise RuntimeError(f"Only {len(new)}/{len(target)} tensors matched from {path}; check the checkpoint layout.")
    encoder.backbone.load_state_dict(new, strict=False)
    print(f"Loaded {len(new)}/{len(target)} encoder tensors from {path}")
    return encoder
