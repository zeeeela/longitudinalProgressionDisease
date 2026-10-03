"""DenseNet-121 for five report-derived diagnosis findings.

Input: [B, 3, H, W]. Output: [B, 5, 4].
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet121, DenseNet121_Weights

DISEASES = (
    "Pleural effusion",
    "Cardiomegaly",
    "Edema",
    "Pneumonia",
    "Pulmonary edema",
)
STATES = ("not_mentioned", "uncertain", "negative", "positive")


class CXRBackbone(nn.Module):
    """ImageNet initialization plus a trainable five-finding diagnosis head."""

    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between zero and one.")
        network = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.features = network.features
        self.feature_dim = network.classifier.in_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, len(DISEASES) * len(STATES)),
        )
        self._backbone_frozen = False

    def embed(self, images):
        """Return [B, 1024] image features, before the diagnosis head."""
        maps = F.relu(self.features(images), inplace=False)
        return F.adaptive_avg_pool2d(maps, output_size=1).flatten(1)

    def forward(self, images):
        """Raw logits; softmax belongs in evaluation, not before the loss."""
        return self.head(self.embed(images)).reshape(-1, len(DISEASES), len(STATES))

    def freeze_backbone(self):
        """Freeze features, including batch-normalization running statistics."""
        self._backbone_frozen = True
        self.features.requires_grad_(False)
        self.features.eval()

    def unfreeze_backbone(self):
        self._backbone_frozen = False
        self.features.requires_grad_(True)
        self.features.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.features.eval()
        return self


def diagnosis_loss(logits, targets):
    """Mean cross entropy across valid disease labels; ignore -100 targets."""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (len(DISEASES), len(STATES)):
        raise ValueError("Expected logits shaped [B, 5, 4].")
    if targets.shape != logits.shape[:2]:
        raise ValueError("Expected targets shaped [B, 5].")
    if not torch.any(targets != -100):
        return logits.sum() * 0
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)
