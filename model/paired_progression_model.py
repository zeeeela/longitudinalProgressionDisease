"""Trainable shared DenseNet for prior/current disease progression."""
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .cxr_backbone import CXRBackbone, DISEASES
except ImportError:
    from cxr_backbone import CXRBackbone, DISEASES

PROGRESSION_STATES = ('improving', 'stable', 'worsening')


class PairedProgressionModel(nn.Module):
    def __init__(self, diagnosis_checkpoint=None, dropout=0.2):
        super().__init__()
        # pretrained=False means genuinely random weights for the scratch case.
        encoder = CXRBackbone(pretrained=False)
        if diagnosis_checkpoint is not None:
            checkpoint = torch.load(diagnosis_checkpoint, map_location='cpu', weights_only=True)
            if 'diseases' in checkpoint and list(checkpoint['diseases']) != list(DISEASES):
                raise ValueError('Diagnosis checkpoint disease order differs.')
            encoder.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
        self.features = encoder.features
        self.feature_dim = encoder.feature_dim
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, len(DISEASES)*3))
        self.initialization = 'diagnosis_checkpoint' if diagnosis_checkpoint is not None else 'random'

    def forward(self, prior, current):
        if prior.shape != current.shape:
            raise ValueError('Prior and current batches must have matching image shapes.')
        batch_size = prior.shape[0]
        maps = self.features(torch.cat([prior, current], dim=0))
        features = F.adaptive_avg_pool2d(F.relu(maps, inplace=False), 1).flatten(1)
        difference = features[batch_size:] - features[:batch_size]
        return self.classifier(difference).reshape(batch_size, len(DISEASES), 3)
