"""Frozen checkpoint features + one PCA/logistic-regression model per disease."""
import numpy as np
import torch
from torch import nn

try:
    from .cxr_backbone import CXRBackbone, DISEASES
except ImportError:
    from cxr_backbone import CXRBackbone, DISEASES

LABELS = ('improving', 'stable', 'worsening')


class PairFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'diseases' in checkpoint and list(checkpoint['diseases']) != list(DISEASES):
            raise ValueError('Checkpoint disease order differs from the loader/model.')
        self.backbone = CXRBackbone(pretrained=False)
        self.backbone.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Feature extraction must never update batch-normalization statistics.
        return super().train(False)

    @torch.inference_mode()
    def forward(self, prior_image, current_image):
        return self.backbone.embed(current_image) - self.backbone.embed(prior_image)


def fit_disease_classifier(train_x, train_y, val_x, val_y, *, components=50,
                           c_values=(0.1, 0.5, 1.0, 10.0), seed=42):
    """Fit PCA on training data only; select regularization by validation macro F1."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    if len(np.unique(train_y)) < 2 or len(train_x) < 3:
        raise ValueError('Need at least three training pairs and two training classes.')
    if not len(val_y):
        raise ValueError('No validation labels for this disease.')
    if not c_values or any(c <= 0 for c in c_values) or components < 1:
        raise ValueError('Positive PCA component count and C values required.')
    pca = PCA(n_components=min(components, len(train_x)-1, train_x.shape[1]),
              svd_solver='full', random_state=seed)
    train_z, val_z = pca.fit_transform(train_x), None
    val_z = pca.transform(val_x)
    best, best_score, trials = None, -1.0, []
    for c in c_values:
        classifier = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=seed)
        classifier.fit(train_z, train_y)
        score = f1_score(val_y, classifier.predict(val_z), labels=[0, 1, 2], average='macro', zero_division=0)
        trials.append({'C': float(c), 'val_macro_f1': float(score)})
        if score > best_score:
            best, best_score = classifier, score
    return Pipeline([('pca', pca), ('classifier', best)]), trials


def predict_probabilities(classifier, features):
    """Always return three columns in LABELS order, including absent train classes."""
    probabilities = np.zeros((len(features), len(LABELS)), dtype=np.float64)
    probabilities[:, classifier.classes_.astype(int)] = classifier.predict_proba(features)
    return probabilities
