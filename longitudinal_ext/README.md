# Progression experiment modules

Start with [main_notebook_clean_ext_multigpu_local.ipynb](../main_notebook_clean_ext_multigpu_local.ipynb).
It compares A, A2, and B-F on a shared reviewed image-pair dataset.

| Module | Purpose |
|---|---|
| `encoders.py` | DenseNet, TorchXRayVision, RAD-DINO, BiomedCLIP, and timm encoders |
| `mimic_iv_features.py` | Study-aligned measurements, changes, medication orders, and care-setting fields |
| `pair_data.py` | Pair loading, training-only selection/scaling, missing-data flags, and shuffled control |
| `fusion_model.py` | Image-feature difference, optional clinical branch, and five disease heads |
| `train_fusion.py` | Masked cross-entropy, AdamW, cosine learning-rate schedule, validation-loss checkpoints |
| `frozen_screen.py` | Optional frozen encoders with PCA/logistic regression |
| `tabular_baseline.py` | Optional clinical-only logistic regression |
| `experiments.py` | Assemble the encoder, loaders, model, and training run |
| `metrics_io.py` | Metrics, predictions, settings, and confusion matrices |
| `reporting.py` | Summaries and confusion plots |

Each disease has three progression outputs: improving, stable, and worsening.
A target of `-100` is ignored for that disease. Patient groups are disjoint.

The current feature-coverage threshold is 5% of training pairs. Missing values use
the training-median equivalent after scaling. Selected non-delta inputs receive
missing-data flags. Data, caches, pretrained weights, and results stay outside Git.

See the root README for paths, dependencies, and commands, and
`mimic_feature_validation.md` for feature availability and timing limitations.
