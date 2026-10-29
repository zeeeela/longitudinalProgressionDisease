# longitudinal_ext

Copy this folder to `C:\Users\Zila\Downloads\BN5212\midterm_prognosis\longitudinal_ext\` (next to `model/`, `data_loader/`).
Use `main_notebook_clean_ext.ipynb` (your notebook + new Section 11).

Extra packages: `pip install pydicom pillow scikit-learn matplotlib`; encoders as needed:
`torchxrayvision` (xrv:*), `transformers` (hf:*), `open_clip_torch` (biomedclip), `timm` (timm:*).
Pretrained weights download on first use (needs internet; on an offline HPC pre-download to TORCH_HOME / HF_HOME).

| file | role |
|---|---|
| encoders.py | `build_encoder(name)`: DenseNet-ImageNet, TorchXRayVision, RAD-DINO, BiomedCLIP, any timm model |
| mimic_iv_features.py | time-aligned labs/vitals/fluids/drugs/stay flags per (prior,current) pair + coverage report |
| pair_data.py | pair dataset, train-only robust scaling, missing indicators, shuffled-tab control |
| fusion_model.py | shared encoder -> (current-prior) [+ tabular MLP] -> masked 5x3 head |
| train_fusion.py | masked CE, per-epoch cosine LR, best-val-loss checkpoint, same result files as the notebook |
| frozen_screen.py | frozen features + PCA + logistic regression for any encoder (fast screening) |
| tabular_baseline.py | tabular-only logistic regression |
| experiments.py / reporting.py | `run_experiment(...)`, mean/sd summaries, confusion plots |

Not verified on real data: column names of your pair CSV (auto-detected: `input_*`/`target_*` path columns; override with
`input_col`/`target_col`), MIMIC-IV file layout (`hosp/`, `icu/`, .csv or .csv.gz), and pretrained-weight loading
(blocked in my sandbox; the pipeline was tested end-to-end on synthetic DICOMs/MIMIC tables with random weights).

