# Longitudinal chest X-ray progression

Compare a patient's earlier and later chest X-rays to classify each disease as
**improving, stable, or worsening**. Both examinations are inputs, so this is
classification of observed change rather than forecasting an unseen examination.

## Start here

Open [main_notebook_clean_ext_multigpu_local.ipynb](main_notebook_clean_ext_multigpu_local.ipynb)
from the project folder and select the `longitudinal_progression` Python environment.
The notebook contains short steps with explanations between code cells.

1. Set the data paths and run options in Section 1.
2. Load the original pair table and accepted reviews in Section 2.
3. Prepare the MIMIC-IV clinical inputs in Section 3.
4. Select the saved diagnosis checkpoint used by A2 in Section 4.
5. Run or load A, A2, and B-F in Section 5, then read Sections 6-7 for results.

`RESUME_RUN=True` reopens the latest run. `REUSE_COMPLETED=True` loads matching
completed results. Set `RESUME_RUN=False` for a new run. Optional encoder screening,
clinical logistic regression, and data extraction are grouped at the end.

## The seven experiments

| ID | Image model / starting weights | Clinical inputs |
|---|---|---|
| A | DenseNet-121, ImageNet | None |
| A2 | DenseNet-121, saved diagnosis checkpoint | None |
| B | DenseNet-121, ImageNet | Matched to each pair |
| C | RAD-DINO | None |
| D | RAD-DINO | Matched to each pair |
| E | RAD-DINO | Shuffled separately within each split |
| F | No image model; tabular neural network | Matched to each pair |

Image models use the difference between current and prior image features. The
optional Section 8 screens keep encoders frozen and fit simpler classifiers.

The disease targets are Pleural effusion, Cardiomegaly, Edema, Pneumonia, and
Pulmonary edema. Edema and Pulmonary edema overlap under the labeling rules.
Targets are `0 = improving`, `1 = stable`, `2 = worsening`, and `-100 = ignore`.

## Local data and reviewed labels

Data files are excluded from Git and must be supplied locally. The usual layout is:

```text
BN5212/
  MIMIC-CXR/p10/p10/...
  MIMIC-IV/3.1/hosp/...
  MIMIC-IV/3.1/icu/...
  midterm_prognosis/
    labelling/
      prognosis_patient_splits.csv
      prognosis_labels_hybrid/
        all_disease_pairs.csv
        reviewed_only_excluding_7_conflicts.csv
```

A2 also needs a saved diagnosis checkpoint. Optional diagnosis training uses the
local `cxr_labels_selected_everpositive` label, review, and report-summary tables.
The original labeling and base-pair preparation utilities remain in `labelling/`
and `data_loader/`.

`prepare_reviewed_training.py` merges selected reviews by patient, prior study,
current study, and disease. It preserves patient assignments and rejects duplicate
or conflicting reviews. One image pair can have several disease labels.

The completed reviewed-data run contains 937 pairs and 1,500 disease labels:
307 accepted reviews confirm 257 existing labels and add 50 labels, including
labels on 22 new pairs. Seven conflicting progression decisions and manual
`no_comparison` decisions were excluded from the selected review file.

Every run keeps its data snapshot, merge audit, clinical features, model files,
settings, predictions, and evaluations under `training_outputs/`. Its location is
recorded in `training_outputs/latest_reviewed_run.txt`. These files stay local.

## Clinical inputs and evaluation

The current reviewed-data run retains 47 numeric inputs and 17 missing-data flags.
Selection and scaling use training rows only. Missing values are filled with the
training-median equivalent after scaling. Delta features have no dedicated
missing-data flags. A clinical-data row does not imply that all measurements exist.

The 80% coverage experiment has not been added yet. Under the current rules that
cutoff would retain 17 numeric inputs plus 2 missing-data flags.

Choose models using validation scores and report held-out test scores. The main
comparison averages the five disease-specific macro-F1 scores. Defaults are 20
epochs and seed 42; one seed does not establish consistent generalization.
See [mimic_feature_validation.md](mimic_feature_validation.md) for extraction
limitations and [historical_extension_results.md](historical_extension_results.md)
for archived original scores.

## Source files

| File or folder | Purpose |
|---|---|
| `longitudinal_ext/` | Encoders, clinical features, fusion models, training, and reporting |
| `prepare_reviewed_training.py` | Merge accepted reviews into the image-pair table |
| `local_notebook_helpers.py` | Reuse data snapshots and matching saved runs |
| `local_data_archives.py` | Optional ZIP extraction |
| `audit_mimic_features.py` | Report coverage and exact retained inputs |
| `execute_local_training.py` | Execute the notebook and save outputs after each cell |
| `prefetch_training_encoders.py` | Download and check the six optional image encoders |
| `summarize_reviewed_extensions.py` | Compare completed runs with historical results |
| `run_extension.py` | Run one A-F experiment per process with explicit input paths |
| `vanda_extensions.pbs` | Optional PBS job-array template |
| `model/` | Diagnosis model and optional training for A2 |
| `data_loader/` | Diagnosis loader, DICOM utilities, and base-pair preparation |
| `labelling/` | Vocabulary, report labeling, and progression-label utilities |

Older notebooks and removed implementations are preserved locally in the ignored
`archive/` folder. The document builder in `docs/` describes an earlier workflow;
this README and the current notebook are the maintained run guide.

## Terminal commands

Install `requirements.txt` in the intended Python environment. Use a CUDA-enabled
PyTorch installation for NVIDIA GPU training.

```powershell
python execute_local_training.py --run-dir training_outputs/YOUR_RUN --extensions-only
```

For a single experiment, pass the reviewed manifest and matching clinical table:

```powershell
python run_extension.py --experiment D_new_img_tab --manifest training_outputs/YOUR_RUN/data/all_disease_pairs_reviewed.csv --tab-csv training_outputs/YOUR_RUN/features/pair_tabular_features.csv --out-root training_outputs/YOUR_RUN
```

Notebook outputs are cleared in the source copy for Git. Executed notebooks and
full local results remain under `training_outputs/`.
