"""Build the repository workflow handover document using python-docx."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'Repository_workflow_summary.docx'
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.top_margin = sec.bottom_margin = Inches(.7)
sec.left_margin = sec.right_margin = Inches(.75)
for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2']:
    s = doc.styles[name]
    s.font.name = 'Calibri'
    s.font.color.rgb = RGBColor(0, 0, 0)
normal = doc.styles['Normal']
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(7)
normal.paragraph_format.line_spacing = 1.08
doc.styles['Title'].font.size = Pt(27)
doc.styles['Heading 1'].font.size = Pt(19)
doc.styles['Heading 2'].font.size = Pt(13)
for name in ['Heading 1', 'Heading 2']:
    doc.styles[name].paragraph_format.space_before = Pt(12)
    doc.styles[name].paragraph_format.space_after = Pt(6)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = footer.add_run('BN5212 workflow  |  ')
run.font.size = Pt(9)
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)


def p(text, style=None):
    return doc.add_paragraph(text, style)


def h(text):
    doc.add_heading(text, level=2)


def page(title):
    doc.add_page_break()
    doc.add_heading(title, level=1)


def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    for cell, text in zip(t.rows[0].cells, headers):
        cell.text = text
    for row in rows:
        for cell, text in zip(t.add_row().cells, row):
            cell.text = str(text)
    for i, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        no_split = OxmlElement('w:cantSplit'); trpr.append(no_split)
        if i == 0:
            repeat = OxmlElement('w:tblHeader'); trpr.append(repeat)
        for j, cell in enumerate(row.cells):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr = cell._tc.get_or_add_tcPr()
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E7EDF2' if i == 0 else 'FFFFFF'); pr.append(shade)
            borders = OxmlElement('w:tcBorders')
            for side in ['top', 'left', 'bottom', 'right']:
                el = OxmlElement('w:' + side); el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4'); el.set(qn('w:color'), 'D9D9D9'); borders.append(el)
            pr.append(borders)
            margins = OxmlElement('w:tcMar')
            for side in ['top', 'left', 'bottom', 'right']:
                el = OxmlElement('w:' + side); el.set(qn('w:w'), '90'); el.set(qn('w:type'), 'dxa'); margins.append(el)
            pr.append(margins)
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(2)
                para.paragraph_format.line_spacing = 1.03
                for r in para.runs:
                    r.font.size = Pt(10)
                    r.bold = i == 0
    p('')
    return t


def code(text):
    para = p(text)
    para.paragraph_format.space_after = Pt(8)
    for r in para.runs:
        r.font.name = 'Consolas'; r.font.size = Pt(9)


doc.add_heading('Chest X ray progression repository workflow', 0)
p('BN5212 project summary\nRepository state reviewed on 2 November 2026', 'Subtitle')
p('This repository builds a research pipeline that compares a patient’s prior and current chest X-rays and classifies disease-specific change. It combines report-derived labels, paired-image models and optional MIMIC-IV clinical features. This document explains how the stages connect, which files drive them, how experiments run, and what the current results support.')
p('The outcome is longitudinal progression classification: both examinations are inputs. Despite “prognosis” in several filenames, the models do not forecast an unseen future examination. Clinical-feature fusion has mixed results and is not yet established as a reliable improvement.')
h('Workflow at a glance')
for text in [
    '1  Select patients with repeated reports and build the disease vocabulary.',
    '2  Label disease presence from Findings and Impression text.',
    '3  Order studies using DICOM timestamps and form consecutive study pairs.',
    '4  Assign five progression targets and reuse patient-level data splits.',
    '5  Train a diagnosis backbone and compare frozen, fine-tuned and scratch image models.',
    '6  Align MIMIC-IV features, screen encoders and compare image and clinical fusion models.',
    '7  Evaluate held-out patients and save metrics, checkpoints and confusion matrices.'
]: p(text)
h('Entry points')
table(['Entry point', 'Use'], [
    ['main_notebook_clean_ext_multigpu.ipynb', 'Original HPC workflow with saved outputs and Section 11 extensions.'],
    ['main_notebook_clean_ext_multigpu_local.ipynb', 'Windows copy with corrected paths, disabled training defaults and feature audit.'],
    ['run_extension.py', 'Run one Section 11 experiment and seed per process.'],
    ['vanda_extensions.pbs', 'PBS template for queued, independent one-GPU experiments.'],
], [3.25, 3.52])
p('The older notebooks remain historical variants. README.md principally describes the original image-only workflow; longitudinal_ext/README.md describes the extension modules.')

page('Data and label preparation')
h('Inputs and cohort')
p('MIMIC-CXR supplies reports and DICOM images. MIMIC-IV supplies a diagnosis dictionary for vocabulary construction and, separately, clinical events for the fusion experiments. The cohort helper selects patient folders with at least two report files; multiple images within one study are not multiple visits.')
p('The vocabulary builder maps ICD diagnosis names to grouped phenotype names. Those dictionary entries support report matching; they are not patient-level diagnostic ground truth. The selected targets are Pleural effusion, Cardiomegaly, Edema, Pneumonia and Pulmonary edema.')
h('From reports to paired targets')
p('scan_disease_v2.py scans Findings and Impression for positive, negative, uncertain and unmentioned disease states. Review queues retain ambiguous cases and support manual overrides. Unmentioned is not treated as disease absence.')
p('build_prognosis_labels.py uses DICOM dates and times to order studies. Five ordered studies yield four consecutive pairs. Patients affected by unresolved study ordering are excluded from pairing. Comparison language and explicit presence transitions produce candidate progression labels.')
table(['Evidence', 'Candidate interpretation'], [
    ['Improved, decreased or resolved', 'Improving'],
    ['Stable or unchanged', 'Stable'],
    ['Worsened, increased or new', 'Worsening'],
    ['Positive to explicit negative', 'Improving'],
    ['Explicit negative to positive', 'Worsening'],
    ['Positive to positive without trend, or unmentioned', 'No progression inferred'],
    ['Conflicting trend and presence transition', 'Mixed and flagged for review'],
], [3.4, 3.37])
p('The notebook uses automatic candidates. Mixed and no-comparison labels are ignored per disease: 0 = improving, 1 = stable, 2 = worsening, and -100 = ignore. A pair remains usable if at least one disease has a valid target. A report’s comparison may refer to a different earlier study, so label-reference validity still needs review.')
h('Images and patient splits')
p('One DICOM is selected per study, preferring the first AP/PA image in sorted file order. Preprocessing handles rescaling, padding, grayscale polarity, percentile intensity scaling and resizing. The original models use 224 by 224 images with ImageNet normalization; extension encoders apply their own normalization.')
table(['Split', 'Pairs', 'Patients'], [('Train',704,212),('Validation',92,40),('Test',119,39),('Total',915,291)], [3.0,1.88,1.89])
p('prognosis_patient_splits.csv is shared across diagnosis and progression. The paired manifest has no patient overlap across splits. Pair counts differ from individual-study diagnosis counts. Path remapping retains only pairs with both images; all 915 pairs were available locally.')

page('Image models and extension experiments')
h('Diagnosis pretraining')
p('CXRBackbone uses ImageNet-initialized DenseNet-121 to predict four report-derived states for each of five diseases from one study. train_backbone.py uses masked cross-entropy and AdamW, selecting the checkpoint with the lowest validation loss. That checkpoint initializes the pretrained progression comparisons.')
h('Original progression comparison')
table(['Experiment', 'How it works'], [
    ['Frozen diagnosis features', 'Freeze DenseNet; subtract prior from current features; fit training-only PCA and a separate logistic-regression classifier per disease. Choose regularization using validation macro F1.'],
    ['Paired fine-tuning', 'Load the diagnosis encoder; train the shared encoder and a new five-disease progression head on feature differences.'],
    ['Paired scratch training', 'Use the same paired architecture with random weights and no diagnosis or ImageNet initialization.'],
], [2.0,4.77])
p('The paired network produces [batch, 5, 3] logits. Both images pass through one shared encoder. Masked cross-entropy excludes -100 targets. Neural runs use 20 epochs and choose the checkpoint by validation loss. Fine-tuning starts at backbone/head rates 1e-5/1e-4; scratch uses 1e-4/1e-4. Cosine decay reaches one tenth of each initial rate. This comparison changes backbone learning rate as well as initialization.')
h('Section 11 extension workflow')
p('Build clinical features and inspect coverage; run a tabular logistic-regression baseline; screen frozen encoders; then run image and fusion ablations. Encoder adapters support ImageNet DenseNet, TorchXRayVision, RAD-DINO, BiomedCLIP and timm backbones.')
table(['Experiment key', 'Inputs and initialization'], [
    ['A_base_img', 'ImageNet DenseNet images'],
    ['A2_base_diag_img', 'Diagnosis-pretrained DenseNet images'],
    ['B_base_img_tab', 'ImageNet DenseNet images plus clinical features'],
    ['C_new_img', 'New encoder images; saved notebook uses RAD-DINO'],
    ['D_new_img_tab', 'New encoder images plus clinical features'],
    ['E_new_img_tab_shuffled', 'Same fusion model with clinical rows shuffled within each split'],
    ['F_tab_nn', 'Clinical-feature neural network only'],
], [2.6,4.17])
p('PairedFusionNet joins image feature differences with an optional clinical MLP, then applies LayerNorm and a shared multilayer head. It uses AdamW and validation-loss checkpoint selection. This head differs from the original paired model. The original notebook used seed 42; the batch launcher maps seven experiments across seeds 42, 43 and 44. Those additional runs have not been executed in this review.')

page('MIMIC IV feature construction')
p('mimic_iv_features.py aligns events to prior and current DICOM study times. It takes the latest blood-test value within 48 hours, median vital values within 24 hours, 24-hour fluid totals, prescription activity and starts, and hospital/ICU stay indicators. Demographics and the interval between X-rays provide additional context.')
h('Features that reached the local model input')
p('The executed local reconstruction retained 47 numeric columns and 17 missingness indicators, giving 64 inputs. Each measurement in the first five rows below supplies a current value and a current-minus-prior difference. Raw prior values are not directly supplied.')
table(['Group', 'Retained measurements', 'Numeric inputs'], [
    ['Blood tests', 'WBC, creatinine, BUN, troponin T, lactate, hemoglobin, albumin, sodium',16],
    ['Vitals and weight', 'Heart rate, respiratory rate, SpO2, systolic blood pressure, FiO2, weight',12],
    ['Fluid balance', 'Net fluid over 24 hours',2],
    ['Medication orders', 'Antibiotic/diuretic active flags and new-order counts over 24 hours',8],
    ['Care setting', 'Inpatient and ICU flags',4],
    ['Other', 'Age, sex, log X-ray gap, log new antibiotic/diuretic order counts between studies',5],
], [1.25,4.52,1.0])
p('Seventeen missingness indicators cover the current values of eight blood tests, six vitals/weight measurements, fluid balance, age and sex. No delta missingness indicators are added.')
h('Selection and preprocessing')
p('The extractor generates 77 columns: current, prior and delta versions of 24 measurements, plus five other fields. The model considers 53 candidates after excluding raw prior values. Training-only filtering removes columns below 5% coverage or with fewer than two distinct values. Retained columns are median-imputed, robustly scaled and clipped; the same fitted transformation is applied to validation and test.')
p('CRP and Celsius temperature were excluded for low training coverage. NT-proBNP was excluded because the regex expects “NT-proBNP” while the local dictionary contains “NTproBNP” (item 50963). This is a mapping defect, not evidence that the test is absent from MIMIC.')
h('Why these groups were considered')
p('Renal tests, fluid balance, weight and diuretic orders provide congestion context; inflammatory tests and antibiotic orders provide infection context; oxygenation and vital signs provide severity context. These are plausible exploratory predictors, not proven independent contributors. Care location, prescriptions and missingness can also encode clinical practice rather than disease biology. The 24/48-hour windows and 5% threshold remain design choices requiring validation.')
p('All 24 reconstructed coverage rows match the saved HPC coverage table to three decimals, and the actual local fusion loaders reproduce the 64 inputs. Exact historical HPC settings are unavailable. The audit preserves the original extraction rules; it does not silently correct them.')

page('Evaluation and interpretation')
h('What is measured')
p('Evaluation masks invalid targets separately for each disease. Outputs include accuracy, macro F1, class support and three-by-three confusion matrices; extension metrics also include balanced accuracy, while frozen baselines may include AUROC/AUPRC when defined. Macro F1 uses all three class labels. Validation guides model selection; test results describe held-out patient performance.')
h('Saved extension results')
p('The values below are mean disease macro F1 from the original notebook’s saved HPC outputs. They are rounded, single-seed results and were not retrained during the local audit.')
table(['Model', 'Validation', 'Test'], [
    ['ImageNet DenseNet images', '.379', '.316'],
    ['ImageNet DenseNet plus clinical', '.381', '.378'],
    ['Diagnosis-pretrained DenseNet images', '.428', '.414'],
    ['RAD-DINO images', '.410', '.409'],
    ['RAD-DINO plus clinical', '.453', '.380'],
    ['RAD-DINO plus shuffled clinical', '.405', '.435'],
    ['Clinical neural network only', '.312', '.290'],
], [4.17,1.3,1.3])
p('Adding clinical inputs increased DenseNet test macro F1 by .062 but reduced RAD-DINO test macro F1 by .029. The shuffled RAD-DINO control exceeded aligned fusion on test. These results do not establish a consistent benefit from clinical fusion. The frozen screening winner was NIH-XRV (.575 validation), whereas RAD-DINO scored .414; the later hardcoded RAD-DINO choice should not be described as the screening winner.')
h('Limits that affect the conclusion')
for text in [
    'Small and imbalanced evaluation sets: the five test targets have 55, 57, 28, 12 and 27 eligible pairs. Pneumonia has no stable test examples. Edema and Pulmonary edema overlap under the report-matching rules.',
    'Timing: lab charttime can precede result availability. Fluid input amounts are assigned to infusion start even if delivery extends past the X-ray. The current extractor therefore does not establish strict prospective availability.',
    'Clinical meaning: prescriptions are order proxies, not confirmed administrations. Stay endpoints are used internally to derive active-stay flags despite the module’s stronger “no discharge time” wording.',
    'Reproducibility: original threaded jobs reset shared random-number generators. Indexed CUDA strings disable mixed precision in the extension trainer. Feature caches do not encode extraction-code or cohort versions.',
    'Uncertainty: only one historical seed is shown; no confidence intervals establish significance. Potential encoder pretraining overlap and report-label validity remain unresolved.'
]: p(text, 'List Bullet')
p('A stronger follow-up would correct extraction, version the caches, repeat matched ablations across seeds, and estimate paired differences using patient-clustered uncertainty intervals. Further tuning after inspecting test results should be treated as exploratory.')

page('Running locally and on Vanda')
h('Local workflow')
p('The Windows copy resolves the project under C:\\Users\\Zila\\Downloads\\BN5212\\midterm_prognosis. Its sibling directories contain MIMIC-CXR and MIMIC-IV/3.1. Use the longitudinal_progression environment. Training and extraction switches begin disabled, and Windows data loaders use zero subprocess workers.')
p('Run setup and path checks, reuse or intentionally rebuild labels, prepare pair loaders, and inspect feature coverage. Enable only the experiments needed. Local hardware is one RTX 5060 Laptop GPU with about 8 GB VRAM; begin with a small batch size.')
code('conda activate longitudinal_progression\npython run_extension.py --experiment A_base_img --batch-size 2\npython run_extension.py --experiment D_new_img_tab --batch-size 2\npython run_extension.py --list-tasks')
p('The launcher validates paths and patient splits, then runs one Section 11 experiment. --dry-run checks configuration without loading a model. It uses process-local image caching and the unindexed “cuda” device string. It does not build labels, reconstruct clinical features or train a diagnosis backbone.')
h('Vanda workflow')
p('Vanda uses PBS. Upload the launcher/template to the HPC repository, use a Linux environment, and prepare the Linux-remapped pair manifest before training. The default launcher searches extensions_outputs/all_disease_pairs_local.csv and the available clinical-feature CSV. Explicit --manifest, --tab-csv and --checkpoint overrides are supported.')
code('ssh YOUR_NUSNETID@vanda.nus.edu.sg\nqstat -Q\nexport PYTHON_BIN=/actual/environment/bin/python\n# Submit from the HPC repository directory\nqsub -q YOUR_GPU_QUEUE -v PYTHON_BIN vanda_extensions.pbs\nqsub -q YOUR_GPU_QUEUE -v PYTHON_BIN \\\n  -J \'0-20%2\' vanda_extensions.pbs')
p('The array maps seven experiment keys to seeds 42, 43 and 44. Each task requests one GPU, four CPU cores, 24 GB RAM and four hours. %2 permits at most two simultaneous tasks. Confirm the permitted GPU queue, account allocation and resource limits on Vanda; the template is not a verified account-specific submission. Weights must be available to the compute environment.')
code('qstat -t -u "$USER"\nqstat -f JOB_ID\ntail -f logs/JOB_ID_task0.log\nqdel JOB_ID')
p('Submitted jobs continue without the laptop connection. qsub -a YYYYMMDDhhmm sets an earliest eligible start time in the scheduler’s timezone, not a guaranteed start. The trainer has no implemented automatic mid-run resume, so request enough walltime; best-checkpoint saving is not a complete restart mechanism.')
h('What multi GPU means in this repository')
p('The original notebook assigns independent experiments to GPUs using threads. The local copy defaults to serial execution; PBS arrays provide isolated processes for concurrent experiments. Neither mechanism distributes one model across GPUs. That would require DistributedDataParallel, distributed sampling and coordinated evaluation/checkpoint handling. Simply requesting more GPUs does not add those capabilities.')

page('Code map and saved artifacts')
table(['Location', 'Responsibility'], [
    ['check_number_of_files.py', 'Identify patients with repeated report files.'],
    ['labelling/', 'Vocabulary, report-state extraction, positive counts, review queues and chronological progression candidates.'],
    ['data_loader/', 'DICOM preparation, original diagnosis and paired datasets, manifest validation and patient splits.'],
    ['model/', 'DenseNet diagnosis model, frozen PCA/logistic regression and original paired neural training.'],
    ['longitudinal_ext/', 'Encoder adapters, image cache, clinical extraction/scaling, fusion, screening, baselines, metrics and reporting.'],
    ['audit_mimic_features.py', 'Reconstruct original clinical-feature selection locally without model training.'],
    ['prepare_local_notebook.py', 'Generate the Windows copy and archive historical output tables.'],
    ['run_extension.py and vanda_extensions.pbs', 'One experiment per process and PBS array submission template.'],
], [2.5,4.27])
h('Artifacts to preserve')
p('Labels and manifests live under labelling/. The key pair input is prognosis_labels_hybrid/all_disease_pairs.csv; patient assignments come from prognosis_patient_splits.csv. Keep manual review decisions when rebuilding generated label files.')
p('training_outputs/ contains separate timestamped runs. Depending on the experiment, these contain best_backbone.pt, best_pair_model.pt or best_model.pt, fitted classifiers, settings.json, history.csv, metrics.csv, confusion_matrices.json and predictions.csv. Frozen original runs use metrics.json for their saved confusion matrices.')
p('extensions_outputs/ contains remapped manifests, decoded-image caches and clinical features. local_feature_audit/ contains pair_tabular_features.csv, feature_selection_audit.csv, feature_coverage.csv and retained_features.json. Rebuild in a new cache directory after changing the cohort, time logic or feature definitions.')
h('Reading and provenance')
p('README.md and longitudinal_ext/README.md explain the original and extension designs. historical_extension_results.md preserves the saved HPC result tables. mimic_feature_validation.md records the detailed clinical-feature audit, exact retained groups and evidence limitations. The original notebook was preserved; the local copy clears historical execution outputs and labels the archived results separately.')
p('Completed local checks covered notebook syntax, setup and pair loaders, image availability, exact feature reconstruction, finite model-input arrays, and launcher configuration. Training was not rerun and no Vanda job was submitted during this review.')
h('External documentation')
for label, url in [
    ('MIMIC lab result timing', 'https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html'),
    ('MIMIC infusion intervals', 'https://mimic.mit.edu/docs/iv/modules/icu/inputevents.html'),
    ('MIMIC prescription meaning', 'https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html'),
    ('NUS Vanda access and PBS overview', 'https://mle4217-5219.matsci.dev/orientation/vanda/'),
    ('PBS submission reference', 'https://github.com/openpbs/openpbs/blob/master/doc/man1/qsub.1B'),
]:
    para = p(label + '\n' + url)
    para.paragraph_format.space_after = Pt(5)
    for r in para.runs: r.font.size = Pt(9)

doc.core_properties.title = 'Chest X ray progression repository workflow'
doc.core_properties.subject = 'BN5212 repository workflow and execution summary'
doc.core_properties.author = ''
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(OUT)
