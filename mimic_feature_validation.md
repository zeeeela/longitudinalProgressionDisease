## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-02
- Verification Status: ANALYZED
- Version Label: local_feature_audit_v1
- Sources: repository code, original notebook's saved HPC outputs, local MIMIC-IV dictionaries/tables, linked documentation.
- Model training was not rerun. Local feature reconstruction is distinct from recovery of historical HPC settings.

## Findings

There is no consistent demonstrated improvement from adding MIMIC-IV features. The historical notebook reports these mean disease macro F1 scores (rounded to three decimals):

| Encoder | Validation image only | Validation + tabular | Difference | Test image only | Test + tabular | Difference |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet DenseNet | .379 | .381 | +.002 | .316 | .378 | +.062 |
| RAD-DINO | .410 | .453 | +.043 | .409 | .380 | -.029 |

RAD-DINO with shuffled tabular rows scores .405 validation / .435 test, exceeding correctly aligned fusion on test by .055. This undermines a claim that aligned clinical information reliably improves generalization; it is not proof that clinical features have no value. Only seed 42 was run, and no uncertainty intervals were reported. Differences above are approximate because the source tables are rounded.

The diagnosis-pretrained extension image-only model scores .428 validation / .414 test. The tabular neural model scores .312 / .290. A separate tabular logistic-regression baseline has stronger test performance (approximately .410 averaged over diseases), indicating substantial dependence on classifier/training choice. The 2x2 fusion experiments use the extension head; do not equate them with the original Sections 7-8 architecture.

These results are archived verbatim in `historical_extension_results.md`. The test split has 119 pairs, but only 55/57/28/12/27 eligible targets for the five diseases; pneumonia has no stable test examples. Edema and Pulmonary edema overlap, so the mean is not five independent clinical endpoints. Original frozen-encoder screening selected NIH-XRV numerically (.575 validation), not RAD-DINO (.414); the later hardcoded RAD-DINO choice is not justified as the screening winner.

## What features are selected

**Completed local reconstruction: 47 numeric features + 17 missingness indicators = 64 model inputs**, using all 915 original pairs and 704 training rows. This is an executed reconstruction of feature selection, not a model-training rerun or certified recovery of historical HPC settings.

The retained measurements below each contribute current and current-minus-prior values (42 numeric columns):

- Blood tests (8): WBC, creatinine, BUN, troponin T, lactate, hemoglobin, albumin, sodium.
- Vitals/weight (6): heart rate, respiratory rate, SpO2, systolic blood pressure, FiO2, weight.
- Fluid balance (1): net fluid over 24 hours.
- Medication order features (4): diuretic active, diuretic new-order count over 24 hours, antibiotic active, antibiotic new-order count over 24 hours.
- Care setting (2): inpatient and ICU flags.

The remaining five numeric columns are age, male indicator, log X-ray gap, and log counts of new diuretic/antibiotic prescription rows between visits. Seventeen missingness indicators correspond to the current values of the eight blood tests, six vitals/weight, fluid balance, age, and sex. No raw prior columns or delta missingness indicators enter the model.

Excluded current/delta columns: NT-proBNP (0% training coverage; dictionary mismatch), CRP (1.14% current / 0.43% delta), temperature Celsius (2.84% / 1.14%). All other candidates met the existing availability/variation rules. Troponin delta barely exceeds the threshold (6.11%); retention does not establish reliability or predictive importance.

The extractor generates 24 measurements at current/prior time and their differences, plus five demographics/interval fields: 77 columns. The model considers 24 current + 24 delta + 5 other columns, or 53 candidates. Prior values are not directly supplied. Train-only filtering removes coverage below 5% and variables with fewer than two distinct nonmissing values. Retained values use median/robust-scale normalization, clipping and median imputation; missingness indicators are appended for retained non-delta columns missing in training. Thus feature filtering is based on availability and variation, not demonstrated predictive importance.

The exact LOCAL reconstruction is exported to `extensions_outputs/local_feature_audit/retained_features.json` and `feature_selection_audit.csv`. Historical HPC `settings.json` files are absent; the original run's exact list cannot be certified from its aggregate coverage table. The copied notebook prints this distinction with the exact local list.

| Candidate measurements | Rationale for considering them | Limits of that rationale |
|---|---|---|
| Creatinine, BUN, sodium | Renal function/electrolyte context for congestion and diuretic treatment | Indirect, nonspecific correlates of radiographic change |
| Weight, fluid balance, diuretic prescriptions, NT-proBNP | Congestion and treatment context, especially edema/effusion | Treatment reflects clinician response; fluid timing and BNP matching need correction |
| WBC, CRP, temperature, antibiotics | Inflammatory and treatment context for pneumonia | Not pneumonia-specific; prescriptions are orders, not confirmed administration |
| HR, RR, SpO2, FiO2, SBP, lactate | Physiological severity and respiratory support | Nonspecific; ICU-only recording and treatment affect values |
| Albumin, hemoglobin, troponin T | Broader illness/cardiovascular context | Exploratory covariates; no demonstrated independent contribution here |
| Age, sex, X-ray gap | Baseline context and time available for progression | Demographic/selection shortcuts possible; not mechanistic proof |
| Inpatient/ICU flags | Context for measurements and acuity | Care-process proxies; informative missingness can encode location rather than disease |

The congestion rationale is consistent with clinical use of natriuretic peptides and diuretics in acute heart failure, but that does not validate these features for this specific radiographic target. [NICE acute heart failure](https://www.nice.org.uk/guidance/cg187/chapter/Recommendations).

The infection/respiratory rationale is consistent with pneumonia assessment using oxygenation and, in appropriate settings, CRP and treatment response. These guidelines do not establish that this notebook's WBC/CRP/window combination improves machine-learning performance. [NICE pneumonia](https://www.nice.org.uk/guidance/ng250/chapter/Recommendations).

The 48-hour lab / 24-hour vital windows, 5% coverage cutoff, and list of drug-name regexes are engineering choices, not validated optimal choices. Current-plus-delta is reasonable for longitudinal change, but delta is missing when either visit lacks data. Medication `new24h` and between-study features count prescription rows; they are not verified counts of distinct treatment episodes. Between-study counts and X-ray gap are log1p-transformed; the `new24h` counts are not.

## Implementation findings affecting validity

1. **NT-proBNP is absent because of a mapping bug.** The code requires `^NT-proBNP$`; the local dictionary has item 50963 named `NTproBNP`. The audit found zero matched item IDs. Its zero coverage is not evidence that the underlying dataset lacks the test. Correct the mapping and invalidate dependent caches before a new scientific run; the reconstruction deliberately preserves the original rules.
2. **Lab result availability is not enforced.** Code uses `charttime` only. MIMIC documents this as usually specimen acquisition, often earlier than availability. A prospective availability claim requires `storetime` handling as well. [MIMIC labevents](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html).
3. **Fluid timing can include future delivery.** Code assigns the entire input amount to `starttime`, ignoring `endtime`. MIMIC defines amount over the infusion interval. An infusion beginning before the X-ray and ending after it can contribute future fluid. Completed-event restrictions or carefully justified interval allocation are needed. Missing input or output is also treated as zero when the other exists, so this is not a complete fluid-balance measurement. [MIMIC inputevents](https://mimic.mit.edu/docs/iv/modules/icu/inputevents.html).
4. **Prescriptions are order proxies.** The code uses prescription starts/stops and name matching, not eMAR administration evidence. Do not describe the features as confirmed medication doses. [MIMIC prescriptions](https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html).
5. **The 'no discharge time' claim is too strong.** `_load_stays` reads discharge/ICU-out times to determine active stay membership. No length-of-stay scalar is passed to the model, but stay endpoints are used internally.
6. **Raw unit consistency is not checked.** Labs/vitals are mapped by labels and plausibility bounds rather than explicit unit validation. FiO2 fractions are converted to percentages, but sparse temperature coverage may also reflect Celsius-only selection. Check original units and temperature Fahrenheit items before interpreting sparsity.
7. **Cache provenance is weak.** Cache filenames do not encode cohort, extraction-code version or time-window parameters. Rebuild in a new cache directory after scientific changes. This audit uses a separate local directory.
8. **Reproducibility is incomplete.** Threaded jobs reset shared Python/NumPy/PyTorch RNGs. Indexed CUDA names disable AMP because of `device == "cuda"`. The local copy defaults to serial execution; original module behavior is preserved and documented.
9. **The outcome is concurrent progression classification.** Current-X-ray measurements are allowed for this definition, but not for forecasting from the prior visit. Labels are report-derived candidates and have not been clinically adjudicated here. RAD-DINO's potential pretraining overlap is already disclosed by the original notebook and remains unresolved for these patients.

## Interpretation and next validation

The feature groups are plausible exploratory inputs, not a clinically or statistically validated selected panel. After correcting extraction, compare image-only, physiological features, care/treatment features, full fusion and shuffled controls on identical patient splits and seeds. Select settings using validation only. Report paired performance differences with patient-clustered bootstrap intervals; repeated pairs from one patient are not independent. Multiple seeds assess training variability but do not replace patient-sampling uncertainty. Because test results have already been inspected, additional tuning should be described as exploratory and confirmed on an untouched evaluation set where possible.

## Fallacy scan

Coverage: 11/11 considered; none of the unavailable analyses is treated as passed.

| Check | Assessment |
|---|---|
| Simpson's paradox | Subgroup reversal cannot be assessed from aggregate tables; disease-specific effects vary |
| Ecological fallacy | Mean disease F1 does not imply benefit for individual patients or every disease |
| Berkson/selection bias | Repeated-X-ray cohort and available report-derived labels limit generalizability |
| Collider bias | ICU/care-selection and treatment proxies could create associations; causal structure unverified |
| Base-rate neglect | Severe imbalance; cardiomegaly accuracy masks poor minority-class performance |
| Regression to the mean | Relevant to clinical change/treatment interpretations; no causal treatment claim supported |
| Survivorship bias | Requiring a later examination selects patients with repeated imaging; attrition not audited |
| Look-elsewhere effect | Multiple encoders and models; no confirmatory significance tests or multiplicity claim |
| Forking paths | Hardcoded encoder differs from screening winner; exploratory choices need recording |
| Correlation versus causation | Added predictive inputs do not establish treatment effects or biological mechanisms |
| Reverse causality | Drugs may respond to disease changes already recognized by clinicians |

## Reproducibility verdict

Historical training: CANNOT_VERIFY from saved output alone. Local paths and feature-selection reconstruction are checked separately. No p-values or confidence intervals can be recovered from rounded aggregate F1 tables, and no statistical significance is claimed.

Local checks completed: notebook code parses; setup/imports and pair loaders execute in the existing `longitudinal_progression` environment; every pair has both image files; all configured required paths and the diagnosis checkpoint exist. The original notebook SHA256 remains `4ffcb3778b540c0b1b3fe4ac378e8c4d64359cf28edb1b830fdcfa1e6423c83c`. No model training was executed, and the original scientific extraction modules were not changed.

The actual fusion loaders reproduce the audited 64 input names exactly, and all train/validation/test tabular arrays are finite. All 24 reconstructed feature-coverage rows match the original notebook's displayed current/prior/delta coverage to its three-decimal precision. This supports consistency with the historical extraction, but does not substitute for missing historical run settings.
