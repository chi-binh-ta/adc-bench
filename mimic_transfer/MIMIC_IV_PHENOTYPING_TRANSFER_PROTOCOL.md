# MIMIC-IV Phenotyping External Transfer — Frozen Protocol

Status: DEMO-SMOKE protocol. This is a new external-transfer family, not F2.12 and does not reopen Helena Stage F.

## Scientific question
Can an early ICU tabular predictor identify rare organ-failure-related phenotypes in the standard MIMIC phenotyping label space, and is performance maintained on low-prevalence labels?

## Critical task correction
The standard MIMIC phenotyping benchmark is multi-label, not exclusive multiclass. An ICU stay may carry several phenotype labels simultaneously. Therefore Helena softmax/G2 is not ported unchanged.

## Public demo scope
The first run uses MIMIC-IV Clinical Database Demo v2.2 (100 patients) only as a pipeline/support smoke test. It cannot support clinical/generalization claims. Full MIMIC-IV requires credentialed PhysioNet access and a signed DUA/CITI training.

## Unit and observation window
- unit: ICU stay (`stay_id`)
- features: structured numeric ICU chart/lab observations from the first 48 hours after ICU `intime`
- labels: hospitalization diagnoses mapped to the HCUP CCS phenotype groups used by the MIMIC benchmark
- demo smoke mapping: ICD-9 diagnoses only, because the canonical benchmark definitions bundled by YerevaNN/vincenzorusso are ICD-9 HCUP CCS. ICD-10-only admissions are not silently remapped in the smoke run.

## Target label space
Use the 25 benchmark HCUP phenotype groups. The rare-organ-failure audit emphasizes:
1. Respiratory failure; insufficiency; arrest (adult)
2. Acute and unspecified renal failure
3. Shock
4. Septicemia (except in labor)
5. Fluid and electrolyte disorders
6. Other liver diseases

These are not a single MODS label. A secondary composite is allowed only as a descriptive endpoint:
`multi_organ_proxy = sum(selected organ-failure labels) >= 2`.
It must never replace per-label results.

## Leakage controls
- no diagnosis codes, diagnosis text, procedures, discharge disposition, discharge time, or post-48h measurements as model features
- feature vocabulary is chosen only from measurement frequency, not target association
- patient-level split: all stays for one `subject_id` stay in one fold
- no TEST/demo evaluation is used to tune a threshold

## Features
For `chartevents` and `labevents` within first 48h:
- retain numeric `valuenum`
- choose the most frequent item IDs (fixed top-N by occurrence; unsupervised)
- aggregate per stay: count, mean, std, min, max, last
Add age at admission/ICU and ICU length metadata only when available before/at the 48h boundary; do not use final LOS as a predictor.

## Baselines
1. prevalence-only constant predictor
2. L2 logistic regression, one-vs-rest per label
3. class-balanced L2 logistic regression as the rare-label baseline
No neural-network tuning is authorized on the 100-patient demo.

## Evaluation
Per label, only if support is sufficient:
- positives >= 5 and negatives >= 5 overall
- every reported CV fold used for AUC/AUPRC must contain both classes
Metrics:
- prevalence
- AUROC
- AUPRC and AUPRC/prevalence lift
- Brier
- sensitivity at a threshold chosen only inside training folds when feasible
Macro summaries are computed only across evaluable labels, with the denominator reported explicitly.
Rare-organ-failure labels are always shown individually.

## Cross-validation
GroupKFold/StratifiedGroupKFold where feasible, grouped by `subject_id`. If support prevents valid grouped CV, downgrade that label to `INSUFFICIENT_SUPPORT` rather than changing the protocol post hoc.

## Demo verdicts
- `DEMO_RARE_LABEL_EVALUABLE`: >=3 of the six organ-failure labels meet support and valid CV requirements.
- `DEMO_SMOKE_ONLY`: pipeline succeeds but fewer than 3 target labels are evaluable.
- `DEMO_PIPELINE_FAIL`: extraction/label/feature invariants fail.

No demo verdict may be interpreted as clinical predictive performance.

## Full-run gate (future, credentialed data only)
A full MIMIC-IV run must use a fresh patient-grouped train/CAL/TEST or nested CV design. Primary tail metrics: macro-AUPRC over the six prespecified organ-failure labels and per-label AUPRC lift. Calibration: macro Brier and per-label reliability. Any new calibration method must be separately preregistered for independent Bernoulli outputs; Helena multiclass G2 is not reused by notation only.
