# MIMIC-IV Demo Phenotyping External Transfer — Result

Status: **DEMO_RARE_LABEL_EVALUABLE (SMOKE-LEVEL ONLY)**.

Valid workflow run: `34500138439`.
Artifact: `10161543963`, SHA256 `914b96a75f83c26499ecf96a155680c65f22265c425bec31b978a21e03b8826f`.

Branch: `mimic-iv-phenotyping-transfer-20260910`.

## Scope and interpretation boundary

This run used the openly available MIMIC-IV Clinical Database Demo v2.2, not the credentialed full MIMIC-IV dataset. The public demo contains 100 subjects in the source dataset; after restricting the smoke phenotype mapping to ICU admissions carrying ICD-9 diagnoses, the analysis contained 86 ICU stays from 67 subjects.

This is a support/pipeline smoke test. It does **not** establish clinical generalization or full-MIMIC performance.

The standard phenotyping task is multi-label rather than exclusive multiclass. Our run is also a custom *early* variant: first-48h structured ICU measurements are used to predict HCUP phenotype labels derived from hospitalization diagnosis codes. The classical phenotyping benchmark typically uses the entire ICU stay, so results are not directly comparable to published full-stay PHE leaderboards.

## Data and leakage controls

Predictors used only numeric values from the first 48 hours after ICU admission:
- `chartevents`: 69,062 retained rows;
- `labevents`: 10,018 retained rows.

Top 40 chart item IDs and top 40 lab item IDs were selected by measurement frequency only. For each selected item, count/mean/std/min/max/last were aggregated, producing 480 candidate features.

Diagnosis codes were used only to construct targets and never entered the predictor. Subject-level grouped cross-validation prevented the same subject from appearing in both train and validation folds.

Because `n=86` stays and `p=480` aggregated features, this is an extreme small-sample/high-dimensional regime. All metrics below should therefore be treated as exploratory smoke estimates.

## Prespecified organ-failure-related targets

All six targets passed the minimal smoke support rule and admitted valid 5-fold subject-grouped CV with the balanced L2 logistic baseline.

| Target | positives | prevalence | AUROC | AUPRC | AP/prevalence lift | Brier | prevalence-baseline Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Acute and unspecified renal failure | 25 | .291 | .746 | .579 | 1.99x | .231 | .206 |
| Fluid and electrolyte disorders | 36 | .419 | .752 | .661 | 1.58x | .246 | .244 |
| Other liver diseases | 18 | .209 | .738 | .433 | 2.07x | .235 | .166 |
| Respiratory failure; insufficiency; arrest (adult) | 21 | .244 | .845 | .642 | 2.63x | .154 | .185 |
| Septicemia (except in labor) | 17 | .198 | .749 | .386 | 1.95x | .253 | .159 |
| Shock | 8 | .093 | .889 | .521 | 5.60x | .217 | .085 |

Six-target macro, balanced L2 logistic:
- AUROC: `0.78636`;
- AUPRC: `0.53702`;
- AUPRC/prevalence lift: `2.637x`;
- Brier: `0.22278`;
- mean OOF prevalence-baseline Brier: `0.17425`.

Unweighted L2 logistic was similar but slightly weaker on mean ranking metrics:
- macro AUROC `0.78308`;
- macro AUPRC `0.52955`;
- macro AUPRC lift `2.611x`;
- macro Brier `0.22412`.

## Main smoke finding

The early structured-EHR features contain clear ranking signal for these organ-failure-related phenotype labels in the public demo: every prespecified label has AUROC above 0.73, and AUPRC is between roughly 1.58x and 5.60x its prevalence.

However probability quality is poor in this high-dimensional tiny-sample run. Balanced logistic improves Brier over the fold-specific prevalence predictor for only one of six targets (respiratory failure). Shock is the most extreme example: AUROC `0.889` and AUPRC `0.521` with only 8 positives, while Brier `0.217` is much worse than the prevalence baseline `0.085`.

Accordingly, the defensible result is **ranking signal exists at smoke level; calibrated rare-event probability prediction is not demonstrated**.

## Multi-organ proxy

A descriptive proxy was also evaluated:

`multi_organ_proxy = at least 2 of the six prespecified HCUP labels`.

It had 36/86 positives (prevalence `0.419`) and balanced-logistic OOF:
- AUROC `0.7767`;
- AUPRC `0.7163`;
- AUPRC lift `1.711x`;
- Brier `0.2264` versus prevalence baseline `0.2436`.

Because prevalence is 41.9%, this endpoint is **not rare**, and because it is a billing-code co-occurrence proxy it must not be called a validated multiple-organ dysfunction syndrome (MODS) endpoint.

## Implication for a genuine rare-MODS study

A clinically meaningful rare-MODS endpoint should be defined prospectively from time-localized organ dysfunction rather than by simply combining discharge diagnosis groups. A natural future direction is SOFA-domain construction (respiratory, coagulation, liver, cardiovascular, CNS, renal) and a preregistered criterion for concurrent failure of multiple organ systems. That is a different label-generation study and should not be silently substituted for 25-PHE.

## Full-MIMIC requirement

The credentialed full MIMIC-IV data are required before any claim about rare organ-failure prediction can be made. The full study should use patient-grouped nested CV or a fresh train/CAL/TEST split, stronger regularization/model selection confined to training data, per-label AUPRC as the primary rare-label metric, and explicit calibration assessment. No Helena multiclass softmax/G2 mechanism should be ported directly because MIMIC PHE is a vector of Bernoulli labels rather than a probability simplex.
