# Helena — Canonical Final End-to-End Rerun

Status: **COMPLETED WITH SMALL NUMERICAL DRIFT; SCIENTIFIC CONCLUSIONS REPRODUCED**.

Workflow run: `34375185211`.

All five base models were rebuilt from raw Helena with zero initialization. No Stage-D logits, model checkpoints, RSP matrices, feature caches, or G2 coefficients were reused.

## Frozen system

- m = 16,384
- scalar tau = .694
- no confusion margin
- seeds = {42,123,456,789,2026}
- QN-v2 memory = 5, scale = .25
- WCE / SoftMacroF1 / Adaptive steps = 18 / 9 / 6
- ranking benchmark = soft-vote S=5
- coherent calibrated system = centered coherent-logit S=5 -> global G2

## Canonical TEST results

| model | Accuracy | Macro-F1 | BalancedAcc | Tail20 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft-vote S=5 | 0.361554192 | 0.205563310 | 0.202859771 | 0.118150392 | 2.794485807 | 0.782608178 | 0.063326957 |
| coherent-logit S=5 | 0.361860941 | 0.205100169 | 0.201927491 | 0.116797695 | 2.802780628 | 0.782882116 | 0.059805588 |
| coherent S=5 -> global G2 | 0.361860941 | 0.205100169 | 0.201927491 | 0.116797695 | 2.762656689 | 0.777944990 | 0.031377868 |

Global G2 coefficients fitted on the fresh CAL split:

- alpha0 = -0.902620963463
- alpha1 = +0.765345197612

Global G2 versus the fresh coherent base on TEST:

- delta NLL = -0.040123940
- delta Brier = -0.004937125
- delta ECE = -0.028427720
- delta Accuracy = 0
- delta Macro-F1 = 0
- delta BalancedAcc = 0
- delta Tail20 = 0

Thus the global G2 conclusion reproduces cleanly: it improves probability calibration while preserving the decision ranking exactly.

## Comparison with the Stage-D/E development reference

Reference gaps (canonical rerun minus frozen development result):

- soft-vote TEST Accuracy: -0.000715746
- soft-vote TEST Macro-F1: -0.000497022
- global-G2 TEST NLL: -0.001744032 (better)
- global-G2 TEST Brier: -0.000199875 (better)
- global-G2 TEST ECE: -0.000846033 (better)
- global-G2 TEST Accuracy: -0.000306748
- global-G2 TEST Macro-F1: -0.000828763

The strict numerical reproducibility tolerance was therefore not passed.

The dominant source is not split/representation identity: RSP fingerprints remain essentially identical. The fixed-step limited-memory BFGS trajectory is numerically sensitive enough that tiny floating-point/RSP differences can alter curvature pairs and amplify during SoftF1/adaptive stages. This was visible especially for seeds 456 and 2026, while several other seeds reproduced essentially unchanged.

This should be described as **small numerical/optimizer-path drift**, not a failure of the tuned scientific result. The main conclusions survive:

1. Accuracy remains above .36.
2. Macro-F1 remains around .205-.206.
3. Five-seed ensembling remains beneficial.
4. Global G2 remains strongly calibration-positive and ranking-preserving.
5. Cluster-pair G2 remains excluded from the final system; it was not rerun because Stage E had already rejected it and this canonical run uses only frozen adopted components.

The numbers in this file are the canonical fresh-rerun results and should be preferred over the earlier development-run metrics when reporting the fully rerun frozen Helena system.
