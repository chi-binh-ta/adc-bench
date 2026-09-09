# Stage E — Cluster-pair G2 — FINAL CLOSURE

Status: **CLOSED**.

System-level winner: **global G2**.

Cluster-pair interpretation: **probability-positive, ranking-negative; NOT ADOPTED as the final system calibrator**.

## Frozen inherited state
- m = 16,384
- scalar tau = .694
- no Stage-C margin
- Stage-D seeds = {42,123,456,789,2026}
- Stage-E structural input = coherent centered-logit S=5 ensemble
- Stage-D benchmark remains soft-vote S=5

## Source-derived geometry
For centered logits z:

    L_ij = z_i - z_j
    u_ij = 1 - 4 sigmoid(L_ij)(1-sigmoid(L_ij))

Global G2:

    R_G,ij = (alpha0 + alpha1 u_ij) L_ij
    t = (1/K) A^T R_G
    z_cal = z - t

Cluster-pair G2 uses six symmetric unordered class-prevalence pair types:

    HH, HM, HT, MM, MT, TT

with

    g_ab(u) = alpha0_ab + alpha1_ab u,

for 12 total parameters.

The source specified Head/Mid/Tail by training prevalence but did not define the cut points. Reconstruction Clone v2 therefore froze Head20/Mid60/Tail20 to align with the existing Tail20 metric.

## Sanity checks
Stage-D coherent S=5 CAL replay was exact:
- NLL gap = 0
- Accuracy gap = 0
- Macro-F1 gap = 0
- Tail20 gap = 0

Complete-graph projection identity:
- max |sum_g t0_g - z_centered| = 1.1805e-05
- max potential gauge error = 3.0994e-05

Both pass.

## 5-fold CAL CV decision

| model | NLL | Brier | ECE | Accuracy | Macro-F1 | BalAcc | Tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| global G2 | 2.743349 | 0.782630 | 0.033662 | 0.356033 | 0.192840 | 0.192797 | 0.105896 |
| cluster-pair G2 | **2.727951** | **0.780894** | **0.022739** | **0.358487** | 0.190448 | 0.189972 | 0.100049 |

Cluster minus global:
- dNLL = -0.015398
- dBrier = -0.001736
- dECE = -0.010923
- dAccuracy = +0.002454
- dMacro-F1 = -0.002392
- dBalancedAcc = -0.002825
- dTail20 = -0.005847

The pre-frozen Stage-E gate required NLL and Brier improvement while allowing at most -0.001 degradation in Accuracy and Macro-F1. Cluster-pair fails because Macro-F1 falls by -0.002392.

Therefore the decision was frozen before TEST as:

    winner = global G2

## Full-CAL coefficients

Global:
- alpha0 = -0.903769
- alpha1 = +0.765460

Cluster-pair:

| pair | alpha0 | alpha1 |
|---|---:|---:|
| HH | -3.024686 | +7.256280 |
| HM | +0.834811 | -0.982242 |
| HT | -1.055933 | -2.288499 |
| MM | -1.735255 | +1.670858 |
| MT | +0.596717 | -0.901126 |
| TT | -9.968361 | +13.222709 |

The very large TT and HH coefficients are a warning that the unregularized 12-parameter calibrator is using strong cluster-specific deformations.

## TEST after decision freeze

| model | NLL | Brier | ECE | Accuracy | Macro-F1 | BalAcc | Tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent S=5 base | 2.804907 | 0.783164 | 0.060577 | 0.362168 | 0.205929 | 0.202301 | 0.116437 |
| **global G2** | **2.764401** | **0.778145** | **0.032224** | **0.362168** | **0.205929** | **0.202301** | **0.116437** |
| cluster-pair G2 | 2.749855 | 0.777691 | **0.016659** | 0.360123 | 0.197262 | 0.193802 | 0.101945 |

Global G2 versus coherent base:
- dNLL = -0.040506
- dBrier = -0.005019
- dECE = -0.028353
- dAccuracy = 0
- dMacro-F1 = 0
- dBalAcc = 0
- dTail20 = 0

Paired bootstrap TEST 95% CI:
- global minus base NLL: [-0.045981, -0.033476]
- global minus base Brier: [-0.005920, -0.004098]

Thus the global G2 calibration gain is clean and ranking-preserving.

Cluster-pair versus global on TEST:
- dNLL = -0.014546
- dBrier = -0.000454
- dECE = -0.015565
- dAccuracy = -0.002045
- dMacro-F1 = -0.008667
- dBalAcc = -0.008499
- dTail20 = -0.014492

Paired bootstrap TEST 95% CI:
- cluster minus global NLL: [-0.020523, -0.010821]
- cluster minus global Brier: [-0.001214, +0.000347]

Interpretation:
- cluster heterogeneity contains real probability-calibration signal: the NLL gain over global G2 is robust;
- the extra Brier gain is not clearly separated from zero;
- the 12-parameter unregularized cluster deformation changes class ordering enough to damage Macro-F1, Balanced Accuracy and Tail20 materially.

## Final Stage-E decision

1. **Global G2 is adopted as the final coherent calibration layer.**
2. **Cluster-pair G2 is not adopted in the final system.**
3. Cluster-pair G2 is retained as a research finding: prevalence-conditioned calibration heterogeneity is real, but the current unconstrained 12-parameter form is too aggressive for decision geometry.
4. If this branch is revisited, the next scientifically justified experiment is shrinkage/hierarchical cluster-pair G2 toward the global coefficients, not a larger unrestricted calibrator.

## Final frozen models after A–E

Pure benchmark / ranking champion:

    soft-vote S=5

    TEST Accuracy = 0.362270
    TEST Macro-F1 = 0.206060

Coherent calibrated system:

    coherent-logit S=5 -> global G2

    TEST NLL = 2.764401
    TEST Brier = 0.778145
    TEST ECE = 0.032224
    TEST Accuracy = 0.362168
    TEST Macro-F1 = 0.205929

Probability-only experimental champion (not adopted):

    coherent-logit S=5 -> cluster-pair G2

    TEST NLL = 2.749855
    TEST ECE = 0.016659

Provenance note: TEST was already opened at the end of Stage D. Stage E did not use Stage-D TEST metrics for fitting or model selection; its global-vs-cluster decision was frozen from CAL cross-validation before the Stage-E TEST evaluation.