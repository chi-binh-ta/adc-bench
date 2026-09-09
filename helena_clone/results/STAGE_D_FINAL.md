# Stage D — Multi-seed Ensemble — FINAL CLOSURE

Status: **CLOSED — POSITIVE**.

## Frozen setup
- m = 16,384
- scalar tau = .694
- QN-v2 fixed-step L-BFGS, memory=5, scale=.25
- no Stage-C margin
- seeds fixed before results: {42, 123, 456, 789, 2026}
- final ensemble size fixed before results: S=5
- S=2 and S=3 are diagnostics only
- two source-derived ensemble rules: soft voting and coherent centered-logit averaging
- CAL used to freeze interpretation; TEST opened only after the CAL decision was written
- G2 is deferred to Stage E

## Five single models on CAL

Mean single-model metrics:
- Accuracy = 0.352433538
- Macro-F1 = 0.189693309
- BalancedAcc = 0.190540978
- Tail20-F1 = 0.093116855
- NLL = 2.835132313
- Brier = 0.793770352
- ECE = 0.061073537

Best individual values across the five seeds:
- best Accuracy = 0.354805726
- best Macro-F1 = 0.194633895
- best Tail20-F1 = 0.101765979
- best NLL = 2.821021557
- best Brier = 0.791761103

No seed was removed, including the weaker seed2026.

## CAL saturation curve

| mode | S | Accuracy | MacroF1 | Tail20 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft_vote | 2 | 0.355010 | 0.194255 | 0.101023 | 2.804308 | 0.789440 | 0.064612 |
| soft_vote | 3 | 0.356646 | 0.193671 | 0.101862 | 2.794271 | 0.788261 | 0.066159 |
| soft_vote | 5 | **0.357464** | **0.194496** | **0.105240** | **2.791106** | **0.788020** | 0.064298 |
| coherent_logit | 2 | 0.356237 | 0.193373 | 0.099214 | 2.804869 | 0.789651 | 0.064196 |
| coherent_logit | 3 | 0.355828 | 0.190617 | 0.100667 | 2.793710 | 0.788264 | 0.062472 |
| coherent_logit | 5 | **0.356033** | **0.192840** | **0.105896** | **2.791123** | **0.788058** | **0.062407** |

Both S=5 variants pass the pre-frozen Stage-D utility gate.

## S=5 soft voting versus single-model baselines on CAL

Versus the mean of the five single models:
- delta Accuracy = +0.005030675
- delta Macro-F1 = +0.004803065
- delta BalancedAcc = +0.005820646
- delta Tail20-F1 = +0.012123288
- delta NLL = -0.044026327
- delta Brier = -0.005750223
- delta ECE = +0.003224426 (worse)

Versus the best individual value metric-by-metric:
- delta Accuracy = +0.002658487
- delta NLL = -0.029915571
- delta Brier = -0.003740974
- delta Tail20-F1 = +0.003474164
- delta Macro-F1 = -0.000137520

Thus the S=5 soft ensemble beats even the best individual seed on Accuracy, NLL, Brier and Tail20, while essentially matching the best individual Macro-F1.

## CAL-frozen benchmark choice

The benchmark ensemble was frozen as:

    soft_vote, S=5, seeds={42,123,456,789,2026}

Reason: relative to coherent-logit S=5, soft voting has higher Accuracy, higher Macro-F1, higher BalancedAcc, slightly lower NLL and slightly lower Brier. Coherent logits have better ECE and slightly higher Tail20, but the pre-frozen rule selects soft voting as the pure benchmark ensemble.

Coherent-logit S=5 is retained as the structural representation for Stage E because G2 operates on the logit/edge field.

## TEST after CAL freeze

| mode | Accuracy | MacroF1 | BalancedAcc | Tail20 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **soft_vote S=5** | **0.362269939** | **0.206060332** | **0.203439344** | **0.118278597** | **2.797028065** | **0.783001257** | 0.064600117 |
| coherent_logit S=5 | 0.362167689 | 0.205928933 | 0.202301239 | 0.116436860 | 2.804907084 | 0.783163818 | **0.060576618** |

The test result was not used to change the seed set, S, or benchmark choice.

Important benchmark milestone under Reconstruction Clone v2:

    soft-vote S=5 Accuracy = 0.36227 > 0.36.

This crosses the earlier Accuracy > .36 target, but it should not be numerically conflated with historical non-clone runs.

## Stage-D decision

- Stage D: CLOSED — POSITIVE.
- Benchmark prediction model: S=5 soft voting.
- Structural model handed to Stage E: S=5 coherent centered-logit ensemble.
- Downstream fixed base remains m=16,384 and scalar tau=.694.
- Stage E may alter calibration geometry only; it must not reopen A/B/C/D.
