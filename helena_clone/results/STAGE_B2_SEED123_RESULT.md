# Stage B2 — Seed 123 confirmation — Final result

Status: COMPLETED. No Stage-B class-conditional policy was confirmed on seed123.

Frozen setup:
- m = 16,384
- seed = 123
- Reconstruction Clone QN-v2: fixed-step L-BFGS, memory=5, scale=.25, no line search
- WCE/SoftF1/Adaptive = 18/9/6
- meta builds per-class P/R and tau_k
- calibration evaluates the three B1-selected policies
- test remains untouched

Seed123 RSP fingerprint:
- mu1 = 6339.458814259362
- mur = 0.010239820133091522
- qmin = 0.0013315379619598389

Base-stage diagnostics:
- WCE meta-NLL = 3.0056447982788086
- SoftF1 meta-NLL = 2.8677737712860107

## Paired confirmation results

| policy | tau0 | rho | lambda | dAcc | dMacroF1 | dBalAcc | dTail20 | dNLL | dBrier | dECE | confirmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| primary | .694 | .15 | 50 | -0.001022495 | -0.000638232 | -0.000883039 | 0 | +0.002159834 | +0.000417859 | +0.000211910 | NO |
| macro | .815 | .15 | 20 | -0.001635992 | -0.002127109 | -0.002257495 | 0 | +0.004373312 | +0.000676632 | -0.000174693 | NO |
| high_tau | .935 | .15 | 20 | -0.001022495 | -0.000317495 | -0.000634662 | 0 | +0.001865387 | +0.000276435 | -0.000665665 | NO |

All three preserve Tail20 exactly on this calibration split, but all three reduce Accuracy, Macro-F1 and Balanced Accuracy relative to their matched scalar controls. Therefore n_confirmed_relaxed=0 and n_confirmed_strict=0.

## Cross-seed interpretation

Seed42 produced positive paired gains for these policies; seed123 reverses the ranking signal. The class-conditional tau_k effect therefore does not replicate across the two confirmation seeds.

Two-seed paired mean deltas (seed42 + seed123)/2:

| policy | mean dAcc | mean dMacroF1 | mean dBalAcc | mean dTail20 | mean dNLL |
|---|---:|---:|---:|---:|---:|
| primary (.694,.15,50) | -0.000306748 | +0.000362288 | -0.000198623 | 0 | +0.001785874 |
| macro (.815,.15,20) | -0.000715746 | +0.000019262 | -0.000564243 | 0 | +0.004556298 |
| high_tau (.935,.15,20) | -0.000408998 | +0.000652168 | +0.000266743 | 0 | +0.001834631 |

No class-conditional policy gives a clean two-seed improvement in the primary Accuracy + Macro-F1 objective. The high_tau policy retains positive mean Macro-F1/BalAcc but loses Accuracy and fails seed-level replication, so it is not robust enough to freeze as Stage-B winner.

## Scalar controls across seed42 and seed123

Two-seed descriptive means:

| scalar tau | Accuracy | Macro-F1 | BalancedAcc | Tail20 | NLL | ECE |
|---:|---:|---:|---:|---:|---:|---:|
| .694 | 0.352352 | 0.192527 | 0.192401 | 0.103344 | 2.832839 | 0.060850 |
| .815 | 0.350307 | 0.192136 | 0.191995 | 0.100364 | 2.840373 | 0.061679 |
| .935 | 0.348875 | 0.192198 | 0.191516 | 0.099823 | 2.848920 | 0.062541 |

Within Reconstruction Clone v2, scalar tau=.694 is the strongest two-seed scalar control: it has the highest mean Accuracy, Macro-F1, Balanced Accuracy, Tail20 and the best NLL/ECE among the three tested scalar values.

## B2 conclusion

B2 result: CLASS-CONDITIONAL tau_k NOT CONFIRMED.

Recommended B3 freeze candidate:
- retain the simpler scalar policy
- tau = .694
- m = 16,384 remains frozen from Stage A

This is a clone-v2 decision, not a reinterpretation of the old historical tau=.935 high-capacity runs. The whole Helena system can later be rerun from scratch under the clone-v2 protocol.
