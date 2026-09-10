# F2.5 — Post-Intervention Calibration Recovery — FINAL

Status: **CLOSED / NO_RECOVERY under the preregistered joint NLL+Brier gate**.

Important secondary finding: the constrained **dose-conditioned G2 (C3)** gives a reproducible NLL/ECE improvement while preserving all F2 ranking gains, but misses the Brier requirement by a small amount. Therefore C3 is scientifically interesting but is **not adopted** under F2.5.

Valid workflow run: `34481003356`.
Artifact: `10153583910`, SHA256 `46cd436c5dd3ab6b5566a8e77f7ecc593364b064bf8d867499382d58b8a4de88`.

## Frozen inherited intervention

F2 intervention is unchanged:

- `lambda=-0.05`;
- same unique-margin V1 soft/hard consensus direction construction;
- same class-level CV seed `20260910`;
- same outer CAL split seed `20260910`;
- same uncertainty gate `a_i=exp(-top2_gap/median_fit_gap)`.

No intervention parameter was reopened.

## Calibration candidates

- `C0`: canonical coherent S5 -> frozen global G2, no intervention.
- `C1`: F2 intervention -> old frozen global G2.
- `C2`: F2 intervention -> refit two global G2 coefficients.
- `C3`: F2 intervention -> dose-conditioned global G2 with four parameters

  `alpha_r(i)=alpha_r+beta_r q_i`,

  where `q_i` is the standardized frozen intervention gate.

C3 is a deliberately rank-1 hidden-state analogue inspired structurally by aspect-conditioned sentiment analysis: one observation may need a different mapping under a different contextual/state regime. No sentiment model or aspect labels are transferred to Helena.

## Reproducibility gate

C1 exactly replays the valid F2 OOF-CAL reference for NLL, Brier, Accuracy, Macro-F1, Balanced Accuracy and Tail20-F1: all replay gaps are exactly zero at reported precision.

## OOF-CAL results

| model | NLL | Brier | ECE | Accuracy | Macro-F1 | BalAcc | Tail20-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 canonical | 2.740368 | **0.782323** | 0.033165 | 0.356442 | 0.193724 | 0.193783 | 0.105896 |
| C1 F2 fixed G2 | 2.741716 | 0.782555 | 0.033778 | **0.357260** | **0.194901** | **0.194121** | **0.112641** |
| C2 refit global G2 | 2.742078 | 0.782612 | 0.034025 | **0.357260** | **0.194901** | **0.194121** | **0.112641** |
| C3 dose-conditioned G2 | **2.739749** | 0.782623 | **0.028647** | **0.357260** | **0.194901** | **0.194121** | **0.112641** |

### C2 conclusion

Naively refitting the two global G2 coefficients does not recover calibration. Relative to C1:

- Delta NLL = `+0.000362`;
- Delta Brier = `+0.000057`;
- Delta ECE = `+0.000247`.

Thus the probability penalty is not simply due to stale values of the two global coefficients.

### C3 conclusion

Dose conditioning changes the result qualitatively.

Relative to C0:

- Delta NLL = `-0.000619` — better than canonical;
- Delta ECE = `-0.004519` — substantially better;
- Delta Brier = `+0.000301` — worse;
- all F2 ranking gains are preserved exactly.

Relative to C1:

- Delta NLL = `-0.001967`;
- Delta ECE = `-0.005132`;
- Delta Brier = `+0.0000686`.

The preregistered gate required both `NLL(Candidate) < NLL(C1)` and `Brier(Candidate) <= Brier(C1)`. C3 fails only the Brier condition. Therefore the frozen decision is correctly:

`status = NO_RECOVERY`, `winner = NONE`.

This is a near miss on the Brier axis, not a negative result for state conditioning as a whole.

## TEST audit after CAL decision freeze

| model | NLL | Brier | ECE | Accuracy | Macro-F1 | BalAcc | Tail20-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 canonical | 2.762657 | **0.777945** | 0.031378 | 0.361861 | 0.205100 | 0.201927 | 0.116798 |
| C1 F2 fixed G2 | 2.764019 | 0.778177 | 0.032313 | **0.362168** | **0.206244** | **0.203481** | **0.117355** |
| C2 refit global G2 | 2.764138 | 0.778156 | 0.032303 | **0.362168** | **0.206244** | **0.203481** | **0.117355** |
| C3 dose-conditioned G2 | **2.762007** | 0.778272 | **0.022731** | **0.362168** | **0.206244** | **0.203481** | **0.117355** |

C3 reproduces the OOF qualitative pattern on TEST:

relative to C0:

- Delta NLL = `-0.000650`;
- Delta ECE = `-0.008647`;
- Delta Brier = `+0.000327`;
- Delta Accuracy = `+0.000307`;
- Delta Macro-F1 = `+0.001144`;
- Delta Balanced Accuracy = `+0.001553`;
- Delta Tail20-F1 = `+0.000557`.

Relative to C1, C3 improves TEST NLL by `-0.002012` and ECE by `-0.009582`, but worsens Brier by `+0.000095`.

Because TEST was not allowed to alter the CAL-frozen gate decision, F2.5 remains NO_RECOVERY.

## Hidden-state coefficient audit

Across the five OOF CAL fits, C3 parameters are stable enough to expose a specific pattern:

- mean alpha0 about `-0.8317`;
- mean alpha1 about `+0.6676`;
- mean beta0 about `-0.0040`;
- mean beta1 about `-0.0603`.

More importantly, `beta1` is negative in all five folds:

`{-0.0919,-0.0304,-0.1029,-0.0141,-0.0623}`.

Full-CAL C3 coefficients:

- alpha0 = `-0.830165`;
- alpha1 = `+0.665607`;
- beta0 = `-0.003133`;
- beta1 = `-0.061625`.

Thus the intervention state mainly modulates the **nonlinear uL component** of G2 rather than the base linear component. This is the strongest mechanistic clue from F2.5.

## Relation to the sentiment-analysis mechanism

The current sentiment repository implements aspect-level sentiment explicitly: a single review can produce different aspect-specific labels, including mixed states. The broader research formulation previously explored `h(x,a)` and a supplementary term `q(x,a)` beyond one global scalar representation.

The Helena C3 result is structurally analogous but narrower:

`global calibration + state-conditioned nonlinear correction`.

The fact that beta1 is consistently nonzero/negative and that NLL/ECE gains replicate OOF-CAL -> TEST is evidence that a single global calibration map misses intervention-state heterogeneity. It does not prove the state is causal or that a learned latent network is needed.

## Scientific closure

1. **F2.5 is CLOSED / NO_RECOVERY under the frozen joint NLL+Brier gate.**
2. C2 refit-global G2 is negative.
3. C3 dose-conditioned G2 is **NLL-positive / ECE-positive / Brier-negative**, while exactly preserving F2 ranking metrics.
4. C3 is not adopted because the Brier safeguard fails.
5. The hidden-state idea is not rejected: it receives a strong structured clue, especially through the stable negative beta1 on the nonlinear G2 basis.
6. Do not relax the F2.5 gate post hoc. Any attempt to reconcile NLL and Brier should be a new checkpoint with a new frozen objective/protocol.

Highest-value next checkpoint: **F2.6 — Proper-Score Geometry of the Hidden-State Calibrator**. Diagnose why the same C3 shift improves log score/NLL and ECE while worsening quadratic/Brier score. Decompose the change by confidence, true-class probability, prevalence group and intervention dose before introducing any new parameter.