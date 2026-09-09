# F2 — Margin-Residual Intervention Test — FINAL

Status: **CLOSED / POSITIVE ON THE PREREGISTERED OOF-CAL GATE**.

System adoption status: **EXPERIMENTAL; NOT ADOPTED into the final coherent system yet** because TEST shows a small but statistically clear NLL/Brier penalty while ranking gains are directionally positive but bootstrap intervals include zero.

Valid workflow run: `34389027159`.
Artifact: `10118906399`, SHA256 `86c97d114b031135f5ffb57c87f4d2d21951a9f0dbee5ae56353844ddb94f76a`.

## Provenance / invalid first attempt

The first F2 workflow run (`34388688304`) is **invalid for scientific closure**. It used class-level CV seed `20260911`, whereas frozen F1.5 used `20260910`. With only 100 classes this changed the residualization folds and broke reproduction of the identified F1.5 component.

Run 2 changed only this compatibility seed to the frozen F1.5 seed. Outer sample cross-fitting, feature families, gate, lambda grid, decision thresholds, canonical logits, global G2 and TEST policy were unchanged.

The corrected run reproduces the F1.5 full-CAL unique-margin residual CV results:

- soft V1 residual CV R2 = `0.07658085` (F1.5: `0.076581`);
- hard V1 residual CV R2 = `0.08371192` (F1.5: `0.083712`).

Therefore run 2 is the valid F2 experiment.

## Frozen intervention

For each outer CAL fold, the unique-margin V1 component is estimated from the 4/5 fit portion only, separately for soft-leakage and hard-confusion residual matrices, then combined after sign alignment.

The consensus class direction is centered and standardized:

`mean(d)=0`, `sd(d)=1`.

For sample i:

`g_i = top1(z_i) - top2(z_i)`

`a_i = exp(-g_i / median(g_fit))`

and

`z_i(lambda) = z_i + lambda * a_i * d`.

Frozen global G2 is recomputed after intervention.

Lambda grid:

`{-0.30,-0.20,-0.15,-0.10,-0.05,-0.025,0,+0.025,+0.05,+0.10,+0.15,+0.20,+0.30}`.

## Direction stability diagnostics

The two independently estimated unique-margin directions are strongly aligned inside every outer fold:

| fold | soft residual CV R2 | hard residual CV R2 | soft-hard aligned dot |
|---:|---:|---:|---:|
| 1 | +0.00574 | +0.02974 | 0.96356 |
| 2 | +0.01607 | +0.06958 | 0.94026 |
| 3 | -0.02087 | +0.05046 | 0.91268 |
| 4 | -0.05554 | -0.01913 | 0.87994 |
| 5 | +0.02292 | +0.01363 | 0.93881 |
| full CAL | +0.07658 | +0.08371 | 0.96090 |

Thus the soft/hard **directional consensus is strong**, even though the class-level residual regression itself is weak in some 4/5 sample perturbations. This is consistent with F1.5 identifying a small unique component rather than the whole V1 axis.

## OOF-CAL decision

Canonical control:

| metric | control |
|---|---:|
| Accuracy | 0.356442 |
| Macro-F1 | 0.193724 |
| BalancedAcc | 0.193783 |
| Tail20-F1 | 0.105896 |
| NLL | 2.740368 |
| Brier | 0.782323 |
| ECE | 0.033165 |

Two nonzero lambdas pass the preregistered gate: `-0.05` and `-0.025`.

Lexicographic winner:

`lambda* = -0.05`.

At `lambda=-0.05`:

| metric | value | delta vs control |
|---|---:|---:|
| Accuracy | 0.357260 | **+0.000818** |
| Macro-F1 | 0.194901 | **+0.001176** |
| BalancedAcc | 0.194121 | **+0.000338** |
| Tail20-F1 | 0.112641 | **+0.006745** |
| NLL | 2.741716 | +0.001348 |
| Brier | 0.782555 | +0.000232 |
| ECE | 0.033778 | +0.000613 |

All hard safeguards pass.

The nearby `lambda=-0.025` also passes, with smaller ranking gains and smaller probability loss. `lambda=-0.10` gives larger Macro-F1 and Tail20 gains, but is correctly rejected because `Delta NLL=+0.003581` exceeds the frozen `+0.003` tolerance. This gives a coherent local trade-off curve rather than a single isolated winning point.

## TEST audit after CAL freeze

TEST control:

| metric | control | F2 lambda=-0.05 | delta |
|---|---:|---:|---:|
| Accuracy | 0.361861 | 0.362168 | **+0.000307** |
| Macro-F1 | 0.205100 | 0.206244 | **+0.001144** |
| BalancedAcc | 0.201927 | 0.203481 | **+0.001553** |
| Tail20-F1 | 0.116798 | 0.117355 | **+0.000557** |
| NLL | 2.762657 | 2.764019 | +0.001362 |
| Brier | 0.777945 | 0.778177 | +0.000232 |
| ECE | 0.031378 | 0.032313 | +0.000935 |

The **sign of all four ranking/class-balance changes replicates on TEST**, but the effect is small.

### Bootstrap 95% intervals

Probability metrics, intervention minus control:

- NLL: `[+0.000756, +0.001968]`;
- Brier: `[+0.000128, +0.000336]`.

These intervals exclude zero: the probability-quality penalty is small but statistically clear under the paired bootstrap.

Ranking/class-balance deltas:

- Accuracy: `[-0.001332, +0.002045]`;
- Macro-F1: `[-0.000674, +0.003156]`;
- BalancedAcc: `[-0.000324, +0.003565]`;
- Tail20-F1: `[-0.005441, +0.006901]`.

All include zero. Therefore TEST provides **directional replication, not strong statistical confirmation of ranking gain**.

## Structural interpretation of the direction

Post-hoc descriptive audit of the frozen full-CAL consensus direction:

- Head20 mean d = `+0.987`;
- Mid60 mean d = `-0.139`;
- Tail20 mean d = `-0.570`;
- Spearman correlation between d and log training support is about `+0.544`.

Because `lambda* < 0`, uncertain samples receive, on average, a **head-logit suppression / lower-prevalence lift**. This explains why Tail20 and balanced metrics improve.

This must not be reinterpreted as prevalence having become the uniquely identified mechanism: F1.5 explicitly found prevalence non-unique after adjustment. The F2 direction is a cross-fitted unique-margin prediction that still inherits correlated class structure.

## Scientific conclusion

F2 answers the narrow actionability question positively:

`identified V1 unique-margin component -> small, reproducible decision-geometry leverage`.

The intervention is not merely descriptive: the CAL-cross-fitted direction changes class decisions in the expected direction and the TEST point estimates replicate the ranking gains.

However the same intervention moves probability estimates away from the current global-G2 optimum. Hence F2 exposes a real **ranking/calibration frontier**:

- negative lambda improves class balance / Macro-F1 over a local range;
- the cost is progressively worse NLL/Brier;
- the preregistered safeguard selects a conservative point before the probability penalty becomes too large.

Therefore:

1. **F2 is CLOSED / POSITIVE** under its preregistered actionability criterion.
2. `lambda*=-0.05` is retained as the F2 experimental intervention.
3. The final production/coherent system remains `coherent S5 -> global G2`; F2 is **not yet adopted** because its TEST ranking CI includes zero while its probability penalty is statistically clear.
4. Do not reopen lambda/gate under F2. Any new intervention form is a new family.

## Highest-value next checkpoint

A natural next test is **F2.5 — Post-Intervention Calibration Recovery**: freeze the F2 direction and `lambda=-0.05`, then refit only the two global-G2 calibration parameters by nested CAL cross-validation. The question is whether the F2 ranking gain can be retained while removing the NLL/Brier penalty. This would test separation between decision geometry and probability calibration without reopening the intervention itself.
