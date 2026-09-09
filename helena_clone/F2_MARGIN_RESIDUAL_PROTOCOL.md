# F2 — Margin-Residual Intervention Test

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

Base system: canonical `coherent-logit S=5 -> global G2` used by F1/F1.5.

## Scientific question

F1.5 identified one robust conditional signal:

`V1 <- margin | all other five observable groups`

in both soft-leakage and hard-confusion residual matrices. F2 tests whether that identified **unique margin component** is actionable when used as a low-dimensional inference-time intervention.

This is not a retraining experiment and it does not reopen Stage C's generic pairwise margin-loss family.

## Leakage / provenance policy

- No Helena TEST labels may be used to construct the intervention direction, choose its sign, tune its strength, or choose the uncertainty gate.
- Intervention selection is performed by **5-fold stratified cross-fitting inside CAL**.
- For each fold, the class direction is estimated only from the 4/5 CAL fit subset and applied to the held-out 1/5 CAL subset.
- The final intervention direction is then refit on all CAL only after `lambda*` is frozen.
- TEST is opened only after that F2 decision.
- Important provenance caveat: Helena TEST has already been used by prior project stages, so F2 TEST is an internal replication audit, not a globally untouched holdout.

## Frozen class-level latent direction

Let `z` denote centered coherent S=5 logits before global G2.

Within each CAL fit fold:

1. Apply the frozen global G2 to obtain probabilities.
2. Build the F1 soft-leakage and hard-confusion residual matrices.
3. Extract their leading right singular factor `V1`.
4. Orient each V1 deterministically so that its Pearson correlation with class `margin_mean` is non-negative. This is only a sign convention.
5. Reuse exactly the six F1 feature families:
   - prevalence;
   - class similarity;
   - margin;
   - seed instability;
   - calibration residual;
   - representation geometry.
6. For margin as the target family, let the other five families be controls `C`.
7. Cross-fit, over the 100 classes, both

   `rV = V1 - Ehat[V1 | C]`

   and each margin feature

   `rX_M = X_M - Ehat[X_M | C]`.

8. Fit ridge `rX_M -> rV` with class-level 5-fold CV over

   `alpha in logspace(-3, 3, 13)`.

9. Use out-of-fold predictions as the unique-margin class component for each residual definition:

   `d_soft`, `d_hard`.

10. Align `d_hard` to `d_soft` by dot-product sign and average unit-norm directions:

   `d_raw = 0.5 * (d_soft/||d_soft|| + aligned(d_hard)/||d_hard||)`.

11. Center and scale across classes:

   `mean(d)=0`, `sd(d)=1`.

The resulting `d in R^K` is the frozen one-dimensional intervention direction for that fold.

## Frozen sample gate

For a sample with centered coherent logits `z_i`, define the top-two gap

`g_i = z_(1) - z_(2) >= 0`.

On each CAL fit fold, define

`s = median(g_fit) + 1e-8`.

The inference-time gate is

`a_i = exp(-g_i / s)`.

Hence ambiguous samples receive the largest intervention and high-margin samples are automatically suppressed. No gate parameter is tuned.

## Intervention

For candidate strength `lambda`,

`z_i(lambda) = z_i + lambda * a_i * d`.

The frozen **global G2 is then recomputed on `z_i(lambda)`** and probabilities are evaluated after G2. Thus F2 keeps the final structural calibration layer rather than bypassing it.

## Frozen lambda grid

`lambda in {-0.30,-0.20,-0.15,-0.10,-0.05,-0.025,0,+0.025,+0.05,+0.10,+0.15,+0.20,+0.30}`.

`lambda=0` is the exact canonical control replay.

## OOF-CAL selection

For every lambda, concatenate the five held-out CAL fold predictions and compute:

- Accuracy;
- Macro-F1;
- Balanced Accuracy;
- Tail20-F1;
- NLL;
- Brier;
- ECE.

Let all deltas be candidate minus `lambda=0`.

A nonzero candidate passes only if all conditions hold:

- `Delta MacroF1 > 0`;
- `Delta BalancedAcc > 0`;
- `Delta Tail20F1 > 0`;
- `Delta Accuracy >= -0.0005`;
- `Delta NLL <= +0.003`;
- `Delta Brier <= +0.0008`.

ECE is logged but is not a hard gate.

Among passing candidates, select lexicographically:

1. maximum Macro-F1;
2. maximum Tail20-F1;
3. maximum Balanced Accuracy;
4. minimum NLL;
5. minimum `|lambda|`.

If no nonzero lambda passes, freeze

`lambda* = 0`

and close F2 as negative.

## Final TEST audit

After the CAL decision is frozen:

1. derive `d_full` from all CAL only;
2. derive `s_full` from all CAL coherent top-two gaps;
3. apply the frozen `lambda*` to TEST coherent logits;
4. re-run frozen global G2;
5. report TEST metrics for control and intervention;
6. report paired bootstrap 95% intervals for changes in sample NLL and Brier and a sample-bootstrap interval for Accuracy/Macro-F1/Tail20-F1 deltas.

TEST cannot change the F2 decision.

## Interpretation rule

- `F2 POSITIVE`: a nonzero lambda passes the preregistered OOF-CAL gate. TEST is then a replication audit.
- `F2 NEGATIVE`: no nonzero lambda passes. Do not reopen lambda or gate form under F2; any alternative gate/direction is a new `F2'` family.

Even a positive F2 establishes **actionability of the conditional predictive component**, not a causal mechanism.
