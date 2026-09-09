# Stage E — Cluster-pair G2 — Frozen Protocol

Status: ACTIVE.

## Inherited frozen state
- Stage A: m = 16,384
- Stage B: scalar tau = .694
- Stage C: no confusion-aware margin
- Stage D: CLOSED — POSITIVE
- Stage-D benchmark: soft-vote S=5
- Stage-E structural input: coherent centered-logit ensemble S=5, seeds {42,123,456,789,2026}
- QN-v2 and all base-model training are frozen; Stage E is calibration geometry only.

## Source-derived G2 geometry
For centered class logits z, define complete-graph pair field

    L_ij = z_i - z_j.

Let

    sigma_ij = sigmoid(L_ij),
    u_ij = 1 - 4 sigma_ij (1-sigma_ij).

Global G2 uses

    R_G,ij = (alpha_0 + alpha_1 u_ij) L_ij.

For complete graph K=100 and W=I,

    t = (1/K) A^T R_G,
    A t = P_G R_G,
    z_cal = z - t.

## Cluster-pair G2
Classes are partitioned by training prevalence. The source specifies Head/Mid/Tail but does not define cut points. Reconstruction Clone v2 therefore freezes the following explicit choice:

- Head20 = 20 most prevalent training classes
- Tail20 = 20 least prevalent training classes
- Mid60 = remaining 60 classes

This is clone-specific and is chosen to align with the existing Tail20 metric.

For unordered cluster pair ab in

    {HH, HM, HT, MM, MT, TT},

use

    g_ab(u) = alpha_{0,ab} + alpha_{1,ab} u,
    g_ab = g_ba.

Then

    R_G,ij = g_{c(i)c(j)}(u_ij) L_ij.

Because the gate is identical on (i,j) and (j,i), R_G remains skew-symmetric. Projection to t uses the same exact complete-graph formula.

Parameter count:
- global G2: 2
- cluster-pair G2: 12

Global G2 is nested inside cluster-pair G2 by setting all six alpha_0 values equal and all six alpha_1 values equal.

## Selection discipline
Stage E is not allowed to select the 12-parameter model from in-sample full-CAL NLL alone.

Use 5-fold stratified CV within the existing CAL split:
1. fit global G2 on 4 folds, evaluate held-out fold;
2. fit cluster-pair G2 on the same 4 folds, evaluate the same held-out fold;
3. aggregate out-of-fold probability and ranking metrics.

Cluster-pair G2 passes the Stage-E CV gate only if:
- CV NLL(cluster) < CV NLL(global),
- CV Brier(cluster) <= CV Brier(global),
- CV Accuracy(cluster) >= CV Accuracy(global) - 0.001,
- CV MacroF1(cluster) >= CV MacroF1(global) - 0.001.

ECE is reported but is not a hard gate because fixed-bin ECE is noisier than NLL/Brier.

After the CV decision is written and frozen:
- refit global and cluster G2 on full CAL for audit;
- open TEST;
- test metrics cannot change the Stage-E winner.

## Sanity checks
- coherent Stage-D CAL logits reconstructed from the five seed artifacts must replay the Stage-D coherent-logit S=5 CAL metrics within numerical tolerance;
- sum of the six cluster alpha0 potential bases must reconstruct centered z (the P_G L identity) to floating-point tolerance;
- no Stage A/B/C/D parameter may be changed.

## Outputs
- Stage-E CV fold results
- full-CAL global and cluster coefficients
- H/M/T class membership
- TEST metrics for base coherent ensemble, global G2 and cluster-pair G2 (reported only after decision freeze)
- final Stage-E closure note.

Important provenance note: TEST was already opened once at the end of Stage D after the Stage-D CAL decision. Stage E does not use those Stage-D TEST values to tune G2, but the overall research process is therefore not globally test-blind anymore. Stage E preserves a stage-local CAL-freeze -> TEST-evaluate discipline.