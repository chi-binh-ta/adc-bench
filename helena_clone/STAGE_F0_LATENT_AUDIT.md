# Stage F0 — Latent Residual Low-Dimensionality Audit

Status: **PROTOCOL FROZEN BEFORE RESULTS**.

## Question
Does the systematic residual error of the frozen canonical Helena system have a low-dimensional structure that generalizes out of sample?

This is an empirical/statistical identification test, not a population-level mathematical proof.

## Frozen upstream system
- canonical rerun workflow: 34375185211
- m = 16,384
- scalar tau = .694
- no Stage-C margin
- seeds = {42,123,456,789,2026}
- coherent centered-logit S=5 ensemble
- global G2 fitted on CAL only
- no A–E parameter may be changed by F0

## Primary residual object
For split s in {CAL, TEST}, true class k, and output coordinate j,

    R_s[k,j] = E[ 1(Y=j) - p_j(X) | Y=k ]

where p is the probability vector after the frozen coherent-S5 -> global-G2 system.

R_s is 100 x 100. Every row sums to zero automatically. No additional row centering is applied.

This tests **systematic class-conditional residual structure**. Raw per-sample one-hot residuals are not assumed low-rank because sampling noise can remain high-dimensional.

## Spectral quantities
Let singular values of R_s be sigma_1 >= ... >= sigma_K and

    e_i = sigma_i^2 / sum_j sigma_j^2.

Report cumulative energy E_s(r) for r in {1,2,3,5,10,20}, plus:

    entropy effective rank = exp(-sum_i e_i log e_i)
    participation-ratio rank = 1 / sum_i e_i^2
    stable rank = ||R||_F^2 / sigma_1^2

## Out-of-sample transfer test
Fit the top-r right singular subspace V_CAL,r on CAL only. Then evaluate on TEST:

    Transfer(r) = ||R_TEST V_CAL,r||_F^2 / ||R_TEST||_F^2.

This is the central anti-PCA-artifact test.

## Random-subspace null
For each r in {1,2,3,5,10}, draw B=5000 Haar/random-QR r-dimensional subspaces Q_r in R^100 with RNG seed 20260909 and compute

    NullTransfer_b(r) = ||R_TEST Q_r||_F^2 / ||R_TEST||_F^2.

Report null mean, 95% and 99% quantiles and empirical p-value

    p = (1 + #{NullTransfer >= Transfer}) / (B+1).

## CAL–TEST subspace stability
Let V_TEST,r be the top-r TEST right-singular subspace. Define

    Overlap(r) = (1/r) ||V_CAL,r^T V_TEST,r||_F^2 in [0,1].

Compare against the same random-subspace null and report empirical p-value.

## Cross-seed corroboration
For each of the five freshly rerun base seeds, form class-conditional residual matrices from that seed's softmax probabilities and report pairwise top-r subspace overlap. This is secondary corroboration only; it does not alter the primary gate.

## Structured-vs-sample-noise diagnostic
For per-sample residual e_i = onehot(y_i)-p_i and TEST class mean mu_k=R_TEST[k], report

    rho_structured = sum_i ||mu_{y_i}||^2 / sum_i ||e_i||^2.

A small rho_structured does NOT by itself reject low-dimensional systematic structure; it only says per-sample noise dominates total residual energy.

## Pre-frozen decision gate
Declare **STRONG LOW-DIMENSIONAL SYSTEMATIC RESIDUAL EVIDENCE** iff there exists a smallest r* <= 10 satisfying all:

1. CAL cumulative energy E_CAL(r*) >= 0.60
2. TEST cumulative energy E_TEST(r*) >= 0.55
3. CAL->TEST Transfer(r*) >= 0.50
4. transfer random-subspace p <= 0.01
5. CAL–TEST subspace Overlap(r*) >= 0.50
6. overlap random-subspace p <= 0.01

If no r <= 10 passes all six conditions, conclude **NO STRONG LOW-DIMENSIONAL EVIDENCE UNDER THE FROZEN F0 GATE** and do not open a latent-factor modeling stage from this test alone.

The thresholds are intentionally strong: at r=10, only 10% of the ambient class-output dimension may be used, while at least half of independent TEST systematic residual energy must transfer through the CAL-learned subspace.
