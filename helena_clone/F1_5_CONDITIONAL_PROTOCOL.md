# F1.5 — Conditional Latent Factor Identification Protocol

Status: **FROZEN BEFORE RESULTS**.

System: canonical `coherent-logit S=5 -> global G2`.

Purpose: determine which observable feature groups explain **unique** variance in stable residual factors after controlling for correlated groups. This is diagnostic/explanatory, not causal identification.

## Factors

Primary: `V1,V2,V3` from both soft-leakage and hard-confusion residual matrices. CAL factors are Hungarian-matched and sign-aligned to TEST as in F1. Factors 4–5 are excluded from semantic identification because F1 stability was weaker.

## Feature groups

- prevalence
- class_similarity
- margin
- seed_instability
- calibration_residual
- representation_geometry

A convenience combined group `geometry = class_similarity + representation_geometry` is used only in predeclared contrasts; it does not replace the six original groups.

## Conditional residualization

For target group G and controls C:

1. On CAL, predict factor loading V from C with ridge using 5-fold cross-fitting; define out-of-fold residual `rV_CAL`.
2. Fit the same control model on full CAL and predict aligned TEST V; define `rV_TEST`.
3. For every feature in G, residualize it against C with the same CAL cross-fit / CAL-full-to-TEST discipline, producing `rXG_CAL` and `rXG_TEST`.
4. Fit ridge `rXG_CAL -> rV_CAL`, choose alpha only by CAL CV, then transfer without refit to TEST.
5. Compute residual CAL CV R2, TEST residual R2, TEST residual correlation, and a 500-permutation p-value obtained by permuting `rV_CAL` before fitting the target model and comparing absolute TEST residual correlation.

This is a predictive conditional-independence diagnostic; it does not prove causality.

## Unique-all-other tests

For every factor and each of the six groups, set G to that group and C to the union of the other five groups. Also compare control-only versus full model on TEST:

`delta_TEST_R2 = R2(full) - R2(control-only)`.

## Predeclared mechanistic contrasts

### V1
- margin | prevalence
- prevalence | margin
- geometry | prevalence + margin
- calibration_residual | prevalence + margin
- seed_instability | prevalence + margin

### V2
- geometry | prevalence + margin
- prevalence | geometry
- margin | geometry
- calibration_residual | prevalence + margin + geometry

### V3
- prevalence | margin + geometry
- margin | prevalence + geometry
- geometry | prevalence + margin

These contrasts are evaluated separately for soft leakage and hard confusion.

## Evidence rule

A conditional target group is a **replicated unique-driver candidate** only if all are true:

- factor CAL/TEST alignment >= 0.75;
- residual CAL 5-fold CV R2 > 0;
- absolute TEST residual correlation >= 0.25;
- conditional permutation p <= 0.05;
- `delta_TEST_R2 > 0` versus control-only.

Stronger status `PRIMARY_UNIQUE` requires the criterion to pass in both residual definitions (soft leakage and hard confusion) for the same factor/mechanistic contrast. Passing in only one matrix is `MATRIX_SPECIFIC_UNIQUE`.

## Interpretation discipline

- Calibration residuals are downstream diagnostics and cannot be called causal drivers.
- Prevalence, margin and geometry are correlated; conditional results, not marginal correlations, determine naming.
- TEST is used only for replication of this diagnostic protocol. It is not a fresh external domain.
- No Helena classifier/calibrator parameters are changed in F1.5.
