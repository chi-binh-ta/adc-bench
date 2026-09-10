# F2.10 — Adaptive Protected-Prefix / Candidate-Set Reliability Identification Audit

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

## Inherited facts

- F2.7: blind non-top diffusion is negative.
- F2.8: runner-up correctness is label-free identifiable; score geometry carries essentially all incremental signal.
- F2.9: protecting rank 2 removes the rank-2 damage exactly and reduces the smallest-eta Brier penalty by about 64%, but remaining damage localizes mainly to true classes at ranks 3--5.

F2.10 is **identification only**. It introduces no diffusion, no eta, no protection threshold and no benchmark-system adoption.

## Scientific question

Can label-free C3 score geometry identify how large a protected candidate prefix should be for each sample?

Let C3 classes be ordered by probability as

`c_(1), c_(2), ..., c_(K)`.

Define cumulative candidate-set targets

`C_m = 1{Y in {c_(1),...,c_(m)}}`, for `m in {2,3,5}`.

Also define conditional rank hazards

`H_r = 1{Y=c_(r) | Y not in {c_(1),...,c_(r-1)}}`, for `r in {2,3,4,5}`.

The cumulative targets answer whether a prefix of size m already contains the true class. The hazards diagnose where additional protected probability mass should stop being treated as disposable residual mass.

## Leakage policy

Use 5 outer CAL folds, stratified by class, seed `20260910`.

For each outer fold:

1. construct frozen F2 intervention and C3 calibrator from the outer-fit CAL subset only;
2. obtain outer-validation C3 probabilities and label-free features;
3. construct outer-fit reliability features through inner 4-fold C3 cross-fitting, seed `20261001`;
4. fit reliability models only on cross-fitted outer-fit features/targets;
5. evaluate on outer-validation;
6. concatenate all outer-validation predictions.

TEST is opened only after the OOF-CAL status is frozen. TEST labels cannot affect feature construction, model fitting or status.

## Frozen feature dictionary

The primary dictionary intentionally follows F2.8's simplification: score geometry first, no target-encoded pair priors.

From sorted C3 probabilities `p1>=...>=p6`:

- `p1,...,p6`;
- adjacent gaps `g12,g23,g34,g45,g56`;
- log ratios `log12,...,log56`;
- cumulative masses `top2_mass`, `top3_mass`, `top5_mass`;
- `p2/(1-p1)`, `p3/(1-p1-p2)`, `p5/(1-top4_mass)` with numerical guards;
- normalized HHI and entropy of the suffix after prefixes 1, 2, 3 and 5;
- top-1 margin state from F2: raw dose `a_i` and standardized C3 state `q_i`.

No CAL label enters these features.

## Frozen models

For each cumulative target C2, C3, C5 and each hazard H2, H3, H4, H5:

- StandardScaler fit on the training fold only;
- LogisticRegression(C=1, L2, solver=lbfgs, max_iter=3000).

No hyperparameter grid is allowed.

A fixed HistGradientBoosting model is run only as a nonlinear corroboration for cumulative targets:

- learning_rate=.05;
- max_iter=150;
- max_leaf_nodes=15;
- l2_regularization=1;
- random_state=20261001.

The logistic family remains primary.

## Evaluation

For every binary target report:

- prevalence;
- AUROC;
- average precision and AP lift;
- binary log loss;
- Brier;
- top-decile enrichment;
- constant-prevalence baseline log loss and delta.

For hazards, metrics are evaluated only on the at-risk subset `Y not in top-(r-1)` by definition.

For cumulative targets C2/C3/C5, run 1000 bootstrap draws, seed `20261002`, for AUROC, AP lift and log-loss delta versus the foldwise constant-prevalence baseline.

## OOF-CAL identification rules

A cumulative target `C_m` is `IDENTIFIED` only if all hold:

1. AUROC >= .62;
2. AP lift >= 1.20;
3. log-loss improvement versus foldwise constant baseline <= -.005;
4. top-decile enrichment >= 1.25;
5. bootstrap lower AUROC bound > .57;
6. bootstrap lower AP-lift bound > 1.10;
7. bootstrap upper log-loss-delta bound < 0.

A hazard `H_r` is `IDENTIFIED` if:

1. at-risk sample count >= 500;
2. both classes occur;
3. AUROC >= .58;
4. AP lift >= 1.15;
5. log-loss delta <= -.003.

OOF-CAL global status:

- `PREFIX_IDENTIFIED` if C2, C3 and C5 all pass;
- `PARTIAL_PREFIX_IDENTIFIED` if exactly two pass;
- `WEAK_PREFIX_SIGNAL` if one passes;
- `NOT_IDENTIFIED` if none pass.

Hazard identification is reported separately and cannot rescue a failed cumulative-prefix status.

## Monotonic candidate-set diagnostic

Because `C2 <= C3 <= C5` samplewise, independently fitted probability estimates should ideally respect

`s2 <= s3 <= s5`.

Report the OOF fraction of samples violating this monotonicity and the average violation magnitude. This is diagnostic only and does not alter the frozen status. No post-hoc isotonic repair is allowed in F2.10.

## TEST internal replication

After CAL status is frozen:

1. rebuild full-CAL C3 and TEST features with no TEST labels;
2. fit each frozen logistic model on all OOF-CAL feature rows/targets;
3. evaluate TEST C2/C3/C5 and hazards;
4. TEST cannot change the CAL-frozen status.

A CAL-identified cumulative target is called internally replicated if TEST has:

- AUROC >= .60;
- AP lift >= 1.15;
- log-loss delta versus frozen CAL-prevalence constant baseline < 0;
- top-decile enrichment >= 1.20.

Final status is `INTERNALLY_REPLICATED_PREFIX_IDENTIFICATION` only when OOF status is `PREFIX_IDENTIFIED` and all three cumulative targets replicate on TEST.

## Interpretation discipline

F2.10 can establish that candidate-set size is statistically predictable from inference-time score geometry. It does not authorize a protection threshold, prefix size rule, diffusion strength or probability correction.

Do not hard-code top-3/top-5 protection, choose score thresholds, tune model hyperparameters or reopen F2/F2.5 parameters inside F2.10. Any adaptive selective-diffusion rule is a later checkpoint.