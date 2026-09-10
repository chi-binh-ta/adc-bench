# F2.10 — Adaptive Protected-Prefix / Candidate-Set Reliability Identification Audit

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

## Inherited facts

F2.7 showed that blind non-top diffusion fails because useful true-class probability can live inside the residual simplex.

F2.8 established that rank-2/runner-up reliability is label-free identifiable within Helena, and that most of the signal is already present in score/residual geometry.

F2.9 protected rank 2 exactly and diffused only rank 3+. This removed the rank-2 true-class damage exactly and reduced the smallest-eta Brier penalty by roughly 64%, but the remaining damage localized mainly to true classes at ranks 3--5.

Therefore the scientifically relevant object is no longer a fixed runner-up state. It is an **adaptive candidate-set / protected-prefix state**.

F2.10 is identification only. It introduces no probability correction, no diffusion strength, no protected-prefix threshold chosen from TEST, and no change to F2/C3.

## Scientific question

Can inference-time, label-free score geometry identify how deep into the ranked class list useful true-class probability is likely to extend?

For each sample, let `rank_y` be the rank of the true class under the frozen F2 + dose-conditioned C3 probability vector. Define the six rank buckets

`B in {1,2,3,4,5,6+}`.

The primary model estimates one coherent categorical distribution

`pi_r(O) = P(B=r | O)`,

with the last component representing `rank_y >= 6`.

This single distribution induces the cumulative candidate-set reliabilities

`C_m(O)=P(rank_y <= m | O)=sum_{r<=m} pi_r(O)`

for `m in {2,3,5}`, and the conditional rank hazards

`H_r(O)=P(rank_y=r | rank_y>=r,O)=pi_r(O)/(1-sum_{k<r}pi_k(O))`

for `r=2,3,4,5`.

The structural purpose is to identify the smallest protected prefix that is plausible per sample before any later selective-diffusion intervention is opened.

## Leakage policy

Use the same Helena lineage and the same nested construction discipline as F2.8.

- Outer CAL: 5-fold stratified-by-class, seed `20260910`.
- Within each outer-fit set: 4-fold C3 cross-fitting, seed `20260928`, to construct reliability-training probabilities/features.
- For an outer-validation sample, F2 direction and C3 calibration are fit only on the corresponding outer-fit CAL subset.
- TEST is opened only after the OOF-CAL identification status is frozen.

No outer-validation or TEST label enters F2 direction fitting, C3 calibration, feature scaling, or rank-bucket model fitting for that held-out sample.

## Frozen score-geometry features

F2.8 showed that score/residual geometry is the only clearly incremental family for runner-up identification. F2.10 therefore deliberately simplifies the primary model to score geometry rather than reopening pair priors, representation geometry, or a flexible latent network.

For sorted C3 probabilities `p1>=...>=p6`:

- `p1,...,p6`;
- adjacent gaps `p1-p2,...,p5-p6`;
- adjacent log ratios `log(p1/p2),...,log(p5/p6)`;
- cumulative masses `top2_mass`, `top3_mass`, `top5_mass`;
- `rank2_non_top_share = p2/(1-p1)`;
- `rank3_5_non_top_share = (p3+p4+p5)/(1-p1)`;
- normalized rank-3+ HHI;
- normalized rank-3+ entropy;
- `rank6plus_mass = 1-top5_mass`.

All denominators use numerical guards only; no label enters these features.

### State-augmented secondary model

A secondary model adds only the already-frozen F2/C3 state variables

- `dose=a_i`;
- `q_state`.

This tests whether the earlier hidden/intervention state adds candidate-set information beyond score geometry. It cannot replace the primary score-only result if the primary fails.

## Frozen learner

Primary:

`StandardScaler -> LogisticRegression(C=1, L2, solver=lbfgs, max_iter=4000)`

with the six rank buckets as the multinomial target.

Secondary state-augmented learner uses the same fixed logistic settings with the two extra state features.

No C grid, class weighting, nonlinear learner, feature selection, or threshold tuning is allowed.

## OOF metrics

### Multinomial rank-bucket level

Report:

- multiclass log loss;
- constant foldwise bucket-prevalence baseline log loss;
- Delta log loss;
- multiclass Brier score `mean(sum_r (pi_r-onehot_r)^2)`;
- exact bucket accuracy;
- macro one-vs-rest AUROC when defined.

### Cumulative candidate-set level

For each `m in {2,3,5}`, using `C_m`:

- prevalence `P(rank_y<=m)`;
- AUROC;
- average precision and AP lift;
- binary log loss and Delta vs foldwise constant baseline;
- binary Brier and Delta vs baseline;
- 1000 bootstrap 95% intervals for AUROC and log-loss Delta.

### Rank-hazard level

For each `r=2,3,4,5`, evaluate only the risk set `rank_y>=r` and use `H_r` to predict `1{rank_y=r}`. Report:

- risk-set size and target prevalence;
- AUROC;
- AP/AP lift;
- binary log loss and Delta vs a frozen risk-set constant baseline.

Hazards are diagnostic components of the coherent multinomial model; they are not separately fitted.

## OOF-CAL identification gate

Status is `PREFIX_IDENTIFIED` only if all conditions hold for the primary score-only multinomial model:

1. multinomial log loss improves over the foldwise constant bucket baseline by at least `0.02`;
2. cumulative `C2`, `C3`, and `C5` each have AUROC >= `0.60`;
3. cumulative `C2`, `C3`, and `C5` each improve binary log loss over their foldwise constant baselines by at least `0.005`;
4. bootstrap 95% lower AUROC bound is > `0.55` for each of `C2`, `C3`, `C5`;
5. bootstrap 95% upper bound for log-loss Delta is < `0` for each of `C2`, `C3`, `C5`;
6. hazard `H2` and `H3` each have AUROC >= `0.56`;
7. at least one of `H4` or `H5` has AUROC >= `0.55`.

If cumulative reliability is identified but the hazard conditions fail, status is `CUMULATIVE_ONLY`.

Otherwise status is `NOT_IDENTIFIED`.

## State augmentation attribution

Compare primary score-only vs state-augmented multinomial model descriptively:

- Delta multinomial log loss;
- Delta macro OVR AUROC;
- cumulative AUC/log-loss changes for C2/C3/C5.

Call state `incremental` only if state augmentation improves multinomial log loss by > `0.002` **and** improves mean cumulative AUROC across C2/C3/C5 by > `0.005`.

This is predictive attribution only.

## Adaptive-prefix diagnostic

No intervention threshold is selected. For fixed confidence targets `rho in {0.60,0.70,0.80}`, use the OOF cumulative probabilities and choose the smallest prefix from `{2,3,5}` whose predicted `C_m >= rho`; if none reaches rho, record `UNRESOLVED` and use prefix 5 only for the empirical coverage summary.

Report for each rho:

- fractions choosing 2, 3, 5, or unresolved;
- mean selected prefix among resolved samples;
- empirical true-class coverage of the selected/fallback prefix;
- unresolved fraction;
- empirical coverage separately within selected-prefix strata.

This diagnostic is meant to characterize whether an adaptive protected-prefix policy is plausible. It does not authorize any diffusion rule.

## TEST internal replication

After OOF-CAL status is frozen:

1. retain the OOF-CAL score-geometry rows and rank-bucket targets;
2. fit the frozen primary and state-augmented multinomial models to all OOF-CAL rows;
3. fit frozen F2 direction and C3 calibration on full CAL, then construct TEST features without TEST labels;
4. evaluate the same rank-bucket, cumulative and hazard metrics on TEST;
5. TEST cannot alter the CAL-frozen status.

Call `PREFIX_IDENTIFIED` internally replicated when TEST satisfies:

- multinomial log loss better than the frozen OOF-CAL bucket-prevalence baseline;
- C2/C3/C5 AUROC each >= `0.58`;
- C2/C3/C5 log loss each better than their frozen OOF-CAL constant baselines;
- H2 and H3 AUROC each >= `0.54`;
- at least one of H4/H5 AUROC >= `0.53`.

## Interpretation discipline

F2.10 may establish that protected-prefix depth is statistically identifiable from existing score geometry. It does not prove that a particular prefix threshold or diffusion map improves proper scores.

Do not hard-code top-3/top-5 protection, tune a confidence threshold, reopen eta, introduce a nonlinear prefix network, or use TEST to choose any future selective-diffusion policy inside F2.10.