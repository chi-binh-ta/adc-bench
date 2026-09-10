# F2.8 — Runner-Up Reliability Identification Audit

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

## Inherited facts

F2.7 established that blind residual-simplex diffusion is negative even though it monotonically reduces wrong-class squared concentration. The failure is concentrated around useful non-top signal, especially samples for which the true class is the runner-up. Under OOF C3, about 20.65% of misclassified samples had the true class at rank 2.

F2.8 is **identification only**. It introduces no probability correction, no new eta, no new intervention strength and no adoption of a classifier into the Helena benchmark system.

## Scientific question

Can an inference-time, label-free observable state distinguish a **useful runner-up** from generic harmful non-top concentration?

For sample i, let `r_i` be the class ranked second by the frozen F2/F2.5 C3 probability vector. Define the binary target

`R_i = 1{Y_i = r_i}`.

Operationally we seek a reliability score

`s_i ≈ P(Y_i = r_i | O_i)`,

where `O_i` contains only information available without the true label at inference.

The primary target is defined over **all samples**, not only misclassified samples, because correctness of the top-1 prediction is unknown at inference. Evaluation conditional on misclassification is diagnostic only.

## Leakage policy and nested construction

Use the frozen Helena CAL partition only for identification/model selection. TEST is an internal replication audit because TEST has already been historically opened in earlier checkpoints.

Outer reliability folds use 5-fold stratified-by-class CAL splitting, seed `20260910`, matching the F2/F2.5 lineage.

For each outer fold:

1. build the frozen F2 intervention and C3 calibrator from the outer-fit CAL subset only;
2. apply it to the outer-validation subset to obtain runner-up labels, probabilities, dose/state and label-free features;
3. construct training features for the outer-fit samples through an **inner 4-fold C3 cross-fit**, seed `20260928`, so the reliability learner is not trained on in-sample C3 probabilities;
4. any target-encoded pair/confusion prior used as a training feature is itself cross-fitted inside the outer-fit reliability set, seed `20260929`;
5. fit the reliability learner on these cross-fitted outer-fit features and evaluate only on outer-validation;
6. concatenate all five outer-validation predictions.

No outer-validation or TEST label may enter C3 construction, pair priors, feature scaling or reliability model fitting for that held-out sample.

## Frozen observable feature groups

### G1 — Score / residual geometry

From C3 probabilities, with `p1 >= p2 >= p3`:

- `p1`, `p2`, `p3`;
- `p1-p2`, `p2-p3`;
- `log(p1/p2)`, `log(p2/p3)`;
- runner-up residual share `p2/(1-p1)`;
- normalized non-top residual HHI;
- normalized non-top residual entropy.

### G2 — Latent/intervention state

- raw frozen F2 dose `a_i`;
- standardized C3 state `q_i` using fit-fold mean/sd only.

### G3 — Seed / ensemble runner-up stability

Relative to the C3 ensemble runner-up class, from the five frozen seed logits only:

- mean and sd of its seed-level rank;
- fraction of seeds placing it exactly rank 2;
- fraction placing it in top 3;
- fraction placing it in top 5;
- sd of its centered seed logit;
- fraction of seeds agreeing with ensemble top-1;
- fraction of seeds ranking ensemble top-1 above ensemble runner-up.

### G4 — Representation/class-pair geometry

Using class prototypes computed **only from the frozen TRAIN split**:

- cosine similarity between top-1 and runner-up class prototypes;
- Euclidean distance between those prototypes;
- log support ratio `log((n_runner+1)/(n_top1+1))`.

No CAL label is used to compute prototype geometry.

### G5 — Cross-fitted confusion/reliability priors

Using only outer-fit reliability targets and a fixed empirical-Bayes shrinkage `kappa=20`:

- ordered `(top1, runner-up)` reliability prior;
- runner-up-class reliability prior;
- top1-class reliability prior;
- `log1p` ordered-pair count.

For outer-fit training samples these priors are 4-fold target-encoded out of fold. For outer-validation they are fit on the entire outer-fit set only.

## Frozen reliability models

### Primary: standardized logistic model

`LogisticRegression(C=1, L2, max_iter=3000, solver=lbfgs)`.

No C grid or class-weight grid is allowed.

Fit:

- `LOGIT_FULL`: all G1--G5;
- `LOGIT_SCORE_ONLY`: G1 only;
- five drop-one ablations `LOGIT_NO_Gk`.

The full logistic model is the primary identification model.

### Secondary nonlinear corroboration

One fixed HistGradientBoosting model on all G1--G5:

- learning_rate `0.05`;
- max_iter `150`;
- max_leaf_nodes `15`;
- l2_regularization `1.0`;
- random_state `20260928`.

No nonlinear hyperparameter may be tuned in F2.8. A nonlinear-only signal cannot be treated as equivalent to primary linear identification; it receives its own status.

## Frozen metrics

For every OOF model report:

- AUROC;
- average precision (AP);
- target prevalence;
- AP lift = AP / prevalence;
- binary log loss;
- binary Brier score;
- top-10%-score positive rate and enrichment over prevalence.

For `LOGIT_FULL` additionally report the same discrimination metrics on the label-defined diagnostic subset `top1 != Y`, plus 1000 paired/bootstrap intervals (seed `20260930`) for:

- AUROC;
- AP lift;
- log-loss difference versus the foldwise constant-prevalence baseline.

## OOF-CAL identification gate

`LOGIT_FULL` is `PRIMARY_IDENTIFIED` only if all hold:

1. AUROC >= `0.60`;
2. AP lift >= `1.25`;
3. log loss improves over foldwise constant-prevalence baseline by at least `0.003`;
4. top-decile enrichment >= `1.50`;
5. misclassified-only AUROC >= `0.56`;
6. bootstrap 95% lower bound for AUROC > `0.55`;
7. bootstrap 95% lower bound for AP lift > `1.10`;
8. bootstrap 95% upper bound for `(logloss_full - logloss_baseline)` < `0`.

If the primary logistic model fails but the fixed nonlinear model satisfies conditions 1--5, status is `NONLINEAR_SIGNAL_ONLY` and no intervention is authorized.

Otherwise F2.8 is `NOT_IDENTIFIED`.

## Group attribution

Group attribution is descriptive, not an extra gate.

For each Gk compare `LOGIT_FULL` to `LOGIT_NO_Gk` using OOF:

- Delta AUROC = `AUC_full - AUC_no_group`;
- Delta AP;
- Delta log loss = `logloss_no_group - logloss_full`.

Call a group `predictively_incremental` only when both

- Delta AUROC > `0.005`, and
- Delta log loss > `0.001`.

This is predictive attribution, not causal identification.

## Final CAL fit and TEST replication

After OOF-CAL status is frozen:

1. retain the complete OOF-CAL feature matrix and targets from the valid outer folds;
2. fit the frozen `LOGIT_FULL` feature scaler/model to all OOF-CAL rows;
3. estimate G5 priors for TEST from all OOF-CAL runner-up targets/pairs only;
4. fit the frozen F2 direction and C3 calibrator on full CAL, then construct TEST G1--G5 without TEST labels;
5. evaluate TEST target `1{Y_test = runnerup_C3}`;
6. TEST cannot alter the CAL-frozen identification status.

Call the primary identification **internally replicated** if TEST additionally satisfies:

- AUROC >= `0.58`;
- AP lift >= `1.15`;
- log loss below the frozen constant OOF-CAL prevalence baseline;
- top-decile enrichment >= `1.30`;
- misclassified-only AUROC >= `0.54`.

Report 1000 bootstrap 95% intervals on TEST for AUROC, AP lift and log-loss difference versus the frozen constant baseline.

## Interpretation discipline

F2.8 can establish that useful runner-up signal is statistically identifiable from label-free observables within the Helena split protocol. It does not establish a causal hidden state, nor does it authorize a probability intervention by itself.

Do not choose a runner-up-protection threshold, retune C3, retune F2, alter kappa, add extra feature families, tune nonlinear models, or optimize an intervention in F2.8. Any runner-up-protected correction must be a later separately frozen checkpoint.