# F2.6 — Proper-Score Geometry of the Hidden-State Calibrator

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

Inherited facts from F2.5:

- F2 intervention is frozen at `lambda=-0.05`;
- C1 = frozen F2 intervention followed by old global G2;
- C3 = same F2 intervention followed by the 4-parameter dose-conditioned G2;
- C3 improves OOF-CAL and TEST NLL/ECE relative to C1/C0 while slightly worsening Brier;
- C3 and C1 have identical ranking/classification decisions.

F2.6 is **diagnostic only**. It introduces no new calibration parameter, intervention parameter, model-selection rule, or adoption gate.

## Scientific question

Why can the same state-conditioned calibrator improve log score (NLL) and ECE while worsening quadratic/Brier score?

The primary comparison is **C3 vs C1**, because both share the exact same intervened logits and argmax decisions; only the probability map differs.

C3 vs C0 is secondary and describes the net intervention+calibration effect.

## Exact proper-score identities

For one-hot truth `y` and probability vector `p`,

`Brier(p,y) = ||p-y||_2^2 = 1 - 2 p_y + ||p||_2^2`.

For candidate `p3` and reference `pr`, define

`Delta p_y = p3_y - pr_y`.

Then exactly

`Delta Brier = -2 Delta p_y + Delta ||p||_2^2`.

A second exact decomposition is

`Brier = (1-p_y)^2 + sum_{j != y} p_j^2`,

so

`Delta Brier = Delta true_sq + Delta wrong_sq`,

where

`Delta true_sq = (1-p3_y)^2 - (1-pr_y)^2`,

`Delta wrong_sq = sum_{j != y} p3_j^2 - sum_{j != y} pr_j^2`.

The log-score difference is

`Delta NLL = -log(p3_y) + log(pr_y) = log(pr_y/p3_y)`.

Thus NLL depends only on the true-class probability, whereas Brier additionally depends on how probability mass is distributed across all wrong classes.

F2.6 will verify these identities numerically at sample level and aggregate the signed contributions.

## Reproduction / leakage policy

Use the same 5-fold stratified CAL split and seeds as F2/F2.5.

Reconstruct OOF C0/C1/C3 exactly using the frozen F2.5 procedure:

1. derive `d_fold` only from each 4/5 CAL fit subset;
2. intervene at fixed `lambda=-0.05`;
3. fit C3 only on intervened fit-fold CAL;
4. apply to the held-out CAL fold;
5. concatenate OOF probabilities.

C1 and C3 OOF metrics must replay F2.5 within numerical tolerance before any diagnostic result is accepted.

For TEST, fit the C3 calibrator on full intervened CAL exactly as F2.5 and apply to TEST. TEST cannot modify bins, hypotheses, or interpretation rules.

## Frozen stratifications

All continuous bin edges are learned from OOF-CAL for the **C1 reference** and then frozen for TEST.

### 1. True-class probability

Five equal-frequency CAL bins of

`p_y(C1)`.

### 2. Reference confidence

Five equal-frequency CAL bins of

`max_j p_j(C1)`.

### 3. Intervention dose

Five equal-frequency CAL bins of the raw frozen F2 gate

`a_i = exp(-top2_gap/scale)`.

### 4. Prevalence group

Use the existing frozen support partition:

- Head20 = 20 highest-support classes;
- Tail20 = 20 lowest-support classes;
- Mid60 = all remaining classes.

Grouping is by the sample's true class.

### 5. Score-conflict state

For each sample under C3 vs C1:

- `BOTH_IMPROVE`: Delta NLL < 0 and Delta Brier < 0;
- `LOG_ONLY`: Delta NLL < 0 and Delta Brier >= 0;
- `BRIER_ONLY`: Delta NLL >= 0 and Delta Brier < 0;
- `BOTH_WORSEN`: Delta NLL >= 0 and Delta Brier >= 0.

Zero differences use the non-improving side as written above.

## Frozen diagnostics

For C3 vs C1 and C3 vs C0 report:

- mean Delta NLL;
- mean Delta Brier;
- mean Delta p_y;
- mean `-2 Delta p_y`;
- mean Delta `||p||^2`;
- mean Delta true-class squared-error term;
- mean Delta wrong-class squared-mass term;
- fraction of samples with increased `p_y`;
- fraction in each score-conflict state.

For C3 vs C1 additionally report all of the above by:

- CAL-frozen true-class-probability quintile;
- CAL-frozen confidence quintile;
- CAL-frozen dose quintile;
- Head/Mid/Tail true-class group;
- score-conflict state.

For each stratum, report both the conditional mean contribution and its share of the total signed NLL/Brier change. Signed shares may exceed 1 or be negative when strata offset each other.

## Tail-sensitivity diagnostic

Because log score weights changes at small `p_y` approximately as `-Delta p_y/p_y`, compute on C3 vs C1:

- Spearman correlation of baseline `p_y(C1)` with Delta NLL and Delta Brier;
- mean Delta NLL/Brier within the bottom 20% and bottom 40% of CAL-frozen `p_y`;
- fraction of total NLL improvement contributed by the bottom 20% and bottom 40% of baseline true-class probability.

This is descriptive, not a new gate.

## State-coupling diagnostic

Compute Spearman correlations of raw intervention dose `a_i` with:

- Delta NLL;
- Delta Brier;
- Delta p_y;
- Delta wrong-class squared mass.

Repeat on OOF-CAL and TEST.

## Bootstrap uncertainty

Use 1000 paired sample bootstrap draws, seed `20260926`, for C3 vs C1 means of:

- Delta NLL;
- Delta Brier;
- Delta p_y;
- Delta true_sq;
- Delta wrong_sq;
- Delta ||p||^2.

Report 95% percentile intervals separately for OOF-CAL and TEST.

## Mechanism classification

F2.6 will not produce a model winner. It will assign descriptive mechanism labels from exact signed means:

1. `TRUE_PROBABILITY_GAIN_WITH_WRONG_CONCENTRATION_COST` if mean Delta p_y > 0, mean Delta wrong_sq > 0, mean Delta NLL < 0 and mean Delta Brier > 0.
2. `LOG_TAIL_GAIN_WITH_TRUE_PROBABILITY_TRADEOFF` if mean Delta NLL < 0, mean Delta Brier > 0, but mean Delta p_y <= 0.
3. `MIXED_PROPER_SCORE_GEOMETRY` otherwise.

A mechanism label is called **replicated** only if the same label occurs on OOF-CAL and TEST.

## Interpretation discipline

This audit can identify *where* and *through which probability geometry* the score disagreement arises. It cannot by itself establish a causal hidden state.

No Brier-weighted refit, multi-objective calibration, extra state feature, or new parameter is allowed in F2.6. Any correction motivated by this audit must begin a separately frozen checkpoint after F2.6 closes.