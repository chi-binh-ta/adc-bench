# F2.6 — Proper-Score Geometry of the Hidden-State Calibrator — FINAL

Status: **CLOSED / MECHANISM REPLICATED CAL -> TEST**.

No model is selected or adopted in F2.6. This checkpoint is diagnostic only.

Valid workflow run: `34482494248`.
Artifact: `10154205057`, SHA256 `8d2e87640d06f271c6941e45503daecc6f03977316d395069feb2d0c1f6edada`.

## Reproduction audit

F2.6 reconstructs OOF C0/C1/C3 probabilities from the frozen F2/F2.5 procedure rather than reading prior summary metrics.

The F2.5 OOF references replay exactly at reported precision:

- C1 NLL/Brier/Macro-F1 gaps = 0;
- C3 NLL/Brier/ECE gaps = 0.

TEST C3 also replays F2.5 within the frozen numerical tolerance.

Both exact Brier identities hold sample-wise with maximum errors around `1e-15`:

`Delta Brier = -2 Delta p_y + Delta ||p||^2`

and

`Delta Brier = Delta true_sq + Delta wrong_sq`.

Therefore the diagnostic decomposition is numerically valid.

## Primary comparison: C3 vs C1

C3 and C1 share the same F2-intervened logits and identical argmax/ranking. Their difference is purely the state-conditioned calibration map.

### Global OOF-CAL decomposition

- mean Delta NLL = `-0.00196711`;
- mean Delta Brier = `+0.00006859`;
- mean Delta p_y = `-0.00159036`;
- mean `-2 Delta p_y` = `+0.00318073`;
- mean Delta `||p||^2` = `-0.00311214`;
- mean Delta true-class squared-error term = `-0.00130671`;
- mean Delta wrong-class squared-mass term = `+0.00137530`;
- fraction of samples with `p_y` increased = `0.42413`.

### Global TEST decomposition

- mean Delta NLL = `-0.00235565`;
- mean Delta Brier = `+0.00009516`;
- mean Delta p_y = `-0.00139509`;
- mean `-2 Delta p_y` = `+0.00279018`;
- mean Delta `||p||^2` = `-0.00269502`;
- mean Delta true-class squared-error term = `-0.00148060`;
- mean Delta wrong-class squared-mass term = `+0.00157576`;
- fraction of samples with `p_y` increased = `0.41953`.

The frozen mechanism label is identical on CAL and TEST:

`LOG_TAIL_GAIN_WITH_TRUE_PROBABILITY_TRADEOFF`.

The label is therefore **replicated** under the preregistered classification rule.

Important nuance: the word `TAIL` in this frozen label refers to the asymmetric/tail-sensitive geometry of the log score, not to the empirical bottom-`p_y` quintile. The bottom 40% of baseline true-class probability does not generate the NLL gain on CAL or TEST.

## Core mechanism: Brier is a cancellation

The most informative exact decomposition is

`Delta Brier = Delta true_sq + Delta wrong_sq`.

On CAL:

- `Delta true_sq = -0.00130671` — improvement;
- `Delta wrong_sq = +0.00137530` — degradation;
- net `Delta Brier = +0.00006859`.

On TEST:

- `Delta true_sq = -0.00148060` — improvement;
- `Delta wrong_sq = +0.00157576` — degradation;
- net `Delta Brier = +0.00009516`.

Thus C3 does **not** worsen Brier because it harms the true-class squared component. It improves that component. Brier becomes slightly worse because the probability mass over wrong classes becomes more quadratically concentrated, and that cost is just larger than the true-class squared-error gain.

Bootstrap 95% intervals show that these two opposing components are individually stable:

CAL:

- Delta true_sq: `[-0.0017533, -0.0008468]`;
- Delta wrong_sq: `[+0.0009939, +0.0018140]`.

TEST:

- Delta true_sq: `[-0.0017955, -0.0011643]`;
- Delta wrong_sq: `[+0.0012852, +0.0018490]`.

Both component signs exclude zero on both splits.

The tiny net Brier difference itself is not statistically resolved by this bootstrap:

- CAL Delta Brier CI: `[-0.0005116, +0.0006564]`;
- TEST Delta Brier CI: `[-0.0002866, +0.0004676]`.

F2.5 still correctly rejected C3 because its adoption gate was based on the preregistered pointwise Brier inequality, not a significance test. F2.6 does not revise that decision.

## Why NLL can improve while mean p_y falls

C3 increases true-class probability for only about 42% of samples, so the arithmetic mean `p_y` decreases.

Nevertheless log loss is nonlinear:

`Delta NLL_i = log(p_ref,y / p_C3,y)`.

Large beneficial changes on selected samples can outweigh more numerous smaller harmful changes. This is visible in the score-conflict states.

### Conflict-state frequencies

CAL:

- BOTH_WORSEN = `52.43%`;
- BOTH_IMPROVE = `34.60%`;
- LOG_ONLY = `7.81%`;
- BRIER_ONLY = `5.15%`.

TEST:

- BOTH_WORSEN = `52.73%`;
- BOTH_IMPROVE = `33.29%`;
- LOG_ONLY = `8.66%`;
- BRIER_ONLY = `5.32%`.

The magnitude asymmetry is decisive. On TEST:

- BOTH_IMPROVE mean Delta NLL = `-0.07976`;
- BOTH_WORSEN mean Delta NLL = `+0.04774`.

Thus a smaller set of strongly improved examples can dominate the mean log score even though more examples worsen.

## True-class probability strata

Using OOF-CAL-frozen quintile cut points of C1 `p_y`, the replicated pattern is non-monotone.

CAL mean Delta NLL by quintile Q1 -> Q5:

`+0.00287, +0.02310, -0.02322, -0.04261, +0.03003`.

TEST:

`-0.00073, +0.02470, -0.02367, -0.04269, +0.02526`.

Therefore the NLL gain is driven primarily by **middle-high probability strata Q3-Q4**, especially Q4, while Q2 and Q5 offset it. It is not a simple low-probability-tail rescue.

Q5 is especially important for the Brier trade-off. Mean Delta p_y is strongly negative:

- CAL Q5: `-0.01994`;
- TEST Q5: `-0.01736`.

and mean Delta Brier is strongly positive:

- CAL Q5: `+0.01273`;
- TEST Q5: `+0.01051`.

Q4 moves in the opposite direction and improves both scores strongly.

## Intervention-state coupling

The state variable is not inert. Spearman correlations replicate closely:

### Dose vs Delta p_y

- CAL rho = `+0.4610`;
- TEST rho = `+0.4487`.

### Dose vs Delta wrong-class squared mass

- CAL rho = `+0.7109`;
- TEST rho = `+0.7149`.

This is the strongest state-coupling result in F2.6.

Higher intervention-state dose causes C3 to move true-class probability upward more strongly **and simultaneously** increases wrong-class quadratic concentration. The second effect explains the Brier tension.

Dose vs Delta NLL is weakly negative but highly stable:

- CAL rho = `-0.1077`;
- TEST rho = `-0.0733`.

Dose vs net Delta Brier is approximately zero:

- CAL rho = `-0.0147`;
- TEST rho = `+0.0125`.

This near-zero net relation is itself consistent with cancellation of the beneficial true-class term and harmful wrong-class term.

## Prevalence decomposition

The calibration recovery and the F2 decision intervention play different roles.

For C3 vs C1, NLL improvement is concentrated in Head classes:

- CAL Head mean Delta NLL = `-0.00504`, contributing about `149%` of the net signed NLL gain before Mid/Tail offsets;
- TEST Head mean Delta NLL = `-0.00524`, contributing about `130%` before offsets.

Mid and Tail do not drive the calibration NLL gain.

This is compatible with F2's earlier decision-side result: F2's intervention improved Macro-F1/Balanced/Tail20, while the state-conditioned calibrator C3 preserves those rankings and mainly reshapes probability quality afterward. The two mechanisms should not be conflated.

## Secondary C3 vs C0 geometry

The full intervention+state-calibration comparison has the same qualitative score conflict.

CAL:

- Delta NLL = `-0.0006186`;
- Delta Brier = `+0.0003006`;
- Delta p_y = `-0.0021201`;
- Delta true_sq = `-0.0005193`;
- Delta wrong_sq = `+0.0008199`.

TEST:

- Delta NLL = `-0.0009933`;
- Delta Brier = `+0.0003274`;
- Delta p_y = `-0.0019602`;
- Delta true_sq = `-0.0006658`;
- Delta wrong_sq = `+0.0009932`.

Again, true-class squared error improves while wrong-class quadratic concentration worsens more.

## Scientific closure

F2.6 resolves the F2.5 score disagreement into a replicated probability-geometry mechanism:

`state-conditioned nonlinear G2`

`-> non-monotone redistribution of p_y across samples`

`-> better log-score on selected high-leverage strata`

`-> better true-class squared-error component`

`-> larger squared concentration over wrong classes`

`-> near-cancellation in Brier, with a small positive point estimate`.

The strongest robust facts are:

1. the exact decompositions hold numerically;
2. mean `p_y` falls while NLL improves on both CAL and TEST;
3. true-class squared error improves with bootstrap intervals below zero on both splits;
4. wrong-class squared mass worsens with bootstrap intervals above zero on both splits;
5. dose strongly predicts both Delta p_y and, even more strongly, wrong-class concentration;
6. the same frozen mechanism label occurs on CAL and TEST.

Therefore F2.6 is **CLOSED / MECHANISM REPLICATED**.

No adoption status changes:

- canonical final coherent system remains unchanged;
- F2 remains an experimental decision intervention;
- C3 remains a non-adopted hidden-state calibrator because F2.5 failed its frozen Brier gate.

The highest-value next checkpoint, if opened, is **F2.7 — Wrong-Class Concentration Control**: freeze F2 and the C3 state variable, then test a separately preregistered mechanism that suppresses only the identified wrong-class quadratic concentration cost rather than retuning the entire calibration map.