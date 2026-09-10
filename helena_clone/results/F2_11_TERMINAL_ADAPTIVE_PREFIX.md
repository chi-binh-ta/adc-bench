# F2.11 — Terminal Coherent Adaptive-Prefix Intervention — FINAL

Status: **CLOSED / TERMINAL_REJECT_LOCK_CANONICAL_V2**.

Valid workflow run: `34495356271`.
Artifact: `10159606990`, SHA256 `3646a7494a2486d5eebdea9ff1014a66ff6d645b79de1da3750955dd2200c4c1`.

This is the terminal Stage F2 verdict. No F2.12+ rescue stage is authorized.

## Final deployment decision

The official Helena deployment system for this study is locked at:

**Canonical v2 = coherent S=5 -> global G2**.

The experimental F2/C3/adaptive-prefix family is not adopted.

F2.5 through F2.11 are retained as **Analytical Insights / Ablation Anatomy**.

## Frozen F2.11 architecture

F2.11 integrated the two intended terminal phases:

1. coherent sequential hazard estimation for the candidate-prefix state;
2. adaptive deeper-tail diffusion after a sample-specific protected prefix.

Sequential hazards were

`h1(O)=P(rank_y=1|O)`

and for `r=2,3,4`,

`hr(O)=P(rank_y=r | rank_y>=r,O)`.

They produce coherent cumulative probabilities

`Fm(O)=1-product_{r<=m}(1-hr(O))`.

For alpha, the protected prefix was

`m*=min{m in {2,3,4}: Fm>=1-alpha}`

with capped fallback `m*=4`.

For all ranks within the selected prefix, probability was preserved exactly. Only the deeper tail was diffused with the inherited F2 dose-gated power transform

`gamma_i=1-eta*a_i`.

Frozen grids:

- alpha `{.05,.10,.15,.20,.25,.30,.35,.40,.45,.50}`;
- eta `{0,.01,.02,.03,.05,.075,.10,.15,.20,.30,.40}`.

## Anti-tuning design

The terminal verdict used 5 outer CAL folds. Within every outer-fit set, C3 probabilities were themselves cross-fitted and the hazard layer was cross-fitted again before alpha/eta selection. The pair chosen inside an outer fold therefore could not use that fold's outer-validation labels.

A nonzero pair was eligible only if it simultaneously satisfied, against Canonical v2 on the selection set:

`Delta NLL <= 0` and `Delta Brier <= 0`,

while exactly preserving the C3 Accuracy, Macro-F1, Balanced Accuracy and Tail20-F1.

## Outer-fold result

**No outer fold found an eligible nonzero pair.**

All five folds therefore used the preregistered identity fallback `eta=0` (`alpha=.25` is irrelevant at eta=0).

| fold | selected alpha | selected eta | nonzero eligible? |
|---:|---:|---:|---|
| 1 | .25 | 0 | no |
| 2 | .25 | 0 | no |
| 3 | .25 | 0 | no |
| 4 | .25 | 0 | no |
| 5 | .25 | 0 | no |

Thus nested F2.11 OOF probabilities equal OOF C3 exactly.

## Nested OOF-CAL terminal verdict

Canonical v2 C0:

- NLL `2.7403678894`;
- Brier `0.7823226652`;
- ECE `0.0331652938`;
- Accuracy `0.3564417178`;
- Macro-F1 `0.1937244679`;
- Balanced Accuracy `0.1937828148`;
- Tail20-F1 `0.1058962558`.

Nested F2.11 = C3:

- NLL `2.7397491932`;
- Brier `0.7826232438`;
- ECE `0.0286465333`;
- Accuracy `0.3572597137`;
- Macro-F1 `0.1949005588`;
- Balanced Accuracy `0.1941208641`;
- Tail20-F1 `0.1126408952`.

Relative to Canonical v2:

`Delta NLL = -0.0006186962`,

but

`Delta Brier = +0.0003005786`.

Therefore the preregistered double proper-score gate fails and OOF status is

**`CAL_TERMINAL_FAIL`**.

Bootstrap 95% intervals versus C0:

- NLL delta `[-0.0028816,+0.0018398]`;
- Brier delta `[-0.0002666,+0.0008987]`.

Bootstrap was descriptive only; the frozen terminal gate uses point-estimate Pareto dominance.

## Full OOF grid anatomy

The global OOF freeze evaluated all 100 valid nonzero `(alpha,eta)` candidates.

Key fact:

**zero of 100 nonzero candidates improved Brier relative even to C3.**

The minimum Brier change versus C3 was still positive:

`min Delta Brier(Candidate-C3) = +1.6361e-6`.

At the same time 54/100 candidates improved NLL relative to C3. Thus the adaptive-prefix family reproduces the same log-score/quadratic-score conflict rather than solving it.

Closest candidate by Brier was

`alpha=.05, eta=.01`.

It used approximately:

- prefix 2: `3.89%`;
- prefix 3: `3.52%`;
- prefix 4: `92.60%`;
- F4 coverage shortfall at the `.95` target: `91.98%`.

Relative to C3 it gave approximately:

- Delta NLL `-9.6e-5`;
- Delta Brier `+1.64e-6`;
- Delta true-class squared term `+5.3e-5`;
- Delta wrong-class squared mass `-5.2e-5`.

This is a near-perfect continuation of the F2.6 cancellation geometry: deeper-tail diffusion reduces wrong-class quadratic concentration, but the accompanying true-class damage remains marginally larger.

The global grid therefore found no eligible deployment pair and froze the identity fallback:

`alpha*=.25, eta*=0`.

## Adaptive-prefix diagnostics

At the identity fallback, alpha does not affect probabilities, but the coherent hazard rule itself assigns on OOF-CAL approximately:

- `m=2`: `17.51%`;
- `m=3`: `5.77%`;
- `m=4`: `76.73%`.

At `alpha=.25`, about `71.64%` of rows have `F4<.75` and therefore hit the capped `m=4` fallback.

This shows that the coherent prefix estimator is informative but often does not reach a high cumulative-confidence threshold by rank 4. F2.10's identification success therefore does not imply that a short discrete protected prefix is sufficient for probability intervention.

## TEST replication after CAL freeze

Because no global nonzero pair was eligible, F2.11 TEST is exactly full-CAL C3.

Canonical v2 TEST:

- NLL `2.7626566887`;
- Brier `0.7779449890`;
- ECE `0.0313778651`;
- Accuracy `0.3618609407`;
- Macro-F1 `0.2051001693`;
- Balanced Accuracy `0.2019274906`;
- Tail20-F1 `0.1167976953`.

F2.11/C3 TEST:

- NLL `2.7620069981`;
- Brier `0.7782723678`;
- ECE `0.0227312318`;
- Accuracy `0.3621676892`;
- Macro-F1 `0.2062443951`;
- Balanced Accuracy `0.2034808168`;
- Tail20-F1 `0.1173547454`.

Relative to Canonical v2:

`Delta NLL = -0.0006496906`,

`Delta Brier = +0.0003273789`.

Thus TEST also fails the terminal double-score gate. Final status is

**`TERMINAL_REJECT_LOCK_CANONICAL_V2`**.

## Scientific interpretation

The terminal result does not say candidate-prefix identification failed. F2.10 showed that it succeeds strongly. F2.11 instead shows that **identifying candidate-set membership is not sufficient to turn the tested discrete prefix-preserving power-diffusion family into a simultaneous NLL/Brier Pareto improvement**.

Across F2.7, F2.9 and F2.11, progressively more selective protection reduces the damage but does not change the sign of the terminal Brier tradeoff. In the closest F2.11 candidate, the beneficial reduction in wrong-class squared concentration and the harmful true-class squared change almost exactly cancel, with the harmful side remaining slightly larger.

A defensible study-level conclusion is therefore:

> Within the tested Helena post-calibration family, probability uncertainty behaves as a distributed rank-spectrum phenomenon rather than a cleanly separable `protected prefix / noise tail` decomposition. Discrete local flattening did not yield a simultaneous NLL/Brier Pareto gain.

This is an empirical conclusion for the tested intervention family, not a universal theorem about multiclass calibration.

## Terminal closure

1. F2.11 is closed.
2. Stage F2 is closed.
3. No F2.12+ tuning/rescue is authorized.
4. Official deployment is **Canonical v2: coherent S=5 -> global G2**.
5. F2 remains evidence of small decision-geometry leverage, but not an adopted deployment intervention.
6. C3 remains an analytical example of NLL/ECE improvement with a Brier tradeoff, not an adopted calibrator.
7. F2.5--F2.11 belong in **Analytical Insights / Ablation Anatomy** and should be presented as a structured anatomy of the multiclass proper-score Pareto boundary.