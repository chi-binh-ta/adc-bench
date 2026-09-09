# F1.5 — Conditional Latent Factor Identification

Status: **COMPLETED**.

Successful workflow run: `34385748628`.

System analyzed: canonical `coherent-logit S=5 -> global G2` residuals.

The protocol was frozen before results. The first workflow attempt failed before producing statistics because the F1-v2 compatibility shim did not export the frozen feature-group definitions; the rerun changed only this compatibility export and left the statistical protocol, features, thresholds, and canonical logits unchanged.

## Question

F1 found strong marginal associations between the leading residual singular factors and prevalence, margin, calibration residuals, seed instability, and representation geometry. F1.5 asks a stricter question:

> Which feature families explain **unique predictive variance** in the stable residual factors after the competing feature families are residualized out?

For target group `G` and controls `C`, both factor and target features were cross-fitted on CAL:

`rV = V - Ehat[V | C]`

`rX_G = X_G - Ehat[X_G | C]`

A CAL-selected ridge model `rX_G -> rV` was then transferred without refit to the aligned TEST factor. The predeclared pass rule required:

- factor CAL/TEST alignment >= .75;
- residual CAL 5-fold CV R2 > 0;
- |TEST residual correlation| >= .25;
- conditional permutation p <= .05;
- positive TEST R2 increment versus the control-only model.

## Main result

Across 60 conditional tests, only **2** passed the complete rule. Both are the same scientific object:

`V1 <- margin | all other five feature groups`

and they pass independently for both soft-leakage and hard-confusion residual matrices.

### Soft-leakage V1: margin unique after all other groups

Controls: prevalence + class similarity + seed instability + calibration residual + representation geometry.

- factor alignment: `0.996672`
- control CAL CV R2: `0.656113`
- residual margin CAL CV R2: `0.076581`
- residual TEST R2: `0.176542`
- residual TEST correlation: `0.473638`
- conditional permutation p: `0.025948`
- control-only TEST R2: `0.689629`
- full-model TEST R2: `0.785493`
- delta TEST R2: `+0.095864`

### Hard-confusion V1: margin unique after all other groups

Same controls.

- factor alignment: `0.979400`
- control CAL CV R2: `0.287086`
- residual margin CAL CV R2: `0.083712`
- residual TEST R2: `0.194351`
- residual TEST correlation: `0.503826`
- conditional permutation p: `0.025948`
- control-only TEST R2: `0.307754`
- full-model TEST R2: `0.371160`
- delta TEST R2: `+0.063406`

Thus margin is the only observable family in the current six-family dictionary that shows **robust unique predictive signal for V1 under the strongest all-other adjustment**.

## All-other unique tests

For factor 1:

| target group | matrices passing | status |
|---|---:|---|
| margin | 2/2 | ROBUST_UNIQUE |
| prevalence | 0/2 | NO_UNIQUE_EVIDENCE |
| class similarity | 0/2 | NO_UNIQUE_EVIDENCE |
| representation geometry | 0/2 | NO_UNIQUE_EVIDENCE |
| calibration residual | 0/2 | NO_UNIQUE_EVIDENCE |
| seed instability | 0/2 | NO_UNIQUE_EVIDENCE |

For factors 2 and 3, **no feature family passed in either residual definition** after controlling the other five families.

## Why this refines F1 rather than contradicting it

F1 established marginal replicated associations. In particular, V1 was strongly associated with prevalence and margin, and V2 had weaker replicated geometric associations. F1.5 shows that these associations are highly shared/collinear.

### V1

The simpler pairwise contrasts did **not** pass the full predeclared rule:

- `margin | prevalence`: NOT_UNIQUE in both residual matrices;
- `prevalence | margin`: NOT_UNIQUE in both;
- `geometry | prevalence + margin`: NOT_UNIQUE;
- `calibration | prevalence + margin`: NOT_UNIQUE;
- `seed instability | prevalence + margin`: NOT_UNIQUE.

For example, soft `margin | prevalence` still had TEST residual correlation `0.506560` and delta TEST R2 `+0.084083`, but its conditional permutation p was `0.137725`; hard `margin | prevalence` had negative residual CAL CV R2 (`-0.053549`) and negative delta TEST R2 (`-0.073855`). Therefore pairwise residualization alone does not isolate a stable margin component.

Once the larger nuisance set is removed, however, the remaining margin component becomes reproducibly predictive in both residual definitions. This suggests that the observable families share substantial common variance and that the robust margin signal is a comparatively small but stable residual component, not the whole V1 axis.

### Prevalence

Prevalence is the strongest marginal correlate of soft V1 in F1, but under all-other adjustment it fails the unique-signal rule:

- soft: residual CAL CV R2 `0.024699`, TEST residual corr `0.482625`, p `0.079840`;
- hard: residual CAL CV R2 `-0.105884`, TEST residual corr `0.301806`, p `0.341317`.

Therefore prevalence should be treated as a strong structural proxy/covariate of V1, not as an independently identified mechanism under the present observable dictionary.

### Geometry and V2

F1's V2 geometric interpretation must be **downgraded**. The marginal associations with centroid separation/prototype scale were replicated, but F1.5 finds no conditional unique geometric signal:

- V2 `geometry | prevalence + margin`: residual CAL CV R2 is negative in both soft (`-0.059042`) and hard (`-0.079442`) matrices.
- No all-other geometry family passes for V2.

Thus V2 remains a stable residual factor, but the currently measured geometry variables do not uniquely explain it. Possible explanations include shared variance with other observables, nonlinear/pairwise geometry not captured by the current class-level summaries, or a genuinely unmeasured latent source. No one of these possibilities is established by F1.5.

### V3

No conditional family passes. Soft V3 has an interesting `geometry | prevalence + margin` TEST residual correlation (`0.322256`, p `0.001996`) but its residual CAL CV R2 is negative (`-0.021127`), so it fails the predeclared generalization rule and is not accepted as unique evidence.

## Updated factor interpretation

1. **V1 — margin-centered dominant residual axis, with prevalence/difficulty as strongly correlated structure.** The strongest unique predictive evidence belongs to margin, not prevalence.
2. **V2 — stable but conditionally unidentified.** Keep the F1 geometric association as a marginal clue only; do not call geometry an identified mechanism.
3. **V3 — stable but conditionally unidentified.** Its secondary prevalence/geometry associations are not uniquely predictive under the present tests.

## Scientific conclusion

The evidence now supports the following decomposition more strongly than the earlier descriptive labeling:

`V1 = shared(prevalence, margin, difficulty, calibration, geometry) + unique_margin_component + unexplained`

where the `unique_margin_component` transfers from CAL to TEST in both independent residual definitions.

For V2 and V3, the present six-family observable dictionary is insufficient for conditional identification.

This is **conditional predictive identification, not causal identification**. In particular, calibration residuals are downstream diagnostics, and controlling correlated/downstream variables can alter interpretation. The result justifies a next experiment focused on whether the unique V1 margin component is actionable, while V2/V3 require richer latent/geometry measurements rather than immediate correction.

Recommended next checkpoint: **F2 — Margin-Residual Intervention Test**. Construct a correction that acts only along the cross-fitted V1 margin-residual direction, freeze its strength on development data, and require improvement in Macro-F1/BalAcc/Tail20 without degrading NLL/Brier beyond a predeclared tolerance. In parallel, treat V2/V3 as discovery targets rather than intervention axes.
