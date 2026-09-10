# F2.7 — Wrong-Class Concentration Control — FINAL

Status: **CLOSED / NO_CONTROL**.

No F2.7 correction is adopted. The frozen winner is the identity:

`eta* = 0`.

Valid final diagnostic-repair workflow run: `34487201181`.
Artifact: `10156162058`, SHA256 `e2adcb2b731548bcd83447b50d6ab8e2f8f307fe8ea9b950af3a64afb47637f4`.

## Scientific question

F2.6 showed that C3's Brier tension is a cancellation:

`Delta Brier = Delta true_sq + Delta wrong_sq`,

with the true-class squared term improving but wrong-class squared probability mass increasing.

F2.7 asked whether a label-free probability transformation could selectively reduce this wrong-class concentration while preserving F2 rankings and C3's NLL/ECE gains.

## Frozen correction: Residual-Simplex Diffusion

For C3 probability vector `p`, predicted class `h=argmax p`, residual mass `m=1-p_h`, and residual simplex `r_j=p_j/m` for `j != h`:

`gamma_i(eta)=1-eta*a_i`,

`r'_j proportional to r_j ** gamma_i`,

`p'_h=p_h`,

`p'_j=m*r'_j`.

The eta grid was frozen at

`{0,.01,.02,.03,.05,.075,.10,.15,.20,.30,.40}`.

The method preserves the predicted top probability and, in exact arithmetic, the weak class order. Its purpose is only to flatten the non-top residual simplex.

## Provenance of implementation attempts

### Run 1 — invalid

Run `34486222599` failed at `eta=0` because exponent-1 recomputation created machine-scale tie perturbations. No scientific eta metrics were produced/read. The identity branch was changed to return `p.copy()` exactly.

### Run 2 — invalid

Run `34486470711` failed at `eta=.01` because exact `argsort` treated floating tie-breaking as a ranking change. No candidate metric table was observed before this failure.

The invariant was then clarified before results: preserve exact argmax and preserve strict probability order up to `64*eps(float64)`; machine-scale ties are one weak-order equivalence class. No eta, transformation, metric gate, split or seed changed.

### Run 3 — scientifically valid selection

Run `34486795737` completed and froze `NO_CONTROL, eta*=0`. C3 replay gaps were exactly zero. A diagnostic-only residual-entropy calculation was later found numerically unstable because it used `1-p_top` under catastrophic cancellation. Entropy was not used in any eligibility or selection rule.

### Run 4 — final diagnostic-repair replay

Run `34487201181` repaired only that entropy calculation and added post-hoc true-class-rank diagnostics. It reproduced the same scientific decision exactly:

`NO_CONTROL`, `eta*=0`, `n_eligible=0`.

Thus run 4 is the final archival artifact while run 3 remains the first valid selection run.

## OOF-CAL result

C3 replayed F2.5/F2.6 exactly:

- NLL `2.7397491932`;
- Brier `0.7826232438`;
- ECE `0.0286465333`;
- Accuracy `0.3572597137`;
- Macro-F1 `0.1949005588`;
- Balanced Accuracy `0.1941208641`;
- Tail20-F1 `0.1126408952`.

No nonzero eta passed the frozen eligibility gate.

### Full eta curve: mechanism succeeds locally but objective fails globally

| eta | Delta NLL vs C3 | Delta Brier vs C3 | Delta wrong_sq | Delta true_sq |
|---:|---:|---:|---:|---:|
| .01 | -0.000026 | +0.000029 | **-0.000358** | **+0.000387** |
| .02 | -0.000010 | +0.000063 | **-0.000713** | **+0.000776** |
| .03 | +0.000049 | +0.000102 | -0.001066 | +0.001168 |
| .05 | +0.000299 | +0.000196 | -0.001765 | +0.001960 |
| .075 | +0.000860 | +0.000342 | -0.002622 | +0.002964 |
| .10 | +0.001704 | +0.000521 | -0.003463 | +0.003983 |
| .15 | +0.004261 | +0.000976 | -0.005088 | +0.006064 |
| .20 | +0.008016 | +0.001562 | -0.006634 | +0.008196 |
| .30 | +0.019277 | +0.003102 | -0.009470 | +0.012573 |
| .40 | +0.035746 | +0.005074 | -0.011935 | +0.017009 |

Residual entropy increases monotonically as intended, from `3.350808` at eta=0 to `3.693705` at eta=.40.

Therefore RSD **does hit the F2.6 target channel**: wrong-class squared concentration falls monotonically. But the true-class squared error increases even faster, so Brier worsens for every nonzero eta.

The smallest intervention already shows the fundamental conflict:

at `eta=.01`,

`Delta wrong_sq=-3.58e-4`,

but

`Delta true_sq=+3.87e-4`,

hence

`Delta Brier=+2.9e-5`.

This is not a strength-selection problem within the frozen family; it is a structural failure of blind residual diffusion.

## Why blind diffusion fails: true-class-rank audit

Post-hoc diagnostic only; it did not participate in selection.

Under OOF C3:

- true class rank 1: `35.73%`;
- rank 2: `13.27%`;
- rank 3--5: `15.19%`;
- rank 6+: `35.81%`.

Among misclassified samples, the true class is the runner-up (`rank2`) in about

`20.65%`.

At the smallest nonzero eta `.01`:

| true-class rank | share | Delta NLL | Delta Brier | Delta p_y | Delta true_sq | Delta wrong_sq |
|---|---:|---:|---:|---:|---:|---:|
| rank1 | .3573 | 0 | **-0.000325** | 0 | 0 | **-0.000325** |
| rank2 | .1327 | **+0.007618** | **+0.002068** | **-0.001400** | **+0.002173** | -0.000105 |
| rank3-5 | .1519 | +0.004247 | +0.000171 | -0.000373 | +0.000672 | -0.000501 |
| rank6+ | .3581 | **-0.004700** | **-0.000434** | +0.000005 | -0.000010 | -0.000424 |

This decomposition explains the negative result.

For correctly classified samples, the construction behaves exactly as intended: `p_y=p_top` is fixed, so the true-class term does not move and Brier improves through reduced wrong-class concentration.

For misclassified samples, however, the true class belongs to the residual simplex. When it is a large residual component — especially the runner-up — flattening with `gamma<1` pushes that useful component downward. The reduction in generic wrong-class concentration is then purchased by a much larger loss in true-class probability.

Thus the object identified in F2.6 as `wrong-class concentration` is not observable label-free at inference. The **non-top residual simplex is a mixture of harmful wrong-class concentration and potentially useful true-class runner-up signal**.

## TEST audit

Because OOF-CAL froze `eta*=0`, F2.7 on TEST is exactly C3:

- NLL `2.7620069981`;
- Brier `0.7782723678`;
- ECE `0.0227312318`;
- Accuracy `0.3621676892`;
- Macro-F1 `0.2062443951`;
- Balanced Accuracy `0.2034808168`;
- Tail20-F1 `0.1173547454`.

No TEST information changed the F2.7 closure.

For C3/F2.7 vs canonical C0 on TEST, bootstrap 95% intervals are:

- Delta NLL: `[-0.0026915,+0.0007477]`;
- Delta Brier: `[-0.0000453,+0.0007155]`;
- Delta true_sq: `[-0.0009815,-0.0003819]`;
- Delta wrong_sq: `[+0.0007332,+0.0012588]`.

This independently restates the F2.6 geometry: true-class squared error is clearly better, wrong-class concentration clearly worse, while the net proper-score tradeoff is small.

## Scientific conclusion

F2.7 is a useful negative result:

`global non-top flattening` is **not** a valid solution to the C3 Brier tension.

The reason is now identified:

`non-top probability mass = harmful wrong-class concentration + useful latent runner-up signal`.

A label-free correction cannot safely act on the entire residual simplex as if all non-top concentration were harmful.

Therefore:

1. **F2.7 CLOSED / NO_CONTROL**;
2. `eta*=0`; no RSD correction is adopted;
3. the F2.6 wrong-class channel is real but not directly observable from non-top concentration alone;
4. any next correction must protect high-value runner-up structure or estimate runner-up reliability without using the true label at inference;
5. do not retune eta or add class-specific diffusion inside F2.7.

## Highest-value next checkpoint

**F2.8 — Runner-Up Reliability Identification Audit** should come before another intervention.

The next scientific question is not "how much should we flatten?" but:

`Can a label-free observable state distinguish useful runner-up concentration from harmful wrong-class concentration?`

Candidate observables should be audited, not immediately optimized: top1-top2 gap, top2 residual share, top2/top3 gap, ensemble disagreement about the runner-up, seed-level runner-up stability, class-pair confusion geometry, and the previously identified latent/margin state.

Only if runner-up reliability is predictably identifiable out of sample should a runner-up-protected control be opened as a later intervention family.