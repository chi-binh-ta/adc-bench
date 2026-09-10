# F2.7 — Wrong-Class Concentration Control

Status: **FROZEN BEFORE RESULTS; numerical ranking invariant amended before any metric result was observed**.

Branch: `helena-reconstruction-clone-20260909`.

## Inherited frozen facts

- F2 intervention direction construction is unchanged.
- `lambda* = -0.05` is unchanged.
- F2 uncertainty/intervention dose `a_i = exp(-top2_gap/median_fit_gap)` is unchanged.
- C3 is the 4-parameter dose-conditioned G2 from F2.5.
- F2.6 established, and replicated CAL -> TEST, that C3 vs C1 improves the true-class squared-error component but increases wrong-class squared probability mass:

  `Delta Brier = Delta true_sq + Delta wrong_sq`,

  with `Delta true_sq < 0` and `Delta wrong_sq > 0` on both OOF-CAL and TEST.

F2.7 does **not** reopen F2, C3, the latent/state definition, or any earlier threshold.

## Scientific question

Can we reduce the wrong-class quadratic concentration created by C3 while preserving:

1. the F2 ranking/class-balance gains;
2. the C3 NLL improvement;
3. the C3 ECE improvement;
4. the true-class probability geometry that F2.6 showed was beneficial?

## Frozen correction family: Residual-Simplex Diffusion (RSD)

For each sample let `p` be its C3 probability vector and

`h = argmax_j p_j`.

Keep the predicted-class probability exactly fixed:

`p'_h = p_h`.

Let residual mass be

`m = 1 - p_h`

and define the normalized residual simplex for `j != h`:

`r_j = p_j / m`.

For a scalar strength `eta >= 0`, define a dose-dependent exponent

`gamma_i(eta) = 1 - eta * a_i`,

where `a_i in (0,1]` is the already-frozen F2 dose.

Then

`r'_j = r_j ** gamma_i / sum_{l != h} r_l ** gamma_i`,

and

`p'_j = m * r'_j` for `j != h`.

Thus F2.7 only diffuses probability mass **within the non-top residual simplex**.

### Structural properties

For all frozen candidate strengths below, `gamma_i > 0`. Therefore in exact arithmetic:

- strict ordering among non-top classes is preserved because `x -> x^gamma` is strictly monotone;
- the largest non-top residual share weakly decreases under `gamma <= 1`;
- the predicted top-class probability is unchanged;
- the class **weak order** and argmax are preserved;
- on correctly classified samples, true-class probability is unchanged and the wrong-class squared mass weakly decreases by simplex flattening.

On misclassified samples the true class lies inside the residual simplex, so its probability may increase or decrease. This is the empirical risk tested by F2.7.

## Frozen eta grid

`eta in {0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40}`.

`eta=0` must replay C3 exactly.

No other functional form or strength is allowed in F2.7.

## OOF-CAL construction

Use the exact 5-fold stratified CAL split and seeds from F2/F2.5/F2.6.

For each fold:

1. derive the frozen F2 direction from 4/5 CAL only;
2. intervene fit and validation logits with `lambda=-0.05`;
3. fit C3 on intervened 4/5 CAL only;
4. apply C3 to the held-out 1/5 CAL;
5. apply every frozen RSD `eta` to the held-out C3 probabilities using that fold's frozen F2 dose;
6. concatenate held-out predictions across folds.

No TEST label may affect eta selection.

## Reproduction gates

Before selection:

- `eta=0` must replay F2.5/F2.6 OOF C3 NLL, Brier, ECE and ranking metrics within `2e-6`;
- for every eta, probability rows must sum to one within `1e-12`;
- argmax must be exactly unchanged sample-wise relative to C3;
- strict probability order must have no reversal larger than the frozen floating tolerance `64 * eps(float64)`; reference ties/near-ties within that tolerance are treated as the same weak order.

### Pre-result numerical-invariant amendment

Two implementation-only attempts failed before producing any eta metric table:

1. run 1 failed at `eta=0` because recomputing an exponent-1 residual simplex perturbed floating-point ties; the identity branch was changed to return `p.copy()` exactly;
2. run 2 failed at `eta=0.01` because exact `argsort` treated tie-breaking among equal/near-equal probabilities as a change of ranking.

No candidate metric or selection result was observed before this amendment. The mathematical invariant intended from the start is preservation of strict order/weak ranking, not arbitrary floating tie-breaking. Therefore the invariant is now frozen as stated above; the eta grid, transformation and all scientific gates remain unchanged.

## Primary mechanism metrics

Relative to C3 report for every eta:

- NLL;
- Brier;
- ECE;
- mean true-class probability;
- mean true-class squared-error term `(1-p_y)^2`;
- mean wrong-class squared mass `sum_{j!=y} p_j^2`;
- entropy of the non-top residual simplex;
- Accuracy, Macro-F1, Balanced Accuracy, Tail20-F1.

The target mechanism requires

`Delta wrong_sq < 0`.

## Eligibility gate

A nonzero eta is eligible only if all conditions hold on OOF-CAL:

1. `wrong_sq(eta) < wrong_sq(C3)`;
2. `Brier(eta) < Brier(C3)`;
3. `Brier(eta) <= Brier(C1)`;
4. `NLL(eta) < NLL(C1)`;
5. `ECE(eta) <= ECE(C1)`;
6. Accuracy, Macro-F1, Balanced Accuracy and Tail20-F1 are exactly equal to C3 within numerical tolerance `1e-12`.

Thus the correction must solve the identified wrong-class mechanism enough to recover the F2.5 Brier gate while retaining C3's probability advantage over C1 and all F2 decisions.

Among eligible eta values choose lexicographically:

1. minimum Brier;
2. minimum NLL;
3. minimum ECE;
4. smallest eta.

If none is eligible, freeze `eta*=0` and close F2.7 `NO_CONTROL`.

## Recovery classification

For the OOF-CAL winner:

- **FULL_PARETO_RECOVERY** if `Brier <= C0`, `NLL <= C0`, and `ECE <= C0`, while decisions and strict/weak ranking remain C3/F2-equivalent;
- **PARTIAL_CONTROL** if eligible but one or more of those three canonical probability inequalities fail;
- **NO_CONTROL** if no nonzero eta is eligible.

The classification is frozen before TEST.

## TEST audit

After eta and status are frozen:

1. derive the full-CAL F2 direction;
2. fit full-CAL C3 exactly as F2.5/F2.6;
3. apply frozen `eta*` to TEST using the full-CAL gate scale;
4. report C0, C1, C3 and F2.7 metrics;
5. decompose Brier into true-class and wrong-class terms;
6. report 1000-pair bootstrap 95% intervals for F2.7 minus C3 and F2.7 minus C0 in NLL, Brier, true_sq and wrong_sq;
7. TEST cannot change eta or the OOF-CAL recovery classification.

## Interpretation discipline

F2.7 is a mechanism-targeted correction test, not a new latent-factor discovery stage. A positive result means the F2.6 wrong-class concentration channel is actionable using a label-free residual-simplex transformation. It does not establish a causal hidden state.

Do not tune a second exponent, class-specific eta, separate head/tail diffusion, or another entropy penalty inside F2.7. Such changes require a new checkpoint.