# F2.9 — Runner-Up-Protected Concentration Control
## Rank-2 Isolation / Selective Rank-3+ Subspace Diffusion

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

## Inherited frozen facts

F2.6 established that C3's Brier tension is a cancellation between an improved true-class squared term and increased wrong-class squared probability mass.

F2.7 showed that blind non-top diffusion is negative even though it monotonically reduces wrong-class concentration. The dominant failure mode is the true class sitting at rank 2: about 20.65% of misclassified OOF-CAL samples had the true class as the runner-up, and blind flattening reduced that useful rank-2 probability.

F2.8 then established that runner-up reliability is label-free identifiable within Helena, with OOF/TEST AUROC about 0.684/0.686. However the present F2.9 variant intentionally tests a simpler structural intervention first: **protect rank 2 unconditionally** and diffuse only rank 3+ probability mass. No runner-up reliability threshold is used in this checkpoint.

Frozen upstream system:

- F2 intervention direction unchanged;
- `lambda*=-0.05` unchanged;
- F2 dose `a_i=exp(-top2_gap/median_fit_gap)` unchanged;
- C3 dose-conditioned global G2 unchanged;
- no C3 coefficient family or F2 parameter is retuned.

## Scientific question

Is the rank-2 contamination identified in F2.7 sufficient to explain the failure of blind residual diffusion?

Equivalently: if top-1 and runner-up probabilities are held exactly fixed, can diffusion restricted to the rank-3+ subspace reduce harmful quadratic concentration enough to recover Brier while preserving C3 NLL/ECE and all F2 decision gains?

## Frozen transform: Rank-2 Isolated Selective Rank-3+ Diffusion (R2I-R3D)

For each C3 probability vector `p_i`, let

`c_(1), c_(2), ..., c_(K)`

be classes sorted by descending C3 probability, with probabilities

`p_(1) >= p_(2) >= ... >= p_(K)`.

Freeze the first two coordinates exactly:

`p'_(1)=p_(1)`,

`p'_(2)=p_(2)`.

Let the rank-3+ mass be

`m3 = 1 - p_(1) - p_(2)`.

For `r>=3`, define the normalized rank-3+ simplex

`u_(r)=p_(r)/m3`.

For scalar strength `eta>=0`, use the already-frozen F2 dose:

`gamma_i(eta)=1-eta*a_i`.

Then

`u'_(r) = u_(r)^gamma_i / sum_{l>=3} u_(l)^gamma_i`,

and

`p'_(r)=m3*u'_(r)` for `r>=3`.

If `m3` is numerically zero, return the original row unchanged.

### Structural properties

For the frozen eta grid below, `gamma_i>0`. Therefore in exact arithmetic:

- `p_(1)` is unchanged;
- `p_(2)` is unchanged;
- total rank-3+ mass is unchanged;
- weak ordering within rank 3+ is preserved because `x -> x^gamma` is monotone;
- the largest rank-3+ share weakly decreases for `gamma<=1`, hence rank 3 cannot overtake the protected rank 2;
- the full weak class order and argmax are preserved;
- if the true class is rank 1 or rank 2, its probability is unchanged exactly;
- if the true class is rank 1 or rank 2, quadratic concentration among all remaining wrong classes can only decrease or remain equal under ideal simplex flattening.

The only remaining true-class risk is when the true class has rank 3 or worse.

## Frozen eta grid

Use the same strengths as F2.7 for direct comparability:

`eta in {0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40}`.

`eta=0` must be bitwise identity on C3 probabilities.

No additional eta values or functional forms are allowed in F2.9.

## OOF-CAL construction

Use the same 5-fold stratified CAL split and seed `20260910` as F2/F2.5/F2.7.

For each fold:

1. derive the frozen F2 direction from the 4/5 CAL fit portion only;
2. intervene fit and validation coherent logits with `lambda=-0.05`;
3. fit C3 on intervened 4/5 CAL only;
4. apply C3 to the held-out 1/5 CAL;
5. apply every frozen R2I-R3D eta to held-out C3 probabilities using that fold's frozen F2 dose;
6. concatenate held-out probabilities across all folds.

No TEST label may affect eta selection.

## Reproduction and invariance gates

Before scientific selection:

- eta=0 must replay F2.5/F2.6 C3 OOF metrics within `2e-6`;
- probability rows must sum to one within `1e-12`;
- top-1 and rank-2 class identities must be unchanged for every sample;
- top-1 and rank-2 probabilities must be unchanged to absolute tolerance `2e-15`;
- no strict-order reversal larger than `64*eps(float64)` is allowed among rank-3+ classes;
- Accuracy, Macro-F1, Balanced Accuracy and Tail20-F1 must remain numerically equal to C3 within `1e-12`.

Machine-scale tie reordering within weak-order equivalence classes is not a failure.

## Primary mechanism metrics

For each eta report:

- NLL;
- Brier;
- ECE;
- mean true-class probability;
- mean true-class squared term `(1-p_y)^2`;
- mean wrong-class squared mass `sum_{j!=y} p_j^2`;
- mean rank-3+ squared mass `sum_{r>=3} p_(r)^2`;
- mean rank-3+ normalized entropy;
- decision metrics above.

Also report post-hoc diagnostics by true-class rank strata:

- rank 1;
- rank 2;
- rank 3--5;
- rank 6+.

These rank-strata diagnostics do not participate in eta selection.

## Eligibility gate

A nonzero eta is eligible only if all conditions hold on OOF-CAL:

1. `wrong_sq(eta) < wrong_sq(C3)`;
2. `rank3plus_sq(eta) < rank3plus_sq(C3)`;
3. `Brier(eta) < Brier(C3)`;
4. `Brier(eta) <= Brier(C1)`;
5. `NLL(eta) < NLL(C1)`;
6. `ECE(eta) <= ECE(C1)`;
7. `true_sq(eta) <= true_sq(C1)`;
8. Accuracy, Macro-F1, Balanced Accuracy and Tail20-F1 equal C3 within `1e-12`.

Thus F2.9 must solve the concentration channel enough to pass the original F2.5 Brier safeguard while retaining at least C3's probability advantages over the fixed-G2 F2 reference.

Among eligible eta values choose lexicographically:

1. minimum Brier;
2. minimum NLL;
3. minimum ECE;
4. smallest eta.

If none is eligible, freeze `eta*=0` and close `NO_CONTROL`.

## Recovery classification

For the OOF-CAL winner:

- **FULL_PARETO_RECOVERY** if `Brier<=C0`, `NLL<=C0`, and `ECE<=C0`, with all decision metrics preserved;
- **PARTIAL_CONTROL** if eligible but at least one canonical probability inequality fails;
- **NO_CONTROL** if no nonzero eta is eligible.

The classification and eta are frozen before TEST.

## TEST audit

After OOF-CAL freezes eta/status:

1. derive full-CAL F2 direction;
2. fit full-CAL C3 exactly as F2.5/F2.6;
3. apply frozen eta to TEST with the full-CAL F2 dose;
4. report C0, C1, C3 and F2.9 metrics;
5. decompose Brier into true-class and wrong-class terms, and report rank-3+ concentration;
6. report 1000 paired-bootstrap 95% intervals for F2.9 minus C3 and F2.9 minus C0 on NLL, Brier, true_sq, wrong_sq and rank3plus_sq;
7. TEST cannot change eta or closure status.

## Interpretation discipline

F2.9 tests one specific structural hypothesis: **protecting the entire runner-up coordinate removes the main damage mechanism of F2.7**.

A positive result would show that rank-2 isolation is enough to make concentration control actionable. It would not prove that every runner-up should always be protected in other datasets or models.

A negative result would imply that harmful/useful probability mass is still mixed inside rank 3+, or that the NLL/Brier tradeoff cannot be solved by monotone within-subspace diffusion alone.

Do not introduce runner-up reliability thresholds, class-specific eta, separate head/tail eta, learned diffusion maps, or additional calibration parameters inside F2.9. Those require a later separately frozen checkpoint.