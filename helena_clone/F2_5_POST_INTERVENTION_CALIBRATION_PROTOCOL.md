# F2.5 — Post-Intervention Calibration Recovery

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

Inherited system: canonical `coherent-logit S=5 -> global G2`.

Inherited intervention from valid F2 run `34389027159`:

- class direction construction is unchanged;
- class-level CV seed = `20260910`;
- sample outer-CAL seed = `20260910`;
- uncertainty gate `a_i = exp(-g_i / median(g_fit))` is unchanged;
- **lambda is frozen at `lambda*=-0.05`**;
- no intervention parameter may be retuned in F2.5.

## Scientific question

F2 produced a small but reproducible ranking/class-balance gain while worsening NLL and Brier. F2.5 asks whether that probability penalty is merely a consequence of applying a calibration map optimized for the pre-intervention logits.

The primary hypothesis is:

`decision intervention` and `probability calibration` can be separated.

If so, refitting only calibration after the frozen F2 intervention should retain the decision gain while recovering NLL/Brier.

## Sentiment-project structural analogy

The user's Aspect-aware Hybrid Retrieval & Sentiment Analysis project treats sentiment as context-dependent: the same review can carry different sentiment states under different aspects. The associated research formulation considered a representation of the form

`h(x,a)` and `L(x,a)=A z(x,a) + C q(x,a)`, where `q` is a supplementary hidden/nonlinear state.

F2.5 imports only the structural lesson — **conditioning on a small state can matter** — not the sentiment model itself.

For Helena the intervention produces a sample-dependent intervention dose through `a_i`. Therefore F2.5 includes one tightly constrained experimental analogue:

`alpha_r(i) = alpha_r + beta_r q_i`,

where `q_i` is a fixed standardized function of the already-frozen intervention gate. No new network, learned latent embedding, class partition, or intervention feature is introduced.

This branch is called `dose-conditioned G2`. It is an experimental hidden-state analogue, not evidence that Helena and sentiment share the same causal mechanism.

## Cross-fitting / leakage policy

Use the same 5-fold stratified CAL split as F2.

For every outer CAL fold:

1. derive the F2 direction `d_fold` from the 4/5 fit portion only using the exact valid F2 procedure;
2. compute `scale_fold = median(top2_gap(z_fit))+1e-8`;
3. intervene on both fit and validation coherent logits with **fixed `lambda=-0.05`**;
4. fit each candidate calibrator on the intervened 4/5 fit portion only;
5. apply that fitted calibrator, without refitting, to the held-out 1/5 CAL portion;
6. concatenate held-out probabilities across all five folds.

TEST labels may not affect intervention construction, calibrator selection, or coefficients. Helena TEST is already historically opened, so final TEST is an internal replication audit only.

## Models

### C0 — canonical control

No F2 intervention; frozen canonical global G2 coefficients.

### C1 — F2 fixed-G2 reference

Frozen F2 intervention with `lambda=-0.05`, followed by the old canonical global G2 coefficients. This must numerically replay valid F2 OOF-CAL metrics.

### C2 — post-intervention global-G2 refit

On each outer fit fold, refit only two parameters:

`theta_G=(alpha0, alpha1)`

by multiclass NLL on intervened fit logits. The complete-graph global G2 geometry is unchanged.

### C3 — dose-conditioned global G2

Let `a_i` be the frozen F2 uncertainty gate. On the outer fit portion define

`q_i = (a_i - mean(a_fit)) / (sd(a_fit)+1e-8)`.

The validation `q_i` uses the fit-fold mean and sd.

For global G2 bases `T0(z), T1(z)`, fit four parameters

`theta_Q=(alpha0, alpha1, beta0, beta1)`

with correction

`t_i = (alpha0 + beta0 q_i) T0_i + (alpha1 + beta1 q_i) T1_i`.

Equivalently use four fixed bases `[T0,T1,q*T0,q*T1]` and optimize multiclass NLL.

No hyperparameter controls the hidden-state form. It is deliberately rank-1 and tied only to the already-frozen intervention dose.

## Optimization

Both C2 and C3 use L-BFGS-B with analytic gradient, `maxiter=500`, `ftol=1e-13`, `gtol=1e-9`, `maxls=50`.

Initialization:

- C2 starts from canonical F2 global coefficients `[-0.9026209634632217, 0.765345197612019]`;
- C3 starts from the fitted C2 coefficients with `beta0=beta1=0`.

## OOF-CAL recovery gate

Let all deltas be relative to C0 canonical control unless stated otherwise.

A recalibrated candidate C2/C3 is eligible only if:

1. `NLL(candidate) < NLL(C1)`;
2. `Brier(candidate) <= Brier(C1)`;
3. `MacroF1(candidate) > MacroF1(C0)`;
4. `BalancedAcc(candidate) > BalancedAcc(C0)`;
5. `Tail20F1(candidate) > Tail20F1(C0)`;
6. `Accuracy(candidate) >= Accuracy(C0)-0.0005`.

Thus recalibration must actually improve probability quality over F2 while preserving a positive decision advantage over the canonical system.

Among eligible C2/C3 candidates choose:

1. minimum NLL;
2. minimum Brier;
3. minimum ECE;
4. maximum Macro-F1.

If neither C2 nor C3 is eligible, F2.5 closes `NO_RECOVERY` and C1 remains experimental only.

## Recovery classification

For the frozen OOF-CAL winner:

- **FULL_RECOVERY**: eligible and both `NLL <= C0 NLL` and `Brier <= C0 Brier`;
- **PARTIAL_RECOVERY**: eligible but one or both probability metrics remain worse than C0;
- **NO_RECOVERY**: no recalibrated candidate is eligible.

C3 is adopted over C2 only through the same predeclared OOF-CAL rule. Its extra hidden-state parameters get no preference.

## Final TEST audit

After winner and recovery status are frozen:

1. derive `d_full` and gate scale from all CAL only using the frozen F2 procedure;
2. intervene full CAL and TEST at `lambda=-0.05`;
3. refit the selected calibrator on full intervened CAL only;
4. evaluate C0, C1, C2 and C3 on TEST for audit, but TEST cannot change the winner;
5. report NLL, Brier, ECE, Accuracy, Macro-F1, Balanced Accuracy and Tail20-F1;
6. paired bootstrap 95% intervals compare the frozen winner against C0 and C1 for NLL/Brier and ranking metrics.

## Interpretation discipline

F2.5 tests whether post-intervention probability loss is calibratable. A positive result supports architectural separation of decision geometry and calibration. It does **not** establish a causal latent mechanism.

The dose-conditioned C3 branch is specifically a constrained test of the sentiment-project lesson that one global map may be insufficient when the same observation occupies different context/state regimes. If C3 does not beat C2 under cross-fitting, the hidden-state extension is rejected for F2.5 rather than enlarged.