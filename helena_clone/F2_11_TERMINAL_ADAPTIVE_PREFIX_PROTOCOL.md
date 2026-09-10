# F2.11 — Terminal Coherent Adaptive-Prefix Intervention

Status: **FROZEN BEFORE RESULTS**.

Branch: `helena-reconstruction-clone-20260909`.

This is the **terminal checkpoint of Stage F2**. No F2.12+ rescue/tuning chain is authorized by this protocol. F2.11 integrates rule identification, hyperparameter selection, intervention execution, and the final packaging verdict.

## Inherited facts

- Canonical v2 deployment baseline is `coherent S=5 -> global G2` (C0).
- F2 froze the decision intervention at `lambda=-0.05` and produced small positive ranking/class-balance changes.
- F2.5/F2.6 found a dose-conditioned four-parameter C3 G2 map that improves NLL/ECE but leaves a small Brier penalty.
- F2.7 blind non-top diffusion failed.
- F2.8 showed runner-up reliability is label-free identifiable.
- F2.9 showed protecting rank 2 removes much of the diffusion damage but useful probability mass extends through ranks 3--5.
- F2.10 established internally replicated candidate-prefix identification for `P(Y in Top-m | O)` at `m={2,3,5}` and useful rank hazards through at least ranks 2--4.

F2.11 does not reopen F2 direction, lambda, C3 basis, C3 dose state, or seed ensemble.

## Terminal scientific question

Can a **coherent sequential rank-hazard model** choose a protected prefix per sample, after which diffusion of only the remaining deeper tail produces a genuine proper-score Pareto improvement over Canonical v2 while preserving F2 ranking gains?

If YES, package the final Helena system as an adaptive-prefix F2/C3 system.
If NO, lock Canonical v2 as the official deployment system and archive F2.5--F2.11 as Analytical Insights / Ablation Anatomy.

## Phase 1 — coherent sequential hazard model

For a C3 probability vector, let `rank_y` be the rank of the true class. Fit four binary logistic hazards from label-free F2.10 score-geometry/state features:

`h_1(O) = P(rank_y = 1 | O)`

and for `r=2,3,4`,

`h_r(O) = P(rank_y = r | rank_y >= r, O)`.

Each hazard uses standardized logistic regression only:

`LogisticRegression(C=1, L2, solver=lbfgs, max_iter=3000)`.

No C grid, nonlinear rescue model, pair prior, representation feature, or TEST-guided modification is allowed.

The coherent cumulative candidate-set probabilities are constructed by survival factorization:

`F_m(O) = 1 - product_{r=1..m} (1-h_r(O))`, for `m=2,3,4`.

Thus `F_2 <= F_3 <= F_4` by construction.

## Observable dictionary

Use exactly the frozen F2.10 feature dictionary:

- sorted C3 probabilities `p1..p6`;
- consecutive gaps `g12..g56`;
- consecutive log-ratios `lr12..lr56`;
- top-2/top-3/top-5 cumulative probability mass;
- conditional shares `p2_resid1`, `p3_resid2`, `p5_resid4`;
- suffix HHI/entropy after ranks 1,2,3,5;
- frozen F2 dose `a_i` and standardized C3 state `q_i`.

No target-encoded pair prior is used in F2.11.

## Adaptive protected-prefix rule

For risk tolerance `alpha`, define threshold `q_alpha = 1-alpha` and

`m_i*(alpha) = min {m in {2,3,4}: F_m(O_i) >= q_alpha}`.

If the set is empty, use capped fallback `m_i*=4` and record a coverage-shortfall flag `F_4 < 1-alpha`.

Frozen alpha grid:

`{0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}`.

No post-hoc alpha interpolation is allowed.

## Phase 2 — selective deeper-tail diffusion

For each sample sort C3 probabilities as

`p_(1) >= ... >= p_(K)`.

Preserve the selected prefix exactly:

`p'_(r)=p_(r)` for `r <= m_i*`.

Let deeper-tail mass be

`M_i = sum_{r>m_i*} p_(r)`

and normalized tail

`u_(r)=p_(r)/M_i` for `r>m_i*`.

Use the inherited F2 dose-gated power diffusion

`gamma_i = 1 - eta*a_i`,

`u'_(r) proportional to u_(r)^gamma_i`,

then restore the exact same total tail mass `M_i`.

Frozen eta grid, inherited from F2.7/F2.9:

`{0,.01,.02,.03,.05,.075,.10,.15,.20,.30,.40}`.

Any pair producing non-positive gamma, row-sum failure, protected-prefix drift, top-1 change, or a strict rank-order reversal greater than `64*eps_float64` is invalid rather than repaired.

Because top-1 probability and class are preserved for every valid pair, Accuracy, Macro-F1, Balanced Accuracy, Tail20-F1, and confidence-based ECE must remain exactly those of C3 up to numerical tolerance.

## Nested-CAL selection — mandatory anti-tuning design

Use outer 5-fold stratified-by-class CAL splitting, seed `20260910`.

For each outer fold:

1. construct C3 probabilities/features for outer-fit samples by inner 4-fold C3 cross-fitting using the frozen F2.10 lineage;
2. create a second 4-fold selection cross-fit on those outer-fit rows, seed `20261011`, to obtain held-out hazard probabilities `h1..h4` for every outer-fit sample;
3. from those held-out hazards compute coherent `F2,F3,F4` and evaluate every frozen `(alpha,eta)` pair on the outer-fit selection predictions;
4. compare candidates to Canonical v2 probabilities on the same outer-fit samples;
5. freeze one `(alpha,eta)` pair without seeing outer-validation;
6. fit hazards on all cross-fitted outer-fit feature rows and apply them to outer-validation C3 features;
7. apply the frozen pair to outer-validation probabilities;
8. concatenate all outer-validation predictions.

Thus the terminal OOF estimate evaluates a rule whose `(alpha,eta)` was chosen without using the labels of its outer-validation fold.

## Inner pair eligibility and selection

Within each outer-fit selection layer, a nonzero-eta pair is eligible only if all hold relative to Canonical v2 C0 on that selection set:

- `Delta NLL = NLL(candidate)-NLL(C0) <= 0`;
- `Delta Brier = Brier(candidate)-Brier(C0) <= 0`;
- Accuracy equals C3 within `1e-12`;
- Macro-F1 equals C3 within `1e-12`;
- Balanced Accuracy equals C3 within `1e-12`;
- Tail20-F1 equals C3 within `1e-12`;
- all probability/ranking invariants pass.

Among eligible pairs choose lexicographically:

1. minimum Brier;
2. minimum NLL;
3. minimum eta;
4. minimum alpha.

If no nonzero pair is eligible, freeze fallback `eta=0, alpha=0.25` for that outer fold. Alpha is irrelevant at eta=0.

## Terminal OOF-CAL verdict

After concatenating outer-validation predictions, compare nested F2.11 to Canonical v2 C0.

`CAL_TERMINAL_PASS` requires all:

- `Delta NLL <= 0`;
- `Delta Brier <= 0`;
- Accuracy equals C3 OOF within `1e-12`;
- Macro-F1 equals C3 OOF within `1e-12`;
- Balanced Accuracy equals C3 OOF within `1e-12`;
- Tail20-F1 equals C3 OOF within `1e-12`;
- top-1 probability/class invariants pass.

Paired bootstrap with 1000 draws, seed `20261012`, is reported for Delta NLL and Delta Brier versus C0 but is descriptive rather than an added pass gate. The binary terminal decision follows the preregistered point-estimate Pareto gate above.

## Full-CAL deployment-pair freeze

Independently of the nested OOF verdict, use the complete outer-OOF C3 probabilities and outer-OOF coherent hazard predictions to evaluate the same frozen `(alpha,eta)` grid on all CAL rows.

Freeze one global deployment pair using the identical eligibility/lexicographic rule. If none is eligible, global deployment pair is `eta=0, alpha=0.25`.

This full-CAL OOF pair is frozen before TEST is opened.

## TEST replication and final packaging verdict

After CAL decisions are frozen:

1. fit full-CAL F2 direction and C3 calibrator using the inherited procedure;
2. train h1..h4 on OOF-CAL features/targets only;
3. construct TEST features without TEST labels;
4. compute coherent TEST `F2,F3,F4`;
5. apply the globally frozen `(alpha,eta)` pair;
6. compare to Canonical v2 TEST and verify ranking invariants.

`TEST_TERMINAL_PASS` requires the same point-estimate gate:

- Delta NLL <= 0 versus C0 TEST;
- Delta Brier <= 0 versus C0 TEST;
- Accuracy/Macro-F1/Balanced Accuracy/Tail20-F1 exactly equal C3 TEST within tolerance.

Final status:

- **TERMINAL_ADOPT_ADAPTIVE_PREFIX** only if both `CAL_TERMINAL_PASS` and `TEST_TERMINAL_PASS` hold and the global frozen eta is nonzero.
- Otherwise **TERMINAL_REJECT_LOCK_CANONICAL_V2**.

TEST is historically opened from earlier Helena checkpoints, so TEST replication is internal benchmark replication, not a pristine external confirmation.

## Packaging rule — no further rescue stages

If `TERMINAL_ADOPT_ADAPTIVE_PREFIX`:

Official Helena packaged system becomes

`F2 intervention (lambda=-0.05) -> dose-conditioned C3 G2 -> coherent adaptive protected-prefix deeper-tail diffusion`.

If `TERMINAL_REJECT_LOCK_CANONICAL_V2`:

Official deployment system is permanently locked for this study at

`Canonical v2 = coherent S=5 -> global G2`.

F2.5 through F2.11 move to **Analytical Insights / Ablation Anatomy**, supporting the conclusion that multiclass probability uncertainty along the rank spectrum may resist discrete local flattening under a simultaneous NLL/Brier Pareto requirement.

No threshold relaxation, new prefix set, new alpha/eta grid, isotonic repair, extra latent network, or F2.12+ tuning is permitted after the F2.11 result.