# F2.10 — Adaptive Protected-Prefix / Candidate-Set Reliability Identification — FINAL

Status: **CLOSED / INTERNALLY_REPLICATED_PREFIX_IDENTIFICATION**.

Valid workflow run: `34493739696`.
Artifact: `10158925964`, SHA256 `72dbb71476429b306d22241c04336887b3ff137676f1f4e21fdec491a4d4b3e0`.

No adaptive prefix threshold, diffusion strength or probability correction is adopted in F2.10.

## Scientific question

F2.9 showed that protecting rank 2 eliminates the runner-up damage but leaves a dominant residual failure when the true class is at ranks 3--5. F2.10 therefore asked whether the size of the useful candidate prefix is itself predictable from label-free score geometry.

For frozen C3 ranking `c_(1),...,c_(K)`, cumulative targets were

`C_m = 1{Y in top-m}`, for `m in {2,3,5}`,

and conditional hazards were

`H_r = 1{Y=c_(r) | Y not in top-(r-1)}`, for `r=2,3,4,5`.

## Leakage control

The audit used:

- 5 outer CAL folds, seed `20260910`;
- inner 4-fold C3 cross-fitting, seed `20261001`, to create reliability-training features;
- TEST opened only after the OOF-CAL status was frozen.

The primary feature dictionary was deliberately score-centric following F2.8: sorted C3 probabilities `p1...p6`, adjacent gaps/log-ratios, cumulative top-m masses, suffix HHI/entropy, residual-share summaries, and the already-frozen F2/C3 dose-state `(a,q)`.

No target-encoded pair priors were used.

## OOF-CAL cumulative-prefix identification

All three cumulative targets passed every frozen criterion.

| target | prevalence | AUROC | AP | AP lift | Delta logloss vs constant | top-10% enrichment |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 0.489980 | **0.756791** | 0.765406 | **1.5621x** | **-0.113347** | **1.8781x** |
| C3 | 0.564417 | **0.759852** | 0.813786 | **1.4418x** | **-0.112515** | **1.6957x** |
| C5 | 0.641922 | **0.751608** | 0.846970 | **1.3194x** | **-0.097504** | **1.4973x** |

Bootstrap 95% intervals:

### C2
- AUROC `[0.743435, 0.770089]`;
- AP lift `[1.52109, 1.60032]`;
- logloss delta `[-0.12541,-0.10097]`.

### C3
- AUROC `[0.746526,0.773410]`;
- AP lift `[1.41176,1.47292]`;
- logloss delta `[-0.12567,-0.09833]`.

### C5
- AUROC `[0.737875,0.765513]`;
- AP lift `[1.29747,1.34286]`;
- logloss delta `[-0.10879,-0.08643]`.

Hence OOF-CAL froze

`PREFIX_IDENTIFIED`.

## Coverage geometry

The actual OOF candidate-set coverage is:

- true class in top 2: `48.998%`;
- true class in top 3: `56.442%`;
- true class in top 5: `64.192%`.

Therefore the incremental true-class mass is not confined to the runner-up:

- exactly rank 3 contributes about `7.44` percentage points beyond top 2;
- ranks 4--5 contribute about `7.75` additional points beyond top 3.

On TEST the same structure persists:

- top 2: `48.875%`;
- top 3: `55.706%`;
- top 5: `63.824%`;

so rank 3 adds about `6.83` points and ranks 4--5 add another `8.12` points.

This directly supports F2.9's conclusion that useful candidate-set probability extends materially through ranks 3--5.

## Hazard identification

OOF-CAL conditional hazards:

| hazard | at-risk n | prevalence | AUROC | AP lift | Delta logloss | status |
|---|---:|---:|---:|---:|---:|---|
| H2 | 3143 | 0.20649 | **0.70608** | **2.0227x** | **-0.04760** | IDENTIFIED |
| H3 | 2494 | 0.14595 | **0.69090** | **2.1403x** | **-0.02918** | IDENTIFIED |
| H4 | 2130 | 0.09671 | **0.63245** | **1.6000x** | **-0.00489** | IDENTIFIED |
| H5 | 1924 | 0.08992 | 0.59054 | 1.3891x | **+0.00305** | NOT IDENTIFIED |

Thus there is clear rank-specific reliability through at least rank 4 under the frozen CAL rule. H5 fails because its logistic prediction worsens binary log loss relative to the constant baseline, despite AUROC/AP enrichment being positive.

TEST H5 happens to look better (AUROC `0.62727`, logloss delta `-0.00820`), but TEST is replication-only and cannot rescue the OOF failure. H5 therefore remains **not identified** in F2.10.

## TEST cumulative replication

All three CAL-identified cumulative targets satisfy the frozen TEST replication conditions:

| target | prevalence | AUROC | AP lift | Delta logloss | top-10% enrichment |
|---|---:|---:|---:|---:|---:|
| C2 | 0.488753 | **0.774235** | **1.5878x** | **-0.124852** | **1.8787x** |
| C3 | 0.557055 | **0.769884** | **1.4565x** | **-0.113320** | **1.6887x** |
| C5 | 0.638241 | **0.759117** | **1.3256x** | **-0.095030** | **1.4947x** |

Therefore final status is

**`INTERNALLY_REPLICATED_PREFIX_IDENTIFICATION`**.

This is internal same-distribution/split-lineage replication, not external-domain invariance.

## Monotonicity / coherence

The three cumulative models were fit independently, so no structural constraint forced

`s2 <= s3 <= s5`.

Nevertheless OOF-CAL violation rate was only

`5.276%`,

with mean total violation magnitude

`0.000363`.

On TEST the violation rate fell to

`4.458%`,

with mean total violation

`0.000275`.

Thus the independent cumulative estimators already form an approximately coherent nested reliability ladder. However the nonzero violations mean F2.10 does **not** authorize direct use as an adaptive prefix decision rule without a separate coherence/decision checkpoint.

## Nonlinear corroboration

The fixed HistGradientBoosting models were consistently weaker than the primary logistic models:

- C2 HGB AUROC `0.73926` vs logistic `0.75679`;
- C3 HGB `0.73958` vs logistic `0.75985`;
- C5 HGB `0.73195` vs logistic `0.75161`.

Hence the identification does not depend on a flexible nonlinear learner. The useful candidate-prefix state is largely captured by a low-complexity function of current score geometry.

## Scientific interpretation

F2.7--F2.10 now support the following progression:

1. blind non-top diffusion confounds useful and harmful residual mass;
2. runner-up reliability is label-free identifiable;
3. protecting rank 2 removes most of the initial damage but not all of it;
4. the remaining useful candidate-set signal extends through ranks 3--5;
5. whether the true class is already inside top 2, top 3 or top 5 is strongly predictable from inference-time score geometry;
6. rank hazards H2--H4 are also predictably identifiable, while H5 is not stable under the frozen OOF rule.

The correct latent object is therefore no longer a single runner-up flag. It is better represented as a **candidate-prefix survival process** over rank:

`S_m(O) = P(Y notin top-m | O)`

or equivalently cumulative inclusion probabilities

`F_m(O) = P(Y in top-m | O)`.

The hazards

`h_r(O)=P(Y=rank r | Y notin top-(r-1),O)`

provide the natural discrete-time factorization

`P(Y=rank r | O) = h_r(O) * prod_{j<r}(1-h_j(O))`.

F2.10 empirically supports this viewpoint through ranks 2--4, with weaker/unstable evidence at rank 5.

## Closure

1. **F2.10 CLOSED / INTERNALLY_REPLICATED_PREFIX_IDENTIFICATION.**
2. C2, C3 and C5 all pass OOF-CAL and replicate on TEST.
3. H2, H3 and H4 are identified on OOF-CAL; H5 is not.
4. Candidate-prefix geometry is mostly predictable from low-complexity score features.
5. Independent cumulative predictions are already nearly monotone, but not exactly coherent.
6. No adaptive protection threshold or diffusion rule is authorized here.
7. Do not use TEST to select prefix thresholds or rescue H5.

## Highest-value next checkpoint

The next justified step is **F2.11 — Coherent Candidate-Prefix Survival Model / Decision Audit** before reopening diffusion.

The goal should be to replace three independent cumulative logits by one coherent discrete-rank survival model whose outputs satisfy

`0 <= F2 <= F3 <= F5 <= 1`

by construction, then cross-fit a decision rule for the smallest protected prefix. Only after that rule is frozen on CAL should a later intervention test selective diffusion outside the predicted protected prefix.