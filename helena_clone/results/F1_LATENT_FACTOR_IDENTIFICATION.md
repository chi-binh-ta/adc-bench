# F1 — Latent Factor Identification

Status: **COMPLETED**.

System analyzed: canonical `coherent-logit S=5 -> global G2` residuals.

Residual matrices:
- soft leakage: `E[p_j | Y=k]`, diagonal removed;
- hard confusion: `P(argmax p=j | Y=k)`, diagonal removed.

Primary interpretation rank: top 5 right singular vectors `V_1,...,V_5`. CAL factors were matched to TEST factors with Hungarian matching on absolute subspace correlation and sign-aligned before interpretation.

## Factor stability

### Soft leakage
| CAL factor | matched TEST factor | alignment |
|---:|---:|---:|
| 1 | 1 | 0.9967 |
| 2 | 2 | 0.9666 |
| 3 | 3 | 0.8224 |
| 4 | 4 | 0.7639 |
| 5 | 5 | 0.4674 |

### Hard confusion
| CAL factor | matched TEST factor | alignment |
|---:|---:|---:|
| 1 | 1 | 0.9794 |
| 2 | 2 | 0.8787 |
| 3 | 3 | 0.8781 |
| 4 | 4 | 0.6145 |
| 5 | 5 | 0.3529 |

Therefore factors 1–3 are the most defensible individually; factor 4 is only moderately stable; factor 5 is not individually stable enough to name.

## Factor 1 — dominant head-attractor / difficulty-margin axis

The sign of singular vectors is arbitrary. The raw SVD orientation has negative association with prevalence; equivalently one may flip `V1` so that the positive direction corresponds to high-prevalence/head classes.

### Soft leakage V1 replicated associations
- log train support: rho CAL `-0.930`, TEST `-0.939`
- class NLL: `+0.666`, `+0.661`
- median margin: `-0.639`, `-0.657`
- mean margin: `-0.605`, `-0.598`
- near-zero margin rate: `-0.629`, `-0.648`
- negative-margin rate: `+0.500`, `+0.473`
- representation nearest-prototype cosine: `-0.590`, `-0.490`
- representation nearest-prototype distance: `+0.525`, `+0.382`
- representation separation ratio: `+0.517`, `+0.392`
- class Brier: `+0.469`, `+0.446`
- seed true-probability std: `-0.362`, `-0.269`
- absolute predicted-class calibration gap: `+0.355`, `+0.384`

All listed associations satisfy the predeclared CAL/TEST replication criterion after BH-FDR.

Group-level transfer for soft V1:
- prevalence: CAL CV R2 `0.718`, TEST R2 `0.691`, TEST corr `0.832`;
- margin: CAL CV R2 `0.450`, TEST R2 `0.576`, TEST corr `0.771`, permutation p `0.004`;
- calibration residual: CAL CV R2 `0.323`, TEST R2 `0.393`, corr `0.644`, p `0.014`;
- representation geometry: CAL CV R2 `0.195`, TEST R2 `0.140`, corr `0.384`, p `0.044`;
- raw class similarity: weak but transferable, CAL CV R2 `0.036`, TEST R2 `0.091`, corr `0.302`, p `0.002`;
- seed-instability group alone does not have positive CAL CV R2.

### Hard confusion V1 replicated associations
- log train support: rho CAL `-0.849`, TEST `-0.872`
- class NLL: `+0.769`, `+0.800`
- mean margin: `-0.758`, `-0.792`
- median margin: `-0.746`, `-0.801`
- negative-margin rate: `+0.728`, `+0.752`
- near-zero margin rate: `-0.721`, `-0.781`
- class Brier: `+0.673`, `+0.717`
- seed true-probability std: `-0.581`, `-0.536`
- q10 margin: `-0.570`, `-0.685`
- absolute predicted-class calibration gap: `+0.417`, `+0.424`
- representation prototype norm: `-0.396`, `-0.377`

Group-level margin model is the strongest robust explanatory group under the predeclared transfer rule:
- CAL CV R2 `0.356`
- TEST R2 `0.370`
- TEST corr `0.626`
- permutation p `0.026`.

Interpretation: the dominant residual mode is substantially a prevalence/difficulty/margin axis. After flipping the arbitrary singular-vector sign, high-loading destinations are predominantly head/common classes with stronger decision margins and lower classwise loss; residual probability/confusion flow is preferentially attracted toward this head side. This is not merely a calibration artifact because it appears in both soft leakage and hard argmax confusion.

## Factor 2 — geometric isolation / prototype-scale axis

Soft leakage V2 is highly stable (`0.9666`) and has replicated associations with:
- raw mean-5 centroid distance: rho CAL `-0.304`, TEST `-0.307`;
- representation prototype norm: `-0.351`, `-0.344`.

Hard-confusion V2 is also stable (`0.8787`) and independently associates with representation prototype norm:
- rho CAL `+0.283`, TEST `+0.319`.

Because factor sign is arbitrary across the two residual definitions, the common evidence is magnitude/association with prototype scale rather than the displayed sign.

Interpretation: factor 2 is the best candidate for a genuinely geometric latent factor distinct from the dominant prevalence-margin mode, but the effect is moderate and should not yet be treated as fully identified.

## Factor 3 — secondary prevalence mode in hard confusion

Hard-confusion V3 alignment is `0.8781` and has replicated association with prevalence:
- log support: rho CAL `-0.304`, TEST `-0.452`.

Soft-leakage V3 is stable enough (`0.8224`) but none of the six observable families passes the replicated feature criterion.

Interpretation: hard confusion contains a second prevalence-related direction, while the corresponding soft factor remains unexplained by the current observable set.

## Factors 4–5

- Soft V4 alignment `0.764`, hard V4 `0.614`: moderate stability but no feature-level association survives the full replication criterion.
- Soft V5 alignment `0.467`, hard V5 `0.353`: too unstable for individual semantic labeling.

These factors should remain **unidentified residual modes** rather than being given post-hoc names.

## Scientific conclusion

The prior F0 audit established that Helena residual error is low-dimensional. F1 now shows that the low-rank structure is not homogeneous:

1. `V1` is largely identifiable as a head/prevalence + decision-margin/difficulty axis, with calibration and representation geometry as correlated manifestations.
2. `V2` carries a weaker but replicated geometric signal involving raw class separation and learned-logit prototype scale.
3. `V3` contains a secondary prevalence component in hard confusion, but is not fully explained across both residual definitions.
4. `V4–V5` remain unidentified; no interpretation should be forced.
5. Seed instability correlates with V1 but is not an independent explanatory group under CAL cross-validation, so it should be treated as a consequence/co-traveler rather than a primary factor at this stage.

Important caveat: these are explanatory associations, not causal identification. Prevalence, margin, calibration residual and representation geometry are strongly correlated with one another. The next justified audit is conditional variance decomposition / residualization: determine which groups explain unique V1/V2 variance after controlling for prevalence and margin, before designing a latent-factor correction layer.
