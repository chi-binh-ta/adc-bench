# Stage F0 — Latent Residual Low-Dimensionality Audit — RESULT

Status: **NO STRONG LOW-DIMENSIONAL EVIDENCE UNDER THE FROZEN F0 GATE**.

This is empirical/statistical evidence on the finite Helena sample, not a population theorem.

## Residual object

`R[k,j] = E[1(Y=j)-p_j(X) | Y=k]` for the frozen coherent-S5 -> global-G2 system.

## Spectral summary

| split | eff-rank entropy | PR-rank | stable rank | E(3) | E(5) | E(10) | E(20) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAL | 93.279 | 90.372 | 53.422 | 0.0458 | 0.0719 | 0.1357 | 0.2601 |
| TEST | 93.211 | 90.420 | 53.626 | 0.0449 | 0.0705 | 0.1333 | 0.2570 |

## Transfer and stability gate

| r | CAL energy | TEST energy | CAL->TEST transfer | p-transfer | overlap | p-overlap | seed median overlap | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.0187 | 0.0186 | 0.0182 | 0.00020 | 0.9418 | 0.00020 | 0.9787 | NO |
| 2 | 0.0324 | 0.0319 | 0.0305 | 0.00020 | 0.6737 | 0.00020 | 0.8798 | NO |
| 3 | 0.0458 | 0.0449 | 0.0422 | 0.00020 | 0.4580 | 0.00020 | 0.7420 | NO |
| 5 | 0.0719 | 0.0705 | 0.0663 | 0.00020 | 0.4376 | 0.00020 | 0.8008 | NO |
| 10 | 0.1357 | 0.1333 | 0.1262 | 0.00020 | 0.3735 | 0.00020 | 0.7733 | NO |

## Structured fraction

- CAL: 0.849649
- TEST: 0.843707

This fraction measures how much total per-sample squared residual energy is carried by class-conditional mean residuals; it is not itself the low-rank decision gate.

## Decision

No rank r<=10 passes all six pre-frozen conditions.
Therefore the current evidence is insufficient to justify a latent-factor model solely from low-rank systematic residual structure.

## Scientific interpretation

The negative low-rank result is not the same as saying the residual is unstructured. In fact, the first one or two residual modes are highly reproducible across CAL/TEST and across the five independent optimizer seeds, and every tested CAL-learned subspace captures more TEST energy than a same-dimensional random subspace (Monte Carlo p=0.00020). However, these reproducible modes carry very little of the total systematic residual energy: only about 1.9% at r=1, 3.2% at r=2, 7.1% at r=5, and 13.3% at r=10 on TEST.

The systematic residual matrix is therefore best described as **structured but high-rank**, not globally low-rank. The entropy effective rank is about 93 and the participation-ratio rank about 90 out of an ambient 100-dimensional output space.

A future latent search should therefore not begin with a single global low-rank factor model. More plausible next hypotheses are local/block latent structure, confusion-graph communities, sparse factors, conditional factors, or nonlinear manifolds.
