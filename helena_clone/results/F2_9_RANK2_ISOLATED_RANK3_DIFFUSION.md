# F2.9 — Rank-2 Isolation / Selective Rank-3+ Subspace Diffusion — FINAL

Status: **CLOSED / NO_CONTROL**.

Frozen winner:

`eta*=0`.

Valid workflow run: `34491033175`.
Artifact: `10157762621`, SHA256 `ae1525370464bd8e45aa98d2da807ddbe94d3ab92267bd357105814612012f91`.

No F2.9 probability correction is adopted.

## Scientific question

F2.7 showed that blind non-top diffusion fails largely because the true class is often the runner-up and blind flattening reduces useful rank-2 probability. F2.8 showed that runner-up reliability is label-free identifiable.

F2.9 tested the simpler structural intervention requested here before using a reliability threshold:

- preserve rank 1 exactly;
- preserve rank 2 exactly;
- diffuse only the rank-3+ probability subspace.

The question was whether **rank-2 isolation alone** is sufficient to convert the F2.7 concentration-control failure into a valid Brier/NLL/ECE recovery.

## Frozen transform

For C3 probabilities sorted as

`p_(1) >= p_(2) >= ... >= p_(K)`, keep

`p'_(1)=p_(1)` and `p'_(2)=p_(2)`.

For rank `r>=3`, normalize the remaining mass

`u_(r)=p_(r)/(1-p_(1)-p_(2))`,

then use

`gamma_i=1-eta*a_i`,

`u'_(r) proportional to u_(r)^gamma_i`,

and restore the original total rank-3+ mass.

The eta grid was inherited from F2.7:

`{0,.01,.02,.03,.05,.075,.10,.15,.20,.30,.40}`.

## Reproduction and invariants

C3 replay gaps for NLL, Brier, ECE and Macro-F1 were exactly zero at reported precision.

For the frozen TEST identity winner the maximum top-1 and top-2 probability deltas are exactly zero. The transform passed the protected-rank and weak-order invariants throughout the OOF sweep.

Because top-1 class and top-1 probability are unchanged, Accuracy/Macro-F1/Balanced Accuracy/Tail20-F1 and confidence-based ECE are unchanged throughout the entire eta grid.

## OOF-CAL result

C3 reference:

- NLL `2.7397491932`;
- Brier `0.7826232438`;
- ECE `0.0286465333`;
- mean true squared term `0.6876360123`;
- mean wrong squared mass `0.0949872314`;
- mean rank-3+ squared mass `0.0150804308`.

No nonzero eta passed the preregistered eligibility gate.

### Eta curve

| eta | Delta NLL vs C3 | Delta Brier vs C3 | Delta wrong_sq | Delta true_sq | Delta rank3+_sq |
|---:|---:|---:|---:|---:|---:|
| .01 | **-0.000033** | +0.000010 | **-0.000150** | +0.000161 | **-0.000167** |
| .02 | **-0.000040** | +0.000023 | **-0.000299** | +0.000321 | **-0.000333** |
| .03 | -0.000020 | +0.000037 | -0.000446 | +0.000482 | -0.000497 |
| .05 | +0.000101 | +0.000070 | -0.000736 | +0.000806 | -0.000819 |
| .075 | +0.000403 | +0.000121 | -0.001089 | +0.001211 | -0.001213 |
| .10 | +0.000877 | +0.000183 | -0.001434 | +0.001617 | -0.001596 |
| .15 | +0.002346 | +0.000338 | -0.002094 | +0.002432 | -0.002330 |
| .20 | +0.004521 | +0.000532 | -0.002715 | +0.003246 | -0.003020 |
| .30 | +0.011046 | +0.001023 | -0.003840 | +0.004864 | -0.004266 |
| .40 | +0.020528 | +0.001630 | -0.004813 | +0.006443 | -0.005340 |

Thus rank-3+ diffusion does exactly reduce the requested concentration channel, and for very small eta even gives a small NLL improvement. Nevertheless Brier is worse for every nonzero eta because the increase in the true-class squared term remains slightly larger than the reduction in wrong-class squared mass.

The smallest Brier penalty occurs at eta=.01:

`Delta wrong_sq=-1.5004e-4`,

`Delta true_sq=+1.6051e-4`,

hence

`Delta Brier=+1.0472e-5`.

The best NLL occurs at eta=.02:

`Delta NLL=-3.9577e-5`,

but

`Delta Brier=+2.2721e-5`.

Neither is eligible because F2.9 was required to improve Brier rather than merely reduce its penalty.

## Direct comparison with F2.7

Rank-2 isolation materially improves the failure geometry, but not enough to reverse it.

At eta=.01, F2.7 had approximately:

- Delta true_sq `+0.0003869`;
- Delta wrong_sq `-0.0003580`;
- Delta Brier `+0.0000289`.

F2.9 at the same eta has:

- Delta true_sq `+0.0001605`;
- Delta wrong_sq `-0.0001500`;
- Delta Brier `+0.0000105`.

Therefore protecting rank 2 reduces the true-class damage by about **58.5%** and the net Brier penalty by about **63.8%** at eta=.01. Similar reductions persist at eta=.02 and .05.

This confirms that runner-up contamination was a major part of F2.7's failure, but not the entire cause.

## True-class-rank mechanism

Post-hoc rank-strata diagnostics were not used for eta selection.

At eta=.01:

| true-class rank | share | Delta NLL | Delta Brier | Delta p_y | Delta true_sq | Delta wrong_sq |
|---|---:|---:|---:|---:|---:|---:|
| rank 1 | 0.3573 | 0 | **-0.000102** | 0 | 0 | **-0.000102** |
| rank 2 | 0.1327 | 0 | **-0.000193** | 0 | 0 | **-0.000193** |
| rank 3--5 | 0.1519 | **+0.006816** | **+0.000859** | **-0.000551** | **+0.000994** | -0.000135 |
| rank 6+ | 0.3581 | **-0.002984** | **-0.000162** | -0.000014 | +0.000027 | **-0.000189** |

This is the decisive F2.9 result.

### Rank 1 and rank 2

The transform behaves as desired. The true-class probability is protected exactly, true-class squared error is unchanged, and Brier improves entirely by reducing concentration elsewhere.

In particular the rank-2 failure mode found in F2.7 is eliminated exactly:

`Delta p_y(rank2)=0`,

`Delta true_sq(rank2)=0`.

### Rank 3--5

The dominant remaining damage moves to the mid-rank region. When the true class is rank 3--5, it is often still a relatively large component inside the rank-3+ simplex. Flattening pushes these large residual components downward toward the rest of the tail:

`Delta p_y=-0.000551` at eta=.01,

producing

`Delta true_sq=+0.000994`.

This dominates the wrong-class concentration benefit for that stratum.

### Rank 6+

Deep-tail true classes tend to benefit from flattening: low residual components receive probability mass and the stratum has negative Delta NLL and negative Delta Brier.

Therefore the useful/harmful mixture is not merely

`rank2 vs rank3+`.

The empirical geometry is closer to

`protected high/mid-rank candidate mass + diffusible deep-tail mass`.

## Refined structural conclusion

F2.7 established:

`non-top mass = harmful concentration + useful runner-up signal`.

F2.9 refines this to:

`non-top mass = useful candidate-set signal (at least ranks 2--5) + diffusible deeper-tail concentration`.

Rank 2 is important but is not the complete boundary between useful and harmful probability concentration.

This is why unconditional rank-2 protection substantially reduces the F2.7 damage but does not produce a Pareto recovery.

## TEST audit

OOF-CAL froze `eta*=0`, therefore F2.9 on TEST is exactly C3:

- NLL `2.7620069981`;
- Brier `0.7782723678`;
- ECE `0.0227312318`;
- Accuracy `0.3621676892`;
- Macro-F1 `0.2062443951`;
- Balanced Accuracy `0.2034808168`;
- Tail20-F1 `0.1173547454`.

TEST did not select or rescue any nonzero eta.

## Scientific closure

1. **F2.9 CLOSED / NO_CONTROL**.
2. `eta*=0`; no rank-3+ diffusion is adopted.
3. Rank-2 isolation works mechanistically: it completely removes rank-2 true-class damage.
4. It reduces F2.7's net Brier penalty by roughly 64% at the smallest eta, so runner-up contamination was a major failure source.
5. The remaining failure localizes primarily to true classes at rank 3--5.
6. Therefore a hard boundary `protect rank<=2 / diffuse rank>=3` is still too crude.
7. Do not add an F2.8 reliability threshold or protect top-5 post hoc inside F2.9.

## Highest-value next checkpoint

The next checkpoint should return to identification before another intervention:

**F2.10 — Adaptive Protected-Prefix / Candidate-Set Reliability Identification Audit**.

Instead of asking only whether rank 2 is correct, estimate label-free quantities such as

`P(Y in top-m | O)` for `m in {2,3,5}`

or rank hazards

`P(Y = rank r | Y not in ranks < r, O)` for `r=2,...,5`.

The goal is to identify the smallest protected prefix needed per sample before opening any adaptive selective-diffusion family. The F2.9 data do not justify simply hard-coding top-5 protection without this audit.