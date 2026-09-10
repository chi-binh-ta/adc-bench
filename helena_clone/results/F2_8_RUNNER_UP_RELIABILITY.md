# F2.8 — Runner-Up Reliability Identification Audit — FINAL

Status: **CLOSED / INTERNALLY_REPLICATED_IDENTIFICATION**.

Valid workflow run: `34489209150`.
Artifact: `10157051714`, SHA256 `5f6fbd0ed08de8a395b4c62c99cf518ecca49beff3fc9c6f4ce00606a1b48563`.

No probability correction, runner-up-protection threshold, F2 intervention parameter or C3 coefficient family is adopted/retuned in F2.8.

## Question

F2.7 showed that blind non-top diffusion fails because the non-top simplex mixes harmful wrong-class concentration with useful runner-up signal. F2.8 asks whether that useful runner-up component is identifiable from label-free inference-time observables.

For C3 runner-up class `r_i`, the target is

`R_i = 1{Y_i = r_i}`,

and the desired score is

`s_i ≈ P(Y_i = r_i | O_i)`.

The primary target is over all samples; a misclassified-only analysis is diagnostic because top-1 correctness is unknown at inference.

## Leakage control

The audit used a nested construction:

- 5 outer CAL folds, seed `20260910`;
- within each outer-fit set, 4-fold C3 cross-fitting, seed `20260928`, for reliability-training features;
- cross-fitted target encoding for pair/confusion priors, seed `20260929`;
- TEST opened only after the OOF-CAL identification status was frozen.

The reconstructed OOF C3 replayed F2.5/F2.6 exactly for NLL, Brier, ECE and Macro-F1; all replay gaps were zero at reported precision.

## OOF-CAL primary result

Runner-up-is-true prevalence:

`0.1327198364`.

Among samples for which C3 top-1 is wrong, runner-up-is-true prevalence:

`0.2064906141`.

The preregistered full standardized logistic reliability model achieved:

- AUROC `0.6837668384`;
- AP `0.2517517576`;
- AP lift `1.8968660934`;
- binary log loss `0.3648161393`;
- foldwise constant-prevalence baseline log loss `0.3920369508`;
- Delta log loss vs baseline `-0.0272208116`;
- binary Brier `0.1088709817`;
- top-decile runner-up-positive rate `0.3190184049`;
- top-decile enrichment `2.4036979969`;
- misclassified-only AUROC `0.6777121795`;
- misclassified-only AP `0.3914786414`.

Bootstrap 95% intervals, 1000 draws:

- AUROC `[0.6620070, 0.7041835]`;
- AP lift `[1.7226232, 2.1050117]`;
- Delta log loss vs baseline `[-0.0340455, -0.0202723]`.

All frozen primary identification conditions pass. Therefore OOF-CAL status was frozen as

`PRIMARY_IDENTIFIED`.

## TEST internal replication

TEST runner-up-is-true prevalence:

`0.1265848671`.

Among TEST misclassified samples:

`0.1984610452`.

The CAL-frozen full logistic reliability model achieved on TEST:

- AUROC `0.6859751058`;
- AP `0.2356146840`;
- AP lift `1.8613179402`;
- log loss `0.3554745991`;
- frozen constant-CAL-prevalence baseline log loss `0.3800084079`;
- Delta log loss `-0.0245338088`;
- Brier `0.1052712586`;
- top-decile positive rate `0.2862985685`;
- top-decile enrichment `2.2617124394`;
- misclassified-only AUROC `0.6835830372`;
- misclassified-only AP `0.3827375712`.

TEST bootstrap 95% intervals:

- AUROC `[0.6706348, 0.7023225]`;
- AP lift `[1.7342534, 2.0202742]`;
- Delta log loss vs frozen constant baseline `[-0.0296719, -0.0197257]`.

All preregistered TEST replication conditions pass. Final status is therefore

**`INTERNALLY_REPLICATED_IDENTIFICATION`**.

This is internal replication within the Helena distribution/split lineage, not external-domain invariance.

## Which observable groups carry the signal?

Feature groups were:

- G1 score/residual geometry;
- G2 F2 dose / C3 latent state;
- G3 seed/ensemble runner-up stability;
- G4 TRAIN-prototype pair geometry;
- G5 cross-fitted confusion/reliability priors.

Drop-one attribution relative to the preregistered full logistic model:

| group removed | Delta AUC full - no-group | Delta AP | Delta logloss no-group - full | incremental? |
|---|---:|---:|---:|---|
| G1 score | **+0.042274** | **+0.035485** | **+0.011771** | **YES** |
| G2 state | +0.000322 | -0.000109 | -0.000023 | no |
| G3 seed | +0.000568 | +0.002770 | +0.000172 | no |
| G4 representation | -0.001622 | +0.000208 | -0.000506 | no |
| G5 pair priors | -0.002108 | +0.001820 | -0.000414 | no |

Only G1 passes the frozen predictive-increment criterion.

The score-only logistic model is itself strong:

- OOF AUROC `0.6856324769`;
- AP `0.2498882353`;
- AP lift `1.8828250703`;
- log loss `0.3648999667`;
- misclassified-only AUROC `0.6747250412`.

Its AUROC is slightly higher than the full model's, while the full model has slightly better AP/log loss. F2.8 does not post-hoc replace the preregistered primary model; the comparison instead shows that almost all identifiable runner-up reliability is already contained in current C3 score geometry.

The fixed nonlinear HGB model was weaker than the logistic model (OOF AUROC `0.6647093510`; TEST AUROC `0.6681472031`), so the identification is not dependent on a flexible nonlinear learner.

## Post-hoc univariate diagnostic

This diagnostic was computed after closure and did not affect any gate. It is archived separately in `results/f2_8_univariate_feature_diagnostics.csv`.

Strongest individual OOF observables by oriented AUROC were:

1. runner-up probability `p2`: `0.693384`;
2. `p2-p3` gap: `0.669732`;
3. runner-up residual share `p2/(1-p1)`: `0.648357`;
4. residual HHI: `0.645381`;
5. residual entropy, reverse-oriented: `0.640414`;
6. seed mean rank of the ensemble runner-up, reverse-oriented: `0.612340`;
7. `log(p2/p3)`: `0.610039`.

F2 latent/intervention dose and q-state are individually nonzero but weaker (`~0.590` AUROC) and add essentially no conditional predictive value once score geometry is present.

Thus a useful runner-up is characterized primarily by a **strong, separated second probability mass**, especially relative to rank 3, rather than by generic ensemble instability or class-pair priors.

## Scientific interpretation

F2.7's mixture problem is now partially resolved:

`non-top mass = harmful concentration + useful runner-up signal`,

and the useful component is statistically distinguishable from label-free score geometry with substantial out-of-fold and TEST replication.

The important result is not that a new latent network has been found. The opposite simplification is supported:

`runner-up reliability is largely observable in the existing probability geometry`.

This also refines the role of the earlier latent state q. q was useful for state-conditioned C3 calibration in F2.5/F2.6, but it is not the main source of incremental information for identifying whether the runner-up itself is correct.

## Closure

1. **F2.8 CLOSED / INTERNALLY_REPLICATED_IDENTIFICATION.**
2. Runner-up correctness can be predicted label-free within Helena with OOF AUROC about `0.684` and TEST AUROC about `0.686`.
3. The signal remains similarly strong within truly misclassified samples (`0.678` OOF, `0.684` TEST).
4. Top-decile enrichment is about `2.40x` OOF and `2.26x` TEST.
5. G1 score/residual geometry is the only clearly incremental feature family under the frozen criterion.
6. No intervention is authorized in F2.8.
7. Do not post-hoc choose a protection threshold from TEST.

## Highest-value next checkpoint

A later **F2.9 — Runner-Up-Protected Concentration Control** may now be scientifically justified, but it must be separately frozen.

The natural intervention should use a cross-fitted reliability score to protect high-reliability runner-up mass while diffusing only residual mass judged unlikely to contain the true class. Threshold/soft-gating form and control strength must be chosen from CAL cross-fitting only; TEST must remain replication-only. Because G1 carries essentially all predictive information, a complexity audit should decide before F2.9 whether to freeze the preregistered LOGIT_FULL score or deliberately open a separate simplification checkpoint for a score-geometry-only reliability function.