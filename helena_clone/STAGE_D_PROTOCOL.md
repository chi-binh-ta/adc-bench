# Stage D — Multi-seed Ensemble Protocol

Status: FROZEN BEFORE RESULTS.

## Inherited state
- Stage A closed: m = 16,384.
- Stage B closed: scalar tau = .694.
- Stage C closed early: no confusion-aware margin winner.
- QN-v2 remains fixed-step L-BFGS, memory=5, scale=.25, no line search.
- Each seed uses WCE_18 -> SoftMacroF1_9 -> Adaptive_tau=.694,6.

## Source-derived ensemble rules
The source proposes two ensemble constructions after the single-model architecture is frozen:

1. Soft voting

    p_soft(y|x) = (1/S) sum_s p_s(y|x).

2. Coherent logit ensemble

    ztilde_s = z_s - mean_k(z_sk),
    zbar = (1/S) sum_s ztilde_s,
    p_logit = softmax(zbar).

The source recommends training 5–10 full-data seeds and testing both constructions. If soft voting is clearly better for pure metrics, it may be used as the benchmark ensemble while coherent logits remain the structural/theoretical representation for subsequent G2 work.

## Clone-v2 frozen seed set

    seeds = [42, 123, 456, 789, 2026]

The seed set is fixed before results. No seed is dropped or selected by performance.

The train/meta/cal/test split itself remains the frozen numeric-label split. The experiment seed changes the nested landmark trajectory and RSP sketch, not the data split, which is required so predictions are aligned sample-by-sample for ensembling.

## Ensemble sizes
Final ensemble size is fixed at S=5.

Prefixes S=2 and S=3, in the fixed order [42,123,456,789,2026], are logged only as saturation diagnostics. They cannot be selected as the final ensemble size.

## Calibration-first discipline
1. Train all five seeds independently.
2. Save aligned calibration and test logits for each seed.
3. Aggregate CAL first.
4. Compute per-seed metrics plus soft-vote and coherent-logit ensemble metrics at S in {2,3,5}.
5. Freeze the Stage-D interpretation using CAL only.
6. Only after that decision is frozen, evaluate the already-frozen S=5 ensemble variants on TEST. Test metrics cannot change the ensemble method or seed set.

## Stage-D gate
Let `mean_single` be the arithmetic mean of the five single-model CAL metrics.

An S=5 ensemble variant passes the Stage-D utility gate if:
- NLL < mean_single_NLL,
- Brier < mean_single_Brier,
- Accuracy >= mean_single_Accuracy - 5e-4,
- MacroF1 >= mean_single_MacroF1 - 5e-4,
- Tail20F1 >= mean_single_Tail20F1 - 5e-4.

This makes NLL/Brier the expected variance-reduction gain while preventing the ensemble from buying probability quality by materially damaging ranking/class-balance.

## Soft-vote vs coherent-logit interpretation
- If one method is no worse than the other by 5e-4 in Accuracy and is strictly better in both NLL and Brier, it is the benchmark ensemble.
- If neither method satisfies that relation, record a Pareto tie rather than cherry-picking one metric.
- Regardless of benchmark choice, coherent logits remain the Stage-E structural input because G2 acts on a logit/edge field.

## G2 separation
The historical source places G2 after the coherent-logit average. Exact historical G2 internals were not recovered in Reconstruction Clone v2. Therefore Stage D is evaluated pre-G2; Stage E owns the subsequent calibration geometry. This is an explicit clone-specific modularization, not a claim about the lost historical runner.
