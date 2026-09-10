# Helena Stage F — Terminal Closure and Packaging Decision

Status: **LOCKED AFTER F2.11**.

Terminal run: `34495356271`.
Terminal artifact: `10159606990`.
Terminal closure report: `results/F2_11_TERMINAL_ADAPTIVE_PREFIX.md`.

## Official deployed system

**Canonical v2**

`coherent centered-logit S=5 -> global G2`

with the frozen global G2 coefficients inherited from Stage E:

`alpha0=-0.9026209634632217`

`alpha1=+0.765345197612019`.

This remains the official packaged Helena model for the present study.

## Why the F2 system is not deployed

F2 established small reproducible ranking/class-balance leverage from the unique-margin residual direction. The later C3 state-conditioned calibration recovered/improved NLL and ECE, but Brier remained slightly worse than Canonical v2.

F2.7--F2.11 progressively tested whether the Brier loss could be removed by protecting useful ranked candidate mass and flattening only the remaining residual tail:

- blind non-top diffusion;
- rank-2-protected diffusion;
- runner-up reliability identification;
- candidate-prefix reliability identification;
- terminal coherent adaptive-prefix diffusion.

F2.10 strongly identified the candidate-prefix state, but F2.11 found no nonzero alpha/eta pair satisfying the simultaneous Canonical-v2 Pareto gate for NLL and Brier under nested CAL selection.

Therefore identification success did not translate into an admissible probability intervention.

## Final empirical anatomy

The Stage F evidence supports the following empirical decomposition for Helena within the tested family:

`decision geometry` and `probability geometry` are partially separable;

`state-conditioned G2` can improve log-score/ECE while worsening quadratic score;

`wrong-class concentration` is real but overlaps useful probability mass assigned to plausible non-top candidates;

protecting progressively larger discrete ranked prefixes reduces this damage but does not eliminate the true-class / wrong-class cancellation;

adaptive prefix identification is possible, yet discrete prefix-preserving power diffusion still fails the joint NLL/Brier Pareto criterion.

A concise study-level statement is:

> Within the tested Helena post-calibration family, probability uncertainty is distributed along the rank spectrum rather than cleanly decomposing into a finite useful prefix and a harmless noise tail. Local rank-subspace flattening does not provide a simultaneous NLL/Brier Pareto improvement.

This statement is specific to the tested model/intervention family and is not a universal theorem for multiclass calibration.

## Packaging of Stage F results

The main method/result section should retain Canonical v2 as the final Helena system.

F2.5--F2.11 should be packaged under **Analytical Insights / Ablation Anatomy** with the following narrative spine:

1. intervention reveals small ranking leverage;
2. state-conditioned calibration exposes the NLL/Brier tradeoff;
3. proper-score decomposition localizes the tradeoff;
4. naive concentration control fails because non-top mass contains useful candidate information;
5. runner-up and candidate-prefix states are statistically identifiable;
6. even a coherent adaptive-prefix terminal intervention does not cross the joint Pareto boundary.

This provides a principled negative result rather than an abandoned tuning chain.

## Frozen closure policy

- Do not reopen alpha, eta, lambda, prefix set, C3 basis, or hazard family inside Stage F2.
- Do not create F2.12+ as a rescue continuation.
- Any future work on alternative multiclass calibration geometry must be declared a **new research family/stage**, with a new protocol and fresh validation strategy.
- Because Helena TEST has been used throughout the diagnostic lineage, future model selection requires a fresh nested/repeated-CV design or new holdout rather than further tuning against this TEST split.