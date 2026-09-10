# F2.7 diagnostic repair note

This note is written **after run 3 produced the frozen OOF-CAL decision `NO_CONTROL`**.

Run 3 (`34486795737`) is scientifically valid for model selection: C3 replayed exactly, the weak-order invariant passed, all frozen eta candidates were evaluated, and no nonzero eta passed the preregistered eligibility gate.

A diagnostic-only defect was then noticed: `mean_residual_entropy` used `1-p_top` as a denominator. For samples where `p_top` rounded to exactly 1 in float64 while tiny non-top probabilities remained representable, catastrophic cancellation made the normalized residual vector invalid and generated huge negative entropy values.

`mean_residual_entropy` was **not used anywhere in the eligibility gate, eta selection, recovery classification, TEST opening rule, or bootstrap**. Therefore it cannot have affected the `NO_CONTROL` decision.

A repair rerun will:

1. compute residual-simplex entropy using the directly summed non-top probability mass;
2. keep every scientific gate, eta value, transformation, seed, split and comparison unchanged;
3. add post-hoc, non-selection diagnostics by the true class's C3 rank (`rank1`, `rank2`, `rank3-5`, `rank6+`) to explain why decreasing wrong-class squared mass can still worsen Brier.

These rank-strata diagnostics are explicitly post-hoc and cannot change the F2.7 decision.