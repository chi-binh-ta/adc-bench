# Helena — Canonical Final Rerun Protocol

Status: FROZEN / NO TUNING.

This run retrains the final Helena Reconstruction Clone v2 system from raw data and zero initialization. It does not reuse Stage-D logits, checkpoints, feature caches, or fitted G2 coefficients.

## Frozen architecture

- representation rank/capacity: m = 16,384
- numeric-label split semantics
- split random_state = 42 with counts train/meta/cal/test = 45,637 / 4,889 / 4,890 / 9,780
- RBF landmark pool and RSP randomization vary only with the fixed model seed
- RSP rank = 256, lambda_spec = 1e-3
- QN-v2 fixed-step L-BFGS: memory = 5, scale = .25, no hidden line search
- WCE steps = 18
- SoftMacroF1 steps = 9, lambda_F = 1
- adaptive steps = 6
- adaptive policy = scalar tau = .694
- no Stage-C confusion margin
- seeds fixed: {42, 123, 456, 789, 2026}

## Final ensemble outputs

Ranking benchmark:

    soft-vote S=5

Coherent calibrated system:

    centered coherent-logit average S=5 -> global G2

Global G2 is fitted on CAL only after all five base models are trained. TEST cannot tune any parameter.

## Reproducibility audit

The final aggregate compares rerun metrics against the previously frozen Stage-D/E outputs. These comparisons are diagnostics only; the rerun is never modified to chase the historical numbers.

Frozen reference values:
- soft-vote S5 TEST Accuracy = .36226993865030677
- soft-vote S5 TEST Macro-F1 = .20606033209756158
- coherent S5 -> global G2 TEST NLL = 2.7644007205963135
- coherent S5 -> global G2 TEST Brier = .7781448650949503
- coherent S5 -> global G2 TEST ECE = .03222390116518077
- coherent S5 -> global G2 TEST Accuracy = .3621676891615542
- coherent S5 -> global G2 TEST Macro-F1 = .20592893268833737

This is the canonical end-to-end rerun after Stages A-E are closed.
