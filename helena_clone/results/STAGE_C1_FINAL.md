# Stage C1 — Seed42 Confusion-Aware Margin — FINAL

Status: CLOSED — NO WINNER.

Frozen inherited setup:
- m = 16,384
- scalar tau = .694 (Stage B frozen)
- QN-v2: fixed-step L-BFGS, memory=5, scale=.25
- seed = 42
- confusion sets built from meta predictions of the frozen scalar Stage-B baseline
- r in {3,5}
- lambda_M in {.025,.05,.10,.20}
- clone-specific margin height m_ij=.5 on targeted confusion pairs
- calibration split used for candidate screening; test untouched.

Control replay check:
- accuracy gap = 0
- Macro-F1 gap = 0
- BalancedAcc gap = 0
- Tail20 gap = 0
- NLL/Brier/ECE gaps = 0

Therefore the Stage-C runner exactly reproduces the frozen baseline when lambda_M=0; the negative result is attributable to the margin intervention rather than replay drift.

C1 gate:
1. delta Accuracy > 0
2. delta targeted-confusion-rate < 0
3. delta Macro-F1 >= -5e-4
4. delta Tail20-F1 >= -5e-4

## Calibration results

| role | r | lambda_M | Accuracy | MacroF1 | BalAcc | Tail20 | target-confusion | dAcc | dMacro | dTail | dTargetConf | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| control | 3 | 0.000 | 0.354601 | 0.197362 | 0.196521 | 0.112449 | 0.234356 |  |  |  |  |  |
| candidate | 3 | 0.025 | 0.354192 | 0.196309 | 0.195379 | 0.108493 | 0.236605 | -0.000409 | -0.001054 | -0.003956 | +0.002249 | NO |
| candidate | 3 | 0.050 | 0.354192 | 0.194737 | 0.193069 | 0.105933 | 0.240900 | -0.000409 | -0.002625 | -0.006516 | +0.006544 | NO |
| candidate | 3 | 0.100 | 0.355010 | 0.195780 | 0.192412 | 0.100377 | 0.244581 | +0.000409 | -0.001583 | -0.012072 | +0.010225 | NO |
| candidate | 3 | 0.200 | 0.351125 | 0.188911 | 0.185798 | 0.108420 | 0.257260 | -0.003476 | -0.008452 | -0.004028 | +0.022904 | NO |
| candidate | 5 | 0.025 | 0.353988 | 0.195585 | 0.194749 | 0.105055 | 0.287321 | -0.000613 | -0.001778 | -0.007393 | +0.002454 | NO |
| candidate | 5 | 0.050 | 0.353988 | 0.194627 | 0.193335 | 0.105055 | 0.289366 | -0.000613 | -0.002735 | -0.007393 | +0.004499 | NO |
| candidate | 5 | 0.100 | 0.352761 | 0.194007 | 0.190706 | 0.104266 | 0.298773 | -0.001840 | -0.003355 | -0.008183 | +0.013906 | NO |
| candidate | 5 | 0.200 | 0.351943 | 0.189814 | 0.187001 | 0.104817 | 0.307566 | -0.002658 | -0.007548 | -0.007631 | +0.022699 | NO |

## Result

- 0/8 candidates pass C1.
- Every candidate increases the targeted confusion rate relative to its matched control.
- The smallest confusion deterioration is still positive: +0.00224949 for r=3, lambda_M=.025.
- The only candidate with positive Accuracy delta is r=3, lambda_M=.10 (dAcc=+0.000409), but it simultaneously worsens Macro-F1 by -0.001583, BalancedAcc by -0.004109, Tail20 by -0.012072, and targeted confusion by +0.010225.
- Larger lambda_M often improve NLL/ECE, but this occurs while ranking/class-balance and targeted confusion deteriorate. This is the wrong trade-off for Stage C.

Scientific conclusion:

The tested confusion-aware margin formulation does not solve the intended confusion problem under Reconstruction Clone v2. Its calibration-side gains at larger lambda_M do not justify the ranking and confusion degradation. No candidate is handed to a second confirmation seed.

Ledger decision:
- C0 protocol: CLOSED
- C1 seed42 pilot: CLOSED — NO WINNER
- C2 confirmation: SKIPPED / NOT OPENED because C1 produced no candidate
- Current Stage-C formulation: CLOSED EARLY / NEGATIVE

Downstream frozen state remains m=16,384 and scalar tau=.694.
