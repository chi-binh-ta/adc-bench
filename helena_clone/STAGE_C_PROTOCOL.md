# Stage C — Confusion-Aware Margin Protocol

Status: C0 FROZEN; C1 seed42 pilot READY.

Source-supported design:
- Stage C is the former E3 roadmap item.
- It runs on the best Stage-B base.
- Do not use a generic all-pairs margin.
- For each class k, construct C_k as the top-r classes most frequently confused with k on meta-validation.
- Candidate r values: {3,5}.
- Margin loss:

  L_margin = (1/n) sum_i log[1 + sum_{j in C_{y_i}} exp(z_j - z_{y_i} + m_{y_i j})].

- Full fine-tuning objective:

  L = L_WCE + lambda_F L_SoftF1 + lambda_M L_margin.

- Source-proposed lambda_M grid: {0.025, 0.05, 0.10, 0.20}.
- Primary Stage-C objective: Accuracy and reduction of high-confusion errors; NLL is secondary.

Clone-v2 frozen inheritance:
- m = 16,384
- scalar tau = 0.694 from Stage B closure
- QN-v2: fixed-step L-BFGS two-loop, memory=5, scale=.25, no hidden line search
- WCE / SoftF1 / Adaptive schedule = 18 / 9 / 6
- train/meta/cal/test split discipline unchanged

Clone-specific reconstruction choice because the source names m_{kj} but does not specify its numerical construction:

  m_{kj} = 0.5 if j in C_k, else 0.

This fixed height deliberately avoids opening another hyperparameter dimension. lambda_M controls the effective strength.

## C1 seed42 pilot

1. Train frozen Stage-B scalar baseline:
   WCE_18 -> SoftF1_9 -> Adaptive_tau=.694,6.
2. Use baseline META predictions to build directed confusion sets C_k for r=3 and r=5. The true class itself is excluded. Ranking within a row uses off-diagonal confusion counts; equivalently row-normalized rates give the same top-r ordering.
3. Freeze these confusion sets before candidate training.
4. For each (r,lambda_M), restart from the SAME WCE checkpoint and run 9 SoftF1+margin QN steps, then the same scalar Adaptive_tau=.694,6 continuation.
5. Include a lambda_M=0 control replay from the same WCE checkpoint to verify the Stage-C runner reproduces the frozen scalar path.
6. Evaluate candidates on CAL only. TEST stays untouched.

Candidate grid:
- r in {3,5}
- lambda_M in {.025,.05,.10,.20}
- 8 margin candidates + 1 scalar replay control.

Primary metrics:
- Accuracy
- targeted confusion rate = fraction of calibration observations whose prediction lies in C_y
- Macro-F1
- Balanced Accuracy
- Tail20-F1

Secondary metrics:
- NLL, Brier, ECE
- MacroPrecision, MacroRecall, WeightedF1

Pilot success rule:
- Accuracy must improve versus scalar replay control; and
- targeted confusion rate must decrease; and
- Macro-F1 and Tail20 may not fall by more than 5e-4 each.

If no candidate satisfies that rule, Stage C can close quickly as negative. If one or more pass, carry at most two candidates to seed123/seed456 screen without retuning the grid.
