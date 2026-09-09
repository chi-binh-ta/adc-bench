# Stage B — FINAL CLOSURE

Status: CLOSED.

Frozen downstream ranking policy for Reconstruction Clone v2:

- representation capacity: m = 16,384
- ranking control: scalar tau = 0.694
- class-conditional tau_k: NOT CONFIRMED across seed42 -> seed123

Evidence summary:

1. B1 seed42: several class-conditional policies produced positive calibration-side paired gains versus matched rho=0 controls.
2. B2 seed123: none of the three preselected policies replicated. All three reduced Accuracy, Macro-F1 and Balanced Accuracy versus their scalar controls; Tail20 was unchanged.
3. Two-seed scalar comparison among tau in {.694,.815,.935} favored tau=.694 on mean Accuracy, Macro-F1, Balanced Accuracy, Tail20, NLL and ECE.

Frozen Stage-B decision:

    tau* = 0.694 (scalar)

This is a Reconstruction Clone v2 design decision. It does not reinterpret the historical high-capacity tau=.935 experiments. The full Helena system can later be rerun from scratch under the clone-v2 protocol.

Downstream rule: Stage C, D and E must inherit m=16,384 and scalar tau=.694 unless a later stage explicitly modifies a different module. Stage C does NOT reopen Stage B.
