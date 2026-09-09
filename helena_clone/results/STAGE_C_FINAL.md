# Stage C — FINAL STATUS

Status: CLOSED EARLY / NEGATIVE for the tested confusion-aware margin formulation.

Inherited frozen state:
- m = 16,384
- scalar tau = .694

C0: CLOSED — protocol frozen.
C1: CLOSED — seed42 pilot, 0/8 candidates passed.
C2: SKIPPED / NOT OPENED because C1 produced no candidate worth seed confirmation.

Key evidence:
- lambda_M=0 replay reproduced the frozen baseline exactly on Accuracy, Macro-F1, Balanced Accuracy, Tail20, NLL, Brier and ECE.
- All eight margin candidates increased the targeted confusion rate.
- No candidate satisfied the joint Accuracy + targeted-confusion gate while preserving Macro-F1 and Tail20.
- Larger margin weights sometimes improved NLL/ECE, but at the cost of worse class-balance/ranking metrics and worse targeted confusion.

Decision:
- do not carry this confusion-aware margin module downstream;
- downstream stages inherit the Stage-B scalar baseline unchanged: m=16,384 and tau=.694;
- a future redesigned confusion mechanism would be a new experiment family, not a continuation of the closed C1 grid.

Exact C1 evidence:
- `helena_clone/results/stageC1_seed42_9rows_cal.csv`
- `helena_clone/results/STAGE_C1_FINAL.md`
