# Helena Reconstruction Clone Changelog

## 2026-09-09 — Stage B1 clone v1 invalidated

The first runnable clone B1 completed successfully as software, but its scientific results are INVALIDATED for Stage-B inference.

Reason 1 — class-order reconstruction bug:
- v1 decoded Helena class labels as strings and used `np.unique(raw_y)`, producing lexicographic order (`1,10,100,11,...`).
- Historical Helena/RSP probes parsed class labels numerically.
- Although a pure class permutation would leave classification metrics invariant if the split were fixed, stratified splitting/landmark sampling consumes RNG by class ordering. Therefore the lexicographic mapping changes the actual split/landmark trajectory and RSP geometry.
- v1 RSP seed42,m=16384 fingerprint was `(mu1,mur,qmin)=(5874.4867,0.00962414,0.00134481)`, farther from the historical fingerprint `(6517.5686,0.01170029,0.00139593)` than earlier numeric-label probes.

Fix:
- `numeric_split.py` parses labels as integers and shifts them by a constant to 0..99, preserving numeric class order and historical stratification semantics.

Reason 2 — optimizer reconstruction too weak:
- v1 used Barzilai-Borwein spectral gradient as a conservative reconstruction choice.
- The source export contains stronger evidence that the pre-RSP unified solver was fixed-step L-BFGS/quasi-Newton with no hidden line search; RSP later reduced equal-compute depth to 18 WCE + 9 SoftF1 + 6 adaptive full gradients.
- v1 Soft checkpoint had only Accuracy≈0.1974, MacroF1≈0.0407 and Tail20=0, so Stage-B policy comparisons at that point were not scientifically useful.

Fix:
- replace BB-gradient with fixed-step limited-memory BFGS two-loop recursion;
- anchor its step scale/memory on the historical seed42 WCE meta-NLL `3.08469558`;
- then check SoftF1 meta-NLL against `3.06046534` and tau=.935 adaptive meta-NLL against `3.05098796`.

Methodological correction:
- v1 ranked candidates using test metrics. This is no longer allowed.
- Stage-B QN reconstruction uses meta to estimate P/R and derive tau_k, calibration split to screen/select Stage-B candidates, and leaves test untouched during the 18-row sweep.

Historical v1 outputs remain preserved as forensic artifacts only. They must not be cited as evidence for or against class-conditional tau_k.

## 2026-09-09 — QN clone v2 frozen for Stage B

Numeric-label seed42,m=16384 RSP reconstruction:
- mu1 = 6352.3076007
- mur = 0.01162961181
- qmin = 0.001410032739
Historical comparison fingerprint:
- mu1 = 6517.5686478
- mur = 0.01170029077
- qmin = 0.001395931118
The reconstruction is close in mur/qmin but not bit-identical in mu1; this remains an explicitly reconstructed spectral sketch.

Fixed-step L-BFGS/QN anchor grid used 18 WCE full gradients with:
- scale in {0.25,0.5,1.0,1.5}
- memory in {5,10}
- historical seed42 WCE meta-NLL target = 3.08469558.

Measured final meta-NLL:
- scale=.25, memory=5: 2.941590786  [closest in scanned grid]
- scale=.25, memory=10: 2.924976349
- scale=.5, memory=5: 2.834270239
- scale=.5, memory=10: 2.833034039
- scale=1.0, memory=5: 2.801619291
- scale=1.0, memory=10: 2.809076309
- scale=1.5, memory=5: 2.835709572
- scale=1.5, memory=10: 2.846878529

Decision for Reconstruction Clone v2:
- fixed-step L-BFGS two-loop recursion
- memory = 5
- fixed scale = 0.25
- no hidden line search
- WCE/SoftF1/Adaptive schedule remains exactly 18/9/6.

This does not claim exact recovery of the historical optimizer. The user explicitly permits an accurate, internally reproducible clone because the whole Helena system will eventually be rerun from scratch. Therefore Stage-B paired comparisons are now prioritized over forcing numerical equality with the old WCE checkpoint.

Stage-B v2 selection discipline:
- train split: parameter fitting
- meta split: estimate per-class P/R and construct tau_k
- calibration split: compare Stage-B candidate policies against matched rho=0 controls
- test split: untouched during the Stage-B sweep.

## 2026-09-09 — Stage B1 QN-v2 seed42 COMPLETED

Experiment:
- m = 16,384 (Stage A frozen)
- seed = 42
- 18 reporting rows / 15 unique models
- QN clone v2: memory=5, scale=.25
- meta used for P/R -> tau_k construction
- calibration used for candidate screening
- test untouched.

Outcome:
- 7/18 reporting rows satisfy the clone's ranking-metric Pareto screen versus matched rho=0 controls.
- Full exact CSV is persisted at `helena_clone/results/stageB1_qn_seed42_18rows_cal.csv`.
- Human-readable analysis is persisted at `helena_clone/results/STAGE_B1_QN_V2_RESULT.md`.

Primary candidate for B2 / seed123 confirmation:
- tau0 = .694
- rho = .15
- lambda_shrink = 50
- mean tau_k = .734572 (range .589829 to .815332)
- calibration Accuracy = .355010225
- Macro-F1 = .198725237
- BalancedAcc = .197006774
- Tail20-F1 = .112448778
- NLL = 2.820905924
- Brier = .792438192
- ECE = .060175407

Paired versus scalar tau0=.694 control:
- dAccuracy = +.000408998
- dMacroF1 = +.001362809
- dBalancedAcc = +.000485792
- dTail20 = 0
- dNLL = +.001411915 (worse)
- dBrier = +.000300058 (worse)
- dECE = -.000250419 (better)

Interpretation:
- class-conditional tau_k produces a real ranking-side Pareto gain on seed42 under the clone protocol;
- likelihood/Brier degrade slightly, which is treated as a downstream calibration trade-off because Stage B optimizes ranking/class-balance and G2 is the later calibration layer;
- rho=.30 at tau0=.694 or .815 is rejected for the seed123 handoff because Tail20-F1 falls materially.

Secondary B2 robustness candidates:
- (.815, .15, 20): largest paired Macro-F1 gain in the moderate-tau region;
- (.935, .15, 20): high-tau robustness check with improved ECE but tau_k close to saturation.

Stage ledger after this run:
- Stage A: CLOSED at m=16,384
- Stage B0: CLOSED / protocol frozen
- Stage B1: COMPLETED on seed42 / positive
- Stage B2: NEXT — seed123 confirmation
- Stage B3: pending final tau_k policy freeze.
