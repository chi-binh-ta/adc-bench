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
