# Helena Reconstruction Clone — 2026-09-09

Status: RECONSTRUCTED / PROVENANCE-PRESERVING CLONE, not claimed bit-identical to the lost high-capacity runner.

Purpose: preserve everything recoverable from the exported ChatGPT text, current conversation, and surviving experiment artifacts so Stage B and later Helena reruns can continue without relying on hidden session state.

## 1. Frozen scientific decisions

- Dataset: Helena, 27 numeric features, 100 classes, N=65,196.
- Canonical split counts: train=45,637; meta=4,889; calibration=4,890; test=9,780.
- Stage A is CLOSED at m*=16,384 for all downstream stages B–E.
- High-capacity classifier pipeline:
  NCAP-RSP -> WCE_18 -> SoftMacroF1_9 -> Adaptive_6 -> G2.
- High-capacity ranking mode used for Stage-A closure: tau=.935.
- Seeds used historically: 42, 123, 456.
- No warm-start across ranks. Each (seed,m) starts from zero in RSP coordinates. Within a rank, WCE -> SoftF1 -> Adaptive is warm-started stage-to-stage.

## 2. Data split reconstruction

Recovered conversation-level reconstruction:

1. Stratified 70/30 split, random_state=42.
2. Split the 30% remainder 50/50 into validation and test, random_state=42.
3. Split validation 50/50 into meta and calibration, random_state=42.

This produces the surviving counts 45,637 / 4,889 / 4,890 / 9,780.

## 3. Raw-feature preprocessing

- Standardize the 27 raw Helena variables using StandardScaler fit ONLY on the train split.
- RBF kernel: K(x,c)=exp(-gamma ||x-c||^2), gamma=0.02423.
- High-rank representation is the raw landmark kernel basis, not dense K_CC^{-1/2} whitening.
- At high rank, each landmark column is normalized by the full-train column mean/std.
- Expected diagnostic after normalization: train feature-column means approx 0 and stds approx 1.
- Corrected cache layout is C-order for writer AND reader.
- Historical bug: old seed123/seed456 cache was written order='F' and read order='C'; those results were rejected.
- Corrected high-capacity runner name recovered from history: run_rsp_highcap_tau0935_corder.py.
- Feature construction block was increased from 192 to 1024 landmark columns per pass without changing the mathematical mean/std definition.

## 4. Landmark construction

High-rank regime switched away from k-means++ to a fixed nested stratified-random landmark trajectory to avoid landmark-selection cost becoming the experiment.

Reconstruction used by this clone:

- Build one StratifiedShuffleSplit pool of 24,576 train indices for each experiment seed.
- m=16,384 uses the prefix C_16384 of that pool.
- The same trajectory nests 16,384 ⊂ 20,480 ⊂ 24,576.

This is strongly supported by the surviving high-rank notes, although the exact internal ordering routine of the lost runner is not proven bit-identical.

## 5. NCAP-RSP geometry

Recovered formula:

Ghat_m ~= V_r diag(mu_1,...,mu_r) V_r^T, r=256.

lambda_spec = 1e-3.

P_m = I + V_r [ diag( sqrt((mu_r+lambda)/(mu_j+lambda)) ) - I ] V_r^T.

Softmax parameterization:

W = P_m Theta.

All spectral factors are positive, so P_m is invertible and the function class is unchanged; only optimization geometry changes.

### Spectral-sketch uncertainty

The exact row-sketch recipe was not recovered. For seed42,m=16384 the historical fingerprint was:

- mu1 = 6517.5686478
- mur = 0.01170029077
- qmin = 0.001395931118

Several forensic variants were tested. A 1024-row stratified sketch with seed42 and randomized SVD came close in mur/qmin but not mu1. Therefore this clone records spectral-sketch construction as RECONSTRUCTED rather than SOURCE-EXACT.

Clone default: 1024 stratified train rows, row random_state=experiment seed, randomized_svd rank=256, n_iter=8, random_state=0.

## 6. Objectives

### WCE

Historical fixed choice:

alpha_WCE = 0.10.

Class weight before normalization:

w_k = pi_k^{-0.10}.

### Soft Macro-F1

Recovered soft counts for class k:

TP_k^s = sum_i y_ik p_ik
FP_k^s = sum_i (1-y_ik) p_ik
FN_k^s = sum_i y_ik (1-p_ik)

softF1_k = 2 TP_k^s / (2TP_k^s + FP_k^s + FN_k^s + eps).

L_softMacroF1 = 1 - (1/K) sum_k softF1_k.

lambda_F = 1.

The historical gradient implementation passed finite-difference audit.

### P/R adaptive rule

Recovered final multiplicative rule:

w_k(tau) proportional to
pi_k^{-0.10}
((Rbar+eps)/(R_k+eps))^{eta_R(tau)}
((P_k+eps)/(Pbar+eps))^{eta_P(tau)}

with eps=0.02,
eta_R(tau)=0.05+0.15 tau,
eta_P(tau)=0.20-0.15 tau.

Interpretation:
- low recall raises class weight;
- low precision lowers class weight, avoiding further over-prediction.

Historical adaptive factors were normalized and earlier experiments used clipping; for the clone, class weights are normalized to mean one and the multiplicative adaptive factor is clipped to [0.5,3.0]. The [0.5,3.0] clip is source-supported from the preceding adaptive pipeline but not explicitly re-observed in the final high-rank runner, so it is tagged RECONSTRUCTED.

## 7. Stage B — class-conditional tau_k

This is the next frozen roadmap stage after m*=16,384.

Low-dimensional policy:

tau_k = clip[ tau0 + rho log((Ptilde_k+eps)/(Rtilde_k+eps)), 0, 1 ].

Shrunk precision/recall:

Ptilde_k = (TP_k + lambda Pbar)/(TP_k+FP_k+lambda)
Rtilde_k = (TP_k + lambda Rbar)/(TP_k+FN_k+lambda)

Then:

eta_R,k = 0.05 + 0.15 tau_k
eta_P,k = 0.20 - 0.15 tau_k.

Frozen B1 grid from the current experiment conversation:

- tau0 in {.694, .815, .935}
- rho in {0, .15, .30}
- lambda in {20, 50}

Total reporting rows = 3*3*2 = 18.
Because rho=0 makes tau_k=tau0 independent of lambda, there are 15 unique trained models but 18 audit rows.

Nested scalar controls are exactly the rho=0 rows.

B1 protocol:
- seed=42
- m=16,384
- rebuild the frozen WCE->SoftF1 checkpoint once
- derive P/R statistics on meta
- branch each Stage-B candidate from the SAME SoftF1 checkpoint
- run exactly 6 adaptive full-gradient refinement steps per candidate
- compare each candidate directly against its rho=0 scalar control at the same tau0
- primary paired metrics: Accuracy, Macro-F1, Balanced Accuracy, Tail20-F1; secondary: NLL/Brier/ECE and G2-NLL.
- only if class-conditional policy produces a Pareto gain on seed42 should the winning region be moved to seed123.

## 8. G2 calibration

Recovered high-level form:

g_G(u)=a0+a1 u

and the full-scale experiments explicitly rejected silently replacing this with the earlier one-parameter approximation a(1-u) at aggressive tau.

G2 is fit on the calibration half AFTER ranking is complete and must not change top-1 ranking metrics.

Historical high-capacity seed42,m=16384,tau=.935 endpoint:

- base NLL = 3.0589647293
- G2 NLL = 2.9153313637
- Accuracy = 0.3317996
- Macro-F1 = 0.1451121
- BalancedAcc ≈ 0.149154
- Tail20-F1 ≈ 0.0679774
- g2_a0 = -1.50616822
- g2_a1 = 1.34225192

The exact definition of scalar u used inside the original G2 code was not recovered from the export. Therefore the clone runner does NOT invent a replacement for G2 inside the training loop. It reports base ranking/NLL for Stage-B selection and leaves a clearly marked G2 hook. Historical G2 values remain in the audit ledger for comparison.

## 9. Optimization schedule

Source-exact fixed-compute high-rank schedule:

- WCE: exactly 18 full-gradient steps
- SoftMacroF1: exactly 9 full-gradient steps
- Adaptive: exactly 6 full-gradient steps

Surviving history also documents the transition away from strong-Wolfe/L-BFGS toward deterministic fixed-step / quasi-Newton continuation because line search became the high-rank bottleneck. Earlier unified protocols used deep fixed-step QN continuation and no hidden line search.

The exact high-rank per-step learning rate/update equation was not recovered.

Clone choice: deterministic full-batch Barzilai-Borwein spectral-gradient update with safeguards, no line search. This is explicitly a RECONSTRUCTION CHOICE chosen because it is consistent with the surviving fixed-step quasi-Newton description and is easy to reproduce. It must not be confused with the lost canonical optimizer.

## 10. Certainty ledger

SOURCE-EXACT / strongly source-supported:
- dataset size and split counts
- gamma=.02423
- alpha_WCE=.10
- lambda_F=1
- RSP rank=256
- lambda_spec=1e-3
- W=P Theta formula
- high-rank raw kernel basis + column normalization
- nested high-rank landmark trajectory
- stratified-random high-rank landmark strategy
- C-order correction
- 18/9/6 fixed full-gradient schedule
- no rank warm-start
- adaptive eta_R/eta_P mapping
- P/R multiplicative adaptive weight rule
- Stage-B low-dimensional tau_k policy form
- Stage A frozen at m=16384
- G2 remains affine 2-parameter in the final full-scale system

CURRENT-CONVERSATION FROZEN:
- B1 grid tau0={.694,.815,.935}, rho={0,.15,.30}, lambda={20,50}
- 18 audit rows / 15 unique models
- rho=0 nested scalar controls
- seed42 first, winners only to seed123

RECONSTRUCTED CHOICES:
- exact spectral row-sketch recipe
- exact high-rank optimizer update / step sizes
- adaptive-factor clipping in the final high-rank runner
- exact internal class-weight normalization convention

NOT RECOVERED — DO NOT FABRICATE:
- original Theta checkpoints
- original logits/probability checkpoint arrays
- exact G2 internal scalar u implementation
- bit-identical randomized-SVD/sketch configuration of run_rsp_highcap_tau0935_corder.py

## 11. Scientific use of the clone

This clone is suitable for:
- continuing Stage B with internally consistent paired controls;
- finding promising tau_k regions;
- designing Stage C/D/E;
- later doing a clean full Helena rerun of all frozen stages with this clone as the new canonical implementation.

It is NOT suitable for claiming exact numerical continuation of the lost seed42 Stage-A checkpoint. All new results must be labeled CLONE / RECONSTRUCTED until the full rerun establishes a new canonical baseline.
