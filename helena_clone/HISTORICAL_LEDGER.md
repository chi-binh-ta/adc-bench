# Historical Helena Ledger

This file preserves numerical checkpoints and decisions recovered from the conversation/export. It is comparison evidence, not a claim that the reconstructed clone is bit-identical.

## Frozen high-capacity protocol

Pipeline: NCAP-RSP -> WCE_18 -> SoftMacroF1_9 -> Adaptive_tau,6 -> G2

Frozen hyperparameters:
- gamma = 0.02423
- alpha_WCE = 0.10
- lambda_SoftF1 = 1
- tau = .935 for high-capacity Stage-A ranking probe
- eta_R(.935)=0.19025
- eta_P(.935)=0.05975
- RSP rank=256
- lambda_spec=1e-3

## Seed 42 — corrected high-capacity

| m | Accuracy | Macro-F1 | Tail20-F1 | G2-NLL |
|---:|---:|---:|---:|---:|
| 12,288 | 0.32955 | 0.14228 | 0.06372 | 2.91797 |
| 16,384 | 0.331800 | 0.145112 | 0.067977 | 2.915331 |
| 20,480 | 0.331800 | 0.145231 | 0.068324 | 2.914212 |
| 24,576 | 0.330879 | 0.144502 | 0.065048 | 2.914564 |

Seed42 m=16,384 additional exact/near-exact values:
- BalancedAcc ≈ 0.149154
- base NLL = 3.0589647293
- G2-NLL = 2.9153313637
- mu1 = 6517.5686478
- mur = 0.01170029077
- qmin = 0.001395931118
- WCE meta NLL = 3.08469558, best/final iter 18
- Soft meta NLL = 3.06046534, best/final iter 9
- Adaptive meta NLL = 3.05098796, best/final iter 6
- g2_a0 = -1.50616822
- g2_a1 = 1.34225192

Seed42 deltas:
- 12,288 -> 16,384: Macro +0.00283; Tail +0.00426; Accuracy +0.00225; BalancedAcc +0.00196; G2-NLL -0.00264.
- 16,384 -> 20,480: Macro +0.000119; Tail +0.000347; Accuracy 0; BalancedAcc about -0.000150; G2-NLL about -0.001119.
- 20,480 -> 24,576: Macro -0.000729; Tail -0.003276; Accuracy -0.00092; G2-NLL +0.000352 (worse).

## Seed 123 — corrected C-order high-capacity

| m | Accuracy | Macro-F1 | Tail20-F1 | G2-NLL |
|---:|---:|---:|---:|---:|
| 16,384 | 0.331902 | 0.147584 | 0.074151 | 2.946963 |
| 20,480 | 0.331288 | 0.148361 | 0.078400 | 2.946098 |
| 24,576 | 0.332618 | 0.149435 | 0.078164 | 2.947291 |

Seed123 deltas:
- 16,384 -> 20,480: Macro +0.000778; Tail +0.004249; G2-NLL improves.
- 20,480 -> 24,576: Macro +0.001074; Tail -0.000236; G2-NLL worsens 2.946098 -> 2.947291.

## Two-seed 16k descriptive endpoint

Supported seeds: 42 and 123. Seed456 16k was never certified.

Means (DESCRIPTIVE 2-SEED, not 3-seed estimates):
- Accuracy 0.331851
- Macro-F1 0.146348
- Tail20 0.071064
- G2-NLL 2.931147

Scientific interpretation at closure:
- m=16,384 is the beginning of the saturation region.
- It is a frozen ranking-capacity design decision, not a 3-seed validated unique optimum.

## Valid seed456 m=20,480 checkpoint

- seed = 456
- m = 20,480
- tau = .935
- eta_R = .19025
- eta_P = .05975
- lambda_spec = .001
- spec_r = 256
- mu1 = 7923.125364095147
- mur = 0.013920819699026
- qmin = 0.0013722968287765
- wce_meta_nll = 3.1315245628356934
- soft_meta_nll = 3.108982801437378
- adapt_meta_nll = 3.099998235702514
- g2_a0 = -1.4935934115171576
- g2_a1 = 1.355399540707502
- cal_g2_nll = 2.939905684069534
- base_nll = 3.069826364517212
- base_brier = 0.8184759020805359
- base_ece = 0.0810147721868899
- Accuracy = 0.3315950920245398
- Macro-F1 = 0.1436005187388093
- BalancedAcc = 0.149151056280595
- WeightedF1 = 0.2796063293807168
- MacroPrecision = 0.1937931813207834
- MacroRecall = 0.149151056280595
- Tail20-F1 = 0.0777853970741901
- G2-NLL = 2.925111532211304
- G2-Brier = 0.8052121996879578
- G2-ECE = 0.0236019136555538

G2 paired NLL improvement at this checkpoint:
2.9251115322 - 3.0698263645 = -0.1447148323 (~4.7141% relative reduction).

## Lower high-rank 3-seed screen

Complete grid: 3 seeds x m={4096,8192,12288} x tau={.400,.694,.935} = 27/27 after later completion.

At tau=.935, 3-seed means:
- m=4096: MacroF1 0.13900; Tail20 0.06645
- m=8192: MacroF1 0.14283; Tail20 0.06974
- m=12288: MacroF1 0.15119; Tail20 0.07620

Seed456 m=12,288:
- tau=.400: Acc .34448, Macro .16288, Bal .16644, Tail .08392, G2-NLL 2.83868
- tau=.694: Acc .34427, Macro .16361, Bal .16671, Tail .08436, G2-NLL 2.83663
- tau=.935: Acc .34407, Macro .16460, Bal .16749, Tail .08664, G2-NLL 2.83482

This established 12,288 > 8,192 under the full 3-seed pipeline before opening the high-capacity continuation.

## Forensic reconstruction attempts — spectral fingerprint

These attempts were made after the exact high-capacity runner was lost. They are preserved so we do not repeat them.

Canonical seed456,m=20,480 fingerprint:
- mu1 = 7923.1253641
- mur = 0.01392081970
- qmin = 0.00137229683

Sketch-size sweep, nested landmarks, seed456:
- S=1024: 8303.3743467, .01429071621, .00135702103
- S=1536: 7990.3255692, .01777735952, .00153297473
- S=2048: 7580.4278706, .01870213028, .00161216580
- S=2560: 7643.4534153, .02116147942, .00170276429
- S=3072: 7750.0169312, .02238554455, .00173709028

S=1024 was closest in mur/qmin but not mu1.

Seed456 row-sketch sweep at S=1024:
- rs0: 7871.2108692, .01305899838, .00133646131
- rs1: 7688.9954384, .01292225720, .00134561245
- rs7: 8101.9451490, .01359991013, .00134239528
- rs42: 7882.5058465, .01429254363, .00139286070
- rs123: 7824.8310104, .01326309111, .00135011051
- rs31415: 7472.9917506, .01137003318, .00128658402
- rs456: 8303.3743494, .01439230012, .00136152126
- rs457: 7989.8687604, .01348267596, .00134633940

No row seed reproduced all three canonical components.

Seed42,m=16,384 reproduction probes in the latest forensic pass:
- canonical: 6517.5686478, .01170029077, .001395931118
- row_rs=42, S=1024, randomized-SVD n_iter=8: 6352.3076004, .01162961164, .00141003276
- row_rs=7 gave mu1 6520.1238558 but mur .01095437 and qmin .00135405, so it was rejected as a canonical match.

Conclusion: exact runner reconstruction remained uncertified; therefore the new branch intentionally becomes a new provenance-preserving clone rather than pretending to be the old runner.
