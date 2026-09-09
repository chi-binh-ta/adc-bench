import argparse, json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import run_stageB1_clone as B
import run_stageE_cluster_g2 as E
from numeric_split import load_split_numeric

SEEDS = [42, 123, 456, 789, 2026]
RANKS = [1, 2, 3, 5, 10, 20]
NULL_RANKS = [1, 2, 3, 5, 10]
NULL_B = 5000
RNG_SEED = 20260909
EXPECTED_ALPHA = np.array([-0.9026209634632217, 0.765345197612019], dtype=np.float64)


def centered(z):
    return z - z.mean(axis=1, keepdims=True)


def load_seed_logits(artifact_dir):
    cal, test = {}, {}
    for seed in SEEDS:
        p = artifact_dir / f'stageD_seed{seed}_predictions.npz'
        if not p.exists():
            raise FileNotFoundError(p)
        d = np.load(p)
        cal[seed] = d['cal_logits'].astype(np.float64)
        test[seed] = d['test_logits'].astype(np.float64)
    return cal, test


def residual_matrix(y, p, K):
    R = np.zeros((K, K), dtype=np.float64)
    support = np.bincount(y, minlength=K)
    if np.any(support == 0):
        raise RuntimeError(f'zero-support class in residual split: {np.where(support==0)[0].tolist()}')
    for k in range(K):
        R[k] = -p[y == k].mean(axis=0)
        R[k, k] += 1.0
    return R, support


def svd_info(R):
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    s2 = s * s
    total = float(s2.sum())
    e = s2 / total if total > 0 else np.zeros_like(s2)
    nz = e[e > 0]
    entropy_rank = float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0
    pr_rank = float(1.0 / np.sum(e * e)) if np.sum(e * e) > 0 else 0.0
    stable_rank = float(total / s2[0]) if len(s2) and s2[0] > 0 else 0.0
    return {
        'U': U, 's': s, 'V': Vt.T, 'energy': e, 'cum': np.cumsum(e),
        'entropy_effective_rank': entropy_rank,
        'participation_ratio_rank': pr_rank,
        'stable_rank': stable_rank,
        'fro2': total,
    }


def subspace_overlap(Va, Vb, r):
    return float(np.linalg.norm(Va[:, :r].T @ Vb[:, :r], ord='fro') ** 2 / r)


def transfer_energy(R, V, r):
    den = float(np.sum(R * R))
    return float(np.sum((R @ V[:, :r]) ** 2) / den)


def random_null(Rtest, Vtest, r, rng, Bn=NULL_B):
    den = float(np.sum(Rtest * Rtest))
    transfer = np.empty(Bn, dtype=np.float64)
    overlap = np.empty(Bn, dtype=np.float64)
    vt = Vtest[:, :r]
    for b in range(Bn):
        q, _ = np.linalg.qr(rng.standard_normal((Rtest.shape[1], r)), mode='reduced')
        transfer[b] = np.sum((Rtest @ q) ** 2) / den
        overlap[b] = np.linalg.norm(q.T @ vt, ord='fro') ** 2 / r
    return transfer, overlap


def structured_fraction(y, p, R):
    # sum_i ||E[e|Y=y_i]||^2 / sum_i ||e_i||^2 without materializing one-hot matrix
    support = np.bincount(y, minlength=p.shape[1]).astype(np.float64)
    numerator = float(np.sum(support * np.sum(R * R, axis=1)))
    # ||onehot-p||^2 = 1 - 2 p_y + ||p||^2
    denominator = float(np.sum(1.0 - 2.0 * p[np.arange(len(y)), y] + np.sum(p * p, axis=1)))
    return numerator / denominator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', type=Path, required=True)
    args = ap.parse_args()
    out = Path(__file__).with_name('outputs')
    out.mkdir(exist_ok=True)

    X, y, tr, meta, cal, te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    ytr, ycal, ytest = y[tr], y[cal], y[te]
    K = int(y.max() + 1)
    train_support = np.bincount(ytr, minlength=K).astype(np.float64)

    zcal_seed, ztest_seed = load_seed_logits(args.artifact_dir)
    zcal = np.mean([centered(zcal_seed[s]) for s in SEEDS], axis=0)
    ztest = np.mean([centered(ztest_seed[s]) for s in SEEDS], axis=0)

    # Reconstruct the frozen final global G2 from CAL only.
    clusters, _, _, _ = E.cluster_partition(train_support)
    Tc = E.build_potential_bases(zcal, clusters)
    Tg = E.global_bases(Tc)
    alpha, opt = E.fit_g2(zcal, Tg, ycal, [0.0, 0.0])
    alpha_gap = (alpha - EXPECTED_ALPHA).tolist()
    if np.max(np.abs(alpha - EXPECTED_ALPHA)) > 1e-3:
        raise RuntimeError(f'canonical global-G2 reconstruction drift too large: alpha={alpha}, expected={EXPECTED_ALPHA}')
    pcal = E.corrected_probs(zcal, Tg, alpha)
    Tt = E.build_potential_bases(ztest, clusters)
    Tgt = E.global_bases(Tt)
    ptest = E.corrected_probs(ztest, Tgt, alpha)

    Rcal, ncal = residual_matrix(ycal, pcal, K)
    Rtest, ntest = residual_matrix(ytest, ptest, K)
    Sc = svd_info(Rcal)
    St = svd_info(Rtest)

    # Spectrum table.
    spectrum_rows = []
    for split, S in [('cal', Sc), ('test', St)]:
        for i, (sig, en, cu) in enumerate(zip(S['s'], S['energy'], S['cum']), 1):
            spectrum_rows.append({'split': split, 'rank': i, 'singular_value': sig,
                                  'energy_fraction': en, 'cumulative_energy': cu})
    pd.DataFrame(spectrum_rows).to_csv(out / 'stageF0_residual_spectrum.csv', index=False)

    # Per-seed base-model TEST residual subspaces for corroboration.
    seed_V = {}
    seed_rank_summaries = []
    for seed in SEEDS:
        p = E.softmax64(ztest_seed[seed])
        Rs, _ = residual_matrix(ytest, p, K)
        Ss = svd_info(Rs)
        seed_V[seed] = Ss['V']
        seed_rank_summaries.append({
            'seed': seed,
            'entropy_effective_rank': Ss['entropy_effective_rank'],
            'participation_ratio_rank': Ss['participation_ratio_rank'],
            'stable_rank': Ss['stable_rank'],
            **{f'cum_r{r}': float(Ss['cum'][r-1]) for r in RANKS},
        })
    pd.DataFrame(seed_rank_summaries).to_csv(out / 'stageF0_seed_spectral_summary.csv', index=False)

    pairwise_rows = []
    seed_median = {}
    for r in NULL_RANKS:
        vals = []
        for a, b in combinations(SEEDS, 2):
            ov = subspace_overlap(seed_V[a], seed_V[b], r)
            vals.append(ov)
            pairwise_rows.append({'rank': r, 'seed_a': a, 'seed_b': b, 'overlap': ov})
        seed_median[r] = float(np.median(vals))
    pd.DataFrame(pairwise_rows).to_csv(out / 'stageF0_seed_pairwise_overlap.csv', index=False)

    rng = np.random.default_rng(RNG_SEED)
    test_rows = []
    pass_ranks = []
    for r in NULL_RANKS:
        cal_energy = float(Sc['cum'][r-1])
        test_energy = float(St['cum'][r-1])
        transfer = transfer_energy(Rtest, Sc['V'], r)
        overlap = subspace_overlap(Sc['V'], St['V'], r)
        null_t, null_o = random_null(Rtest, St['V'], r, rng)
        p_t = float((1 + np.sum(null_t >= transfer)) / (NULL_B + 1))
        p_o = float((1 + np.sum(null_o >= overlap)) / (NULL_B + 1))
        passed = bool(cal_energy >= .60 and test_energy >= .55 and transfer >= .50 and
                      p_t <= .01 and overlap >= .50 and p_o <= .01)
        if passed:
            pass_ranks.append(r)
        test_rows.append({
            'rank': r,
            'cal_cumulative_energy': cal_energy,
            'test_cumulative_energy': test_energy,
            'cal_to_test_transfer': transfer,
            'transfer_null_mean': float(null_t.mean()),
            'transfer_null_q95': float(np.quantile(null_t, .95)),
            'transfer_null_q99': float(np.quantile(null_t, .99)),
            'transfer_p': p_t,
            'cal_test_overlap': overlap,
            'overlap_null_mean': float(null_o.mean()),
            'overlap_null_q95': float(np.quantile(null_o, .95)),
            'overlap_null_q99': float(np.quantile(null_o, .99)),
            'overlap_p': p_o,
            'median_pairwise_seed_overlap': seed_median[r],
            'primary_gate_pass': passed,
        })
    rank_df = pd.DataFrame(test_rows)
    rank_df.to_csv(out / 'stageF0_rank_tests.csv', index=False)

    rstar = min(pass_ranks) if pass_ranks else None
    rho_cal = structured_fraction(ycal, pcal, Rcal)
    rho_test = structured_fraction(ytest, ptest, Rtest)
    summary = {
        'status': 'STAGE_F0_COMPLETE',
        'question': 'Does systematic residual error after the frozen Helena system have transferable low-dimensional structure?',
        'final_system': 'coherent_logit_S5 -> global_G2',
        'global_g2_alpha': alpha.tolist(),
        'global_g2_alpha_gap_vs_canonical': alpha_gap,
        'global_g2_optimizer': opt,
        'residual_definition': 'R[k,j] = E[1(Y=j)-p_j(X) | Y=k]',
        'ambient_dimension': K,
        'cal_support_minmax': [int(ncal.min()), int(ncal.max())],
        'test_support_minmax': [int(ntest.min()), int(ntest.max())],
        'cal_spectral': {
            'entropy_effective_rank': Sc['entropy_effective_rank'],
            'participation_ratio_rank': Sc['participation_ratio_rank'],
            'stable_rank': Sc['stable_rank'],
            **{f'cum_r{r}': float(Sc['cum'][r-1]) for r in RANKS},
        },
        'test_spectral': {
            'entropy_effective_rank': St['entropy_effective_rank'],
            'participation_ratio_rank': St['participation_ratio_rank'],
            'stable_rank': St['stable_rank'],
            **{f'cum_r{r}': float(St['cum'][r-1]) for r in RANKS},
        },
        'rank_tests': test_rows,
        'structured_fraction': {'cal': rho_cal, 'test': rho_test},
        'primary_gate': {
            'r_le_10': True,
            'cal_energy_ge': .60,
            'test_energy_ge': .55,
            'transfer_ge': .50,
            'transfer_p_le': .01,
            'overlap_ge': .50,
            'overlap_p_le': .01,
        },
        'r_star': rstar,
        'strong_low_dimensional_evidence': bool(rstar is not None),
        'interpretation': ('STRONG LOW-DIMENSIONAL SYSTEMATIC RESIDUAL EVIDENCE' if rstar is not None
                           else 'NO STRONG LOW-DIMENSIONAL EVIDENCE UNDER THE FROZEN F0 GATE'),
        'provenance_note': 'F0 uses fresh canonical-run seed logits from workflow 34375185211. No A-E parameter is retuned. TEST is diagnostic evidence only and does not change the frozen system.'
    }
    json.dump(summary, open(out / 'stageF0_summary.json', 'w'), indent=2)

    # Compact human-readable report.
    lines = [
        '# Stage F0 — Latent Residual Low-Dimensionality Audit — RESULT', '',
        f"Status: **{summary['interpretation']}**.", '',
        'This is empirical/statistical evidence on the finite Helena sample, not a population theorem.', '',
        '## Residual object', '',
        '`R[k,j] = E[1(Y=j)-p_j(X) | Y=k]` for the frozen coherent-S5 -> global-G2 system.', '',
        '## Spectral summary', '',
        '| split | eff-rank entropy | PR-rank | stable rank | E(3) | E(5) | E(10) | E(20) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        f"| CAL | {Sc['entropy_effective_rank']:.3f} | {Sc['participation_ratio_rank']:.3f} | {Sc['stable_rank']:.3f} | {Sc['cum'][2]:.4f} | {Sc['cum'][4]:.4f} | {Sc['cum'][9]:.4f} | {Sc['cum'][19]:.4f} |",
        f"| TEST | {St['entropy_effective_rank']:.3f} | {St['participation_ratio_rank']:.3f} | {St['stable_rank']:.3f} | {St['cum'][2]:.4f} | {St['cum'][4]:.4f} | {St['cum'][9]:.4f} | {St['cum'][19]:.4f} |",
        '', '## Transfer and stability gate', '',
        '| r | CAL energy | TEST energy | CAL->TEST transfer | p-transfer | overlap | p-overlap | seed median overlap | pass |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|:---:|',
    ]
    for row in test_rows:
        lines.append(f"| {row['rank']} | {row['cal_cumulative_energy']:.4f} | {row['test_cumulative_energy']:.4f} | {row['cal_to_test_transfer']:.4f} | {row['transfer_p']:.5f} | {row['cal_test_overlap']:.4f} | {row['overlap_p']:.5f} | {row['median_pairwise_seed_overlap']:.4f} | {'YES' if row['primary_gate_pass'] else 'NO'} |")
    lines += ['', '## Structured fraction', '',
              f"- CAL: {rho_cal:.6f}", f"- TEST: {rho_test:.6f}", '',
              'This fraction measures how much total per-sample squared residual energy is carried by class-conditional mean residuals; it is not itself the low-rank decision gate.', '',
              '## Decision', '']
    if rstar is not None:
        lines += [f"The smallest pre-frozen passing rank is **r*={rstar}** (<=10 of 100 dimensions).",
                  'Therefore F0 supports opening a latent-factor identification stage, but the latent factors still need interpretation and causal/structural validation.']
    else:
        lines += ['No rank r<=10 passes all six pre-frozen conditions.',
                  'Therefore the current evidence is insufficient to justify a latent-factor model solely from low-rank systematic residual structure.']
    (out / 'STAGE_F0_RESULT.md').write_text('\n'.join(lines) + '\n')
    print('STAGE_F0_RESULT', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
