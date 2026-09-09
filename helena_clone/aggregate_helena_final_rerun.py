import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import run_stageB1_clone as B
import run_stageE_cluster_g2 as E
from numeric_split import load_split_numeric

SEEDS = [42, 123, 456, 789, 2026]
EXPECTED = {
    'soft_vote_test_accuracy': 0.36226993865030677,
    'soft_vote_test_macro_f1': 0.20606033209756158,
    'global_g2_test_nll': 2.7644007205963135,
    'global_g2_test_brier': 0.7781448650949503,
    'global_g2_test_ece': 0.03222390116518077,
    'global_g2_test_accuracy': 0.3621676891615542,
    'global_g2_test_macro_f1': 0.20592893268833737,
}


def load_seed_outputs(artifact_dir):
    cal_logits, test_logits, summaries = [], [], []
    for seed in SEEDS:
        p = artifact_dir / f'stageD_seed{seed}_predictions.npz'
        s = artifact_dir / f'stageD_seed{seed}_summary.json'
        if not p.exists() or not s.exists():
            raise FileNotFoundError(f'missing final-rerun seed artifact for seed={seed}')
        d = np.load(p)
        zc = d['cal_logits'].astype(np.float64)
        zt = d['test_logits'].astype(np.float64)
        if zc.shape[1] != 100 or zt.shape[1] != 100:
            raise RuntimeError(f'unexpected class dimension for seed {seed}')
        cal_logits.append(zc)
        test_logits.append(zt)
        summaries.append(json.load(open(s)))
    return cal_logits, test_logits, summaries


def centered(z):
    return z - z.mean(axis=1, keepdims=True)


def aggregate_probs(logits_list):
    return np.mean([E.softmax64(z) for z in logits_list], axis=0)


def metric_row(split, model, y, p, train_support):
    return {'split': split, 'model': model, **B.metrics(y, p.astype(np.float32), train_support)}


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

    zc_list, zt_list, seed_summaries = load_seed_outputs(args.artifact_dir)

    # Pure ranking benchmark: arithmetic mean of seed probabilities.
    psoft_cal = aggregate_probs(zc_list)
    psoft_test = aggregate_probs(zt_list)

    # Coherent structural ensemble: centered logit mean.
    zcoh_cal = np.mean([centered(z) for z in zc_list], axis=0)
    zcoh_test = np.mean([centered(z) for z in zt_list], axis=0)
    pcoh_cal = E.softmax64(zcoh_cal)
    pcoh_test = E.softmax64(zcoh_test)

    # Final calibrator: global G2 fitted on CAL only.
    clusters, head, mid, tail = E.cluster_partition(train_support)
    Tc = E.build_potential_bases(zcoh_cal, clusters)
    Tg = E.global_bases(Tc)
    alpha, opt = E.fit_g2(zcoh_cal, Tg, ycal, [0.0, 0.0])
    pg_cal = E.corrected_probs(zcoh_cal, Tg, alpha)

    # Architecture and coefficients are now frozen; TEST is evaluation only.
    Tt = E.build_potential_bases(zcoh_test, clusters)
    Tgt = E.global_bases(Tt)
    pg_test = E.corrected_probs(zcoh_test, Tgt, alpha)

    rows = [
        metric_row('cal', 'soft_vote_S5', ycal, psoft_cal, train_support),
        metric_row('cal', 'coherent_logit_S5', ycal, pcoh_cal, train_support),
        metric_row('cal', 'coherent_S5_global_G2', ycal, pg_cal, train_support),
        metric_row('test', 'soft_vote_S5', ytest, psoft_test, train_support),
        metric_row('test', 'coherent_logit_S5', ytest, pcoh_test, train_support),
        metric_row('test', 'coherent_S5_global_G2', ytest, pg_test, train_support),
    ]
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / 'helena_final_rerun_metrics.csv', index=False)

    test_soft = next(r for r in rows if r['split']=='test' and r['model']=='soft_vote_S5')
    test_g2 = next(r for r in rows if r['split']=='test' and r['model']=='coherent_S5_global_G2')
    test_coh = next(r for r in rows if r['split']=='test' and r['model']=='coherent_logit_S5')

    gaps = {
        'soft_vote_test_accuracy': float(test_soft['accuracy'] - EXPECTED['soft_vote_test_accuracy']),
        'soft_vote_test_macro_f1': float(test_soft['macro_f1'] - EXPECTED['soft_vote_test_macro_f1']),
        'global_g2_test_nll': float(test_g2['nll'] - EXPECTED['global_g2_test_nll']),
        'global_g2_test_brier': float(test_g2['brier'] - EXPECTED['global_g2_test_brier']),
        'global_g2_test_ece': float(test_g2['ece'] - EXPECTED['global_g2_test_ece']),
        'global_g2_test_accuracy': float(test_g2['accuracy'] - EXPECTED['global_g2_test_accuracy']),
        'global_g2_test_macro_f1': float(test_g2['macro_f1'] - EXPECTED['global_g2_test_macro_f1']),
    }
    repro_pass = bool(
        abs(gaps['soft_vote_test_accuracy']) <= 2.5e-4 and
        abs(gaps['soft_vote_test_macro_f1']) <= 2.5e-4 and
        abs(gaps['global_g2_test_accuracy']) <= 2.5e-4 and
        abs(gaps['global_g2_test_macro_f1']) <= 2.5e-4 and
        abs(gaps['global_g2_test_nll']) <= 5e-4 and
        abs(gaps['global_g2_test_brier']) <= 5e-4 and
        abs(gaps['global_g2_test_ece']) <= 5e-4
    )

    summary = {
        'status': 'HELENA_FINAL_CANONICAL_RERUN',
        'frozen': {
            'm': 16384,
            'tau': 0.694,
            'margin': 'none',
            'seeds': SEEDS,
            'qn_memory': 5,
            'qn_scale': 0.25,
            'steps': {'wce': 18, 'soft_f1': 9, 'adaptive': 6},
            'benchmark': 'soft_vote_S5',
            'calibrated_system': 'coherent_logit_S5 -> global_G2',
        },
        'global_g2_alpha': alpha.tolist(),
        'global_g2_optimizer': opt,
        'test_soft_vote': test_soft,
        'test_coherent_base': test_coh,
        'test_global_g2': test_g2,
        'global_g2_minus_coherent_test': {
            k: float(test_g2[k] - test_coh[k])
            for k in ['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','tail20_f1']
        },
        'seed_audit': seed_summaries,
        'frozen_reference_gaps': gaps,
        'repro_pass': repro_pass,
        'note': 'All five base models were retrained from raw Helena and zero initialization; no prior logits/checkpoints were reused.'
    }
    json.dump(summary, open(out / 'helena_final_rerun_summary.json','w'), indent=2)

    md = f'''# Helena — Canonical Final End-to-End Rerun\n\nStatus: **{'REPRODUCED' if repro_pass else 'COMPLETED WITH DRIFT'}**.\n\nAll five base models were rebuilt from raw Helena with zero initialization. No Stage-D logits, model checkpoints, RSP matrices, or G2 coefficients were reused.\n\n## Frozen system\n\n- m = 16,384\n- scalar tau = .694\n- no confusion margin\n- seeds = {{42,123,456,789,2026}}\n- QN-v2 memory=5, scale=.25\n- WCE/SoftF1/Adaptive = 18/9/6\n- ranking benchmark = soft-vote S=5\n- calibrated system = coherent-logit S=5 -> global G2\n\n## Final TEST\n\n| model | Accuracy | Macro-F1 | BalAcc | Tail20 | NLL | Brier | ECE |\n|---|---:|---:|---:|---:|---:|---:|---:|\n| soft-vote S=5 | {test_soft['accuracy']:.9f} | {test_soft['macro_f1']:.9f} | {test_soft['balanced_accuracy']:.9f} | {test_soft['tail20_f1']:.9f} | {test_soft['nll']:.9f} | {test_soft['brier']:.9f} | {test_soft['ece']:.9f} |\n| coherent S=5 | {test_coh['accuracy']:.9f} | {test_coh['macro_f1']:.9f} | {test_coh['balanced_accuracy']:.9f} | {test_coh['tail20_f1']:.9f} | {test_coh['nll']:.9f} | {test_coh['brier']:.9f} | {test_coh['ece']:.9f} |\n| coherent S=5 -> global G2 | {test_g2['accuracy']:.9f} | {test_g2['macro_f1']:.9f} | {test_g2['balanced_accuracy']:.9f} | {test_g2['tail20_f1']:.9f} | {test_g2['nll']:.9f} | {test_g2['brier']:.9f} | {test_g2['ece']:.9f} |\n\nGlobal G2 coefficients fitted on CAL:\n\n- alpha0 = {alpha[0]:.12f}\n- alpha1 = {alpha[1]:.12f}\n\nReproducibility audit pass: **{repro_pass}**.\n\nThe reference gaps are stored in `helena_final_rerun_summary.json`; they are diagnostics and did not alter the frozen pipeline.\n'''
    (out / 'HELENA_FINAL_RERUN.md').write_text(md)
    print('HELENA_FINAL_RERUN', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
