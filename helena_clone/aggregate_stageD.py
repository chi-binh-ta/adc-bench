import json, re
from pathlib import Path
import numpy as np
import pandas as pd

import run_stageB1_clone as B
from numeric_split import load_split_numeric

SEEDS = [42, 123, 456, 789, 2026]
PREFIXES = [2, 3, 5]
TOL = 5e-4
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'helena_clone' / 'stageD_artifacts'
OUT = ROOT / 'helena_clone' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)


def load_logits(seed):
    hits = list(ART.rglob(f'stageD_seed{seed}_predictions.npz'))
    if len(hits) != 1:
        raise RuntimeError(f'expected one prediction artifact for seed {seed}, found {hits}')
    d = np.load(hits[0])
    return d['cal_logits'].astype(np.float32), d['test_logits'].astype(np.float32)


def probs_soft(zs):
    ps = [B.softmax(z.astype(np.float64)) for z in zs]
    return np.mean(ps, axis=0).astype(np.float32)


def probs_coherent(zs):
    centered = [z.astype(np.float64) - z.astype(np.float64).mean(axis=1, keepdims=True) for z in zs]
    zbar = np.mean(centered, axis=0)
    return B.softmax(zbar).astype(np.float32)


def metric_row(split, mode, S, p, y, support, prefix):
    m = B.metrics(y, p, support)
    return {'split': split, 'mode': mode, 'S': S, 'seed_prefix': '-'.join(map(str, prefix)), **m}


def utility_pass(m, mean_single):
    return bool(
        m['nll'] < mean_single['nll'] and
        m['brier'] < mean_single['brier'] and
        m['accuracy'] >= mean_single['accuracy'] - TOL and
        m['macro_f1'] >= mean_single['macro_f1'] - TOL and
        m['tail20_f1'] >= mean_single['tail20_f1'] - TOL
    )


def near_dom(a, b):
    return bool(a['accuracy'] >= b['accuracy'] - TOL and a['nll'] < b['nll'] and a['brier'] < b['brier'])


def main():
    X, y, tr, meta, cal, te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    _, train_support = B.class_priors(y[tr], int(y.max()+1))

    cal_logits = {}
    test_logits = {}
    rows = []
    single_cal = []
    for s in SEEDS:
        zc, zt = load_logits(s)
        cal_logits[s] = zc
        test_logits[s] = zt
        pc = B.softmax(zc.astype(np.float64)).astype(np.float32)
        r = metric_row('cal', 'single', 1, pc, y[cal], train_support, [s])
        r['seed'] = s
        rows.append(r)
        single_cal.append(r)

    metric_names = ['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','weighted_f1','macro_precision','macro_recall','tail20_f1']
    mean_single = {k: float(np.mean([r[k] for r in single_cal])) for k in metric_names}
    best_single = {
        'accuracy': float(max(r['accuracy'] for r in single_cal)),
        'nll': float(min(r['nll'] for r in single_cal)),
        'brier': float(min(r['brier'] for r in single_cal)),
        'macro_f1': float(max(r['macro_f1'] for r in single_cal)),
        'tail20_f1': float(max(r['tail20_f1'] for r in single_cal)),
    }

    cal_final = {}
    for S in PREFIXES:
        prefix = SEEDS[:S]
        zs = [cal_logits[s] for s in prefix]
        ps = probs_soft(zs)
        pl = probs_coherent(zs)
        rs = metric_row('cal', 'soft_vote', S, ps, y[cal], train_support, prefix)
        rl = metric_row('cal', 'coherent_logit', S, pl, y[cal], train_support, prefix)
        rows += [rs, rl]
        if S == 5:
            cal_final['soft_vote'] = rs
            cal_final['coherent_logit'] = rl

    passes = {k: utility_pass(v, mean_single) for k,v in cal_final.items()}
    soft = cal_final['soft_vote']; coh = cal_final['coherent_logit']
    if near_dom(soft, coh) and not near_dom(coh, soft):
        benchmark_choice = 'soft_vote'
    elif near_dom(coh, soft) and not near_dom(soft, coh):
        benchmark_choice = 'coherent_logit'
    else:
        benchmark_choice = 'pareto_tie'

    cal_decision = {
        'status': 'STAGE_D_CAL_FROZEN',
        'seeds': SEEDS,
        'final_S': 5,
        'mean_single_cal': mean_single,
        'best_single_cal': best_single,
        'soft_vote_cal': soft,
        'coherent_logit_cal': coh,
        'utility_pass': passes,
        'benchmark_choice': benchmark_choice,
        'test_policy': 'test may now be opened; its metrics cannot change seed set, S, or benchmark_choice',
    }
    with open(OUT/'stageD_cal_decision.json','w') as f:
        json.dump(cal_decision, f, indent=2)
    print('STAGE_D_CAL_DECISION', json.dumps(cal_decision), flush=True)

    # TEST IS OPENED ONLY AFTER cal_decision has been written/frozen above.
    print('TEST_OPENED_AFTER_CAL_FREEZE', flush=True)
    test_final = {}
    for mode in ['soft_vote','coherent_logit']:
        zs = [test_logits[s] for s in SEEDS]
        p = probs_soft(zs) if mode == 'soft_vote' else probs_coherent(zs)
        r = metric_row('test', mode, 5, p, y[te], train_support, SEEDS)
        rows.append(r)
        test_final[mode] = r

    df = pd.DataFrame(rows)
    df.to_csv(OUT/'stageD_ensemble_results.csv', index=False)

    final = {
        **cal_decision,
        'test_final_frozen_variants': test_final,
        'stageD_pass': bool(any(passes.values())),
        'stageE_structural_input': 'coherent_logit_S5',
    }
    with open(OUT/'stageD_final_summary.json','w') as f:
        json.dump(final, f, indent=2)

    md = []
    md.append('# Stage D — Multi-seed Ensemble — Final')
    md.append('')
    md.append(f"Status: {'PASS' if final['stageD_pass'] else 'NO ENSEMBLE WINNER'}.")
    md.append('')
    md.append('Frozen seeds: `42, 123, 456, 789, 2026`; final S=5. Prefix S=2,3 is diagnostic only.')
    md.append('')
    md.append('## Calibration decision (frozen before test)')
    md.append('')
    md.append('| mode | Acc | MacroF1 | Tail20 | NLL | Brier | ECE | utility pass |')
    md.append('|---|---:|---:|---:|---:|---:|---:|:---:|')
    for mode in ['soft_vote','coherent_logit']:
        r=cal_final[mode]
        md.append(f"| {mode} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['tail20_f1']:.6f} | {r['nll']:.6f} | {r['brier']:.6f} | {r['ece']:.6f} | {'YES' if passes[mode] else 'NO'} |")
    md.append('')
    md.append(f"Benchmark choice frozen on CAL: **{benchmark_choice}**.")
    md.append('')
    md.append('## Test results after freeze')
    md.append('')
    md.append('| mode | Acc | MacroF1 | BalAcc | Tail20 | NLL | Brier | ECE |')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for mode in ['soft_vote','coherent_logit']:
        r=test_final[mode]
        md.append(f"| {mode} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['balanced_accuracy']:.6f} | {r['tail20_f1']:.6f} | {r['nll']:.6f} | {r['brier']:.6f} | {r['ece']:.6f} |")
    md.append('')
    md.append('Stage E structural input is fixed to the S=5 coherent-logit ensemble, regardless of which benchmark variant wins, because G2 acts on the logit/edge field.')
    with open(OUT/'STAGE_D_FINAL.md','w') as f:
        f.write('\n'.join(md)+'\n')

    print('STAGE_D_FINAL', json.dumps(final), flush=True)

if __name__ == '__main__':
    main()
