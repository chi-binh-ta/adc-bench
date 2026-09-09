import json
from pathlib import Path
import numpy as np
import pandas as pd

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageB1_qn import train_qn, nll_only

# Stage B2 is a confirmation run. Do not tune anything on seed123.
SEED = 123
B.SEED = SEED

# Three B1-selected policies and their matched scalar controls.
CANDIDATES = [
    {"label": "primary",  "tau0": 0.694, "rho": 0.15, "lambda_shrink": 50.0},
    {"label": "macro",    "tau0": 0.815, "rho": 0.15, "lambda_shrink": 20.0},
    {"label": "high_tau", "tau0": 0.935, "rho": 0.15, "lambda_shrink": 20.0},
]

# Confirmation tolerance carried over from B1's relaxed Pareto screen.
TOL = 5e-4


def make_policy_stats(pi, Praw, Rraw, tp, fp, fn, tau0, rho, lam):
    Pbar = float(np.mean(Praw))
    Rbar = float(np.mean(Rraw))
    Ptilde = (tp + lam * Pbar) / (tp + fp + lam)
    Rtilde = (tp + lam * Rbar) / (tp + fn + lam)
    tauk = np.clip(tau0 + rho * np.log((Ptilde + B.EPS) / (Rtilde + B.EPS)), 0.0, 1.0)
    etaR = 0.05 + 0.15 * tauk
    etaP = 0.20 - 0.15 * tauk
    factor = ((Rbar + B.EPS) / (Rraw + B.EPS)) ** etaR * ((Praw + B.EPS) / (Pbar + B.EPS)) ** etaP
    factor = np.clip(factor, B.CLIP_LO, B.CLIP_HI)
    return tauk, etaR, etaP, B.normalized_class_weights((pi ** (-B.ALPHA)) * factor, YTRAIN)


def run_adaptive(label, tau0, rho, lam, theta, bias, F, V, qminus, pi, Praw, Rraw, tp, fp, fn, train_support, ycal, ymeta):
    tauk, etaR, etaP, cw = make_policy_stats(pi, Praw, Rraw, tp, fp, fn, tau0, rho, lam)
    th, bi, hist = train_qn(
        f"B2_{label}_t{tau0}_r{rho}_l{lam}",
        F['train'], YTRAIN, theta, bias, V, qminus, B.ADAPT_STEPS,
        lambda tt, bb, cw=cw: B.wce_loss_grad(F['train'], YTRAIN, tt, bb, V, qminus, cw),
    )
    pcal = B.predict_proba(F['cal'], th, bi, V, qminus)
    pmeta = B.predict_proba(F['meta'], th, bi, V, qminus)
    mcal = B.metrics(ycal, pcal, train_support)
    return {
        "seed": SEED,
        "m": B.M,
        "label": label,
        "tau0": tau0,
        "rho": rho,
        "lambda_shrink": lam,
        "tauk_mean": float(tauk.mean()),
        "tauk_min": float(tauk.min()),
        "tauk_max": float(tauk.max()),
        "etaR_mean": float(etaR.mean()),
        "etaP_mean": float(etaP.mean()),
        "meta_nll_after": nll_only(ymeta, pmeta),
        **{"cal_" + k: v for k, v in mcal.items()},
    }, th, bi, hist


def main():
    global YTRAIN
    print("STAGE_B2_START", json.dumps({"seed": SEED, "m": B.M, "candidates": CANDIDATES}), flush=True)

    X, y, tr, meta, cal, te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    YTRAIN = y[tr]
    Xt = X[tr]

    # Frozen Stage-A representation, but experiment-seed-specific nested landmarks/RSP.
    li, C = B.build_landmarks(Xt, YTRAIN)
    paths, mean, std = B.build_feature_cache(X, (tr, meta, cal, te), C)
    sizes = dict(train=len(tr), meta=len(meta), cal=len(cal), test=len(te))
    F = B.open_features(paths, sizes)
    V, qminus, rsp = B.build_rsp(F['train'], YTRAIN)

    K = int(y.max() + 1)
    pi, train_support = B.class_priors(YTRAIN, K)
    cw_base = B.normalized_class_weights(pi ** (-B.ALPHA), YTRAIN)

    theta = np.zeros((B.M, K), np.float32)
    bias = np.zeros(K, np.float32)

    # QN-v2 is frozen from seed42; no re-anchoring on seed123.
    theta, bias, hw = train_qn(
        'B2_WCE', F['train'], YTRAIN, theta, bias, V, qminus, B.WCE_STEPS,
        lambda th, bi: B.wce_loss_grad(F['train'], YTRAIN, th, bi, V, qminus, cw_base),
    )
    pmeta_wce = B.predict_proba(F['meta'], theta, bias, V, qminus)
    wce_meta_nll = nll_only(y[meta], pmeta_wce)

    theta, bias, hs = train_qn(
        'B2_SoftMacroF1', F['train'], YTRAIN, theta, bias, V, qminus, B.SOFT_STEPS,
        lambda th, bi: B.softf1_loss_grad(F['train'], YTRAIN, th, bi, V, qminus, cw_base, B.LAM_F),
    )
    pmeta_soft = B.predict_proba(F['meta'], theta, bias, V, qminus)
    soft_meta_nll = nll_only(y[meta], pmeta_soft)
    Praw, Rraw, Fraw, supp, tp, fp, fn = B.pr_from_probs(y[meta], pmeta_soft, K)

    pd.DataFrame({
        'cls': np.arange(K), 'precision': Praw, 'recall': Rraw, 'f1': Fraw,
        'support': supp, 'tp': tp, 'fp': fp, 'fn': fn, 'pi': pi,
    }).to_csv(B.OUT / 'stageB2_seed123_meta_pr_soft.csv', index=False)

    rows = []
    training_log = hw + hs
    checkpoints = {}

    for c in CANDIDATES:
        tau0 = float(c['tau0'])
        rho = float(c['rho'])
        lam = float(c['lambda_shrink'])
        label = c['label']

        # Matched scalar control. rho=0 makes lambda irrelevant, but keep same lambda for audit clarity.
        ctrl, th0, bi0, h0 = run_adaptive(
            label + '_scalar', tau0, 0.0, lam, theta, bias, F, V, qminus,
            pi, Praw, Rraw, tp, fp, fn, train_support, y[cal], y[meta]
        )
        cand, th1, bi1, h1 = run_adaptive(
            label, tau0, rho, lam, theta, bias, F, V, qminus,
            pi, Praw, Rraw, tp, fp, fn, train_support, y[cal], y[meta]
        )
        training_log += h0 + h1

        # Paired deltas: candidate - matched scalar control.
        for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'tail20_f1', 'nll', 'brier', 'ece']:
            cand['delta_cal_' + metric] = cand['cal_' + metric] - ctrl['cal_' + metric]
        cand['confirmed_relaxed_pareto'] = bool(
            (cand['delta_cal_accuracy'] > 0 or cand['delta_cal_macro_f1'] > 0 or cand['delta_cal_tail20_f1'] > 0)
            and cand['delta_cal_accuracy'] >= -TOL
            and cand['delta_cal_macro_f1'] >= -TOL
            and cand['delta_cal_tail20_f1'] >= -TOL
        )
        cand['confirmed_strict_ranking'] = bool(
            cand['delta_cal_accuracy'] >= 0
            and cand['delta_cal_macro_f1'] > 0
            and cand['delta_cal_balanced_accuracy'] >= 0
            and cand['delta_cal_tail20_f1'] >= 0
        )

        ctrl['role'] = 'control'
        cand['role'] = 'candidate'
        rows.extend([ctrl, cand])
        checkpoints[label] = (th1, bi1)

        print('B2_PAIR', json.dumps({
            'label': label,
            'control': {k: ctrl[k] for k in ['cal_accuracy','cal_macro_f1','cal_balanced_accuracy','cal_tail20_f1','cal_nll','cal_ece']},
            'candidate': {k: cand[k] for k in ['cal_accuracy','cal_macro_f1','cal_balanced_accuracy','cal_tail20_f1','cal_nll','cal_ece']},
            'deltas': {k: cand[k] for k in cand if k.startswith('delta_cal_')},
            'relaxed': cand['confirmed_relaxed_pareto'],
            'strict': cand['confirmed_strict_ranking'],
        }), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(B.OUT / 'stageB2_seed123_6rows_cal.csv', index=False)
    json.dump(training_log, open(B.OUT / 'stageB2_seed123_training_log.json', 'w'), indent=2)

    candidates_df = df[df.role == 'candidate'].copy()
    confirmed = candidates_df[candidates_df.confirmed_relaxed_pareto == True].copy()

    # B2 does not select by seed123 alone; it only confirms/rejects the B1 shortlist.
    summary = {
        'status': 'STAGE_B2_SEED123_CONFIRMATION',
        'seed': SEED,
        'm': B.M,
        'rsp': rsp,
        'wce_meta_nll': wce_meta_nll,
        'soft_meta_nll': soft_meta_nll,
        'n_pairs': len(CANDIDATES),
        'n_confirmed_relaxed': int(candidates_df.confirmed_relaxed_pareto.sum()),
        'n_confirmed_strict': int(candidates_df.confirmed_strict_ranking.sum()),
        'pairs': candidates_df.to_dict(orient='records'),
        'decision_rule': {
            'relaxed_tolerance': TOL,
            'b3_note': 'Use seed42 + seed123 paired deltas jointly; do not retune on seed123.'
        }
    }
    json.dump(summary, open(B.OUT / 'stageB2_seed123_summary.json', 'w'), indent=2)
    print('STAGE_B2_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
