import os, json
from pathlib import Path
import numpy as np

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageB1_qn import train_qn, nll_only

SEED = int(os.environ['STAGE_D_SEED'])
TAU = 0.694
B.SEED = SEED


def predict_logits(F, theta, bias, V, qminus):
    W = B.apply_P(theta, V, qminus)
    out = np.empty((len(F), bias.shape[0]), np.float32)
    for s in range(0, len(F), B.SAMPLE_BLOCK):
        e = min(s + B.SAMPLE_BLOCK, len(F))
        out[s:e] = (np.asarray(F[s:e], dtype=np.float32) @ W + bias).astype(np.float32)
    return out


def scalar_adaptive_weights(pi, ytr, Praw, Rraw):
    Pbar = float(np.mean(Praw))
    Rbar = float(np.mean(Rraw))
    etaR = 0.05 + 0.15 * TAU
    etaP = 0.20 - 0.15 * TAU
    factor = ((Rbar + B.EPS) / (Rraw + B.EPS)) ** etaR * ((Praw + B.EPS) / (Pbar + B.EPS)) ** etaP
    factor = np.clip(factor, B.CLIP_LO, B.CLIP_HI)
    cw = B.normalized_class_weights((pi ** (-B.ALPHA)) * factor, ytr)
    return cw, etaR, etaP


def main():
    print('STAGE_D_SEED_START', json.dumps({'seed': SEED, 'm': B.M, 'tau': TAU}), flush=True)
    X, y, tr, meta, cal, te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    ytr = y[tr]
    Xt = X[tr]

    _, C = B.build_landmarks(Xt, ytr)
    paths, _, _ = B.build_feature_cache(X, (tr, meta, cal, te), C)
    sizes = dict(train=len(tr), meta=len(meta), cal=len(cal), test=len(te))
    F = B.open_features(paths, sizes)
    V, qminus, rsp = B.build_rsp(F['train'], ytr)

    K = int(y.max() + 1)
    pi, train_support = B.class_priors(ytr, K)
    cw_base = B.normalized_class_weights(pi ** (-B.ALPHA), ytr)
    theta = np.zeros((B.M, K), np.float32)
    bias = np.zeros(K, np.float32)

    theta, bias, hw = train_qn(
        f'D_WCE_seed{SEED}', F['train'], ytr, theta, bias, V, qminus, B.WCE_STEPS,
        lambda th, bi: B.wce_loss_grad(F['train'], ytr, th, bi, V, qminus, cw_base),
    )
    wce_meta_nll = nll_only(y[meta], B.predict_proba(F['meta'], theta, bias, V, qminus))

    theta, bias, hs = train_qn(
        f'D_SoftF1_seed{SEED}', F['train'], ytr, theta, bias, V, qminus, B.SOFT_STEPS,
        lambda th, bi: B.softf1_loss_grad(F['train'], ytr, th, bi, V, qminus, cw_base, B.LAM_F),
    )
    pmeta = B.predict_proba(F['meta'], theta, bias, V, qminus)
    soft_meta_nll = nll_only(y[meta], pmeta)
    Praw, Rraw, _, _, _, _, _ = B.pr_from_probs(y[meta], pmeta, K)
    cw_adapt, etaR, etaP = scalar_adaptive_weights(pi, ytr, Praw, Rraw)

    theta, bias, ha = train_qn(
        f'D_Adaptive_tau0694_seed{SEED}', F['train'], ytr, theta, bias, V, qminus, B.ADAPT_STEPS,
        lambda th, bi: B.wce_loss_grad(F['train'], ytr, th, bi, V, qminus, cw_adapt),
    )

    zcal = predict_logits(F['cal'], theta, bias, V, qminus)
    ztest = predict_logits(F['test'], theta, bias, V, qminus)
    pcal = B.softmax(zcal.astype(np.float64)).astype(np.float32)
    cal_metrics = B.metrics(y[cal], pcal, train_support)
    adapt_meta_nll = nll_only(y[meta], B.predict_proba(F['meta'], theta, bias, V, qminus))

    pred_path = B.OUT / f'stageD_seed{SEED}_predictions.npz'
    np.savez_compressed(pred_path, cal_logits=zcal, test_logits=ztest)
    summary = {
        'status': 'STAGE_D_SINGLE_SEED',
        'seed': SEED,
        'm': B.M,
        'tau': TAU,
        'etaR': etaR,
        'etaP': etaP,
        'rsp': rsp,
        'wce_meta_nll': wce_meta_nll,
        'soft_meta_nll': soft_meta_nll,
        'adapt_meta_nll': adapt_meta_nll,
        'cal_metrics': cal_metrics,
        'prediction_file': pred_path.name,
    }
    with open(B.OUT / f'stageD_seed{SEED}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(B.OUT / f'stageD_seed{SEED}_training.json', 'w') as f:
        json.dump(hw + hs + ha, f, indent=2)
    print('STAGE_D_SEED_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
