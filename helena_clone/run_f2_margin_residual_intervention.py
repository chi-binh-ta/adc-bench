import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageE_cluster_g2 import cluster_partition
import run_latent_factor_identification_v2 as F

SEEDS = [42, 123, 456, 789, 2026]
ALPHA_G2 = np.array([-0.9026209634632217, 0.765345197612019], dtype=np.float64)
LAMBDAS = np.array([-0.30,-0.20,-0.15,-0.10,-0.05,-0.025,0.0,0.025,0.05,0.10,0.15,0.20,0.30], dtype=np.float64)
ALPHAS = np.logspace(-3, 3, 13)
OUTER_SEED = 20260910
CLASS_SEED = 20260911
BOOT_SEED = 20260912
N_BOOT = 1000

GROUPS = F.GROUPS
MARGIN_COLS = GROUPS['margin']
CONTROL_GROUPS = ['prevalence','class_similarity','seed_instability','calibration_residual','representation_geometry']
CONTROL_COLS = []
for _g in CONTROL_GROUPS:
    CONTROL_COLS.extend(GROUPS[_g])


def softmax64(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_seed_logits(artifact_dir):
    cal, test = [], []
    for seed in SEEDS:
        d = np.load(artifact_dir / f'stageD_seed{seed}_predictions.npz')
        zc = d['cal_logits'].astype(np.float64)
        zt = d['test_logits'].astype(np.float64)
        zc -= zc.mean(axis=1, keepdims=True)
        zt -= zt.mean(axis=1, keepdims=True)
        cal.append(zc); test.append(zt)
    return np.stack(cal), np.stack(test)


def global_bases_fast(z):
    # Exact complete-graph global G2 bases without materializing 12 cluster bases.
    z = z.astype(np.float64, copy=True)
    z -= z.mean(axis=1, keepdims=True)
    n, K = z.shape
    t0 = z.copy()
    t1 = np.zeros_like(z)
    invK = 1.0 / K
    for i in range(K):
        for j in range(i + 1, K):
            L = z[:, i] - z[:, j]
            s = expit(L)
            u = 1.0 - 4.0 * s * (1.0 - s)
            r = invK * u * L
            t1[:, i] += r
            t1[:, j] -= r
    return np.stack([t0, t1], axis=2)


def global_probs(z):
    z = z.astype(np.float64, copy=True)
    z -= z.mean(axis=1, keepdims=True)
    T = global_bases_fast(z)
    t = np.einsum('nkj,j->nk', T, ALPHA_G2, optimize=True)
    return softmax64(z - t)


def metrics(y, p, train_support):
    return B.metrics(y, p.astype(np.float32), train_support)


def impute_pair(Fit, cols):
    X = Fit[cols].to_numpy(dtype=np.float64, copy=True)
    for j in range(X.shape[1]):
        finite = np.isfinite(X[:, j])
        med = float(np.median(X[finite, j])) if finite.any() else 0.0
        X[~finite, j] = med
    return X


def ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


def choose_alpha(X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=CLASS_SEED)
    best = None
    for a in ALPHAS:
        score = float(np.mean(cross_val_score(ridge(a), X, y, cv=cv, scoring='r2')))
        if best is None or score > best[0]:
            best = (score, float(a))
    return best


def oof_predict(X, y, alpha):
    cv = KFold(n_splits=5, shuffle=True, random_state=CLASS_SEED)
    return cross_val_predict(ridge(alpha), X, y, cv=cv)


def residual_margin_design(Fc):
    XC = impute_pair(Fc, CONTROL_COLS)
    XM = impute_pair(Fc, MARGIN_COLS)
    RM = np.empty_like(XM)
    nuisance_alphas = []
    for j in range(XM.shape[1]):
        _, a = choose_alpha(XC, XM[:, j])
        nuisance_alphas.append(a)
        RM[:, j] = XM[:, j] - oof_predict(XC, XM[:, j], a)
    return XC, RM, nuisance_alphas


def orient_v1(v, Fc):
    m = Fc['margin_mean'].to_numpy(dtype=np.float64)
    c = float(np.corrcoef(v, m)[0,1]) if np.std(m) > 0 and np.std(v) > 0 else 0.0
    if np.isfinite(c) and c < 0:
        return -v, -c
    return v, abs(c) if np.isfinite(c) else 0.0


def unique_margin_component(v, Fc, XC, RM):
    _, ac = choose_alpha(XC, v)
    rv = v - oof_predict(XC, v, ac)
    cv_r2, am = choose_alpha(RM, rv)
    d = oof_predict(RM, rv, am)
    return d, {'control_alpha':ac, 'margin_alpha':am, 'margin_residual_cv_r2':cv_r2,
               'rv_sd':float(np.std(rv)), 'd_sd':float(np.std(d))}


def derive_direction(X, y, tr, split_idx, z, zseeds, p, K, train_support, rawgeo):
    Fc = F.make_features(X, y, tr, split_idx, z, zseeds, p, K, train_support, rawgeo)
    soft, hard = F.residual_mats(y[split_idx], p, K)
    _, ss, Vs = F.svd_factors(soft)
    _, sh, Vh = F.svd_factors(hard)
    vs, orient_s = orient_v1(Vs[:,0].astype(np.float64), Fc)
    vh, orient_h = orient_v1(Vh[:,0].astype(np.float64), Fc)
    XC, RM, nuisance_alphas = residual_margin_design(Fc)
    ds, diag_s = unique_margin_component(vs, Fc, XC, RM)
    dh, diag_h = unique_margin_component(vh, Fc, XC, RM)
    ns = float(np.linalg.norm(ds)); nh = float(np.linalg.norm(dh))
    if ns < 1e-12 or nh < 1e-12:
        raise RuntimeError('Degenerate unique-margin direction')
    us = ds / ns; uh = dh / nh
    raw_dot = float(np.dot(us, uh))
    if raw_dot < 0:
        uh = -uh
    aligned_dot = float(np.dot(us, uh))
    d = 0.5 * (us + uh)
    d -= d.mean()
    sd = float(d.std())
    if sd < 1e-12:
        raise RuntimeError('Degenerate combined intervention direction')
    d /= sd
    diag = {
        'soft_singular1':float(ss[0]), 'hard_singular1':float(sh[0]),
        'soft_v1_marginmean_abs_corr':orient_s, 'hard_v1_marginmean_abs_corr':orient_h,
        'soft_component':diag_s, 'hard_component':diag_h,
        'soft_hard_raw_dot':raw_dot, 'soft_hard_aligned_dot':aligned_dot,
        'combined_mean':float(d.mean()), 'combined_sd':float(d.std()),
        'nuisance_alphas':nuisance_alphas,
    }
    return d, ds, dh, diag


def top2_gap(z):
    top2 = np.partition(z, -2, axis=1)[:, -2:]
    return top2.max(axis=1) - top2.min(axis=1)


def gate_scale(z):
    return float(np.median(top2_gap(z)) + 1e-8)


def intervene(z, d, scale, lam):
    z = z.astype(np.float64, copy=True)
    z -= z.mean(axis=1, keepdims=True)
    a = np.exp(-top2_gap(z) / scale)
    zz = z + float(lam) * a[:,None] * d[None,:]
    zz -= zz.mean(axis=1, keepdims=True)
    return zz, a


def metric_delta(candidate, base):
    keys = ['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','tail20_f1']
    return {k:float(candidate[k]-base[k]) for k in keys}


def passes_gate(delta):
    return bool(
        delta['macro_f1'] > 0 and
        delta['balanced_accuracy'] > 0 and
        delta['tail20_f1'] > 0 and
        delta['accuracy'] >= -0.0005 and
        delta['nll'] <= 0.003 and
        delta['brier'] <= 0.0008
    )


def sample_losses(y, p):
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    one = np.eye(p.shape[1], dtype=np.float64)[y]
    brier = np.sum((p - one)**2, axis=1)
    return nll, brier


def paired_mean_ci(a, b, rng, Bn=N_BOOT):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d); vals = np.empty(Bn)
    for r in range(Bn):
        ix = rng.integers(0, n, n)
        vals[r] = d[ix].mean()
    return [float(np.quantile(vals,.025)), float(np.quantile(vals,.975))]


def bootstrap_metric_delta(y, p1, p0, train_support, rng, Bn=N_BOOT):
    keys = ['accuracy','macro_f1','balanced_accuracy','tail20_f1']
    vals = {k:np.empty(Bn, dtype=np.float64) for k in keys}
    n = len(y)
    for b in range(Bn):
        ix = rng.integers(0, n, n)
        m1 = metrics(y[ix], p1[ix], train_support)
        m0 = metrics(y[ix], p0[ix], train_support)
        for k in keys:
            vals[k][b] = m1[k] - m0[k]
    return {k:[float(np.quantile(v,.025)), float(np.quantile(v,.975))] for k,v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', type=Path, required=True)
    args = ap.parse_args()
    out = Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)

    X, y, tr, meta, cal, te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    K = int(y.max() + 1)
    train_support = np.bincount(y[tr], minlength=K).astype(np.float64)
    clusters, head, mid, tail = cluster_partition(train_support)
    rawgeo = F.raw_geometry_features(X, y, tr, K)
    zsc, zst = load_seed_logits(args.artifact_dir)
    zcal = zsc.mean(axis=0); ztest = zst.mean(axis=0)
    ycal = y[cal]; ytest = y[te]

    # Canonical control replay.
    pbase_cal = global_probs(zcal)
    base_cal = metrics(ycal, pbase_cal, train_support)
    print('F2_CANONICAL_CAL', json.dumps(base_cal), flush=True)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    poof = {float(lam):np.zeros((len(ycal),K), dtype=np.float64) for lam in LAMBDAS}
    fold_rows = []

    for fold, (fi, vi) in enumerate(outer.split(np.zeros(len(ycal)), ycal), 1):
        fit_global = cal[fi]
        zfit = zcal[fi]; zval = zcal[vi]
        zsfit = zsc[:,fi,:]
        pfit = global_probs(zfit)
        d, ds, dh, ddiag = derive_direction(X, y, tr, fit_global, zfit, zsfit, pfit,
                                             K, train_support, rawgeo)
        scale = gate_scale(zfit)
        val_gate = np.exp(-top2_gap(zval) / scale)
        row = {
            'fold':fold, 'n_fit':len(fi), 'n_val':len(vi), 'gate_scale':scale,
            'val_gate_mean':float(val_gate.mean()), 'val_gate_q90':float(np.quantile(val_gate,.90)),
            'direction_min':float(d.min()), 'direction_max':float(d.max()),
            **{f'diag_{k}':json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in ddiag.items()}
        }
        fold_rows.append(row)
        print('F2_FOLD_DIRECTION', json.dumps({'fold':fold,'gate_scale':scale,'diag':ddiag}), flush=True)
        for lam in LAMBDAS:
            zv, _ = intervene(zval, d, scale, lam)
            poof[float(lam)][vi] = global_probs(zv)

    pd.DataFrame(fold_rows).to_csv(out/'f2_fold_diagnostics.csv', index=False)

    # lambda=0 must replay the canonical CAL system exactly up to numerical tolerance.
    oof0 = poof[0.0]
    replay = metrics(ycal, oof0, train_support)
    replay_gap = {k:float(replay[k]-base_cal[k]) for k in ['nll','brier','accuracy','macro_f1','balanced_accuracy','tail20_f1']}
    if max(abs(v) for v in replay_gap.values()) > 5e-7:
        raise RuntimeError(f'F2 lambda=0 replay mismatch: {replay_gap}')

    rows=[]
    for lam in LAMBDAS:
        m = metrics(ycal, poof[float(lam)], train_support)
        dlt = metric_delta(m, replay)
        passed = False if lam == 0 else passes_gate(dlt)
        rows.append({'lambda':float(lam), **m,
                     **{f'delta_{k}':v for k,v in dlt.items()},
                     'passes_nonzero_gate':passed})
    tab = pd.DataFrame(rows)
    tab.to_csv(out/'f2_oof_lambda_metrics.csv', index=False)

    candidates = [r for r in rows if r['passes_nonzero_gate']]
    if candidates:
        winner = sorted(candidates, key=lambda r:(-r['macro_f1'],-r['tail20_f1'],-r['balanced_accuracy'],r['nll'],abs(r['lambda'])))[0]
        lam_star = float(winner['lambda']); status = 'F2_POSITIVE'
    else:
        winner = next(r for r in rows if r['lambda'] == 0.0)
        lam_star = 0.0; status = 'F2_NEGATIVE'

    decision = {
        'status':status,
        'system':'canonical coherent S5 -> global G2',
        'direction':'cross-fitted V1 unique-margin component, soft/hard consensus',
        'gate':'a=exp(-top2_gap/median_fit_gap)',
        'lambda_grid':[float(x) for x in LAMBDAS],
        'lambda_star':lam_star,
        'oof_control':replay,
        'oof_winner':winner,
        'gate_rule':{
            'macro_f1_gt_0':True,'balanced_accuracy_gt_0':True,'tail20_f1_gt_0':True,
            'accuracy_min_delta':-0.0005,'nll_max_delta':0.003,'brier_max_delta':0.0008
        },
        'n_nonzero_passing':len(candidates),
        'test_policy':'lambda frozen from cross-fitted CAL before F2 TEST audit; TEST cannot change decision',
        'provenance_note':'Helena TEST was already opened by earlier stages; F2 TEST is an internal replication audit.'
    }
    with open(out/'f2_cal_decision.json','w') as f:
        json.dump(decision,f,indent=2)
    print('F2_CAL_DECISION', json.dumps(decision), flush=True)

    # Decision is frozen. Refit direction on all CAL only, then open F2 TEST.
    pcal = global_probs(zcal)
    d_full, ds_full, dh_full, diag_full = derive_direction(X, y, tr, cal, zcal, zsc, pcal,
                                                            K, train_support, rawgeo)
    scale_full = gate_scale(zcal)
    pd.DataFrame({
        'class':np.arange(K), 'train_support':train_support.astype(int),
        'cluster':np.array(['H','M','T'])[clusters],
        'd_consensus':d_full, 'd_soft_raw':ds_full, 'd_hard_raw':dh_full
    }).to_csv(out/'f2_full_direction.csv', index=False)

    print('F2_TEST_OPENED_AFTER_CAL_FREEZE', flush=True)
    p0_test = global_probs(ztest)
    z1_test, gate_test = intervene(ztest, d_full, scale_full, lam_star)
    p1_test = global_probs(z1_test)
    m0 = metrics(ytest, p0_test, train_support)
    m1 = metrics(ytest, p1_test, train_support)
    delta_test = metric_delta(m1, m0)
    pd.DataFrame([
        {'model':'control_global_g2', **m0},
        {'model':'f2_margin_residual', **m1},
        {'model':'delta_f2_minus_control', **delta_test}
    ]).to_csv(out/'f2_test_metrics.csv', index=False)

    rng = np.random.default_rng(BOOT_SEED)
    nll1, br1 = sample_losses(ytest,p1_test); nll0, br0 = sample_losses(ytest,p0_test)
    boot = {
        'nll_delta_ci95':paired_mean_ci(nll1,nll0,rng),
        'brier_delta_ci95':paired_mean_ci(br1,br0,rng),
        'ranking_metric_delta_ci95':bootstrap_metric_delta(ytest,p1_test,p0_test,train_support,rng),
        'n_boot':N_BOOT
    }
    with open(out/'f2_test_bootstrap.json','w') as f:
        json.dump(boot,f,indent=2)

    final = {
        'status':status,
        'lambda_star':lam_star,
        'oof_cal_decision':decision,
        'full_cal_direction_diag':diag_full,
        'full_cal_gate_scale':scale_full,
        'test_gate_mean':float(gate_test.mean()),
        'test_gate_q90':float(np.quantile(gate_test,.90)),
        'test_control':m0,
        'test_intervention':m1,
        'test_delta':delta_test,
        'bootstrap':boot,
        'interpretation':'Actionability test only; positive does not establish causal identification.'
    }
    with open(out/'F2_MARGIN_RESIDUAL_INTERVENTION.json','w') as f:
        json.dump(final,f,indent=2)
    print('F2_FINAL', json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
