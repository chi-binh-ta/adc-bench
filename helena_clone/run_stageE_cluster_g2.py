import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import run_stageB1_clone as B
from numeric_split import load_split_numeric

SEEDS = [42, 123, 456, 789, 2026]
PAIR_NAMES = ['HH','HM','HT','MM','MT','TT']
EXPECTED_COH_CAL = {
    'nll': 2.7911226749420166,
    'accuracy': 0.35603271983640083,
    'macro_f1': 0.192839567750495,
    'tail20_f1': 0.10589625579099263,
}


def load_coherent_logits(artifact_dir):
    cal, test = [], []
    for seed in SEEDS:
        path = artifact_dir / f'stageD_seed{seed}_predictions.npz'
        if not path.exists():
            raise FileNotFoundError(path)
        d = np.load(path)
        zc = d['cal_logits'].astype(np.float64)
        zt = d['test_logits'].astype(np.float64)
        zc -= zc.mean(axis=1, keepdims=True)
        zt -= zt.mean(axis=1, keepdims=True)
        cal.append(zc); test.append(zt)
    return np.mean(cal, axis=0), np.mean(test, axis=0)


def cluster_partition(train_support):
    K = len(train_support)
    order = np.lexsort((np.arange(K), train_support))
    tail = order[:20]
    head = order[-20:]
    mid = order[20:-20]
    c = np.empty(K, dtype=np.int64)
    c[head] = 0; c[mid] = 1; c[tail] = 2
    return c, head, mid, tail


def pair_group(ci, cj):
    a, b = sorted((int(ci), int(cj)))
    # 0=H,1=M,2=T
    return {(0,0):0,(0,1):1,(0,2):2,(1,1):3,(1,2):4,(2,2):5}[(a,b)]


def build_potential_bases(z, clusters):
    # T[n,k,2*g] = potential basis from 1{pair group g} * L
    # T[n,k,2*g+1] = potential basis from 1{pair group g} * u * L
    n, K = z.shape
    T = np.zeros((n, K, 12), dtype=np.float32)
    invK = 1.0 / K
    for i in range(K):
        for j in range(i+1, K):
            g = pair_group(clusters[i], clusters[j])
            L = z[:, i] - z[:, j]
            s = expit(L)
            u = 1.0 - 4.0 * s * (1.0 - s)
            r0 = (invK * L).astype(np.float32)
            r1 = (invK * u * L).astype(np.float32)
            j0, j1 = 2*g, 2*g+1
            T[:, i, j0] += r0; T[:, j, j0] -= r0
            T[:, i, j1] += r1; T[:, j, j1] -= r1
    return T


def global_bases(T):
    return np.stack([T[:, :, 0::2].sum(axis=2), T[:, :, 1::2].sum(axis=2)], axis=2).astype(np.float32)


def softmax64(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def corrected_probs(z, T, alpha):
    t = np.einsum('nkj,j->nk', T, alpha, optimize=True)
    return softmax64(z - t)


def loss_grad(alpha, z, T, y):
    t = np.einsum('nkj,j->nk', T, alpha, optimize=True)
    zz = z - t
    p = softmax64(zz)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()
    dz = p
    dz[np.arange(len(y)), y] -= 1.0
    dz /= len(y)
    grad = -np.einsum('nkj,nk->j', T, dz, optimize=True)
    return float(loss), grad


def fit_g2(z, T, y, x0):
    res = minimize(lambda a: loss_grad(a, z, T, y), np.asarray(x0, dtype=np.float64),
                   jac=True, method='L-BFGS-B',
                   options={'maxiter':500,'ftol':1e-13,'gtol':1e-9,'maxls':50})
    if not res.success:
        print('OPT_WARN', json.dumps({'message':res.message,'nit':int(res.nit),'fun':float(res.fun)}), flush=True)
    return res.x.astype(np.float64), {'success':bool(res.success),'nit':int(res.nit),'fun':float(res.fun),'message':str(res.message)}


def metrics(y, p, train_support):
    return B.metrics(y, p.astype(np.float32), train_support)


def group_f1(y, p, head, mid, tail):
    pred = p.argmax(axis=1)
    _, _, F, _ = precision_recall_fscore_support(y, pred, labels=np.arange(p.shape[1]), zero_division=0)
    return {
        'f1_head20': float(F[head].mean()),
        'f1_mid60': float(F[mid].mean()),
        'f1_tail20': float(F[tail].mean()),
        'worst20_class_f1': float(np.sort(F)[:20].mean()),
    }


def sample_losses(y, p):
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    one = np.eye(p.shape[1], dtype=np.float64)[y]
    brier = np.sum((p-one)**2, axis=1)
    return nll, brier


def bootstrap_delta(a, b, rng, Bn=2000):
    # delta = mean(a-b)
    d = np.asarray(a)-np.asarray(b)
    n = len(d); vals = np.empty(Bn)
    for r in range(Bn):
        ix = rng.integers(0, n, n)
        vals[r] = d[ix].mean()
    return [float(np.quantile(vals,.025)), float(np.quantile(vals,.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', type=Path, required=True)
    args = ap.parse_args()

    out = Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
    X, y, tr, meta, cal, te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    ytr, ycal, ytest = y[tr], y[cal], y[te]
    K = int(y.max()+1)
    train_support = np.bincount(ytr, minlength=K).astype(np.float64)
    clusters, head, mid, tail = cluster_partition(train_support)

    zcal, ztest = load_coherent_logits(args.artifact_dir)
    pbase_cal = softmax64(zcal)
    replay = metrics(ycal, pbase_cal, train_support)
    replay_gap = {k: float(replay[k]-EXPECTED_COH_CAL[k]) for k in EXPECTED_COH_CAL}
    print('STAGE_E_REPLAY', json.dumps({'metrics':replay,'gap':replay_gap}), flush=True)
    if max(abs(v) for v in replay_gap.values()) > 2e-6:
        raise RuntimeError(f'Stage-D coherent replay mismatch: {replay_gap}')

    pd.DataFrame({'class':np.arange(K),'train_support':train_support.astype(int),
                  'cluster':np.array(['H','M','T'])[clusters]}).to_csv(out/'stageE_class_clusters.csv', index=False)

    print('STAGE_E_BUILD_BASES_CAL', flush=True)
    Tc = build_potential_bases(zcal, clusters)
    Tg = global_bases(Tc)
    recon = Tc[:,:,0::2].sum(axis=2)
    recon_err = float(np.max(np.abs(recon-zcal)))
    gauge_err = float(np.max(np.abs(Tc.sum(axis=1))))
    print('STAGE_E_SANITY', json.dumps({'PGL_reconstruction_maxabs':recon_err,'potential_gauge_maxabs':gauge_err}), flush=True)
    if recon_err > 1e-4 or gauge_err > 1e-4:
        raise RuntimeError('G2 potential-basis sanity failure')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260909)
    poof_g = np.zeros((len(ycal),K), dtype=np.float64)
    poof_c = np.zeros((len(ycal),K), dtype=np.float64)
    folds=[]; coef_folds=[]
    for fold,(fi,vi) in enumerate(skf.split(np.zeros(len(ycal)),ycal),1):
        ag, og = fit_g2(zcal[fi], Tg[fi], ycal[fi], [0.0,0.0])
        x0c = np.tile(ag, 6)
        ac, oc = fit_g2(zcal[fi], Tc[fi], ycal[fi], x0c)
        pg = corrected_probs(zcal[vi], Tg[vi], ag)
        pc = corrected_probs(zcal[vi], Tc[vi], ac)
        poof_g[vi]=pg; poof_c[vi]=pc
        mg=metrics(ycal[vi],pg,train_support); mc=metrics(ycal[vi],pc,train_support)
        folds.append({'fold':fold,'model':'global',**mg}); folds.append({'fold':fold,'model':'cluster',**mc})
        coef_folds.append({'fold':fold,'global':ag.tolist(),'cluster':ac.tolist(),'opt_global':og,'opt_cluster':oc})
        print('STAGE_E_FOLD', json.dumps({'fold':fold,'global':mg,'cluster':mc}), flush=True)

    cvg=metrics(ycal,poof_g,train_support); cvc=metrics(ycal,poof_c,train_support)
    delta={k:float(cvc[k]-cvg[k]) for k in ['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','tail20_f1']}
    cluster_pass=bool(delta['nll']<0 and delta['brier']<=0 and delta['accuracy']>=-0.001 and delta['macro_f1']>=-0.001)
    winner='cluster_pair_g2' if cluster_pass else 'global_g2'
    decision={
        'status':'STAGE_E_CAL_CV_FROZEN','seeds':SEEDS,'structural_input':'coherent_logit_S5',
        'cluster_definition':'Head20/Mid60/Tail20 by frozen training prevalence',
        'global_cv':cvg,'cluster_cv':cvc,'delta_cluster_minus_global':delta,
        'cluster_pass':cluster_pass,'winner':winner,
        'gate':{'nll_lt_0':True,'brier_le_0':True,'accuracy_tol':-0.001,'macro_f1_tol':-0.001},
        'test_policy':'decision frozen before full-CAL refit and TEST evaluation; test cannot change winner',
        'provenance_note':'Stage-D test was already opened previously; Stage E does not use it for tuning.'
    }
    with open(out/'stageE_cal_cv_decision.json','w') as f: json.dump(decision,f,indent=2)
    pd.DataFrame(folds).to_csv(out/'stageE_cv_folds.csv',index=False)
    with open(out/'stageE_cv_coefficients.json','w') as f: json.dump(coef_folds,f,indent=2)
    print('STAGE_E_CAL_DECISION', json.dumps(decision), flush=True)

    # Decision is now frozen. Full-CAL refit is calibration, then TEST is evaluated.
    ag, og = fit_g2(zcal,Tg,ycal,[0.0,0.0])
    ac, oc = fit_g2(zcal,Tc,ycal,np.tile(ag,6))
    pglobal_cal=corrected_probs(zcal,Tg,ag); pcluster_cal=corrected_probs(zcal,Tc,ac)

    coef_rows=[{'pair':'GLOBAL','alpha0':ag[0],'alpha1':ag[1]}]
    for g,name in enumerate(PAIR_NAMES):
        coef_rows.append({'pair':name,'alpha0':ac[2*g],'alpha1':ac[2*g+1]})
    pd.DataFrame(coef_rows).to_csv(out/'stageE_fullcal_coefficients.csv',index=False)

    print('TEST_OPENED_AFTER_STAGE_E_CV_FREEZE', flush=True)
    Tt=build_potential_bases(ztest,clusters); Tgt=global_bases(Tt)
    pbase_test=softmax64(ztest)
    pglobal_test=corrected_probs(ztest,Tgt,ag)
    pcluster_test=corrected_probs(ztest,Tt,ac)

    rows=[]
    for split, yy, models in [
        ('cal_fullfit',ycal,[('base_coherent',pbase_cal),('global_g2',pglobal_cal),('cluster_pair_g2',pcluster_cal)]),
        ('test',ytest,[('base_coherent',pbase_test),('global_g2',pglobal_test),('cluster_pair_g2',pcluster_test)])]:
        for name,p in models:
            mm=metrics(yy,p,train_support); gf=group_f1(yy,p,head,mid,tail)
            rows.append({'split':split,'model':name,**mm,**gf})
    rdf=pd.DataFrame(rows); rdf.to_csv(out/'stageE_results.csv',index=False)

    # Paired TEST uncertainty; decision remains frozen regardless of these intervals.
    n0,b0=sample_losses(ytest,pbase_test); ng,bg=sample_losses(ytest,pglobal_test); nc,bc=sample_losses(ytest,pcluster_test)
    rng=np.random.default_rng(20260909)
    boot={
        'global_minus_base_nll_ci95':bootstrap_delta(ng,n0,rng),
        'cluster_minus_global_nll_ci95':bootstrap_delta(nc,ng,rng),
        'cluster_minus_base_nll_ci95':bootstrap_delta(nc,n0,rng),
        'global_minus_base_brier_ci95':bootstrap_delta(bg,b0,rng),
        'cluster_minus_global_brier_ci95':bootstrap_delta(bc,bg,rng),
    }

    test_rows={r['model']:r for r in rows if r['split']=='test'}
    final={
        **decision,
        'sanity':{'replay_gap':replay_gap,'PGL_reconstruction_maxabs':recon_err,'potential_gauge_maxabs':gauge_err},
        'fullcal_global_alpha':ag.tolist(),
        'fullcal_cluster_alpha':{name:[float(ac[2*g]),float(ac[2*g+1])] for g,name in enumerate(PAIR_NAMES)},
        'fullcal_optimizer':{'global':og,'cluster':oc},
        'test_results':test_rows,
        'test_paired_bootstrap':boot,
        'cluster_incremental_test':{
            k:float(test_rows['cluster_pair_g2'][k]-test_rows['global_g2'][k])
            for k in ['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','tail20_f1']
        },
        'global_vs_base_test':{
            k:float(test_rows['global_g2'][k]-test_rows['base_coherent'][k])
            for k in ['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','tail20_f1']
        },
    }
    with open(out/'stageE_final_summary.json','w') as f: json.dump(final,f,indent=2)
    print('STAGE_E_FINAL', json.dumps(final), flush=True)


if __name__=='__main__': main()
