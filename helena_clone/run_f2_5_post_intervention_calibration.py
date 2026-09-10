import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

import run_stageB1_clone as B
from numeric_split import load_split_numeric
import run_latent_factor_identification_v2 as F
import run_f2_margin_residual_intervention as F2

LAMBDA_STAR = -0.05
OUTER_SEED = 20260910
BOOT_SEED = 20260925
N_BOOT = 1000
CANON_ALPHA = np.array([-0.9026209634632217, 0.765345197612019], dtype=np.float64)
MODELS = ['C0_canonical','C1_f2_fixed_g2','C2_refit_global_g2','C3_dose_conditioned_g2']


def softmax64(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_bases(z, T, theta):
    t = np.einsum('nkj,j->nk', T, theta, optimize=True)
    return softmax64(z - t)


def loss_grad(theta, z, T, y):
    t = np.einsum('nkj,j->nk', T, theta, optimize=True)
    zz = z - t
    p = softmax64(zz)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()
    dz = p
    dz[np.arange(len(y)), y] -= 1.0
    dz /= len(y)
    grad = -np.einsum('nkj,nk->j', T, dz, optimize=True)
    return float(loss), grad


def fit_theta(z, T, y, x0):
    res = minimize(lambda a: loss_grad(a, z, T, y), np.asarray(x0, np.float64),
                   jac=True, method='L-BFGS-B',
                   options={'maxiter':500,'ftol':1e-13,'gtol':1e-9,'maxls':50})
    return res.x.astype(np.float64), {
        'success':bool(res.success), 'nit':int(res.nit), 'fun':float(res.fun),
        'message':str(res.message)
    }


def standardize_gate(a_fit, a_other):
    mu = float(np.mean(a_fit)); sd = float(np.std(a_fit) + 1e-8)
    return (a_fit-mu)/sd, (a_other-mu)/sd, mu, sd


def dose_bases(T2, q):
    return np.concatenate([T2, T2 * q[:,None,None]], axis=2)


def delta(m, ref):
    keys=['nll','brier','ece','accuracy','macro_f1','balanced_accuracy','tail20_f1']
    return {k:float(m[k]-ref[k]) for k in keys}


def eligible(m, c0, c1):
    return bool(
        m['nll'] < c1['nll'] and
        m['brier'] <= c1['brier'] and
        m['macro_f1'] > c0['macro_f1'] and
        m['balanced_accuracy'] > c0['balanced_accuracy'] and
        m['tail20_f1'] > c0['tail20_f1'] and
        m['accuracy'] >= c0['accuracy'] - 0.0005
    )


def sample_losses(y,p):
    nll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-15))
    one=np.eye(p.shape[1],dtype=np.float64)[y]
    brier=np.sum((p-one)**2,axis=1)
    return nll,brier


def paired_ci(a,b,rng,Bn=N_BOOT):
    d=np.asarray(a)-np.asarray(b); n=len(d); vals=np.empty(Bn)
    for j in range(Bn):
        ix=rng.integers(0,n,n); vals[j]=d[ix].mean()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def boot_metric_delta(y,p1,p0,support,rng,Bn=N_BOOT):
    keys=['accuracy','macro_f1','balanced_accuracy','tail20_f1']
    vals={k:np.empty(Bn) for k in keys}; n=len(y)
    for j in range(Bn):
        ix=rng.integers(0,n,n)
        m1=F2.metrics(y[ix],p1[ix],support); m0=F2.metrics(y[ix],p0[ix],support)
        for k in keys: vals[k][j]=m1[k]-m0[k]
    return {k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in vals.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)

    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(np.float64)
    rawgeo=F.raw_geometry_features(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir)
    zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]

    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED)
    poof={m:np.zeros((len(ycal),K),dtype=np.float64) for m in MODELS}
    fold_rows=[]; coef_rows=[]

    for fold,(fi,vi) in enumerate(outer.split(np.zeros(len(ycal)),ycal),1):
        fit_idx=cal[fi]; zfit=zcal[fi]; zval=zcal[vi]; zsfit=zsc[:,fi,:]
        pfit_pre=F2.global_probs(zfit)
        d,ds,dh,ddiag=F2.derive_direction(X,y,tr,fit_idx,zfit,zsfit,pfit_pre,K,support,rawgeo)
        scale=F2.gate_scale(zfit)
        zfit_i,afit=F2.intervene(zfit,d,scale,LAMBDA_STAR)
        zval_i,aval=F2.intervene(zval,d,scale,LAMBDA_STAR)

        poof['C0_canonical'][vi]=F2.global_probs(zval)
        poof['C1_f2_fixed_g2'][vi]=F2.global_probs(zval_i)

        Tfit=F2.global_bases_fast(zfit_i); Tval=F2.global_bases_fast(zval_i)
        a2,opt2=fit_theta(zfit_i,Tfit,ycal[fi],CANON_ALPHA)
        poof['C2_refit_global_g2'][vi]=apply_bases(zval_i,Tval,a2)

        qfit,qval,qmu,qsd=standardize_gate(afit,aval)
        T4fit=dose_bases(Tfit,qfit); T4val=dose_bases(Tval,qval)
        x03=np.array([a2[0],a2[1],0.0,0.0],dtype=np.float64)
        a3,opt3=fit_theta(zfit_i,T4fit,ycal[fi],x03)
        poof['C3_dose_conditioned_g2'][vi]=apply_bases(zval_i,T4val,a3)

        fold_rows.append({'fold':fold,'gate_scale':scale,'gate_fit_mean':float(afit.mean()),
                          'gate_val_mean':float(aval.mean()),'soft_hard_aligned_dot':ddiag['soft_hard_aligned_dot'],
                          'soft_margin_cv_r2':ddiag['soft_component']['margin_residual_cv_r2'],
                          'hard_margin_cv_r2':ddiag['hard_component']['margin_residual_cv_r2']})
        coef_rows.append({'fold':fold,'model':'C2_refit_global_g2','alpha0':a2[0],'alpha1':a2[1],
                          'beta0':0.0,'beta1':0.0,'opt_success':opt2['success'],'opt_nit':opt2['nit'],'opt_fun':opt2['fun']})
        coef_rows.append({'fold':fold,'model':'C3_dose_conditioned_g2','alpha0':a3[0],'alpha1':a3[1],
                          'beta0':a3[2],'beta1':a3[3],'opt_success':opt3['success'],'opt_nit':opt3['nit'],'opt_fun':opt3['fun']})
        print('F2_5_FOLD',json.dumps({'fold':fold,'C2':a2.tolist(),'C3':a3.tolist(),'q_mu':qmu,'q_sd':qsd}),flush=True)

    pd.DataFrame(fold_rows).to_csv(out/'f2_5_fold_diagnostics.csv',index=False)
    pd.DataFrame(coef_rows).to_csv(out/'f2_5_cv_coefficients.csv',index=False)

    metrics_oof={m:F2.metrics(ycal,p,support) for m,p in poof.items()}
    c0=metrics_oof['C0_canonical']; c1=metrics_oof['C1_f2_fixed_g2']
    # Must reproduce valid F2 OOF reference closely.
    expected_f2={'nll':2.741716146469116,'brier':0.7825546554981124,'accuracy':0.3572597137014315,
                 'macro_f1':0.19490055879898846,'balanced_accuracy':0.19412086409852683,'tail20_f1':0.11264089516721096}
    replay_gap={k:float(c1[k]-v) for k,v in expected_f2.items()}
    if max(abs(v) for v in replay_gap.values())>2e-6:
        raise RuntimeError(f'F2 reference replay mismatch: {replay_gap}')

    rows=[]
    for m in MODELS:
        mm=metrics_oof[m]
        rows.append({'model':m,**mm,**{f'delta_vs_c0_{k}':v for k,v in delta(mm,c0).items()},
                     **{f'delta_vs_c1_{k}':v for k,v in delta(mm,c1).items()},
                     'eligible':eligible(mm,c0,c1) if m in MODELS[2:] else False})
    pd.DataFrame(rows).to_csv(out/'f2_5_oof_metrics.csv',index=False)

    cand=[r for r in rows if r['model'] in MODELS[2:] and r['eligible']]
    if cand:
        w=sorted(cand,key=lambda r:(r['nll'],r['brier'],r['ece'],-r['macro_f1']))[0]
        winner=w['model']
        if w['nll']<=c0['nll'] and w['brier']<=c0['brier']:
            recovery='FULL_RECOVERY'
        else:
            recovery='PARTIAL_RECOVERY'
    else:
        winner='NONE'; recovery='NO_RECOVERY'

    decision={'status':recovery,'winner':winner,'lambda_frozen':LAMBDA_STAR,
              'models':metrics_oof,'f2_replay_gap':replay_gap,
              'gate_rule':{'nll_lt_C1':True,'brier_le_C1':True,'macro_f1_gt_C0':True,
                           'balanced_accuracy_gt_C0':True,'tail20_f1_gt_C0':True,
                           'accuracy_min_vs_C0':-0.0005},
              'test_policy':'winner frozen from OOF-CAL before TEST; TEST cannot change decision',
              'dose_conditioned_note':'C3 is a rank-1 intervention-state conditioning test inspired structurally by aspect-conditioned sentiment, not a transferred sentiment model.'}
    with open(out/'f2_5_cal_decision.json','w') as f: json.dump(decision,f,indent=2)
    print('F2_5_CAL_DECISION',json.dumps(decision),flush=True)

    # Full-CAL refits after decision freeze; all models are audited on TEST regardless of OOF winner.
    pcal_pre=F2.global_probs(zcal)
    dfull,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,K,support,rawgeo)
    scale=F2.gate_scale(zcal)
    zcal_i,acal=F2.intervene(zcal,dfull,scale,LAMBDA_STAR)
    ztest_i,atest=F2.intervene(ztest,dfull,scale,LAMBDA_STAR)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,opt2=fit_theta(zcal_i,Tcal,ycal,CANON_ALPHA)
    p2=apply_bases(ztest_i,Ttest,a2)
    qcal,qtest,qmu,qsd=standardize_gate(acal,atest)
    a3,opt3=fit_theta(zcal_i,dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0,0],float))
    p3=apply_bases(ztest_i,dose_bases(Ttest,qtest),a3)
    ptest={'C0_canonical':F2.global_probs(ztest),'C1_f2_fixed_g2':F2.global_probs(ztest_i),
           'C2_refit_global_g2':p2,'C3_dose_conditioned_g2':p3}
    test_metrics={m:F2.metrics(ytest,p,support) for m,p in ptest.items()}
    pd.DataFrame([{'model':m,**mm} for m,mm in test_metrics.items()]).to_csv(out/'f2_5_test_metrics.csv',index=False)
    pd.DataFrame([{'model':'C2_refit_global_g2','alpha0':a2[0],'alpha1':a2[1],'beta0':0.0,'beta1':0.0},
                  {'model':'C3_dose_conditioned_g2','alpha0':a3[0],'alpha1':a3[1],'beta0':a3[2],'beta1':a3[3]}]).to_csv(out/'f2_5_fullcal_coefficients.csv',index=False)

    boot={}
    if winner!='NONE':
        rng=np.random.default_rng(BOOT_SEED); pw=ptest[winner]
        nw,bw=sample_losses(ytest,pw)
        for ref in ['C0_canonical','C1_f2_fixed_g2']:
            nr,br=sample_losses(ytest,ptest[ref])
            boot[ref]={'nll_delta_ci95':paired_ci(nw,nr,rng),'brier_delta_ci95':paired_ci(bw,br,rng),
                       'ranking_delta_ci95':boot_metric_delta(ytest,pw,ptest[ref],support,rng)}
    with open(out/'f2_5_test_bootstrap.json','w') as f: json.dump(boot,f,indent=2)

    final={'status':recovery,'winner':winner,'lambda_frozen':LAMBDA_STAR,'oof_cal':metrics_oof,
           'test':test_metrics,'full_cal_coefficients':{'C2':a2.tolist(),'C3':a3.tolist()},
           'full_direction_diag':ddiag,'gate_scale':scale,'q_mean_cal':qmu,'q_sd_cal':qsd,
           'bootstrap':boot,'interpretation':'Calibration recovery/actionability only; dose conditioning is not causal latent identification.'}
    with open(out/'F2_5_POST_INTERVENTION_CALIBRATION.json','w') as f: json.dump(final,f,indent=2)
    print('F2_5_TEST_OPENED_AFTER_CAL_FREEZE',flush=True)
    print('F2_5_FINAL',json.dumps(final),flush=True)

if __name__=='__main__':
    main()
