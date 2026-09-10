import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import run_stageB1_clone as B
from numeric_split import load_split_numeric
import run_latent_factor_identification_v2 as F
import run_f2_margin_residual_intervention as F2
import run_f2_5_post_intervention_calibration as F25

ETA_GRID=np.array([0.0,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40],dtype=np.float64)
LAMBDA_STAR=-0.05
OUTER_SEED=20260910
BOOT_SEED=20260927
N_BOOT=1000
TOL=1e-12
RANK_TOL=64*np.finfo(np.float64).eps


def rsd(p,a,eta):
    p=np.asarray(p,dtype=np.float64)
    a=np.asarray(a,dtype=np.float64)
    if float(eta)==0.0:
        return p.copy()
    n,K=p.shape
    h=np.argmax(p,axis=1)
    top=p[np.arange(n),h].copy()
    mass=1.0-top
    gamma=1.0-float(eta)*a
    if np.any(gamma<=0):
        raise RuntimeError('Non-positive RSD exponent')
    logp=np.log(np.maximum(p,1e-300))
    out=np.zeros_like(p)
    for i in range(n):
        mask=np.ones(K,dtype=bool); mask[h[i]]=False
        lr=logp[i,mask]
        x=gamma[i]*lr
        x-=x.max()
        w=np.exp(x); w/=w.sum()
        out[i,mask]=mass[i]*w
        out[i,h[i]]=top[i]
    rowsum=out.sum(axis=1)
    if np.max(np.abs(rowsum-1.0))>TOL:
        raise RuntimeError(f'RSD row-sum error {np.max(np.abs(rowsum-1))}')
    return out


def ranking_equal(p0,p1):
    if not np.array_equal(np.argmax(p0,axis=1),np.argmax(p1,axis=1)):
        return False
    order=np.argsort(-p0,axis=1,kind='stable')
    v=np.take_along_axis(p1,order,axis=1)
    # In the reference weak order, a positive adjacent difference means a
    # reversal. Allow only machine-scale tie noise; strict reversals fail.
    max_reversal=float(np.max(v[:,1:]-v[:,:-1]))
    return bool(max_reversal<=RANK_TOL)


def geometry(y,p):
    idx=np.arange(len(y)); py=p[idx,y]
    true_sq=(1.0-py)**2
    sq=p*p
    wrong_sq=sq.sum(axis=1)-sq[idx,y]
    top=np.argmax(p,axis=1)
    ent=np.empty(len(y),dtype=np.float64)
    for i in range(len(y)):
        mask=np.ones(p.shape[1],dtype=bool); mask[top[i]]=False
        mass=1.0-p[i,top[i]]
        r=p[i,mask]/max(mass,1e-300)
        ent[i]=-np.sum(r*np.log(np.maximum(r,1e-300)))
    return {'py':py,'true_sq':true_sq,'wrong_sq':wrong_sq,'residual_entropy':ent}


def summarize(y,p,support):
    m=F2.metrics(y,p,support); g=geometry(y,p)
    return {**m,'mean_py':float(g['py'].mean()),'mean_true_sq':float(g['true_sq'].mean()),
            'mean_wrong_sq':float(g['wrong_sq'].mean()),'mean_residual_entropy':float(g['residual_entropy'].mean())}


def eligible(mm,c0,c1,c3):
    rankkeys=['accuracy','macro_f1','balanced_accuracy','tail20_f1']
    return bool(
        mm['mean_wrong_sq'] < c3['mean_wrong_sq'] and
        mm['brier'] < c3['brier'] and
        mm['brier'] <= c1['brier'] and
        mm['nll'] < c1['nll'] and
        mm['ece'] <= c1['ece'] and
        all(abs(mm[k]-c3[k])<=TOL for k in rankkeys)
    )


def paired_ci(x1,x0,rng,Bn=N_BOOT):
    d=np.asarray(x1,dtype=np.float64)-np.asarray(x0,dtype=np.float64)
    n=len(d); vals=np.empty(Bn,dtype=np.float64)
    for b in range(Bn):
        ix=rng.integers(0,n,n); vals[b]=d[ix].mean()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def build_oof(X,y,tr,cal,zcal,zsc,support,rawgeo):
    K=zcal.shape[1]; ycal=y[cal]
    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED)
    probs={'C0':np.zeros((len(cal),K),float),'C1':np.zeros((len(cal),K),float),'C3':np.zeros((len(cal),K),float)}
    doses=np.zeros(len(cal),float); fold_rows=[]; coef_rows=[]
    for fold,(fi,vi) in enumerate(outer.split(np.zeros(len(cal)),ycal),1):
        fit_idx=cal[fi]; zfit=zcal[fi]; zval=zcal[vi]; zsfit=zsc[:,fi,:]
        pfit_pre=F2.global_probs(zfit)
        d,ds,dh,ddiag=F2.derive_direction(X,y,tr,fit_idx,zfit,zsfit,pfit_pre,K,support,rawgeo)
        scale=F2.gate_scale(zfit)
        zfit_i,afit=F2.intervene(zfit,d,scale,LAMBDA_STAR)
        zval_i,aval=F2.intervene(zval,d,scale,LAMBDA_STAR)
        probs['C0'][vi]=F2.global_probs(zval)
        probs['C1'][vi]=F2.global_probs(zval_i)
        Tfit=F2.global_bases_fast(zfit_i); Tval=F2.global_bases_fast(zval_i)
        a2,_=F25.fit_theta(zfit_i,Tfit,ycal[fi],F25.CANON_ALPHA)
        qfit,qval,qmu,qsd=F25.standardize_gate(afit,aval)
        T4fit=F25.dose_bases(Tfit,qfit); T4val=F25.dose_bases(Tval,qval)
        a3,opt3=F25.fit_theta(zfit_i,T4fit,ycal[fi],np.array([a2[0],a2[1],0.0,0.0]))
        probs['C3'][vi]=F25.apply_bases(zval_i,T4val,a3)
        doses[vi]=aval
        fold_rows.append({'fold':fold,'gate_scale':scale,'dose_mean_val':float(aval.mean()),
                          'soft_hard_aligned_dot':ddiag['soft_hard_aligned_dot']})
        coef_rows.append({'fold':fold,'alpha0':a3[0],'alpha1':a3[1],'beta0':a3[2],'beta1':a3[3],
                          'opt_success':opt3['success'],'opt_fun':opt3['fun']})
    return probs,doses,pd.DataFrame(fold_rows),pd.DataFrame(coef_rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(np.float64)
    rawgeo=F.raw_geometry_features(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir); zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]

    probs,doses,fold_df,coef_df=build_oof(X,y,tr,cal,zcal,zsc,support,rawgeo)
    fold_df.to_csv(out/'f2_7_fold_diagnostics.csv',index=False); coef_df.to_csv(out/'f2_7_c3_cv_coefficients.csv',index=False)
    base={k:summarize(ycal,p,support) for k,p in probs.items()}
    expected={'nll':2.7397491931915283,'brier':0.7826232437924868,'ece':0.028646533298620415,
              'macro_f1':0.19490055879898846}
    gaps={k:float(base['C3'][k]-v) for k,v in expected.items()}
    if max(abs(v) for v in gaps.values())>2e-6:
        raise RuntimeError(f'C3 replay mismatch {gaps}')

    rows=[]; peta={}
    for eta in ETA_GRID:
        p=rsd(probs['C3'],doses,float(eta)); peta[float(eta)]=p
        if not ranking_equal(probs['C3'],p):
            raise RuntimeError(f'Weak ranking changed at eta={eta}')
        mm=summarize(ycal,p,support)
        rows.append({'eta':float(eta),**mm,
                     'delta_nll_vs_c3':mm['nll']-base['C3']['nll'],
                     'delta_brier_vs_c3':mm['brier']-base['C3']['brier'],
                     'delta_ece_vs_c3':mm['ece']-base['C3']['ece'],
                     'delta_wrong_sq_vs_c3':mm['mean_wrong_sq']-base['C3']['mean_wrong_sq'],
                     'delta_true_sq_vs_c3':mm['mean_true_sq']-base['C3']['mean_true_sq'],
                     'delta_entropy_vs_c3':mm['mean_residual_entropy']-base['C3']['mean_residual_entropy'],
                     'eligible': False if eta==0 else eligible(mm,base['C0'],base['C1'],base['C3'])})
    tab=pd.DataFrame(rows); tab.to_csv(out/'f2_7_oof_eta_metrics.csv',index=False)
    candidates=[r for r in rows if r['eligible']]
    if candidates:
        win=sorted(candidates,key=lambda r:(r['brier'],r['nll'],r['ece'],r['eta']))[0]
        eta_star=float(win['eta'])
        if win['brier']<=base['C0']['brier'] and win['nll']<=base['C0']['nll'] and win['ece']<=base['C0']['ece']:
            status='FULL_PARETO_RECOVERY'
        else:
            status='PARTIAL_CONTROL'
    else:
        eta_star=0.0; win=next(r for r in rows if r['eta']==0.0); status='NO_CONTROL'
    decision={'status':status,'eta_star':eta_star,'eta_grid':ETA_GRID.tolist(),'c0':base['C0'],'c1':base['C1'],'c3':base['C3'],
              'winner':win,'n_eligible':len(candidates),'c3_replay_gap':gaps,
              'rank_tolerance':float(RANK_TOL),
              'test_policy':'eta/status frozen from OOF-CAL before TEST; TEST cannot change closure'}
    with open(out/'f2_7_cal_decision.json','w') as f: json.dump(decision,f,indent=2)
    print('F2_7_CAL_DECISION',json.dumps(decision),flush=True)

    pcal_pre=F2.global_probs(zcal)
    dfull,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,K,support,rawgeo)
    scale=F2.gate_scale(zcal)
    zcal_i,acal=F2.intervene(zcal,dfull,scale,LAMBDA_STAR)
    ztest_i,atest=F2.intervene(ztest,dfull,scale,LAMBDA_STAR)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,_=F25.fit_theta(zcal_i,Tcal,ycal,F25.CANON_ALPHA)
    qcal,qtest,qmu,qsd=F25.standardize_gate(acal,atest)
    a3,opt3=F25.fit_theta(zcal_i,F25.dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0.0,0.0]))
    pc3=F25.apply_bases(ztest_i,F25.dose_bases(Ttest,qtest),a3)
    ptest={'C0':F2.global_probs(ztest),'C1':F2.global_probs(ztest_i),'C3':pc3,'F2_7':rsd(pc3,atest,eta_star)}
    if not ranking_equal(pc3,ptest['F2_7']): raise RuntimeError('TEST weak ranking changed')
    tmetrics={k:summarize(ytest,p,support) for k,p in ptest.items()}
    pd.DataFrame([{'model':k,**v} for k,v in tmetrics.items()]).to_csv(out/'f2_7_test_metrics.csv',index=False)

    rng=np.random.default_rng(BOOT_SEED); boot={}
    gw=geometry(ytest,ptest['F2_7'])
    for ref in ['C3','C0']:
        gr=geometry(ytest,ptest[ref])
        nw=-np.log(np.maximum(gw['py'],1e-15)); nr=-np.log(np.maximum(gr['py'],1e-15))
        bw=gw['true_sq']+gw['wrong_sq']; br=gr['true_sq']+gr['wrong_sq']
        boot[ref]={'nll_delta_ci95':paired_ci(nw,nr,rng),'brier_delta_ci95':paired_ci(bw,br,rng),
                   'true_sq_delta_ci95':paired_ci(gw['true_sq'],gr['true_sq'],rng),
                   'wrong_sq_delta_ci95':paired_ci(gw['wrong_sq'],gr['wrong_sq'],rng)}
    with open(out/'f2_7_test_bootstrap.json','w') as f: json.dump(boot,f,indent=2)
    final={'status':status,'eta_star':eta_star,'oof_decision':decision,'test':tmetrics,'bootstrap':boot,
           'fullcal_c3_coefficients':a3.tolist(),'full_direction_diag':ddiag,
           'interpretation':'Mechanism-targeted residual-simplex diffusion; TEST cannot change OOF closure.'}
    with open(out/'F2_7_WRONG_CLASS_CONCENTRATION_CONTROL.json','w') as f: json.dump(final,f,indent=2)
    print('F2_7_TEST_OPENED_AFTER_CAL_FREEZE',flush=True)
    print('F2_7_FINAL',json.dumps(final),flush=True)

if __name__=='__main__': main()
