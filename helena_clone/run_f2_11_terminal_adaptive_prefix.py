import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import run_stageB1_clone as B
from numeric_split import load_split_numeric
import run_latent_factor_identification_v2 as F
import run_f2_margin_residual_intervention as F2
import run_f2_5_post_intervention_calibration as F25
import run_f2_8_runner_up_reliability as F28
import run_f2_10_adaptive_protected_prefix as F210

OUTER_SEED = 20260910
SELECT_SEED = 20261011
BOOT_SEED = 20261012
N_BOOT = 1000
ALPHA_GRID = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50], dtype=np.float64)
ETA_GRID = np.array([0.0,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40], dtype=np.float64)
TOL = 1e-12
RANK_TOL = 64*np.finfo(np.float64).eps
FALLBACK_ALPHA = 0.25


def fit_hazards(Xtr, rank_tr, Xv):
    Xtr=np.asarray(Xtr,dtype=np.float64); Xv=np.asarray(Xv,dtype=np.float64)
    rank_tr=np.asarray(rank_tr,dtype=int)
    out=np.zeros((len(Xv),4),dtype=np.float64)
    for r in range(1,5):
        risk=rank_tr>=r
        yy=(rank_tr[risk]==r).astype(int)
        if len(np.unique(yy))<2:
            raise RuntimeError(f'hazard h{r} has one training class')
        sc=StandardScaler().fit(Xtr[risk])
        clf=LogisticRegression(C=1.0,max_iter=3000,solver='lbfgs')
        clf.fit(sc.transform(Xtr[risk]),yy)
        out[:,r-1]=clf.predict_proba(sc.transform(Xv))[:,1]
    return out


def crossfit_hazards(X, rank, seed=SELECT_SEED):
    X=np.asarray(X,dtype=np.float64); rank=np.asarray(rank,dtype=int)
    strata=np.minimum(rank,5)
    skf=StratifiedKFold(n_splits=4,shuffle=True,random_state=seed)
    out=np.zeros((len(rank),4),dtype=np.float64)
    for ti,vi in skf.split(np.zeros(len(rank)),strata):
        out[vi]=fit_hazards(X[ti],rank[ti],X[vi])
    return out


def cumulative_from_hazards(h):
    h=np.clip(np.asarray(h,dtype=np.float64),1e-12,1-1e-12)
    surv=np.cumprod(1.0-h,axis=1)
    Fm=1.0-surv
    # columns correspond F1,F2,F3,F4
    if np.any(Fm[:,1]>Fm[:,2]+1e-15) or np.any(Fm[:,2]>Fm[:,3]+1e-15):
        raise RuntimeError('coherent cumulative monotonicity failure')
    return Fm


def choose_prefix(Fm, alpha):
    q=1.0-float(alpha)
    m=np.full(len(Fm),4,dtype=int)
    hit2=Fm[:,1]>=q
    hit3=(~hit2)&(Fm[:,2]>=q)
    m[hit2]=2; m[hit3]=3
    shortfall=Fm[:,3]<q
    return m,shortfall


def adaptive_tail_diffuse(p,dose,Fm,alpha,eta):
    p=np.asarray(p,dtype=np.float64); dose=np.asarray(dose,dtype=np.float64)
    prefix,shortfall=choose_prefix(Fm,alpha)
    if float(eta)==0.0:
        return p.copy(),prefix,shortfall,{
            'valid':True,'max_row_error':0.0,'max_prefix_drift':0.0,
            'max_top1_delta':0.0,'max_reversal':float('-inf')}
    gamma=1.0-float(eta)*dose
    if np.any(gamma<=0):
        return None,prefix,shortfall,{'valid':False,'reason':'nonpositive_gamma'}
    order=np.argsort(-p,axis=1,kind='stable')
    v=np.take_along_axis(p,order,axis=1)
    nv=v.copy()
    for m in [2,3,4]:
        idx=np.where(prefix==m)[0]
        if len(idx)==0: continue
        tail=v[idx,m:]
        mass=tail.sum(axis=1)
        active=mass>1e-300
        if not np.any(active): continue
        ia=idx[active]; t=tail[active]
        x=gamma[ia,None]*np.log(np.maximum(t,1e-300))
        x-=x.max(axis=1,keepdims=True)
        w=np.exp(x); w/=w.sum(axis=1,keepdims=True)
        nv[ia,m:]=mass[active,None]*w
    out=np.zeros_like(p)
    np.put_along_axis(out,order,nv,axis=1)
    rowerr=float(np.max(np.abs(out.sum(axis=1)-1.0)))
    # Exact protected prefix drift in original sorted coordinates.
    drift=0.0
    for m in [2,3,4]:
        idx=np.where(prefix==m)[0]
        if len(idx): drift=max(drift,float(np.max(np.abs(nv[idx,:m]-v[idx,:m]))))
    top0=np.argmax(p,axis=1); top1=np.argmax(out,axis=1)
    top_delta=float(np.max(np.abs(out[np.arange(len(p)),top0]-p[np.arange(len(p)),top0])))
    maxrev=float(np.max(nv[:,1:]-nv[:,:-1]))
    valid=bool(rowerr<=TOL and drift<=TOL and top_delta<=TOL and np.array_equal(top0,top1) and maxrev<=RANK_TOL and np.all(out>=-1e-15))
    return out,prefix,shortfall,{
        'valid':valid,'max_row_error':rowerr,'max_prefix_drift':drift,
        'max_top1_delta':top_delta,'max_reversal':maxrev}


def proper_geometry(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=np.float64)
    ii=np.arange(len(y)); py=p[ii,y]
    true_sq=(1.0-py)**2
    wrong_sq=np.sum(p*p,axis=1)-p[ii,y]**2
    return py,true_sq,wrong_sq


def summarize(y,p,support):
    m=F2.metrics(y,p,support)
    py,ts,ws=proper_geometry(y,p)
    return {**m,'mean_py':float(py.mean()),'mean_true_sq':float(ts.mean()),'mean_wrong_sq':float(ws.mean())}


def ranking_preserved(mm,c3):
    return bool(
        abs(mm['accuracy']-c3['accuracy'])<=TOL and
        abs(mm['macro_f1']-c3['macro_f1'])<=TOL and
        abs(mm['balanced_accuracy']-c3['balanced_accuracy'])<=TOL and
        abs(mm['tail20_f1']-c3['tail20_f1'])<=TOL
    )


def grid_select(y,p3,dose,Fm,p0,support,tag):
    c0=summarize(y,p0,support); c3=summarize(y,p3,support)
    rows=[]; eligible=[]
    for alpha in ALPHA_GRID:
        for eta in ETA_GRID:
            pp,prefix,shortfall,inv=adaptive_tail_diffuse(p3,dose,Fm,float(alpha),float(eta))
            if pp is None or not inv.get('valid',False):
                rows.append({'tag':tag,'alpha':float(alpha),'eta':float(eta),'valid':False,'eligible':False,
                             'prefix2_share':float(np.mean(prefix==2)),'prefix3_share':float(np.mean(prefix==3)),
                             'prefix4_share':float(np.mean(prefix==4)),'coverage_shortfall':float(np.mean(shortfall)),
                             **{k:v for k,v in inv.items() if k!='valid'}})
                continue
            mm=summarize(y,pp,support)
            ok=bool(float(eta)>0 and mm['nll']-c0['nll']<=0 and mm['brier']-c0['brier']<=0 and ranking_preserved(mm,c3))
            row={'tag':tag,'alpha':float(alpha),'eta':float(eta),'valid':True,'eligible':ok,
                 'prefix2_share':float(np.mean(prefix==2)),'prefix3_share':float(np.mean(prefix==3)),
                 'prefix4_share':float(np.mean(prefix==4)),'coverage_shortfall':float(np.mean(shortfall)),
                 **mm,'delta_nll_vs_c0':float(mm['nll']-c0['nll']),
                 'delta_brier_vs_c0':float(mm['brier']-c0['brier']),
                 'delta_nll_vs_c3':float(mm['nll']-c3['nll']),
                 'delta_brier_vs_c3':float(mm['brier']-c3['brier']),
                 'delta_true_sq_vs_c3':float(mm['mean_true_sq']-c3['mean_true_sq']),
                 'delta_wrong_sq_vs_c3':float(mm['mean_wrong_sq']-c3['mean_wrong_sq']),
                 'max_row_error':inv['max_row_error'],'max_prefix_drift':inv['max_prefix_drift'],
                 'max_top1_delta':inv['max_top1_delta'],'max_reversal':inv['max_reversal']}
            rows.append(row)
            if ok: eligible.append(row)
    if eligible:
        win=sorted(eligible,key=lambda r:(r['brier'],r['nll'],r['eta'],r['alpha']))[0]
        selected={'alpha':float(win['alpha']),'eta':float(win['eta']),'eligible':True,'selection_row':win}
    else:
        selected={'alpha':FALLBACK_ALPHA,'eta':0.0,'eligible':False,'selection_row':None}
    return selected,pd.DataFrame(rows),c0,c3


def paired_boot(y,p1,p0,seed=BOOT_SEED):
    rng=np.random.default_rng(seed); y=np.asarray(y,dtype=int); n=len(y); ii=np.arange(n)
    nll1=-np.log(np.maximum(p1[ii,y],1e-15)); nll0=-np.log(np.maximum(p0[ii,y],1e-15))
    one=np.eye(p1.shape[1],dtype=np.float64)[y]
    b1=np.sum((p1-one)**2,axis=1); b0=np.sum((p0-one)**2,axis=1)
    dn=np.empty(N_BOOT); db=np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix=rng.integers(0,n,n); dn[b]=np.mean(nll1[ix]-nll0[ix]); db[b]=np.mean(b1[ix]-b0[ix])
    return {'nll_delta_ci95':[float(np.quantile(dn,.025)),float(np.quantile(dn,.975))],
            'brier_delta_ci95':[float(np.quantile(db,.025)),float(np.quantile(db,.975))]}


def terminal_pass(mm,c0,c3):
    return bool(mm['nll']-c0['nll']<=0 and mm['brier']-c0['brier']<=0 and ranking_preserved(mm,c3))


def build_full_test_c3(X,y,tr,cal,zcal,ztest,zsc,support,rawgeo):
    K=zcal.shape[1]; ycal=y[cal]
    pcal_pre=F2.global_probs(zcal)
    d,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,K,support,rawgeo)
    scale=F2.gate_scale(zcal)
    zcal_i,acal=F2.intervene(zcal,d,scale,-0.05)
    ztest_i,atest=F2.intervene(ztest,d,scale,-0.05)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,_=F25.fit_theta(zcal_i,Tcal,ycal,F25.CANON_ALPHA)
    qcal,qtest,qmu,qsd=F25.standardize_gate(acal,atest)
    a3,_=F25.fit_theta(zcal_i,F25.dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0.0,0.0]))
    ptest=F25.apply_bases(ztest_i,F25.dose_bases(Ttest,qtest),a3)
    return ptest,atest,qtest,{'a3':a3.tolist(),'scale':float(scale),'q_mu':float(qmu),'q_sd':float(qsd),
                              'soft_hard_dot':float(ddiag['soft_hard_aligned_dot'])}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)

    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(np.float64)
    rawgeo=F.raw_geometry_features(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir); zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]
    p0_cal=F2.global_probs(zcal)

    p3_oof=np.zeros((len(cal),K),dtype=np.float64)
    dose_oof=np.zeros(len(cal),dtype=np.float64); q_oof=np.zeros(len(cal),dtype=np.float64)
    X_oof=np.zeros((len(cal),len(F210.FEATURES)),dtype=np.float64)
    rank_oof=np.zeros(len(cal),dtype=int); h_oof=np.zeros((len(cal),4),dtype=np.float64)
    F_oof=np.zeros((len(cal),4),dtype=np.float64)
    nested_pred=np.zeros((len(cal),K),dtype=np.float64)
    fold_rows=[]; grid_frames=[]

    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED)
    for fold,(fi,vi) in enumerate(outer.split(np.zeros(len(cal)),ycal),1):
        fi=np.asarray(fi,dtype=int); vi=np.asarray(vi,dtype=int)
        # Outer-fit C3 probabilities are themselves C3-cross-fitted.
        ptri,atri,qtri=F210.inner_oof_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,fi)
        Xtri,otri=F210.make_features(ptri,atri,qtri)
        ranktri,_,_=F210.targets_from_order(ycal[fi],otri)
        # Second selection cross-fit: hazard prediction for pair selection never sees its own target.
        hsel=crossfit_hazards(Xtri.to_numpy(),ranktri,SELECT_SEED)
        Fsel=cumulative_from_hazards(hsel)
        selected,gdf,c0sel,c3sel=grid_select(ycal[fi],ptri,atri,Fsel,p0_cal[fi],support,f'outer{fold}_selection')
        gdf.insert(0,'fold',fold); grid_frames.append(gdf)

        # Outer-validation C3 and hazard prediction.
        pv,av,qv,diag=F28.fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,fi,vi)
        Xv,ov=F210.make_features(pv,av,qv)
        rankv,_,_=F210.targets_from_order(ycal[vi],ov)
        hv=fit_hazards(Xtri.to_numpy(),ranktri,Xv.to_numpy())
        Fv=cumulative_from_hazards(hv)
        pp,prefix,shortfall,inv=adaptive_tail_diffuse(pv,av,Fv,selected['alpha'],selected['eta'])
        if pp is None or not inv.get('valid',False):
            raise RuntimeError(f'outer fold {fold} selected pair violates invariants: {inv}')

        p3_oof[vi]=pv; dose_oof[vi]=av; q_oof[vi]=qv; X_oof[vi]=Xv.to_numpy(); rank_oof[vi]=rankv
        h_oof[vi]=hv; F_oof[vi]=Fv; nested_pred[vi]=pp
        fold_rows.append({'fold':fold,'selected_alpha':selected['alpha'],'selected_eta':selected['eta'],
                          'inner_pair_eligible':selected['eligible'],'prefix2_share_val':float(np.mean(prefix==2)),
                          'prefix3_share_val':float(np.mean(prefix==3)),'prefix4_share_val':float(np.mean(prefix==4)),
                          'coverage_shortfall_val':float(np.mean(shortfall)),
                          'outer_scale':diag['scale'],'soft_hard_dot':diag['soft_hard_dot'],
                          'max_row_error':inv['max_row_error'],'max_prefix_drift':inv['max_prefix_drift'],
                          'max_top1_delta':inv['max_top1_delta'],'max_reversal':inv['max_reversal']})
        print('F2_11_OUTER_FOLD',json.dumps(fold_rows[-1]),flush=True)

    pd.concat(grid_frames,ignore_index=True).to_csv(out/'f2_11_inner_selection_grid.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'f2_11_outer_fold_selections.csv',index=False)
    pd.DataFrame(X_oof,columns=F210.FEATURES).to_csv(out/'f2_11_oof_features.csv',index=False)

    c0_oof=summarize(ycal,p0_cal,support); c3_oof=summarize(ycal,p3_oof,support); nested=summarize(ycal,nested_pred,support)
    cal_pass=terminal_pass(nested,c0_oof,c3_oof)
    cal_boot=paired_boot(ycal,nested_pred,p0_cal,BOOT_SEED)

    # Global deployment pair is selected from complete outer-OOF predictions after nested verdict metrics exist.
    global_selected,global_grid,_,_=grid_select(ycal,p3_oof,dose_oof,F_oof,p0_cal,support,'global_oof_freeze')
    global_grid.to_csv(out/'f2_11_global_oof_grid.csv',index=False)
    gpp,gprefix,gshort,ginv=adaptive_tail_diffuse(p3_oof,dose_oof,F_oof,global_selected['alpha'],global_selected['eta'])
    global_oof_metrics=summarize(ycal,gpp,support)

    cal_decision={
        'cal_status':'CAL_TERMINAL_PASS' if cal_pass else 'CAL_TERMINAL_FAIL',
        'cal_terminal_pass':bool(cal_pass),'nested_oof_metrics':nested,'c0_oof':c0_oof,'c3_oof':c3_oof,
        'nested_delta_vs_c0':{'nll':float(nested['nll']-c0_oof['nll']),'brier':float(nested['brier']-c0_oof['brier'])},
        'nested_bootstrap_vs_c0':cal_boot,'outer_fold_selections':fold_rows,
        'global_deployment_pair':global_selected,'global_oof_metrics':global_oof_metrics,
        'global_prefix_distribution':{'m2':float(np.mean(gprefix==2)),'m3':float(np.mean(gprefix==3)),
                                      'm4':float(np.mean(gprefix==4)),'coverage_shortfall':float(np.mean(gshort))},
        'global_invariants':ginv,
        'test_policy':'All CAL decisions and global pair frozen before TEST; TEST cannot retune alpha or eta.'}
    with open(out/'f2_11_cal_decision.json','w') as f: json.dump(cal_decision,f,indent=2)
    print('F2_11_CAL_DECISION',json.dumps(cal_decision),flush=True)

    # TEST after CAL freeze.
    p3_test,atest,qtest,full_diag=build_full_test_c3(X,y,tr,cal,zcal,ztest,zsc,support,rawgeo)
    Xtest,otest=F210.make_features(p3_test,atest,qtest)
    rank_test,_,_=F210.targets_from_order(ytest,otest)
    htest=fit_hazards(X_oof,rank_oof,Xtest.to_numpy())
    Ftest=cumulative_from_hazards(htest)
    pfinal,prefix_test,short_test,inv_test=adaptive_tail_diffuse(
        p3_test,atest,Ftest,global_selected['alpha'],global_selected['eta'])
    if pfinal is None or not inv_test.get('valid',False):
        raise RuntimeError(f'TEST frozen pair invariant failure {inv_test}')
    p0_test=F2.global_probs(ztest)
    c0_test=summarize(ytest,p0_test,support); c3_test=summarize(ytest,p3_test,support); final_test=summarize(ytest,pfinal,support)
    test_pass=terminal_pass(final_test,c0_test,c3_test)
    test_boot=paired_boot(ytest,pfinal,p0_test,BOOT_SEED+100)

    if cal_pass and test_pass and global_selected['eta']>0:
        final_status='TERMINAL_ADOPT_ADAPTIVE_PREFIX'
        deployment='F2 intervention lambda=-0.05 -> dose-conditioned C3 G2 -> coherent adaptive protected-prefix deeper-tail diffusion'
    else:
        final_status='TERMINAL_REJECT_LOCK_CANONICAL_V2'
        deployment='Canonical v2 = coherent S=5 -> global G2'

    pd.DataFrame([
        {'split':'OOF_CAL','model':'C0',**c0_oof},
        {'split':'OOF_CAL','model':'C3',**c3_oof},
        {'split':'OOF_CAL','model':'F2_11_nested',**nested},
        {'split':'OOF_CAL','model':'F2_11_global_pair',**global_oof_metrics},
        {'split':'TEST','model':'C0',**c0_test},
        {'split':'TEST','model':'C3',**c3_test},
        {'split':'TEST','model':'F2_11',**final_test},
    ]).to_csv(out/'f2_11_metrics.csv',index=False)
    pd.DataFrame({
        'split':['OOF_CAL']*len(cal)+['TEST']*len(te),
        'prefix_m':np.concatenate([gprefix,prefix_test]),
        'coverage_shortfall':np.concatenate([gshort,short_test]).astype(int)
    }).to_csv(out/'f2_11_prefix_assignments.csv',index=False)

    final={
        'status':final_status,'deployment_system':deployment,
        'cal_terminal_pass':bool(cal_pass),'test_terminal_pass':bool(test_pass),
        'global_alpha':float(global_selected['alpha']),'global_eta':float(global_selected['eta']),
        'global_pair_was_eligible_on_oof':bool(global_selected['eligible']),
        'test_metrics':{'C0':c0_test,'C3':c3_test,'F2_11':final_test},
        'test_delta_vs_c0':{'nll':float(final_test['nll']-c0_test['nll']),'brier':float(final_test['brier']-c0_test['brier'])},
        'test_bootstrap_vs_c0':test_boot,
        'test_prefix_distribution':{'m2':float(np.mean(prefix_test==2)),'m3':float(np.mean(prefix_test==3)),
                                    'm4':float(np.mean(prefix_test==4)),'coverage_shortfall':float(np.mean(short_test))},
        'test_invariants':inv_test,'fullcal_c3':full_diag,
        'terminal_policy':'No F2.12+ rescue stage. If rejected, Canonical v2 is locked and F2.5-F2.11 are analytical/ablation anatomy.'}
    with open(out/'F2_11_TERMINAL_ADAPTIVE_PREFIX.json','w') as f: json.dump(final,f,indent=2)
    print('F2_11_TEST_OPENED_AFTER_CAL_FREEZE',flush=True)
    print('F2_11_FINAL',json.dumps(final),flush=True)

if __name__=='__main__':
    main()
