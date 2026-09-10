import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import run_stageB1_clone as B
from numeric_split import load_split_numeric
import run_latent_factor_identification_v2 as F
import run_f2_margin_residual_intervention as F2
import run_f2_8_runner_up_reliability as F28

OUTER_SEED=20260910
INNER_SEED=20261001
BOOT_SEED=20261002
N_BOOT=1000
EPS=1e-12
CUM_MS=[2,3,5]
HAZ_RS=[2,3,4,5]

FEATURES=[
    'p1','p2','p3','p4','p5','p6',
    'g12','g23','g34','g45','g56',
    'lr12','lr23','lr34','lr45','lr56',
    'top2_mass','top3_mass','top5_mass',
    'p2_resid1','p3_resid2','p5_resid4',
    'hhi_suf1','ent_suf1','hhi_suf2','ent_suf2',
    'hhi_suf3','ent_suf3','hhi_suf5','ent_suf5',
    'dose','q_state'
]


def sorted_info(p):
    order=np.argsort(-p,axis=1,kind='stable')
    vals=np.take_along_axis(p,order,axis=1)
    return order,vals


def suffix_stats(vals,start):
    x=vals[:,start:]
    mass=np.maximum(x.sum(axis=1,keepdims=True),1e-300)
    r=x/mass
    hhi=np.sum(r*r,axis=1)
    ent=-np.sum(r*np.log(np.maximum(r,1e-300)),axis=1)
    return hhi,ent


def make_features(p,dose,q):
    order,v=sorted_info(p)
    feat={}
    for j in range(6): feat[f'p{j+1}']=v[:,j]
    for j in range(5):
        feat[f'g{j+1}{j+2}']=v[:,j]-v[:,j+1]
        feat[f'lr{j+1}{j+2}']=np.log(np.maximum(v[:,j],1e-15)/np.maximum(v[:,j+1],1e-15))
    feat['top2_mass']=v[:,:2].sum(axis=1)
    feat['top3_mass']=v[:,:3].sum(axis=1)
    feat['top5_mass']=v[:,:5].sum(axis=1)
    feat['p2_resid1']=v[:,1]/np.maximum(1-v[:,0],1e-15)
    feat['p3_resid2']=v[:,2]/np.maximum(1-v[:,:2].sum(axis=1),1e-15)
    feat['p5_resid4']=v[:,4]/np.maximum(1-v[:,:4].sum(axis=1),1e-15)
    for s in [1,2,3,5]:
        h,e=suffix_stats(v,s)
        feat[f'hhi_suf{s}']=h; feat[f'ent_suf{s}']=e
    feat['dose']=dose; feat['q_state']=q
    df=pd.DataFrame(feat)[FEATURES]
    if not np.isfinite(df.to_numpy()).all(): raise RuntimeError('nonfinite F2.10 feature')
    return df,order


def targets_from_order(y,order):
    rank=np.empty(len(y),dtype=int)
    inv=np.empty_like(order)
    rr=np.arange(1,order.shape[1]+1,dtype=order.dtype)
    inv[np.arange(len(order))[:,None],order]=rr[None,:]
    rank=inv[np.arange(len(y)),y]
    cum={m:(rank<=m).astype(int) for m in CUM_MS}
    haz={r:{'risk':rank>=r,'target':(rank==r).astype(int)} for r in HAZ_RS}
    return rank,cum,haz


def fit_predict_logit(Xtr,ytr,Xv):
    sc=StandardScaler().fit(Xtr)
    a=sc.transform(Xtr); b=sc.transform(Xv)
    clf=LogisticRegression(C=1.0,max_iter=3000,solver='lbfgs')
    clf.fit(a,ytr)
    return clf.predict_proba(b)[:,1]


def fit_predict_hgb(Xtr,ytr,Xv):
    clf=HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=15,
        l2_regularization=1.0,random_state=INNER_SEED)
    clf.fit(Xtr,ytr)
    return clf.predict_proba(Xv)[:,1]


def metrics(y,s,b):
    y=np.asarray(y,dtype=int); s=np.clip(np.asarray(s,float),1e-12,1-1e-12); b=np.clip(np.asarray(b,float),1e-12,1-1e-12)
    prev=float(y.mean())
    auc=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else float('nan')
    ap=float(average_precision_score(y,s)) if y.sum()>0 else float('nan')
    ll=float(log_loss(y,s,labels=[0,1])); base=float(log_loss(y,b,labels=[0,1]))
    br=float(np.mean((s-y)**2)); n10=max(1,int(math.ceil(.10*len(y))))
    top=float(y[np.argsort(-s,kind='stable')[:n10]].mean())
    return {'n':int(len(y)),'prevalence':prev,'auc':auc,'ap':ap,'ap_lift':float(ap/max(prev,1e-15)),
            'logloss':ll,'baseline_logloss':base,'logloss_delta':ll-base,'brier':br,
            'top10_positive_rate':top,'top10_enrichment':float(top/max(prev,1e-15))}


def boot(y,s,b,seed=BOOT_SEED):
    rng=np.random.default_rng(seed); y=np.asarray(y); s=np.asarray(s); b=np.asarray(b); n=len(y)
    aa=[]; al=[]; dl=[]
    for _ in range(N_BOOT):
        ix=rng.integers(0,n,n); yy=y[ix]
        if len(np.unique(yy))<2 or yy.mean()<=0: continue
        ss=np.clip(s[ix],1e-12,1-1e-12); bb=np.clip(b[ix],1e-12,1-1e-12)
        aa.append(roc_auc_score(yy,ss)); al.append(average_precision_score(yy,ss)/yy.mean())
        dl.append(log_loss(yy,ss,labels=[0,1])-log_loss(yy,bb,labels=[0,1]))
    q=lambda x:[float(np.quantile(x,.025)),float(np.quantile(x,.975))]
    return {'auc_ci95':q(aa),'ap_lift_ci95':q(al),'logloss_delta_ci95':q(dl),'n_valid':len(aa)}


def cum_identified(m,b):
    return bool(m['auc']>=.62 and m['ap_lift']>=1.20 and m['logloss_delta']<=-.005 and
                m['top10_enrichment']>=1.25 and b['auc_ci95'][0]>.57 and b['ap_lift_ci95'][0]>1.10 and b['logloss_delta_ci95'][1]<0)


def haz_identified(m):
    return bool(m['n']>=500 and m['auc']>=.58 and m['ap_lift']>=1.15 and m['logloss_delta']<=-.003)


def inner_oof_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,outer_fit):
    n=len(outer_fit); K=zcal.shape[1]; yy=y[cal[outer_fit]]
    pp=np.zeros((n,K)); aa=np.zeros(n); qq=np.zeros(n)
    skf=StratifiedKFold(n_splits=4,shuffle=True,random_state=INNER_SEED)
    for ti,vi in skf.split(np.zeros(n),yy):
        pe,ae,qe,_=F28.fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,outer_fit[ti],outer_fit[vi])
        pp[vi]=pe; aa[vi]=ae; qq[vi]=qe
    return pp,aa,qq


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(float)
    rawgeo=F.raw_geometry_features(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir); zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]

    oof_X=np.zeros((len(cal),len(FEATURES))); oof_order=np.zeros((len(cal),K),dtype=int)
    cum_pred={m:np.zeros(len(cal)) for m in CUM_MS}; cum_hgb={m:np.zeros(len(cal)) for m in CUM_MS}; cum_base={m:np.zeros(len(cal)) for m in CUM_MS}
    haz_pred={r:np.full(len(cal),np.nan) for r in HAZ_RS}; haz_base={r:np.full(len(cal),np.nan) for r in HAZ_RS}; haz_risk={r:np.zeros(len(cal),bool) for r in HAZ_RS}
    fold_rows=[]

    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED)
    for fold,(fi,vi) in enumerate(outer.split(np.zeros(len(cal)),ycal),1):
        fi=np.asarray(fi); vi=np.asarray(vi)
        ptri,atri,qtri=inner_oof_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,fi)
        Xtri,otri=make_features(ptri,atri,qtri); rtri,ctri,htri=targets_from_order(ycal[fi],otri)
        pv,av,qv,diag=F28.fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,fi,vi)
        Xv,ov=make_features(pv,av,qv); rv,cv,hv=targets_from_order(ycal[vi],ov)
        oof_X[vi]=Xv.to_numpy(); oof_order[vi]=ov
        for m in CUM_MS:
            cum_pred[m][vi]=fit_predict_logit(Xtri,ctri[m],Xv)
            cum_hgb[m][vi]=fit_predict_hgb(Xtri,ctri[m],Xv)
            cum_base[m][vi]=float(ctri[m].mean())
        for r in HAZ_RS:
            trmask=htri[r]['risk']; vmask=hv[r]['risk']; haz_risk[r][vi]=vmask
            yt=htri[r]['target'][trmask]
            if len(np.unique(yt))<2: raise RuntimeError(f'hazard H{r} training fold one class')
            pred=fit_predict_logit(Xtri.loc[trmask],yt,Xv.loc[vmask])
            haz_pred[r][vi[vmask]]=pred; haz_base[r][vi[vmask]]=float(yt.mean())
        fold_rows.append({'fold':fold,'n_fit':len(fi),'n_val':len(vi),'scale':diag['scale'],'q_mu':diag['q_mu'],'q_sd':diag['q_sd'],'soft_hard_dot':diag['soft_hard_dot']})
        print('F2_10_FOLD',json.dumps(fold_rows[-1]),flush=True)

    pd.DataFrame(fold_rows).to_csv(out/'f2_10_fold_diagnostics.csv',index=False)
    pd.DataFrame(oof_X,columns=FEATURES).to_csv(out/'f2_10_oof_features.csv',index=False)

    _,cum_true,haz_true=targets_from_order(ycal,oof_order)
    cum_metrics={}; cum_boot={}; hgb_metrics={}; cum_pass={}
    for m in CUM_MS:
        cum_metrics[m]=metrics(cum_true[m],cum_pred[m],cum_base[m]); cum_boot[m]=boot(cum_true[m],cum_pred[m],cum_base[m],BOOT_SEED+m)
        hgb_metrics[m]=metrics(cum_true[m],cum_hgb[m],cum_base[m]); cum_pass[m]=cum_identified(cum_metrics[m],cum_boot[m])
    haz_metrics={}; haz_pass={}
    for r in HAZ_RS:
        mask=haz_risk[r]; yy=haz_true[r]['target'][mask]
        haz_metrics[r]=metrics(yy,haz_pred[r][mask],haz_base[r][mask]); haz_pass[r]=haz_identified(haz_metrics[r])

    n_pass=sum(cum_pass.values())
    if n_pass==3: cal_status='PREFIX_IDENTIFIED'
    elif n_pass==2: cal_status='PARTIAL_PREFIX_IDENTIFIED'
    elif n_pass==1: cal_status='WEAK_PREFIX_SIGNAL'
    else: cal_status='NOT_IDENTIFIED'
    mono12=cum_pred[2]-cum_pred[3]; mono35=cum_pred[3]-cum_pred[5]
    viol=(mono12>0)|(mono35>0)
    mag=np.maximum(mono12,0)+np.maximum(mono35,0)
    mono={'violation_fraction':float(viol.mean()),'mean_total_violation':float(mag.mean()),'max_total_violation':float(mag.max())}
    decision={'status':cal_status,'cumulative_metrics':cum_metrics,'cumulative_bootstrap':cum_boot,'cumulative_identified':cum_pass,
              'hazard_metrics':haz_metrics,'hazard_identified':haz_pass,'hgb_cumulative':hgb_metrics,'monotonicity':mono,
              'test_policy':'CAL status frozen before TEST; TEST cannot change OOF closure'}
    with open(out/'f2_10_cal_decision.json','w') as f: json.dump(decision,f,indent=2)
    print('F2_10_CAL_DECISION',json.dumps(decision),flush=True)

    # Full-CAL C3 and TEST features after CAL decision freeze.
    allpos=np.arange(len(cal)); ptest,atest,qtest,diag=F28.fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,allpos,np.arange(len(te)))
    # fit_apply_c3 indexes zcal eval positions, so build full-CAL -> TEST explicitly using F2/F2.5 lineage.
    # Recompute here with full CAL fit and ztest evaluation.
    import run_f2_5_post_intervention_calibration as F25
    pcal_pre=F2.global_probs(zcal)
    dfull,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,K,support,rawgeo)
    scale=F2.gate_scale(zcal); zcal_i,acal=F2.intervene(zcal,dfull,scale,-0.05); ztest_i,atest=F2.intervene(ztest,dfull,scale,-0.05)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,_=F25.fit_theta(zcal_i,Tcal,ycal,F25.CANON_ALPHA); qcal,qtest,qmu,qsd=F25.standardize_gate(acal,atest)
    a3,_=F25.fit_theta(zcal_i,F25.dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0.,0.]))
    ptest=F25.apply_bases(ztest_i,F25.dose_bases(Ttest,qtest),a3)
    Xtest,otest=make_features(ptest,atest,qtest); rtest,ctest,htest=targets_from_order(ytest,otest)
    Xoof=pd.DataFrame(oof_X,columns=FEATURES)

    test_cum={}; test_cum_pass={}; test_haz={}
    pred_test_cum={}
    for m in CUM_MS:
        pred=fit_predict_logit(Xoof,cum_true[m],Xtest); pred_test_cum[m]=pred
        b=np.full(len(ytest),float(cum_true[m].mean()))
        mm=metrics(ctest[m],pred,b); test_cum[m]=mm
        test_cum_pass[m]=bool(cum_pass[m] and mm['auc']>=.60 and mm['ap_lift']>=1.15 and mm['logloss_delta']<0 and mm['top10_enrichment']>=1.20)
    for r in HAZ_RS:
        trmask=haz_true[r]['risk']; temask=htest[r]['risk']
        pred=fit_predict_logit(Xoof.loc[trmask],haz_true[r]['target'][trmask],Xtest.loc[temask])
        b=np.full(int(temask.sum()),float(haz_true[r]['target'][trmask].mean()))
        test_haz[r]=metrics(htest[r]['target'][temask],pred,b)

    final_status='INTERNALLY_REPLICATED_PREFIX_IDENTIFICATION' if cal_status=='PREFIX_IDENTIFIED' and all(test_cum_pass.values()) else cal_status
    testmono12=pred_test_cum[2]-pred_test_cum[3]; testmono35=pred_test_cum[3]-pred_test_cum[5]
    testmono={'violation_fraction':float(((testmono12>0)|(testmono35>0)).mean()),'mean_total_violation':float((np.maximum(testmono12,0)+np.maximum(testmono35,0)).mean())}
    final={'status':final_status,'cal_status':cal_status,'test_cumulative':test_cum,'test_cumulative_replication':test_cum_pass,
           'test_hazards':test_haz,'test_monotonicity':testmono,'fullcal_c3_coefficients':a3.tolist(),
           'interpretation':'Identification only. No adaptive prefix rule or diffusion is authorized.'}
    with open(out/'F2_10_ADAPTIVE_PROTECTED_PREFIX.json','w') as f: json.dump(final,f,indent=2)
    pd.DataFrame([{'target':f'C{m}',**cum_metrics[m],'identified':cum_pass[m]} for m in CUM_MS]+[
        {'target':f'H{r}',**haz_metrics[r],'identified':haz_pass[r]} for r in HAZ_RS]).to_csv(out/'f2_10_oof_metrics.csv',index=False)
    pd.DataFrame([{'target':f'C{m}',**test_cum[m],'replicated':test_cum_pass[m]} for m in CUM_MS]+[
        {'target':f'H{r}',**test_haz[r],'replicated':False} for r in HAZ_RS]).to_csv(out/'f2_10_test_metrics.csv',index=False)
    np.savez_compressed(out/'f2_10_oof_predictions.npz',**{f'C{m}':cum_pred[m] for m in CUM_MS},**{f'H{r}':haz_pred[r] for r in HAZ_RS})
    print('F2_10_TEST_OPENED_AFTER_CAL_FREEZE',flush=True)
    print('F2_10_FINAL',json.dumps(final),flush=True)

if __name__=='__main__': main()
