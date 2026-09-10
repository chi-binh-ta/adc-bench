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
import run_f2_5_post_intervention_calibration as F25

LAMBDA_STAR=-0.05
OUTER_SEED=20260910
INNER_SEED=20260928
PAIR_SEED=20260929
BOOT_SEED=20260930
KAPPA=20.0
N_BOOT=1000
EPS=1e-12

GROUPS={
    'G1_score':[
        'p1','p2','p3','gap12','gap23','logratio12','logratio23',
        'runner_residual_share','residual_hhi','residual_entropy'],
    'G2_state':['dose','q_state'],
    'G3_seed':[
        'runner_rank_mean','runner_rank_sd','runner_rank2_frac','runner_top3_frac',
        'runner_top5_frac','runner_logit_sd','ensemble_top1_vote_frac','top1_above_runner_frac'],
    'G4_repr':['prototype_cosine','prototype_distance','log_support_ratio'],
    'G5_pair':['pair_reliability_prior','runner_class_prior','top1_class_prior','log_pair_count'],
}
ALL_COLS=sum(GROUPS.values(),[])


def top3_info(p):
    order=np.argsort(-p,axis=1,kind='stable')[:,:3]
    rows=np.arange(len(p))
    vals=p[rows[:,None],order]
    return order[:,0],order[:,1],order[:,2],vals[:,0],vals[:,1],vals[:,2]


def fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,fit_pos,eval_pos):
    fit_pos=np.asarray(fit_pos,dtype=int); eval_pos=np.asarray(eval_pos,dtype=int)
    K=zcal.shape[1]
    fit_abs=cal[fit_pos]
    zfit=zcal[fit_pos]; zeval=zcal[eval_pos]; zsfit=zsc[:,fit_pos,:]
    pfit_pre=F2.global_probs(zfit)
    d,ds,dh,ddiag=F2.derive_direction(X,y,tr,fit_abs,zfit,zsfit,pfit_pre,K,support,rawgeo)
    scale=F2.gate_scale(zfit)
    zfit_i,afit=F2.intervene(zfit,d,scale,LAMBDA_STAR)
    zeval_i,aeval=F2.intervene(zeval,d,scale,LAMBDA_STAR)
    Tfit=F2.global_bases_fast(zfit_i); Teval=F2.global_bases_fast(zeval_i)
    a2,_=F25.fit_theta(zfit_i,Tfit,y[fit_abs],F25.CANON_ALPHA)
    qfit,qeval,qmu,qsd=F25.standardize_gate(afit,aeval)
    a3,opt3=F25.fit_theta(
        zfit_i,F25.dose_bases(Tfit,qfit),y[fit_abs],
        np.array([a2[0],a2[1],0.0,0.0],dtype=float))
    peval=F25.apply_bases(zeval_i,F25.dose_bases(Teval,qeval),a3)
    return peval,aeval,qeval,{
        'scale':float(scale),'q_mu':float(qmu),'q_sd':float(qsd),
        'a3':a3.tolist(),'opt_success':bool(opt3['success']),
        'soft_hard_dot':float(ddiag['soft_hard_aligned_dot'])}


def build_prototype_geometry(X,y,tr,K):
    sc=StandardScaler().fit(X[tr])
    xt=sc.transform(X[tr]).astype(np.float64)
    cent=np.zeros((K,xt.shape[1]),dtype=np.float64)
    for k in range(K):
        mask=(y[tr]==k)
        if mask.any(): cent[k]=xt[mask].mean(axis=0)
    norms=np.linalg.norm(cent,axis=1)
    den=np.maximum(norms[:,None]*norms[None,:],1e-12)
    cos=(cent@cent.T)/den
    sq=np.sum(cent*cent,axis=1)
    d2=np.maximum(sq[:,None]+sq[None,:]-2*(cent@cent.T),0.0)
    dist=np.sqrt(d2)
    return cos,dist


def seed_features(zs,top1,runner):
    # zs: S x N x K
    S,N,K=zs.shape
    ii=np.arange(N)
    rscore=zs[:,ii,runner]
    ranks=1+np.sum(zs>rscore[:,:,None],axis=2)
    seed_top1=np.argmax(zs,axis=2)
    tscore=zs[:,ii,top1]
    return {
        'runner_rank_mean':ranks.mean(axis=0),
        'runner_rank_sd':ranks.std(axis=0),
        'runner_rank2_frac':(ranks==2).mean(axis=0),
        'runner_top3_frac':(ranks<=3).mean(axis=0),
        'runner_top5_frac':(ranks<=5).mean(axis=0),
        'runner_logit_sd':rscore.std(axis=0),
        'ensemble_top1_vote_frac':(seed_top1==top1[None,:]).mean(axis=0),
        'top1_above_runner_frac':(tscore>rscore).mean(axis=0),
    }


def base_features(p,dose,q,zs,cosmat,distmat,support):
    top1,runner,third,p1,p2,p3=top3_info(p)
    mass=np.maximum(1.0-p1,1e-15)
    N,K=p.shape
    residual_hhi=np.empty(N); residual_entropy=np.empty(N)
    for i in range(N):
        mask=np.ones(K,dtype=bool); mask[top1[i]]=False
        rr=p[i,mask]/mass[i]
        rr=np.maximum(rr,0.0); s=rr.sum()
        if not np.isfinite(s) or s<=0:
            residual_hhi[i]=1.0; residual_entropy[i]=0.0
        else:
            rr=rr/s
            residual_hhi[i]=np.sum(rr*rr)
            residual_entropy[i]=-np.sum(rr*np.log(np.maximum(rr,1e-300)))
    feat={
        'p1':p1,'p2':p2,'p3':p3,
        'gap12':p1-p2,'gap23':p2-p3,
        'logratio12':np.log(np.maximum(p1,1e-15)/np.maximum(p2,1e-15)),
        'logratio23':np.log(np.maximum(p2,1e-15)/np.maximum(p3,1e-15)),
        'runner_residual_share':p2/mass,
        'residual_hhi':residual_hhi,'residual_entropy':residual_entropy,
        'dose':dose,'q_state':q,
        'prototype_cosine':cosmat[top1,runner],
        'prototype_distance':distmat[top1,runner],
        'log_support_ratio':np.log((support[runner]+1.0)/(support[top1]+1.0)),
    }
    feat.update(seed_features(zs,top1,runner))
    return pd.DataFrame(feat),top1,runner


def fit_encoder(top1,runner,target):
    target=np.asarray(target,dtype=float)
    g=float(target.mean())
    def table(keys):
        d={}
        for k,t in zip(keys,target):
            if k not in d: d[k]=[0,0.0]
            d[k][0]+=1; d[k][1]+=float(t)
        return d
    pair=[(int(a),int(b)) for a,b in zip(top1,runner)]
    return {'global':g,'pair':table(pair),'runner':table([int(x) for x in runner]),
            'top1':table([int(x) for x in top1])}


def apply_encoder(enc,top1,runner):
    g=enc['global']; out=np.zeros((len(top1),4),dtype=float)
    for i,(a,b) in enumerate(zip(top1,runner)):
        pc,pp=enc['pair'].get((int(a),int(b)),(0,0.0))
        rc,rp=enc['runner'].get(int(b),(0,0.0))
        tc,tp=enc['top1'].get(int(a),(0,0.0))
        out[i,0]=(pp+KAPPA*g)/(pc+KAPPA)
        out[i,1]=(rp+KAPPA*g)/(rc+KAPPA)
        out[i,2]=(tp+KAPPA*g)/(tc+KAPPA)
        out[i,3]=math.log1p(pc)
    return pd.DataFrame(out,columns=GROUPS['G5_pair'])


def pair_oof(top1,runner,target,n_splits=4,seed=PAIR_SEED):
    out=np.zeros((len(target),4),dtype=float)
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for ti,vi in skf.split(np.zeros(len(target)),target):
        enc=fit_encoder(top1[ti],runner[ti],target[ti])
        out[vi]=apply_encoder(enc,top1[vi],runner[vi]).to_numpy()
    return pd.DataFrame(out,columns=GROUPS['G5_pair'])


def assemble(base,pair):
    df=pd.concat([base.reset_index(drop=True),pair.reset_index(drop=True)],axis=1)
    if list(df.columns)!=ALL_COLS:
        df=df[ALL_COLS]
    vals=df.to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        bad=np.where(~np.isfinite(vals))
        raise RuntimeError(f'nonfinite feature at {bad[0][0]},{bad[1][0]}')
    return df


def fit_logit(Xtr,ytr,Xv,cols):
    sc=StandardScaler().fit(Xtr[cols])
    a=sc.transform(Xtr[cols]); b=sc.transform(Xv[cols])
    clf=LogisticRegression(C=1.0,max_iter=3000,solver='lbfgs')
    clf.fit(a,ytr)
    return clf.predict_proba(b)[:,1]


def fit_hgb(Xtr,ytr,Xv):
    clf=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=150,max_leaf_nodes=15,
        l2_regularization=1.0,random_state=INNER_SEED)
    clf.fit(Xtr[ALL_COLS],ytr)
    return clf.predict_proba(Xv[ALL_COLS])[:,1]


def bin_metrics(y,s,baseline=None,wrong_mask=None):
    y=np.asarray(y,dtype=int); s=np.clip(np.asarray(s,dtype=float),1e-12,1-1e-12)
    prev=float(y.mean())
    auc=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else float('nan')
    ap=float(average_precision_score(y,s)) if y.sum()>0 else float('nan')
    ll=float(log_loss(y,s,labels=[0,1])); br=float(np.mean((s-y)**2))
    n10=max(1,int(math.ceil(0.10*len(y))))
    ix=np.argsort(-s,kind='stable')[:n10]; top_rate=float(y[ix].mean())
    d={'prevalence':prev,'auc':auc,'ap':ap,'ap_lift':float(ap/max(prev,1e-15)),
       'logloss':ll,'brier':br,'top10_positive_rate':top_rate,
       'top10_enrichment':float(top_rate/max(prev,1e-15))}
    if baseline is not None:
        bp=np.clip(np.asarray(baseline,dtype=float),1e-12,1-1e-12)
        d['baseline_logloss']=float(log_loss(y,bp,labels=[0,1]))
        d['logloss_delta_vs_baseline']=ll-d['baseline_logloss']
    if wrong_mask is not None:
        wm=np.asarray(wrong_mask,dtype=bool)
        yy=y[wm]; ss=s[wm]
        d['misclassified_n']=int(wm.sum())
        d['misclassified_prevalence']=float(yy.mean()) if len(yy) else float('nan')
        d['misclassified_auc']=float(roc_auc_score(yy,ss)) if len(np.unique(yy))==2 else float('nan')
        d['misclassified_ap']=float(average_precision_score(yy,ss)) if yy.sum()>0 else float('nan')
    return d


def boot_intervals(y,s,b,seed=BOOT_SEED,Bn=N_BOOT):
    rng=np.random.default_rng(seed); y=np.asarray(y); s=np.asarray(s); b=np.asarray(b)
    auc=[]; lift=[]; lld=[]; n=len(y)
    for _ in range(Bn):
        ix=rng.integers(0,n,n); yy=y[ix]
        if len(np.unique(yy))<2 or yy.mean()<=0: continue
        ss=np.clip(s[ix],1e-12,1-1e-12); bb=np.clip(b[ix],1e-12,1-1e-12)
        auc.append(roc_auc_score(yy,ss))
        lift.append(average_precision_score(yy,ss)/yy.mean())
        lld.append(log_loss(yy,ss,labels=[0,1])-log_loss(yy,bb,labels=[0,1]))
    q=lambda x:[float(np.quantile(x,.025)),float(np.quantile(x,.975))]
    return {'auc_ci95':q(auc),'ap_lift_ci95':q(lift),'logloss_delta_ci95':q(lld),'n_valid':len(auc)}


def build_inner_train_features(X,y,tr,cal,zcal,zsc,support,rawgeo,cosmat,distmat,outer_fit):
    yy=y[cal[outer_fit]]; n=len(outer_fit); K=zcal.shape[1]
    pp=np.zeros((n,K)); aa=np.zeros(n); qq=np.zeros(n)
    skf=StratifiedKFold(n_splits=4,shuffle=True,random_state=INNER_SEED)
    for itr,iva in skf.split(np.zeros(n),yy):
        pe,ae,qe,_=fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,outer_fit[itr],outer_fit[iva])
        pp[iva]=pe; aa[iva]=ae; qq[iva]=qe
    base,t1,r=base_features(pp,aa,qq,zsc[:,outer_fit,:],cosmat,distmat,support)
    target=(yy==r).astype(int)
    pair=pair_oof(t1,r,target,n_splits=4,seed=PAIR_SEED)
    return assemble(base,pair),target,t1,r,pp


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(float)
    rawgeo=F.raw_geometry_features(X,y,tr,K); cosmat,distmat=build_prototype_geometry(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir); zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]

    model_names=['LOGIT_FULL','LOGIT_SCORE_ONLY']+[f'LOGIT_NO_{g}' for g in GROUPS]+['HGB_FULL']
    oof_pred={m:np.zeros(len(cal),dtype=float) for m in model_names}
    base_pred=np.zeros(len(cal),dtype=float); oof_target=np.zeros(len(cal),dtype=int)
    oof_wrong=np.zeros(len(cal),dtype=bool); oof_top1=np.zeros(len(cal),dtype=int); oof_runner=np.zeros(len(cal),dtype=int)
    oof_features=np.zeros((len(cal),len(ALL_COLS)),dtype=float)
    oof_c3=np.zeros((len(cal),K),dtype=float)
    fold_rows=[]

    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED)
    for fold,(fi,vi) in enumerate(outer.split(np.zeros(len(cal)),ycal),1):
        fi=np.asarray(fi); vi=np.asarray(vi)
        Xtr_rel,ttr,t1tr,rtr,ptr=build_inner_train_features(
            X,y,tr,cal,zcal,zsc,support,rawgeo,cosmat,distmat,fi)
        pv,av,qv,diag=fit_apply_c3(X,y,tr,cal,zcal,zsc,support,rawgeo,fi,vi)
        bv,t1v,rv=base_features(pv,av,qv,zsc[:,vi,:],cosmat,distmat,support)
        enc=fit_encoder(t1tr,rtr,ttr)
        pairv=apply_encoder(enc,t1v,rv)
        Xv_rel=assemble(bv,pairv)
        tv=(ycal[vi]==rv).astype(int)

        oof_target[vi]=tv; oof_wrong[vi]=(t1v!=ycal[vi]); oof_top1[vi]=t1v; oof_runner[vi]=rv
        oof_features[vi]=Xv_rel[ALL_COLS].to_numpy(); oof_c3[vi]=pv
        base_pred[vi]=float(ttr.mean())

        oof_pred['LOGIT_FULL'][vi]=fit_logit(Xtr_rel,ttr,Xv_rel,ALL_COLS)
        oof_pred['LOGIT_SCORE_ONLY'][vi]=fit_logit(Xtr_rel,ttr,Xv_rel,GROUPS['G1_score'])
        for g,cols in GROUPS.items():
            keep=[c for c in ALL_COLS if c not in cols]
            oof_pred[f'LOGIT_NO_{g}'][vi]=fit_logit(Xtr_rel,ttr,Xv_rel,keep)
        oof_pred['HGB_FULL'][vi]=fit_hgb(Xtr_rel,ttr,Xv_rel)
        fold_rows.append({'fold':fold,'n_train':len(fi),'n_val':len(vi),'train_runner_rate':float(ttr.mean()),
                          'val_runner_rate':float(tv.mean()),'val_misclassified_rate':float(oof_wrong[vi].mean()),**diag})
        print('F2_8_FOLD',json.dumps(fold_rows[-1]),flush=True)

    pd.DataFrame(fold_rows).to_csv(out/'f2_8_fold_diagnostics.csv',index=False)
    # C3 replay gate.
    c3m=F2.metrics(ycal,oof_c3,support)
    exp={'nll':2.7397491931915283,'brier':0.7826232437924868,'ece':0.028646533298620415,
         'macro_f1':0.19490055879898846}
    replay={k:float(c3m[k]-v) for k,v in exp.items()}
    if max(abs(v) for v in replay.values())>2e-6: raise RuntimeError(f'C3 replay mismatch {replay}')

    metrics={m:bin_metrics(oof_target,p,base_pred,oof_wrong) for m,p in oof_pred.items()}
    base_metrics=bin_metrics(oof_target,base_pred,base_pred,oof_wrong)
    rows=[{'model':'BASELINE',**base_metrics}]+[{'model':m,**metrics[m]} for m in model_names]
    pd.DataFrame(rows).to_csv(out/'f2_8_oof_metrics.csv',index=False)
    boot=boot_intervals(oof_target,oof_pred['LOGIT_FULL'],base_pred)

    attr=[]; fullm=metrics['LOGIT_FULL']
    for g in GROUPS:
        mm=metrics[f'LOGIT_NO_{g}']
        da=fullm['auc']-mm['auc']; dap=fullm['ap']-mm['ap']; dll=mm['logloss']-fullm['logloss']
        attr.append({'group':g,'delta_auc_full_minus_no_group':da,'delta_ap':dap,
                     'delta_logloss_no_group_minus_full':dll,
                     'predictively_incremental':bool(da>0.005 and dll>0.001)})
    pd.DataFrame(attr).to_csv(out/'f2_8_group_attribution.csv',index=False)

    primary=bool(
        fullm['auc']>=0.60 and fullm['ap_lift']>=1.25 and
        (base_metrics['logloss']-fullm['logloss'])>=0.003 and fullm['top10_enrichment']>=1.50 and
        fullm['misclassified_auc']>=0.56 and boot['auc_ci95'][0]>0.55 and
        boot['ap_lift_ci95'][0]>1.10 and boot['logloss_delta_ci95'][1]<0)
    hm=metrics['HGB_FULL']
    nonlinear=bool(hm['auc']>=0.60 and hm['ap_lift']>=1.25 and
        (base_metrics['logloss']-hm['logloss'])>=0.003 and hm['top10_enrichment']>=1.50 and hm['misclassified_auc']>=0.56)
    cal_status='PRIMARY_IDENTIFIED' if primary else ('NONLINEAR_SIGNAL_ONLY' if nonlinear else 'NOT_IDENTIFIED')
    cal_dec={'status':cal_status,'primary_model':'LOGIT_FULL','metrics':metrics,'baseline':base_metrics,
             'bootstrap':boot,'group_attribution':attr,'c3_replay_gap':replay,
             'target_prevalence':float(oof_target.mean()),
             'misclassified_runnerup_rate':float(oof_target[oof_wrong].mean()),
             'test_policy':'CAL status frozen before TEST; TEST cannot rescue identification'}
    with open(out/'f2_8_cal_decision.json','w') as f: json.dump(cal_dec,f,indent=2)
    print('F2_8_CAL_DECISION',json.dumps(cal_dec),flush=True)

    # Archive OOF features/predictions used for final reliability fit.
    fdf=pd.DataFrame(oof_features,columns=ALL_COLS)
    fdf.insert(0,'runner_class',oof_runner); fdf.insert(0,'top1_class',oof_top1); fdf.insert(0,'target_runner_true',oof_target)
    fdf.to_csv(out/'f2_8_oof_features.csv',index=False)
    pdf=pd.DataFrame({'target_runner_true':oof_target,'top1_wrong':oof_wrong.astype(int),'baseline':base_pred})
    for m in model_names: pdf[m]=oof_pred[m]
    pdf.to_csv(out/'f2_8_oof_predictions.csv',index=False)

    # TEST opened only after CAL decision is frozen.
    allpos=np.arange(len(cal),dtype=int); test_dummy=np.arange(len(te),dtype=int)
    # Full-CAL C3 -> TEST, written separately because eval positions are not CAL positions.
    pcal_pre=F2.global_probs(zcal)
    dfull,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,K,support,rawgeo)
    scale=F2.gate_scale(zcal)
    zcal_i,acal=F2.intervene(zcal,dfull,scale,LAMBDA_STAR)
    ztest_i,atest=F2.intervene(ztest,dfull,scale,LAMBDA_STAR)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,_=F25.fit_theta(zcal_i,Tcal,ycal,F25.CANON_ALPHA)
    qcal,qtest,qmu,qsd=F25.standardize_gate(acal,atest)
    a3,_=F25.fit_theta(zcal_i,F25.dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0.0,0.0]))
    ptest=F25.apply_bases(ztest_i,F25.dose_bases(Ttest,qtest),a3)
    btest,t1test,rtest=base_features(ptest,atest,qtest,zst,cosmat,distmat,support)
    # Pair encoder is frozen from all OOF-CAL runner-up assignments/targets.
    fullenc=fit_encoder(oof_top1,oof_runner,oof_target)
    Xtest=assemble(btest,apply_encoder(fullenc,t1test,rtest))
    ttest=(ytest==rtest).astype(int); wrongtest=(t1test!=ytest)
    Xoof=pd.DataFrame(oof_features,columns=ALL_COLS)
    sc=StandardScaler().fit(Xoof[ALL_COLS]); clf=LogisticRegression(C=1.0,max_iter=3000,solver='lbfgs')
    clf.fit(sc.transform(Xoof[ALL_COLS]),oof_target)
    stest=clf.predict_proba(sc.transform(Xtest[ALL_COLS]))[:,1]
    hgb=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=150,max_leaf_nodes=15,l2_regularization=1.0,random_state=INNER_SEED)
    hgb.fit(Xoof[ALL_COLS],oof_target); shtest=hgb.predict_proba(Xtest[ALL_COLS])[:,1]
    frozen_prev=float(oof_target.mean()); bconst=np.full(len(ttest),frozen_prev)
    testm=bin_metrics(ttest,stest,bconst,wrongtest); testh=bin_metrics(ttest,shtest,bconst,wrongtest)
    testboot=boot_intervals(ttest,stest,bconst,seed=BOOT_SEED+1)
    replicated=bool(primary and testm['auc']>=0.58 and testm['ap_lift']>=1.15 and
                    testm['logloss']<testm['baseline_logloss'] and testm['top10_enrichment']>=1.30 and
                    testm['misclassified_auc']>=0.54)
    final_status='INTERNALLY_REPLICATED_IDENTIFICATION' if replicated else cal_status
    test_rows=[{'model':'LOGIT_FULL','dataset':'TEST',**testm},{'model':'HGB_FULL','dataset':'TEST',**testh}]
    pd.DataFrame(test_rows).to_csv(out/'f2_8_test_metrics.csv',index=False)
    tpdf=Xtest.copy(); tpdf.insert(0,'runner_class',rtest); tpdf.insert(0,'top1_class',t1test)
    tpdf.insert(0,'top1_wrong',wrongtest.astype(int)); tpdf.insert(0,'target_runner_true',ttest)
    tpdf['LOGIT_FULL']=stest; tpdf['HGB_FULL']=shtest
    tpdf.to_csv(out/'f2_8_test_predictions.csv',index=False)

    final={'status':final_status,'cal_status':cal_status,'internally_replicated':replicated,
           'oof_primary':fullm,'oof_baseline':base_metrics,'oof_bootstrap':boot,
           'test_primary':testm,'test_nonlinear':testh,'test_bootstrap':testboot,
           'group_attribution':attr,'c3_replay_gap':replay,
           'oof_runner_true_prevalence':float(oof_target.mean()),
           'oof_misclassified_runnerup_rate':float(oof_target[oof_wrong].mean()),
           'test_runner_true_prevalence':float(ttest.mean()),
           'test_misclassified_runnerup_rate':float(ttest[wrongtest].mean()),
           'fullcal_c3_coefficients':a3.tolist(),
           'interpretation':'Identification only. No runner-up protection threshold or probability correction is authorized by F2.8.'}
    with open(out/'F2_8_RUNNER_UP_RELIABILITY.json','w') as f: json.dump(final,f,indent=2)
    with open(out/'f2_8_bootstrap.json','w') as f: json.dump({'OOF_CAL':boot,'TEST':testboot},f,indent=2)
    print('F2_8_TEST_OPENED_AFTER_CAL_FREEZE',flush=True)
    print('F2_8_FINAL',json.dumps(final),flush=True)

if __name__=='__main__': main()
