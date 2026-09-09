import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import run_stageB1_clone as B
from numeric_split import load_split_numeric
from optimizer_anchor_qn import dotpair, lbfgs_dir

QN = json.load(open(Path(__file__).with_name('qn_selected.json'),'r'))
QN_MEM = int(QN['memory'])
QN_SCALE = float(QN['scale'])


def train_qn(name, F, y, theta, bias, V, qminus, steps, grad_fn, memory=QN_MEM, scale=QN_SCALE):
    hist=[]; prev_th=prev_bi=prev_gt=prev_gb=None; logs=[]
    th=theta.copy(); bi=bias.copy()
    for step in range(1,steps+1):
        t0=time.time(); out=grad_fn(th,bi); loss,gt,gb=out[:3]; extra=out[3] if len(out)>3 else {}
        if prev_th is not None:
            sT=(th.astype(np.float64)-prev_th).astype(np.float32)
            sB=(bi.astype(np.float64)-prev_bi).astype(np.float32)
            yT=(gt-prev_gt).astype(np.float32); yB=(gb-prev_gb).astype(np.float32)
            sy=dotpair(sT,sB,yT,yB)
            if sy>1e-12 and np.isfinite(sy):
                hist.append((sT,sB,yT,yB,1.0/sy))
                if len(hist)>memory: hist.pop(0)
        dT,dB,gamma=lbfgs_dir(gt,gb,hist)
        prev_th=th.astype(np.float64).copy(); prev_bi=bi.astype(np.float64).copy(); prev_gt=gt.copy(); prev_gb=gb.copy()
        th=(th.astype(np.float64)+scale*dT).astype(np.float32); bi=(bi.astype(np.float64)+scale*dB).astype(np.float32)
        rec=dict(stage=name,step=step,loss=float(loss),scale=scale,memory=memory,gamma=float(gamma),hist=len(hist),seconds=time.time()-t0,**extra)
        logs.append(rec); print('QN_STEP',json.dumps(rec),flush=True)
        if not np.isfinite(loss) or not np.all(np.isfinite(th)): raise RuntimeError(f'non-finite {name} step {step}')
    return th,bi,logs


def nll_only(y,p):
    return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())


def main():
    if QN.get('status') == 'PENDING_ANCHOR': raise RuntimeError('qn_selected.json is still PENDING_ANCHOR')
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path']); Xt,yt=X[tr],y[tr]
    li,C=B.build_landmarks(Xt,yt); paths,mean,std=B.build_feature_cache(X,(tr,meta,cal,te),C)
    sizes=dict(train=len(tr),meta=len(meta),cal=len(cal),test=len(te)); F=B.open_features(paths,sizes)
    V,qminus,rsp=B.build_rsp(F['train'],yt)
    K=int(y.max()+1); pi,train_support=B.class_priors(yt,K); cw_base=B.normalized_class_weights(pi**(-B.ALPHA),yt)
    theta=np.zeros((B.M,K),np.float32); bias=np.zeros(K,np.float32)

    theta,bias,hw=train_qn('WCE',F['train'],yt,theta,bias,V,qminus,B.WCE_STEPS,
        lambda th,bi:B.wce_loss_grad(F['train'],yt,th,bi,V,qminus,cw_base))
    pmeta_w=B.predict_proba(F['meta'],theta,bias,V,qminus); wce_meta_nll=nll_only(y[meta],pmeta_w)
    print('ANCHOR_STAGE',json.dumps({'stage':'WCE','meta_nll':wce_meta_nll,'historical':3.08469558}),flush=True)

    theta,bias,hs=train_qn('SoftMacroF1',F['train'],yt,theta,bias,V,qminus,B.SOFT_STEPS,
        lambda th,bi:B.softf1_loss_grad(F['train'],yt,th,bi,V,qminus,cw_base,B.LAM_F))
    pmeta=B.predict_proba(F['meta'],theta,bias,V,qminus); soft_meta_nll=nll_only(y[meta],pmeta)
    print('ANCHOR_STAGE',json.dumps({'stage':'SoftMacroF1','meta_nll':soft_meta_nll,'historical':3.06046534}),flush=True)
    np.savez_compressed(B.OUT/'checkpoint_soft_qn_seed42_m16384.npz',theta=theta,bias=bias)

    Praw,Rraw,Fraw,supp,tp,fp,fn=B.pr_from_probs(y[meta],pmeta,K)
    pd.DataFrame(dict(cls=np.arange(K),precision=Praw,recall=Rraw,f1=Fraw,support=supp,tp=tp,fp=fp,fn=fn,pi=pi)).to_csv(B.OUT/'meta_pr_soft_qn.csv',index=False)
    Pbar=float(np.mean(Praw)); Rbar=float(np.mean(Rraw))

    rows=[]; unique={}; all_logs=hw+hs
    for tau0 in B.CFG['stageB1']['tau0']:
      for rho in B.CFG['stageB1']['rho']:
       for lam in B.CFG['stageB1']['lambda_shrink']:
        tau0=float(tau0); rho=float(rho); lam=float(lam); key=(tau0,rho,lam if rho!=0 else 0.0)
        if key not in unique:
            Ptilde=(tp+lam*Pbar)/(tp+fp+lam); Rtilde=(tp+lam*Rbar)/(tp+fn+lam)
            tauk=np.clip(tau0+rho*np.log((Ptilde+B.EPS)/(Rtilde+B.EPS)),0,1)
            etaR=.05+.15*tauk; etaP=.20-.15*tauk
            factor=((Rbar+B.EPS)/(Rraw+B.EPS))**etaR*((Praw+B.EPS)/(Pbar+B.EPS))**etaP
            factor=np.clip(factor,B.CLIP_LO,B.CLIP_HI); cw=B.normalized_class_weights((pi**(-B.ALPHA))*factor,yt)
            th,bi,h=train_qn(f'Adaptive_t{tau0}_r{rho}_l{lam}',F['train'],yt,theta,bias,V,qminus,B.ADAPT_STEPS,
                lambda tt,bb,cw=cw:B.wce_loss_grad(F['train'],yt,tt,bb,V,qminus,cw))
            pcal=B.predict_proba(F['cal'],th,bi,V,qminus); mcal=B.metrics(y[cal],pcal,train_support)
            pmeta_after=B.predict_proba(F['meta'],th,bi,V,qminus); meta_after=nll_only(y[meta],pmeta_after)
            unique[key]=(mcal,tauk,etaR,etaP,meta_after,th,bi,h); all_logs+=h
        mcal,tauk,etaR,etaP,meta_after,th,bi,h=unique[key]
        rows.append(dict(seed=42,m=B.M,tau0=tau0,rho=rho,lambda_shrink=lam,
            tauk_mean=float(tauk.mean()),tauk_min=float(tauk.min()),tauk_max=float(tauk.max()),
            etaR_mean=float(etaR.mean()),etaP_mean=float(etaP.mean()),meta_nll_after=float(meta_after),
            **{'cal_'+k:v for k,v in mcal.items()}))

    df=pd.DataFrame(rows)
    paired=['accuracy','macro_f1','balanced_accuracy','tail20_f1','nll','brier','ece']
    for mc in paired: df['delta_cal_'+mc]=np.nan
    for i,r in df.iterrows():
        ctrl=df[(df.tau0==r.tau0)&(df.rho==0.0)&(df.lambda_shrink==r.lambda_shrink)].iloc[0]
        for mc in paired: df.loc[i,'delta_cal_'+mc]=r['cal_'+mc]-ctrl['cal_'+mc]
    df['pareto_gain_cal']=(
        ((df.delta_cal_accuracy>0)|(df.delta_cal_macro_f1>0)|(df.delta_cal_tail20_f1>0)) &
        (df.delta_cal_accuracy>=-5e-4)&(df.delta_cal_macro_f1>=-5e-4)&(df.delta_cal_tail20_f1>=-5e-4)
    )
    ranked=df.sort_values(['pareto_gain_cal','delta_cal_macro_f1','delta_cal_tail20_f1','delta_cal_accuracy'],ascending=[False,False,False,False])
    df.to_csv(B.OUT/'stageB1_qn_seed42_18rows_cal.csv',index=False); ranked.to_csv(B.OUT/'stageB1_qn_seed42_ranked_cal.csv',index=False)
    json.dump(all_logs,open(B.OUT/'stageB1_qn_training_logs.json','w'),indent=2)
    summary={'status':'STAGE_B1_QN_NUMERIC_SPLIT','qn':QN,'rsp':rsp,'wce_meta_nll':wce_meta_nll,'soft_meta_nll':soft_meta_nll,
             'n_rows':len(df),'n_unique':len(unique),'n_pareto_cal':int(df.pareto_gain_cal.sum()),'top5_cal':ranked.head(5).to_dict(orient='records')}
    json.dump(summary,open(B.OUT/'stageB1_qn_summary.json','w'),indent=2); print('B1_QN_SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__': main()
