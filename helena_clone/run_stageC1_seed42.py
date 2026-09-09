import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import run_stageB1_clone as B
import run_stageB1_qn as Q
from numeric_split import load_split_numeric

SEED = 42
TAU = 0.694
MARGIN_HEIGHT = 0.5
R_GRID = [3, 5]
LM_GRID = [0.025, 0.05, 0.10, 0.20]


def scalar_adaptive_weights(pi, yt, Praw, Rraw):
    Pbar = float(np.mean(Praw)); Rbar = float(np.mean(Rraw))
    etaR = 0.05 + 0.15 * TAU
    etaP = 0.20 - 0.15 * TAU
    factor = ((Rbar + B.EPS)/(Rraw + B.EPS))**etaR * ((Praw + B.EPS)/(Pbar + B.EPS))**etaP
    factor = np.clip(factor, B.CLIP_LO, B.CLIP_HI)
    return B.normalized_class_weights((pi**(-B.ALPHA))*factor, yt)


def top_confusion_sets(y, pred, K, r):
    cm = confusion_matrix(y, pred, labels=np.arange(K)).astype(np.int64)
    sets = np.empty((K, r), dtype=np.int64)
    for k in range(K):
        row = cm[k].copy(); row[k] = -1
        # stable descending count, class index breaks ties deterministically
        order = np.lexsort((np.arange(K), -row))
        order = order[order != k]
        sets[k] = order[:r]
    return sets, cm


def targeted_confusion_rate(y, pred, confsets):
    hit = np.zeros(len(y), dtype=bool)
    for i, (yy, pp) in enumerate(zip(y, pred)):
        hit[i] = int(pp) in confsets[int(yy)]
    return float(hit.mean())


def margin_loss_grad(F, y, theta, bias, V, qminus, confsets, margin_height=MARGIN_HEIGHT):
    K = bias.shape[0]
    W = B.apply_P(theta, V, qminus)
    gW = np.zeros_like(W, dtype=np.float64); gb = np.zeros(K, np.float64)
    total = 0.0; n = len(y)
    for s in range(0, n, B.SAMPLE_BLOCK):
        e = min(s + B.SAMPLE_BLOCK, n)
        Xb = np.asarray(F[s:e], dtype=np.float32); yy = y[s:e]
        z = (Xb @ W + bias).astype(np.float64)
        dz = np.zeros_like(z)
        for a in range(len(yy)):
            k = int(yy[a]); js = confsets[k]
            vals = z[a, js] - z[a, k] + margin_height
            vmax = max(0.0, float(np.max(vals)))
            ev = np.exp(vals - vmax)
            denom = np.exp(-vmax) + ev.sum()
            total += vmax + np.log(denom)
            q = ev / denom
            dz[a, js] += q
            dz[a, k] -= q.sum()
        gW += Xb.T.astype(np.float64) @ dz
        gb += dz.sum(axis=0)
    total /= n; gW /= n; gb /= n
    gtheta = B.apply_P(gW.astype(np.float32), V, qminus).astype(np.float64)
    return float(total), gtheta, gb


def soft_margin_grad(F, y, theta, bias, V, qminus, cw_base, confsets, lambda_m):
    base = B.softf1_loss_grad(F, y, theta, bias, V, qminus, cw_base, B.LAM_F)
    loss0, gt0, gb0, extra = base
    lm, gtm, gbm = margin_loss_grad(F, y, theta, bias, V, qminus, confsets)
    extra = dict(extra); extra.update(margin=float(lm), lambda_m=float(lambda_m))
    return float(loss0 + lambda_m*lm), gt0 + lambda_m*gtm, gb0 + lambda_m*gbm, extra


def eval_with_target(F, y, theta, bias, V, qminus, support, confsets):
    p = B.predict_proba(F, theta, bias, V, qminus)
    m = B.metrics(y, p, support)
    pred = p.argmax(axis=1)
    m['target_confusion_rate'] = targeted_confusion_rate(y, pred, confsets)
    return m, p


def main():
    print('STAGE_C1_START', json.dumps({'seed':SEED,'m':B.M,'tau':TAU,'r':R_GRID,'lambda_m':LM_GRID,'margin_height':MARGIN_HEIGHT}), flush=True)
    X,y,tr,meta,cal,te = load_split_numeric(B.ROOT, B.CFG['dataset']['path'])
    Xt,yt = X[tr], y[tr]
    _,C = B.build_landmarks(Xt,yt)
    paths,_,_ = B.build_feature_cache(X,(tr,meta,cal,te),C)
    sizes=dict(train=len(tr),meta=len(meta),cal=len(cal),test=len(te)); F=B.open_features(paths,sizes)
    V,qminus,rsp=B.build_rsp(F['train'],yt)
    K=int(y.max()+1); pi,train_support=B.class_priors(yt,K); cw_base=B.normalized_class_weights(pi**(-B.ALPHA),yt)

    # Shared WCE checkpoint.
    theta0=np.zeros((B.M,K),np.float32); bias0=np.zeros(K,np.float32)
    wce_th,wce_bi,hw=Q.train_qn('C1_WCE',F['train'],yt,theta0,bias0,V,qminus,B.WCE_STEPS,
        lambda th,bi:B.wce_loss_grad(F['train'],yt,th,bi,V,qminus,cw_base))

    # Frozen Stage-B scalar baseline, used only to discover confusion topology.
    soft_th,soft_bi,hs=Q.train_qn('C1_BASE_SOFT',F['train'],yt,wce_th,wce_bi,V,qminus,B.SOFT_STEPS,
        lambda th,bi:B.softf1_loss_grad(F['train'],yt,th,bi,V,qminus,cw_base,B.LAM_F))
    pmeta_soft=B.predict_proba(F['meta'],soft_th,soft_bi,V,qminus)
    Praw,Rraw,_,_,_,_,_=B.pr_from_probs(y[meta],pmeta_soft,K)
    cw_ad=scalar_adaptive_weights(pi,yt,Praw,Rraw)
    base_th,base_bi,ha=Q.train_qn('C1_BASE_ADAPT_tau0694',F['train'],yt,soft_th,soft_bi,V,qminus,B.ADAPT_STEPS,
        lambda th,bi:B.wce_loss_grad(F['train'],yt,th,bi,V,qminus,cw_ad))
    pmeta_base=B.predict_proba(F['meta'],base_th,base_bi,V,qminus); pred_meta=pmeta_base.argmax(axis=1)
    base_cal_p=B.predict_proba(F['cal'],base_th,base_bi,V,qminus); base_cal=B.metrics(y[cal],base_cal_p,train_support)

    conf_by_r={}; cm_by_r={}
    for r in R_GRID:
        cs,cm=top_confusion_sets(y[meta],pred_meta,K,r); conf_by_r[r]=cs; cm_by_r[r]=cm
        np.save(B.OUT/f'C1_confsets_seed42_r{r}.npy',cs)
        pd.DataFrame(cm).to_csv(B.OUT/f'C1_meta_confusion_seed42_r{r}.csv',index=False)

    rows=[]; all_logs=hw+hs+ha
    # One replay control (lambda=0); use r=3 only since margin inactive.
    control_conf=conf_by_r[3]
    ctrl_soft_th,ctrl_soft_bi,hc1=Q.train_qn('C1_CONTROL_SOFT_lambda0',F['train'],yt,wce_th,wce_bi,V,qminus,B.SOFT_STEPS,
        lambda th,bi:soft_margin_grad(F['train'],yt,th,bi,V,qminus,cw_base,control_conf,0.0))
    pmeta_ctrl=B.predict_proba(F['meta'],ctrl_soft_th,ctrl_soft_bi,V,qminus)
    P0,R0,_,_,_,_,_=B.pr_from_probs(y[meta],pmeta_ctrl,K); cw0=scalar_adaptive_weights(pi,yt,P0,R0)
    ctrl_th,ctrl_bi,hc2=Q.train_qn('C1_CONTROL_ADAPT',F['train'],yt,ctrl_soft_th,ctrl_soft_bi,V,qminus,B.ADAPT_STEPS,
        lambda th,bi:B.wce_loss_grad(F['train'],yt,th,bi,V,qminus,cw0))
    ctrl_m,_=eval_with_target(F['cal'],y[cal],ctrl_th,ctrl_bi,V,qminus,train_support,control_conf)
    all_logs += hc1+hc2
    # replay check against separately built baseline
    replay_gap={k:float(ctrl_m[k]-base_cal[k]) for k in ['accuracy','macro_f1','balanced_accuracy','tail20_f1','nll','brier','ece']}
    print('C1_CONTROL_REPLAY_GAP',json.dumps(replay_gap),flush=True)
    rows.append(dict(role='control',r=3,lambda_m=0.0,margin_height=MARGIN_HEIGHT,**{'cal_'+k:v for k,v in ctrl_m.items()}))

    for r in R_GRID:
        confsets=conf_by_r[r]
        ctrl_target=targeted_confusion_rate(y[cal], B.predict_proba(F['cal'],ctrl_th,ctrl_bi,V,qminus).argmax(axis=1), confsets)
        for lm in LM_GRID:
            th,bi,h1=Q.train_qn(f'C1_SOFT_MARGIN_r{r}_lm{lm}',F['train'],yt,wce_th,wce_bi,V,qminus,B.SOFT_STEPS,
                lambda tt,bb,r=r,lm=lm:soft_margin_grad(F['train'],yt,tt,bb,V,qminus,cw_base,conf_by_r[r],lm))
            pmeta=B.predict_proba(F['meta'],th,bi,V,qminus)
            Pc,Rc,_,_,_,_,_=B.pr_from_probs(y[meta],pmeta,K); cwa=scalar_adaptive_weights(pi,yt,Pc,Rc)
            th,bi,h2=Q.train_qn(f'C1_ADAPT_r{r}_lm{lm}',F['train'],yt,th,bi,V,qminus,B.ADAPT_STEPS,
                lambda tt,bb,cwa=cwa:B.wce_loss_grad(F['train'],yt,tt,bb,V,qminus,cwa))
            met,_=eval_with_target(F['cal'],y[cal],th,bi,V,qminus,train_support,confsets)
            rec=dict(role='candidate',r=r,lambda_m=lm,margin_height=MARGIN_HEIGHT,**{'cal_'+k:v for k,v in met.items()})
            rec.update(delta_cal_accuracy=met['accuracy']-ctrl_m['accuracy'],
                       delta_cal_macro_f1=met['macro_f1']-ctrl_m['macro_f1'],
                       delta_cal_balanced_accuracy=met['balanced_accuracy']-ctrl_m['balanced_accuracy'],
                       delta_cal_tail20_f1=met['tail20_f1']-ctrl_m['tail20_f1'],
                       delta_cal_nll=met['nll']-ctrl_m['nll'],
                       delta_cal_brier=met['brier']-ctrl_m['brier'],
                       delta_cal_ece=met['ece']-ctrl_m['ece'],
                       control_target_confusion_rate=float(ctrl_target),
                       delta_target_confusion_rate=float(met['target_confusion_rate']-ctrl_target))
            rec['pass_C1']=bool(rec['delta_cal_accuracy']>0 and rec['delta_target_confusion_rate']<0 and rec['delta_cal_macro_f1']>=-5e-4 and rec['delta_cal_tail20_f1']>=-5e-4)
            rows.append(rec); all_logs += h1+h2
            print('C1_RESULT',json.dumps(rec),flush=True)

    df=pd.DataFrame(rows)
    df.to_csv(B.OUT/'stageC1_seed42_9rows_cal.csv',index=False)
    cand=df[df.role=='candidate'].copy().sort_values(['pass_C1','delta_cal_accuracy','delta_target_confusion_rate'],ascending=[False,False,True])
    cand.to_csv(B.OUT/'stageC1_seed42_ranked_cal.csv',index=False)
    summary={'status':'STAGE_C1_SEED42','seed':SEED,'m':B.M,'tau':TAU,'margin_height':MARGIN_HEIGHT,'rsp':rsp,
             'replay_gap':replay_gap,'n_candidates':8,'n_pass':int(cand.pass_C1.sum()),'top':cand.head(4).to_dict(orient='records')}
    json.dump(summary,open(B.OUT/'stageC1_seed42_summary.json','w'),indent=2)
    json.dump(all_logs,open(B.OUT/'stageC1_seed42_training_logs.json','w'),indent=2)
    print('STAGE_C1_SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__': main()
