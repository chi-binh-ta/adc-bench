import json, time
import numpy as np
from pathlib import Path
import run_stageB1_clone as B
from numeric_split import load_split_numeric

TARGET_WCE_META_NLL = 3.08469558
SCALES = [0.25, 0.5, 1.0, 1.5]
MEMORIES = [5, 10]


def dotpair(a1,a2,b1,b2):
    return float(np.sum(a1*b1, dtype=np.float64) + np.sum(a2*b2, dtype=np.float64))


def lbfgs_dir(gt, gb, hist):
    qt = gt.astype(np.float64).copy(); qb = gb.astype(np.float64).copy()
    alphas=[]
    for sT,sB,yT,yB,rho in reversed(hist):
        a = rho*dotpair(sT,sB,qt,qb)
        alphas.append(a)
        qt -= a*yT; qb -= a*yB
    if hist:
        sT,sB,yT,yB,rho = hist[-1]
        yy = dotpair(yT,yB,yT,yB)
        sy = dotpair(sT,sB,yT,yB)
        gamma = sy/max(yy,1e-30)
        gamma = float(np.clip(gamma,1e-6,1e6))
    else:
        gamma = 1.0
    rt = gamma*qt; rb = gamma*qb
    for (sT,sB,yT,yB,rho),a in zip(hist, reversed(alphas)):
        beta = rho*dotpair(yT,yB,rt,rb)
        rt += sT*(a-beta); rb += sB*(a-beta)
    return -rt, -rb, gamma


def meta_nll(Fmeta, ymeta, theta, bias, V, qminus):
    p=B.predict_proba(Fmeta,theta,bias,V,qminus)
    return float(-np.log(np.maximum(p[np.arange(len(ymeta)),ymeta],1e-12)).mean())


def run_cfg(Ftr,ytr,Fmeta,ymeta,V,qminus,cw,scale,mem):
    K=int(ytr.max()+1)
    th=np.zeros((B.M,K),np.float32); bi=np.zeros(K,np.float32)
    hist=[]; prev_th=prev_bi=prev_gt=prev_gb=None
    log=[]
    for step in range(1,B.WCE_STEPS+1):
        t0=time.time()
        loss,gt,gb=B.wce_loss_grad(Ftr,ytr,th,bi,V,qminus,cw)
        if prev_th is not None:
            sT=(th.astype(np.float64)-prev_th).astype(np.float32)
            sB=(bi.astype(np.float64)-prev_bi).astype(np.float32)
            yT=(gt-prev_gt).astype(np.float32); yB=(gb-prev_gb).astype(np.float32)
            sy=dotpair(sT,sB,yT,yB)
            if sy>1e-12 and np.isfinite(sy):
                hist.append((sT,sB,yT,yB,1.0/sy))
                if len(hist)>mem: hist.pop(0)
        dT,dB,gamma=lbfgs_dir(gt,gb,hist)
        prev_th=th.astype(np.float64).copy(); prev_bi=bi.astype(np.float64).copy()
        prev_gt=gt.copy(); prev_gb=gb.copy()
        th=(th.astype(np.float64)+scale*dT).astype(np.float32)
        bi=(bi.astype(np.float64)+scale*dB).astype(np.float32)
        rec=dict(scale=scale,memory=mem,step=step,train_loss=float(loss),gamma=float(gamma),hist=len(hist),seconds=time.time()-t0)
        log.append(rec); print('ANCHOR_STEP',json.dumps(rec),flush=True)
        if not np.isfinite(loss) or not np.all(np.isfinite(th)):
            return None,log
    mnll=meta_nll(Fmeta,ymeta,th,bi,V,qminus)
    rec=dict(scale=scale,memory=mem,meta_nll=mnll,target=TARGET_WCE_META_NLL,abs_error=abs(mnll-TARGET_WCE_META_NLL))
    print('ANCHOR_RESULT',json.dumps(rec),flush=True)
    return (th,bi,rec),log


def main():
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path']); Xt,yt=X[tr],y[tr]
    li,C=B.build_landmarks(Xt,yt)
    paths,mean,std=B.build_feature_cache(X,(tr,meta,cal,te),C)
    sizes=dict(train=len(tr),meta=len(meta),cal=len(cal),test=len(te))
    F=B.open_features(paths,sizes)
    V,qminus,fp=B.build_rsp(F['train'],yt)
    K=int(y.max()+1); pi,support=B.class_priors(yt,K); cw=B.normalized_class_weights(pi**(-B.ALPHA),yt)
    results=[]; logs=[]; best=None
    for mem in MEMORIES:
        for scale in SCALES:
            out,lg=run_cfg(F['train'],yt,F['meta'],y[meta],V,qminus,cw,scale,mem)
            logs.extend(lg)
            if out is not None:
                th,bi,rec=out; results.append(rec)
                if best is None or rec['abs_error']<best[2]['abs_error']:
                    best=(th.copy(),bi.copy(),rec)
    results=sorted(results,key=lambda r:r['abs_error'])
    Path(B.OUT).mkdir(parents=True,exist_ok=True)
    json.dump({'status':'NUMERIC_SPLIT_QN_ANCHOR','rsp':fp,'target_wce_meta_nll':TARGET_WCE_META_NLL,'results':results},open(B.OUT/'optimizer_anchor_results.json','w'),indent=2)
    json.dump(logs,open(B.OUT/'optimizer_anchor_steps.json','w'),indent=2)
    if best:
        th,bi,rec=best
        np.savez_compressed(B.OUT/'checkpoint_wce_anchor_best.npz',theta=th,bias=bi)
        print('ANCHOR_BEST',json.dumps(rec),flush=True)

if __name__=='__main__': main()
