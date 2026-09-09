import json
import numpy as np
from pathlib import Path
import run_stageB1_clone as B
from numeric_split import load_split_numeric
from optimizer_anchor_qn import run_cfg

TARGET=3.08469558

def main():
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path']); Xt,yt=X[tr],y[tr]
    li,C=B.build_landmarks(Xt,yt); paths,mean,std=B.build_feature_cache(X,(tr,meta,cal,te),C)
    sizes=dict(train=len(tr),meta=len(meta),cal=len(cal),test=len(te)); F=B.open_features(paths,sizes)
    V,qminus,fp=B.build_rsp(F['train'],yt)
    K=int(y.max()+1); pi,support=B.class_priors(yt,K); cw=B.normalized_class_weights(pi**(-B.ALPHA),yt)
    out,logs=run_cfg(F['train'],yt,F['meta'],y[meta],V,qminus,cw,1.0,10)
    result={'status':'QUICK_NUMERIC_QN','rsp':fp,'target':TARGET,'anchor':None if out is None else out[2]}
    Path(B.OUT).mkdir(parents=True,exist_ok=True)
    json.dump(result,open(B.OUT/'optimizer_anchor_quick.json','w'),indent=2)
    if out is not None:
        np.savez_compressed(B.OUT/'checkpoint_wce_quick_qn.npz',theta=out[0],bias=out[1])
    print('QUICK_ANCHOR_FINAL',json.dumps(result),flush=True)

if __name__=='__main__': main()
