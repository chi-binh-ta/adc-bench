import numpy as np, pandas as pd, json, os
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

ARFF='tmpdata/helena.arff'; GAMMA=0.02423; MMAX=24576; M=16384; S=1024; BS=1024
TARGET=(6517.568647791651,0.0117002907677287,0.0013959311181679)

def load():
    data,_=arff.loadarff(ARFF); df=pd.DataFrame(data)
    y=df['class'].apply(lambda v:int(v.decode() if isinstance(v,(bytes,bytearray)) else v)).to_numpy(np.int64)
    X=df[[f'V{i}' for i in range(1,28)]].to_numpy(np.float32); idx=np.arange(len(y))
    tr,tmp=train_test_split(idx,test_size=.30,random_state=42,stratify=y)
    va,te=train_test_split(tmp,test_size=.50,random_state=42,stratify=y[tmp])
    meta,cal=train_test_split(va,test_size=.50,random_state=42,stratify=y[va])
    sc=StandardScaler().fit(X[tr]); X=sc.transform(X).astype(np.float32)
    return X,y,tr,meta,cal,te

def rbf(A,C):
    d=np.maximum(np.einsum('ij,ij->i',A,A)[:,None]+np.einsum('ij,ij->i',C,C)[None,:]-2.0*(A@C.T),0.0)
    return np.exp(-GAMMA*d).astype(np.float32)

def main():
    X,y,tr,meta,cal,te=load(); Xt=X[tr]; yt=y[tr]
    ss=StratifiedShuffleSplit(n_splits=1,train_size=MMAX,random_state=42)
    li,_=next(ss.split(np.zeros(len(yt)),yt)); li=li[:M]; C=Xt[li]
    mean=np.empty(M,np.float32); std=np.empty(M,np.float32)
    for j in range(0,M,BS):
        K=rbf(Xt,C[j:j+BS]); mean[j:j+BS]=K.mean(0); std[j:j+BS]=K.std(0)
    std=np.maximum(std,1e-6)
    rowsets=[]
    rr=StratifiedShuffleSplit(n_splits=1,train_size=S,random_state=42); si,_=next(rr.split(np.zeros(len(yt)),yt)); rowsets.append(('sss42',si))
    for rs in [0,42,123,31415]:
        rng=np.random.RandomState(rs); rowsets.append((f'rng{rs}',rng.choice(len(yt),S,replace=False)))
    rowsets.append(('pool_prefix',li[:S]))
    rowsets.append(('train_prefix',np.arange(S)))
    out=[]
    for name,si in rowsets:
        F=np.empty((S,M),np.float32)
        for j in range(0,M,BS):
            F[:,j:j+BS]=(rbf(Xt[si],C[j:j+BS])-mean[j:j+BS])/std[j:j+BS]
        for niter,rs in [(4,0),(4,42),(8,0),(8,42)]:
            _,sv,_=randomized_svd(F/np.sqrt(float(S)),n_components=256,n_iter=niter,random_state=rs,flip_sign=False)
            mu1=float(sv[0]**2); mur=float(sv[-1]**2); q=float(np.sqrt((mur+1e-3)/(mu1+1e-3)))
            score=abs(mu1/TARGET[0]-1)+abs(mur/TARGET[1]-1)+abs(q/TARGET[2]-1)
            rec=dict(rows=name,n_iter=niter,svd_rs=rs,mu1=mu1,mur=mur,qmin=q,target_score=float(score))
            out.append(rec); print('FP',json.dumps(rec),flush=True)
    out.sort(key=lambda z:z['target_score'])
    print('BEST',json.dumps(out[:10],indent=2),flush=True)
    os.makedirs('tmpout',exist_ok=True); json.dump(out,open('tmpout/stageB1_fingerprint_m16384.json','w'),indent=2)

if __name__=='__main__': main()
