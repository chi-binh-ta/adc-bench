import numpy as np, pandas as pd, json, os, time
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

ARFF='tmpdata/helena.arff'; GAMMA=0.02423; M=20480; S=3072

def load():
    data,_=arff.loadarff(ARFF)
    df=pd.DataFrame(data)
    y=df['class'].apply(lambda v:int(v.decode() if isinstance(v,(bytes,bytearray)) else v)).to_numpy(np.int64)
    X=df[[f'V{i}' for i in range(1,28)]].to_numpy(np.float32)
    idx=np.arange(len(y))
    tr,tmp=train_test_split(idx,test_size=0.30,random_state=42,stratify=y)
    va,te=train_test_split(tmp,test_size=0.50,random_state=42,stratify=y[tmp])
    meta,cal=train_test_split(va,test_size=0.50,random_state=42,stratify=y[va])
    sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]).astype(np.float32); Xall=sc.transform(X).astype(np.float32)
    return Xall,y,tr,meta,cal,te,Xtr

def lm_sss(ytr, seed, m):
    s=StratifiedShuffleSplit(n_splits=1, train_size=m, random_state=seed)
    ii,_=next(s.split(np.zeros(len(ytr)), ytr))
    return ii

def rbf(A,C):
    a2=np.einsum('ij,ij->i',A,A)[:,None]
    c2=np.einsum('ij,ij->i',C,C)[None,:]
    D=np.maximum(a2+c2-2.0*(A@C.T),0.0)
    return np.exp(-GAMMA*D, dtype=np.float32)

def probe(seed):
    Xall,y,tr,meta,cal,te,Xtr=load(); ytr=y[tr]
    li=lm_sss(ytr,seed,M); C=Xtr[li]
    # train column moments in blocks
    means=np.empty(M,np.float32); stds=np.empty(M,np.float32)
    bs=512
    for j in range(0,M,bs):
        K=rbf(Xtr,C[j:j+bs])
        means[j:j+bs]=K.mean(0)
        stds[j:j+bs]=K.std(0)
    stds=np.maximum(stds,1e-6)
    outs=[]
    for rs in [0,42,123,456,seed,seed+1,31415]:
        sss=StratifiedShuffleSplit(n_splits=1, train_size=S, random_state=rs)
        si,_=next(sss.split(np.zeros(len(ytr)),ytr))
        F=np.empty((S,M),np.float32)
        for j in range(0,M,bs):
            K=rbf(Xtr[si],C[j:j+bs]); F[:,j:j+bs]=(K-means[j:j+bs])/stds[j:j+bs]
        # singular values of F/sqrt(S): squared = covariance eigenvalues
        _,sv,_=randomized_svd(F/np.sqrt(float(S)), n_components=256, n_iter=4, random_state=0, flip_sign=False)
        outs.append({'row_rs':int(rs),'mu1':float(sv[0]**2),'mur':float(sv[-1]**2),'qmin':float(np.sqrt((sv[-1]**2+1e-3)/(sv[0]**2+1e-3)))})
    return {'seed':seed,'m':M,'split_sizes':[len(tr),len(meta),len(cal),len(te)],'landmark_first10':li[:10].tolist(),'variants':outs}

res={'seed42':probe(42),'seed456':probe(456)}
os.makedirs('tmpout',exist_ok=True)
json.dump(res,open('tmpout/probe_rsp_protocol.json','w'),indent=2)
print(json.dumps(res,indent=2))
