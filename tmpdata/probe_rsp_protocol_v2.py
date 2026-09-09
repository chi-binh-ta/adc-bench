import numpy as np, pandas as pd, json, os
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

ARFF='tmpdata/helena.arff'; GAMMA=0.02423; MMAX=24576; M=20480; S=3072; BS=512

def load():
 data,_=arff.loadarff(ARFF); df=pd.DataFrame(data)
 y=df['class'].apply(lambda v:int(v.decode() if isinstance(v,(bytes,bytearray)) else v)).to_numpy(np.int64)
 X=df[[f'V{i}' for i in range(1,28)]].to_numpy(np.float32); idx=np.arange(len(y))
 tr,tmp=train_test_split(idx,test_size=.30,random_state=42,stratify=y)
 va,te=train_test_split(tmp,test_size=.50,random_state=42,stratify=y[tmp])
 meta,cal=train_test_split(va,test_size=.50,random_state=42,stratify=y[va])
 sc=StandardScaler(); sc.fit(X[tr]); X=sc.transform(X).astype(np.float32)
 return X,y,tr,meta,cal,te

def rbf(A,C):
 D=np.maximum(np.einsum('ij,ij->i',A,A)[:,None]+np.einsum('ij,ij->i',C,C)[None,:]-2*(A@C.T),0)
 return np.exp(-GAMMA*D).astype(np.float32)

def lm_pool(ytr,seed):
 s=StratifiedShuffleSplit(n_splits=1,train_size=MMAX,random_state=seed)
 ii,_=next(s.split(np.zeros(len(ytr)),ytr)); return ii[:M]

def rowsets(ytr,seed):
 out=[]
 for rs in [0,42,123,456,31415]:
  ss=StratifiedShuffleSplit(n_splits=1,train_size=S,random_state=rs); ii,_=next(ss.split(np.zeros(len(ytr)),ytr)); out.append((f'sss{rs}',ii))
 for rs in [0,42,123,456,31415]:
  rng=np.random.RandomState(rs); ii=rng.choice(len(ytr),S,replace=False); out.append((f'rng{rs}',ii))
 return out

def probe(seed):
 X,y,tr,meta,cal,te=load(); Xt=X[tr]; yt=y[tr]; li=lm_pool(yt,seed); C=Xt[li]
 mean=np.empty(M,np.float32); std=np.empty(M,np.float32)
 for j in range(0,M,BS):
  K=rbf(Xt,C[j:j+BS]); mean[j:j+BS]=K.mean(0); std[j:j+BS]=K.std(0)
 std=np.maximum(std,1e-6)
 vals=[]
 for name,si in rowsets(yt,seed):
  F=np.empty((S,M),np.float32)
  for j in range(0,M,BS):
   F[:,j:j+BS]=(rbf(Xt[si],C[j:j+BS])-mean[j:j+BS])/std[j:j+BS]
  _,sv,_=randomized_svd(F/np.sqrt(float(S)),n_components=256,n_iter=8,random_state=123,flip_sign=False)
  vals.append(dict(rows=name,mu1=float(sv[0]**2),mur=float(sv[-1]**2),qmin=float(np.sqrt((sv[-1]**2+1e-3)/(sv[0]**2+1e-3)))))
 return dict(seed=seed,landmark_first10=li[:10].tolist(),variants=vals)

res={'seed42':probe(42),'seed456':probe(456)}
os.makedirs('tmpout',exist_ok=True); json.dump(res,open('tmpout/probe_rsp_protocol_v2.json','w'),indent=2); print(json.dumps(res,indent=2))
