import numpy as np, pandas as pd, json
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

ARFF='tmpdata/helena.arff'; G=.02423; M=16384; S=1024; BS=1024
TARGET=(6517.568647791651,0.0117002907677287,0.0013959311181679)
D,_=arff.loadarff(ARFF); df=pd.DataFrame(D)
y=df['class'].apply(lambda v:int(v.decode() if isinstance(v,(bytes,bytearray)) else v)).to_numpy(np.int64)
X=df[[f'V{i}' for i in range(1,28)]].to_numpy(np.float32); idx=np.arange(len(y))
tr,tmp=train_test_split(idx,test_size=.30,random_state=42,stratify=y)
va,te=train_test_split(tmp,test_size=.50,random_state=42,stratify=y[tmp]); meta,cal=train_test_split(va,test_size=.50,random_state=42,stratify=y[va])
sc=StandardScaler().fit(X[tr]); X=sc.transform(X).astype(np.float32); Xt=X[tr]; yt=y[tr]
def rbf(A,C):
 d=np.maximum(np.einsum('ij,ij->i',A,A)[:,None]+np.einsum('ij,ij->i',C,C)[None,:]-2*(A@C.T),0); return np.exp(-G*d).astype(np.float32)
rr=StratifiedShuffleSplit(n_splits=1,train_size=S,random_state=42); si,_=next(rr.split(np.zeros(len(yt)),yt))
for pool in [16384,20480,24576]:
 ss=StratifiedShuffleSplit(n_splits=1,train_size=pool,random_state=42); li,_=next(ss.split(np.zeros(len(yt)),yt)); C=Xt[li[:M]]
 mean=np.empty(M,np.float32); std=np.empty(M,np.float32)
 for j in range(0,M,BS):
  K=rbf(Xt,C[j:j+BS]); mean[j:j+BS]=K.mean(0); std[j:j+BS]=K.std(0)
 std=np.maximum(std,1e-6); F=np.empty((S,M),np.float32)
 for j in range(0,M,BS): F[:,j:j+BS]=(rbf(Xt[si],C[j:j+BS])-mean[j:j+BS])/std[j:j+BS]
 for niter,rs in [(4,0),(8,0),(8,42)]:
  _,sv,_=randomized_svd(F/np.sqrt(float(S)),n_components=256,n_iter=niter,random_state=rs,flip_sign=False)
  mu1=float(sv[0]**2); mur=float(sv[-1]**2); q=float(np.sqrt((mur+1e-3)/(mu1+1e-3)))
  score=abs(mu1/TARGET[0]-1)+abs(mur/TARGET[1]-1)+abs(q/TARGET[2]-1)
  print('POOLFP',json.dumps(dict(pool=pool,n_iter=niter,svd_rs=rs,mu1=mu1,mur=mur,qmin=q,target_score=score)),flush=True)
