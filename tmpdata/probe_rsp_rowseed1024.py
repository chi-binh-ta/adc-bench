import numpy as np,pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
G=.02423; MMAX=24576; M=20480; S=1024; BS=512
D,_=arff.loadarff('tmpdata/helena.arff'); df=pd.DataFrame(D)
y=df['class'].apply(lambda v:int(v.decode() if isinstance(v,(bytes,bytearray)) else v)).to_numpy(np.int64)
X=df[[f'V{i}' for i in range(1,28)]].to_numpy(np.float32); idx=np.arange(len(y))
tr,tmp=train_test_split(idx,test_size=.30,random_state=42,stratify=y); va,te=train_test_split(tmp,test_size=.50,random_state=42,stratify=y[tmp]); meta,cal=train_test_split(va,test_size=.50,random_state=42,stratify=y[va])
sc=StandardScaler().fit(X[tr]); X=sc.transform(X).astype(np.float32); Xt=X[tr]; yt=y[tr]
def rbf(A,C):
 d=np.maximum(np.einsum('ij,ij->i',A,A)[:,None]+np.einsum('ij,ij->i',C,C)[None,:]-2*(A@C.T),0); return np.exp(-G*d).astype(np.float32)
seed=456
ss=StratifiedShuffleSplit(n_splits=1,train_size=MMAX,random_state=seed); li,_=next(ss.split(np.zeros(len(yt)),yt)); C=Xt[li[:M]]
mean=np.empty(M,np.float32); std=np.empty(M,np.float32)
for j in range(0,M,BS):
 K=rbf(Xt,C[j:j+BS]); mean[j:j+BS]=K.mean(0); std[j:j+BS]=K.std(0)
std=np.maximum(std,1e-6)
for rs in [0,1,7,42,123,31415,456,457]:
 rr=StratifiedShuffleSplit(n_splits=1,train_size=S,random_state=rs); si,_=next(rr.split(np.zeros(len(yt)),yt))
 F=np.empty((S,M),np.float32)
 for j in range(0,M,BS): F[:,j:j+BS]=(rbf(Xt[si],C[j:j+BS])-mean[j:j+BS])/std[j:j+BS]
 _,sv,_=randomized_svd(F/np.sqrt(float(S)),n_components=256,n_iter=10,random_state=0,flip_sign=False)
 print('ROWSEED',rs,float(sv[0]**2),float(sv[-1]**2),float(np.sqrt((sv[-1]**2+1e-3)/(sv[0]**2+1e-3))),flush=True)
