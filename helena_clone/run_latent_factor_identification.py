import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageE_cluster_g2 import build_potential_bases, global_bases, corrected_probs, cluster_partition

SEEDS=[42,123,456,789,2026]
ALPHA_G2=np.array([-0.9026209634632217,0.765345197612019],dtype=np.float64)
R=5
N_PERM_CORR=2000
N_PERM_GROUP=500
RNG_SEED=20260910

GROUPS={
 'prevalence':['log_train_support','support_quantile'],
 'class_similarity':['raw_nn_cosine','raw_nn_euclid','raw_mean5_euclid','raw_sep_ratio'],
 'margin':['margin_mean','margin_q10','margin_median','margin_negative_rate','margin_nearzero_rate'],
 'seed_instability':['seed_trueprob_std','seed_margin_std','seed_disagreement','seed_vote_entropy'],
 'calibration_residual':['pred_cal_gap','pred_abs_cal_gap','class_nll','class_brier'],
 'representation_geometry':['rep_proto_norm','rep_nn_cosine','rep_nn_euclid','rep_within_scatter','rep_sep_ratio'],
}


def softmax64(z):
 z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)


def load_seed_logits(artifact_dir):
 cal=[]; test=[]
 for seed in SEEDS:
  d=np.load(artifact_dir/f'stageD_seed{seed}_predictions.npz')
  zc=d['cal_logits'].astype(np.float64); zt=d['test_logits'].astype(np.float64)
  zc-=zc.mean(axis=1,keepdims=True); zt-=zt.mean(axis=1,keepdims=True)
  cal.append(zc); test.append(zt)
 return np.stack(cal),np.stack(test)


def final_probs(z,clusters):
 T=global_bases(build_potential_bases(z,clusters))
 return corrected_probs(z,T,ALPHA_G2)


def residual_mats(y,p,K):
 pred=p.argmax(axis=1)
 soft=np.zeros((K,K)); hard=np.zeros((K,K))
 for k in range(K):
  ix=y==k
  if not ix.any(): continue
  soft[k]=p[ix].mean(axis=0)
  hard[k]=np.bincount(pred[ix],minlength=K)/ix.sum()
 np.fill_diagonal(soft,0.0); np.fill_diagonal(hard,0.0)
 return soft,hard


def raw_geometry_features(X,y,tr,K):
 cent=np.zeros((K,X.shape[1])); scatter=np.zeros(K)
 for k in range(K):
  Xi=X[tr][y[tr]==k]
  cent[k]=Xi.mean(axis=0)
  scatter[k]=np.mean(np.sum((Xi-cent[k])**2,axis=1))
 n=np.linalg.norm(cent,axis=1,keepdims=True); C=cent/np.maximum(n,1e-15)
 cos=C@C.T; np.fill_diagonal(cos,-np.inf)
 nncos=cos.max(axis=1)
 d=np.sqrt(np.maximum(((cent[:,None,:]-cent[None,:,:])**2).sum(axis=2),0))
 np.fill_diagonal(d,np.inf)
 nnd=d.min(axis=1); mean5=np.sort(d,axis=1)[:,:5].mean(axis=1)
 sep=nnd/np.sqrt(np.maximum(scatter,1e-15))
 return {'raw_nn_cosine':nncos,'raw_nn_euclid':nnd,'raw_mean5_euclid':mean5,'raw_sep_ratio':sep}


def prototype_geometry(z,y,K):
 cent=np.zeros((K,z.shape[1])); scatter=np.zeros(K)
 for k in range(K):
  Z=z[y==k]
  cent[k]=Z.mean(axis=0)
  scatter[k]=np.mean(np.sum((Z-cent[k])**2,axis=1))
 norm=np.linalg.norm(cent,axis=1)
 C=cent/np.maximum(norm[:,None],1e-15)
 cos=C@C.T; np.fill_diagonal(cos,-np.inf)
 nncos=cos.max(axis=1)
 d=np.sqrt(np.maximum(((cent[:,None,:]-cent[None,:,:])**2).sum(axis=2),0))
 np.fill_diagonal(d,np.inf); nnd=d.min(axis=1)
 sep=nnd/np.sqrt(np.maximum(scatter,1e-15))
 return {'rep_proto_norm':norm,'rep_nn_cosine':nncos,'rep_nn_euclid':nnd,'rep_within_scatter':scatter,'rep_sep_ratio':sep}


def margin_features(z,y,K):
 out={k:np.full(K,np.nan) for k in ['margin_mean','margin_q10','margin_median','margin_negative_rate','margin_nearzero_rate']}
 for k in range(K):
  Z=z[y==k]
  true=Z[:,k]
  tmp=Z.copy(); tmp[:,k]=-np.inf
  m=true-tmp.max(axis=1)
  out['margin_mean'][k]=m.mean(); out['margin_q10'][k]=np.quantile(m,.10); out['margin_median'][k]=np.median(m)
  out['margin_negative_rate'][k]=np.mean(m<0); out['margin_nearzero_rate'][k]=np.mean(np.abs(m)<.5)
 return out


def seed_instability_features(zseeds,y,K):
 S,n,_=zseeds.shape
 probs=np.stack([softmax64(zseeds[s]) for s in range(S)])
 pred=np.argmax(zseeds,axis=2)
 out={k:np.full(K,np.nan) for k in ['seed_trueprob_std','seed_margin_std','seed_disagreement','seed_vote_entropy']}
 for k in range(K):
  ix=np.where(y==k)[0]
  if len(ix)==0: continue
  tp=probs[:,ix,k]
  out['seed_trueprob_std'][k]=np.std(tp,axis=0).mean()
  margins=[]
  for s in range(S):
   Z=zseeds[s,ix]; true=Z[:,k]; tmp=Z.copy(); tmp[:,k]=-np.inf; margins.append(true-tmp.max(axis=1))
  margins=np.stack(margins)
  out['seed_margin_std'][k]=np.std(margins,axis=0).mean()
  dis=[]; ent=[]
  for jj in range(len(ix)):
   votes=np.bincount(pred[:,ix[jj]],minlength=K).astype(float)/S
   dis.append(1.0-votes.max())
   nz=votes[votes>0]; ent.append(-np.sum(nz*np.log(nz))/np.log(S))
  out['seed_disagreement'][k]=np.mean(dis); out['seed_vote_entropy'][k]=np.mean(ent)
 return out


def calibration_features(p,y,K):
 pred=p.argmax(axis=1); conf=p.max(axis=1)
 one=np.eye(K)[y]; brier=np.sum((p-one)**2,axis=1); nll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-15))
 out={k:np.full(K,np.nan) for k in ['pred_cal_gap','pred_abs_cal_gap','class_nll','class_brier']}
 for k in range(K):
  ixp=pred==k
  if ixp.any():
   gap=conf[ixp].mean()-np.mean(y[ixp]==k)
   out['pred_cal_gap'][k]=gap; out['pred_abs_cal_gap'][k]=abs(gap)
  ixy=y==k
  if ixy.any():
   out['class_nll'][k]=nll[ixy].mean(); out['class_brier'][k]=brier[ixy].mean()
 return out


def make_features(X,y,tr,split_idx,z,zseeds,p,K,train_support,rawgeo):
 sup=train_support.astype(float)
 rank=rankdata(sup,method='average')
 F={'log_train_support':np.log(np.maximum(sup,1)),'support_quantile':(rank-.5)/K}
 F.update(rawgeo); F.update(prototype_geometry(z,y[split_idx],K)); F.update(margin_features(z,y[split_idx],K))
 F.update(seed_instability_features(zseeds,y[split_idx],K)); F.update(calibration_features(p,y[split_idx],K))
 return pd.DataFrame({'class':np.arange(K),**F})


def svd_factors(M):
 U,s,Vt=np.linalg.svd(M,full_matrices=False)
 return U[:,:R],s[:R],Vt[:R].T


def match_factors(Vc,Vt):
 A=np.abs(Vc.T@Vt)
 row,col=linear_sum_assignment(-A)
 order=np.empty(R,dtype=int); order[row]=col
 Vm=Vt[:,order].copy(); align=[]
 for r in range(R):
  d=float(Vc[:,r]@Vm[:,r])
  if d<0: Vm[:,r]*=-1; d=-d
  align.append(d)
 return Vm,order,np.asarray(align),A


def ranked_corr_perm(v,X,rng,nperm):
 vr=rankdata(v); vr=(vr-vr.mean())/(vr.std(ddof=1)+1e-15)
 vals=[]; names=[]
 for name,x in X.items():
  x=np.asarray(x,float); mask=np.isfinite(x)
  if mask.sum()<20: vals.append((name,np.nan,np.nan)); continue
  xr=rankdata(x[mask]); xr=(xr-xr.mean())/(xr.std(ddof=1)+1e-15)
  vv=vr[mask]
  rho=float(np.dot(vv,xr)/(len(vv)-1))
  null=np.empty(nperm)
  for b in range(nperm):
   null[b]=np.dot(rng.permutation(vv),xr)/(len(vv)-1)
  p=float((1+np.sum(np.abs(null)>=abs(rho)))/(nperm+1))
  vals.append((name,rho,p))
 return vals


def bh_q(pvals):
 p=np.asarray(pvals,float); q=np.full_like(p,np.nan)
 ok=np.isfinite(p); idx=np.where(ok)[0]
 if not len(idx): return q
 order=idx[np.argsort(p[idx])]; m=len(order); prev=1.0
 for j in range(m-1,-1,-1):
  i=order[j]; val=p[i]*m/(j+1); prev=min(prev,val); q[i]=min(prev,1.0)
 return q


def corr_table(matrix,Vc,Vt,Fc,Ft,rng):
 rows=[]
 feature_cols=[c for c in Fc.columns if c!='class']
 for r in range(R):
  for split,V,F in [('cal',Vc,Fc),('test',Vt,Ft)]:
   vals=ranked_corr_perm(V[:,r],{c:F[c].values for c in feature_cols},rng,N_PERM_CORR)
   qs=bh_q([x[2] for x in vals])
   for (name,rho,p),q in zip(vals,qs):
    group=next(g for g,cols in GROUPS.items() if name in cols)
    rows.append({'matrix':matrix,'factor':r+1,'split':split,'group':group,'feature':name,'spearman_rho':rho,'perm_p':p,'bh_q':q})
 return rows


def impute_matrix(F,cols):
 X=F[cols].to_numpy(float)
 for j in range(X.shape[1]):
  med=np.nanmedian(X[:,j]); X[~np.isfinite(X[:,j]),j]=med
 return X


def group_models(matrix,Vc,Vt,Fc,Ft,rng):
 rows=[]; alphas=np.logspace(-3,3,13); cv=KFold(n_splits=5,shuffle=True,random_state=RNG_SEED)
 for r in range(R):
  for group,cols in GROUPS.items():
   Xc=impute_matrix(Fc,cols); Xt=impute_matrix(Ft,cols); yv=Vc[:,r]; yt=Vt[:,r]
   best=None
   for a in alphas:
    model=make_pipeline(StandardScaler(),Ridge(alpha=float(a)))
    sc=float(np.mean(cross_val_score(model,Xc,yv,cv=cv,scoring='r2')))
    if best is None or sc>best[0]: best=(sc,float(a))
   cv_r2,alpha=best
   model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(Xc,yv)
   pred=model.predict(Xt)
   ss=np.sum((yt-yt.mean())**2); test_r2=float(1-np.sum((yt-pred)**2)/max(ss,1e-30))
   test_corr=float(np.corrcoef(pred,yt)[0,1]) if np.std(pred)>0 else 0.0
   null=[]
   for _ in range(N_PERM_GROUP):
    mp=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(Xc,rng.permutation(yv))
    pp=mp.predict(Xt); c=float(np.corrcoef(pp,yt)[0,1]) if np.std(pp)>0 else 0.0; null.append(abs(c))
   p=float((1+np.sum(np.asarray(null)>=abs(test_corr)))/(N_PERM_GROUP+1))
   rows.append({'matrix':matrix,'factor':r+1,'group':group,'n_features':len(cols),'alpha':alpha,'cal_cv_r2':cv_r2,'test_r2':test_r2,'test_corr':test_corr,'test_corr_perm_p':p})
 return rows


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
 out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
 X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
 K=int(y.max()+1); train_support=np.bincount(y[tr],minlength=K)
 clusters,_,_,_=cluster_partition(train_support)
 zsc,zst=load_seed_logits(args.artifact_dir)
 zc=zsc.mean(axis=0); zt=zst.mean(axis=0)
 pc=final_probs(zc,clusters); pt=final_probs(zt,clusters)
 softC,hardC=residual_mats(y[cal],pc,K); softT,hardT=residual_mats(y[te],pt,K)
 rawgeo=raw_geometry_features(X,y,tr,K)
 Fc=make_features(X,y,tr,cal,zc,zsc,pc,K,train_support,rawgeo)
 Ft=make_features(X,y,tr,te,zt,zst,pt,K,train_support,rawgeo)
 Fc.to_csv(out/'latent_factor_class_features_cal.csv',index=False); Ft.to_csv(out/'latent_factor_class_features_test.csv',index=False)
 rng=np.random.default_rng(RNG_SEED)
 corr_rows=[]; group_rows=[]; load_rows=[]; summaries={}
 for name,Mc,Mt in [('soft_leakage',softC,softT),('hard_confusion',hardC,hardT)]:
  Uc,sc,Vc=svd_factors(Mc); Ut,st,Vt0=svd_factors(Mt)
  Vt,match,align,A=match_factors(Vc,Vt0)
  corr_rows.extend(corr_table(name,Vc,Vt,Fc,Ft,rng)); group_rows.extend(group_models(name,Vc,Vt,Fc,Ft,rng))
  for r in range(R):
   for k in range(K): load_rows.append({'matrix':name,'factor':r+1,'class':k,'V_cal':Vc[k,r],'V_test_aligned':Vt[k,r]})
  summaries[name]={'singular_values_cal':[float(x) for x in sc], 'singular_values_test_top5_raw':[float(x) for x in st],
                   'test_factor_match_1based':[int(x+1) for x in match], 'factor_alignment':[float(x) for x in align],
                   'min_alignment':float(align.min())}
 corr=pd.DataFrame(corr_rows); groups=pd.DataFrame(group_rows); loads=pd.DataFrame(load_rows)
 # Replicated feature associations: FDR-significant in both splits, same sign, and nontrivial magnitude.
 rep=[]
 for (m,f,feat),d in corr.groupby(['matrix','factor','feature']):
  if set(d['split'])!={'cal','test'}: continue
  a=d[d.split.eq('cal')].iloc[0]; b=d[d.split.eq('test')].iloc[0]
  ok=(a.bh_q<=.10 and b.bh_q<=.10 and abs(a.spearman_rho)>=.25 and abs(b.spearman_rho)>=.20 and a.spearman_rho*b.spearman_rho>0)
  if ok: rep.append({'matrix':m,'factor':int(f),'group':a.group,'feature':feat,'rho_cal':a.spearman_rho,'q_cal':a.bh_q,'rho_test':b.spearman_rho,'q_test':b.bh_q})
 repdf=pd.DataFrame(rep)
 corr.to_csv(out/'latent_factor_correlations.csv',index=False); groups.to_csv(out/'latent_factor_group_models.csv',index=False); loads.to_csv(out/'latent_factor_loadings.csv',index=False); repdf.to_csv(out/'latent_factor_replicated_associations.csv',index=False)
 # Group-level winners require positive CAL CV R2 and independently significant CAL->TEST correlation.
 gpass=groups[(groups.cal_cv_r2>0)&(groups.test_corr_perm_p<=.05)&(groups.test_corr.abs()>=.25)].copy()
 winners=[]
 for (m,f),d in gpass.groupby(['matrix','factor']):
  q=d.sort_values(['test_corr_perm_p','test_corr'],ascending=[True,False]).iloc[0]
  winners.append({'matrix':m,'factor':int(f),'group':q.group,'cal_cv_r2':float(q.cal_cv_r2),'test_r2':float(q.test_r2),'test_corr':float(q.test_corr),'perm_p':float(q.test_corr_perm_p)})
 summary={'status':'F1_LATENT_FACTOR_IDENTIFICATION','system':'canonical coherent S5 -> global G2','rank_interpreted':R,
          'factor_stability':summaries,'group_winners':winners,'n_replicated_feature_associations':int(len(repdf)),
          'replicated_feature_associations':rep,'criteria':{
           'feature_replication':'BH q<=.10 on CAL and TEST, |rho_CAL|>=.25, |rho_TEST|>=.20, same sign',
           'group_replication':'CAL 5-fold CV R2>0, |TEST corr|>=.25, permutation p<=.05'},
          'caveat':'Associations are explanatory/diagnostic, not causal. Calibration residual variables are especially non-causal diagnostics. CAL and TEST share the Helena distribution.'}
 with open(out/'F1_LATENT_FACTOR_IDENTIFICATION.json','w') as f: json.dump(summary,f,indent=2)
 print('F1_LATENT_FACTOR_IDENTIFICATION',json.dumps(summary),flush=True)

if __name__=='__main__': main()
