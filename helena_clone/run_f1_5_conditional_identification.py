import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageE_cluster_g2 import build_potential_bases, global_bases, corrected_probs, cluster_partition
import run_latent_factor_identification_v2 as F

SEEDS=[42,123,456,789,2026]
ALPHA_G2=np.array([-0.9026209634632217,0.765345197612019],dtype=np.float64)
ALPHAS=np.logspace(-3,3,13)
R=3
N_PERM=500
RNG_SEED=20260910
CV=KFold(n_splits=5,shuffle=True,random_state=RNG_SEED)

BASE_GROUPS=F.GROUPS
GEOMETRY_COLS=BASE_GROUPS['class_similarity']+BASE_GROUPS['representation_geometry']

CONTRASTS={
 1:[
   ('margin_given_prevalence','margin',['prevalence']),
   ('prevalence_given_margin','prevalence',['margin']),
   ('geometry_given_prevalence_margin','geometry',['prevalence','margin']),
   ('calibration_given_prevalence_margin','calibration_residual',['prevalence','margin']),
   ('seed_instability_given_prevalence_margin','seed_instability',['prevalence','margin']),
 ],
 2:[
   ('geometry_given_prevalence_margin','geometry',['prevalence','margin']),
   ('prevalence_given_geometry','prevalence',['geometry']),
   ('margin_given_geometry','margin',['geometry']),
   ('calibration_given_prevalence_margin_geometry','calibration_residual',['prevalence','margin','geometry']),
 ],
 3:[
   ('prevalence_given_margin_geometry','prevalence',['margin','geometry']),
   ('margin_given_prevalence_geometry','margin',['prevalence','geometry']),
   ('geometry_given_prevalence_margin','geometry',['prevalence','margin']),
 ],
}


def uniq(xs):
 out=[]
 for x in xs:
  if x not in out: out.append(x)
 return out


def group_cols(name):
 if name=='geometry': return GEOMETRY_COLS
 return BASE_GROUPS[name]


def control_cols(names):
 out=[]
 for n in names: out.extend(group_cols(n))
 return uniq(out)


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


def matrices_from_frame(Fc,Ft,cols):
 Xc=Fc[cols].to_numpy(dtype=np.float64,copy=True)
 Xt=Ft[cols].to_numpy(dtype=np.float64,copy=True)
 for j in range(Xc.shape[1]):
  finite=np.isfinite(Xc[:,j])
  med=float(np.median(Xc[finite,j])) if finite.any() else 0.0
  Xc[~np.isfinite(Xc[:,j]),j]=med
  Xt[~np.isfinite(Xt[:,j]),j]=med
 return Xc,Xt


def model(alpha):
 return make_pipeline(StandardScaler(),Ridge(alpha=float(alpha)))


def choose_alpha(X,y):
 best=None
 for a in ALPHAS:
  sc=float(np.mean(cross_val_score(model(a),X,y,cv=CV,scoring='r2')))
  if best is None or sc>best[0]: best=(sc,float(a))
 return best


def crossfit_y(X,y,alpha):
 return cross_val_predict(model(alpha),X,y,cv=CV)


def test_predict(Xc,yc,Xt,alpha):
 return model(alpha).fit(Xc,yc).predict(Xt)


def safe_corr(a,b):
 if np.std(a)<1e-15 or np.std(b)<1e-15: return 0.0
 return float(np.corrcoef(a,b)[0,1])


def residualize_target_features(XGc,XGt,XCc,XCt):
 if XCc.shape[1]==0:
  return XGc.copy(),XGt.copy(),[]
 RGc=np.empty_like(XGc); RGt=np.empty_like(XGt); alphas=[]
 for j in range(XGc.shape[1]):
  _,a=choose_alpha(XCc,XGc[:,j]); alphas.append(a)
  RGc[:,j]=XGc[:,j]-crossfit_y(XCc,XGc[:,j],a)
  RGt[:,j]=XGt[:,j]-test_predict(XCc,XGc[:,j],XCt,a)
 return RGc,RGt,alphas


def conditional_test(matrix_name,factor,alignment,Vc,Vt,Fc,Ft,target_name,controls_names,test_name,rng):
 gcols=group_cols(target_name)
 ccols=control_cols(controls_names)
 XGc,XGt=matrices_from_frame(Fc,Ft,gcols)
 if ccols:
  XCc,XCt=matrices_from_frame(Fc,Ft,ccols)
  control_cv_r2,a_c=choose_alpha(XCc,Vc)
  pred_c_cal=crossfit_y(XCc,Vc,a_c)
  pred_c_test=test_predict(XCc,Vc,XCt,a_c)
  rVc=Vc-pred_c_cal; rVt=Vt-pred_c_test
 else:
  XCc=np.zeros((len(Vc),0)); XCt=np.zeros((len(Vt),0))
  control_cv_r2=0.0; a_c=np.nan
  pred_c_test=np.repeat(Vc.mean(),len(Vt)); rVc=Vc-Vc.mean(); rVt=Vt-Vc.mean()

 RGc,RGt,feature_resid_alphas=residualize_target_features(XGc,XGt,XCc,XCt)
 residual_cv_r2,a_g=choose_alpha(RGc,rVc)
 pred_rg_cal=crossfit_y(RGc,rVc,a_g)
 pred_rg_test=test_predict(RGc,rVc,RGt,a_g)
 residual_test_r2=float(r2_score(rVt,pred_rg_test))
 residual_test_corr=safe_corr(pred_rg_test,rVt)

 # Original-scale control versus full predictive decomposition.
 test_control_r2=float(r2_score(Vt,pred_c_test))
 full_cols=uniq(ccols+gcols)
 XFc,XFt=matrices_from_frame(Fc,Ft,full_cols)
 full_cv_r2,a_full=choose_alpha(XFc,Vc)
 pred_full_test=test_predict(XFc,Vc,XFt,a_full)
 test_full_r2=float(r2_score(Vt,pred_full_test))
 delta_test_r2=test_full_r2-test_control_r2

 # Conditional permutation: keep residualized target design and TEST fixed; destroy CAL target-factor relation.
 null=np.empty(N_PERM)
 for b in range(N_PERM):
  mp=model(a_g).fit(RGc,rng.permutation(rVc))
  null[b]=abs(safe_corr(mp.predict(RGt),rVt))
 perm_p=float((1+np.sum(null>=abs(residual_test_corr)))/(N_PERM+1))

 passes=bool(alignment>=.75 and residual_cv_r2>0 and abs(residual_test_corr)>=.25 and perm_p<=.05 and delta_test_r2>0)
 return {
  'matrix':matrix_name,'factor':factor,'factor_alignment':alignment,'test_name':test_name,
  'target_group':target_name,'controls':'+'.join(controls_names) if controls_names else 'none',
  'target_n_features':len(gcols),'control_n_features':len(ccols),
  'control_alpha':a_c,'control_cal_cv_r2':control_cv_r2,
  'residual_target_alpha':a_g,'residual_cal_cv_r2':residual_cv_r2,
  'residual_test_r2':residual_test_r2,'residual_test_corr':residual_test_corr,
  'conditional_perm_p':perm_p,
  'test_control_r2':test_control_r2,'full_alpha':a_full,'full_cal_cv_r2':full_cv_r2,
  'test_full_r2':test_full_r2,'delta_test_r2':delta_test_r2,
  'pass_unique_driver':passes,
  'feature_residualization_alphas':';'.join(str(x) for x in feature_resid_alphas),
 }


def make_all_other_tests(matrix_name,factor,alignment,Vc,Vt,Fc,Ft,rng):
 rows=[]
 names=list(BASE_GROUPS.keys())
 for target in names:
  ctr=[x for x in names if x!=target]
  rows.append(conditional_test(matrix_name,factor,alignment,Vc,Vt,Fc,Ft,target,ctr,
                               f'all_other__{target}',rng))
 return rows


def build_features(X,y,tr,idx,z,zseeds,p,K,train_support,rawgeo):
 return F.make_features(X,y,tr,idx,z,zseeds,p,K,train_support,rawgeo)


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
 out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
 X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
 K=int(y.max()+1); train_support=np.bincount(y[tr],minlength=K)
 clusters,_,_,_=cluster_partition(train_support)
 zsc,zst=load_seed_logits(args.artifact_dir)
 zc=zsc.mean(axis=0); zt=zst.mean(axis=0)
 pc=final_probs(zc,clusters); pt=final_probs(zt,clusters)
 softC,hardC=F.residual_mats(y[cal],pc,K); softT,hardT=F.residual_mats(y[te],pt,K)
 rawgeo=F.raw_geometry_features(X,y,tr,K)
 Fc=build_features(X,y,tr,cal,zc,zsc,pc,K,train_support,rawgeo)
 Ft=build_features(X,y,tr,te,zt,zst,pt,K,train_support,rawgeo)
 rng=np.random.default_rng(RNG_SEED)

 rows=[]; stability={}
 for matrix_name,Mc,Mt in [('soft_leakage',softC,softT),('hard_confusion',hardC,hardT)]:
  Uc,sc,Vc5=F.svd_factors(Mc); Ut,st,Vt05=F.svd_factors(Mt)
  Vt5,match,align,A=F.match_factors(Vc5,Vt05)
  stability[matrix_name]={'alignment':[float(x) for x in align[:R]],
                          'match_1based':[int(x+1) for x in match[:R]],
                          'singular_values_cal':[float(x) for x in sc[:R]]}
  for rr in range(R):
   factor=rr+1; vc=Vc5[:,rr]; vt=Vt5[:,rr]; al=float(align[rr])
   rows.extend(make_all_other_tests(matrix_name,factor,al,vc,vt,Fc,Ft,rng))
   for cname,target,controls in CONTRASTS[factor]:
    rows.append(conditional_test(matrix_name,factor,al,vc,vt,Fc,Ft,target,controls,cname,rng))

 df=pd.DataFrame(rows)
 df.to_csv(out/'f1_5_conditional_tests.csv',index=False)

 # Mechanistic contrast status aggregated across the two independent residual definitions.
 mech=df[~df.test_name.str.startswith('all_other__')].copy()
 statuses=[]
 for (factor,test_name),d in mech.groupby(['factor','test_name']):
  passm={r.matrix:bool(r.pass_unique_driver) for _,r in d.iterrows()}
  n=sum(passm.values())
  status='PRIMARY_UNIQUE' if n==2 else ('MATRIX_SPECIFIC_UNIQUE' if n==1 else 'NOT_UNIQUE')
  statuses.append({'factor':int(factor),'test_name':test_name,'status':status,
                   'soft_pass':passm.get('soft_leakage',False),'hard_pass':passm.get('hard_confusion',False)})
 status_df=pd.DataFrame(statuses)
 status_df.to_csv(out/'f1_5_mechanistic_status.csv',index=False)

 allother=df[df.test_name.str.startswith('all_other__')].copy()
 unique_summary=[]
 for (factor,target),d in allother.groupby(['factor','target_group']):
  passes=int(d.pass_unique_driver.sum())
  unique_summary.append({'factor':int(factor),'target_group':target,'n_matrices_pass':passes,
                         'status':'ROBUST_UNIQUE' if passes==2 else ('MATRIX_SPECIFIC' if passes==1 else 'NO_UNIQUE_EVIDENCE')})
 unique_df=pd.DataFrame(unique_summary)
 unique_df.to_csv(out/'f1_5_all_other_unique_summary.csv',index=False)

 summary={
  'status':'F1_5_CONDITIONAL_LATENT_FACTOR_IDENTIFICATION',
  'system':'canonical coherent S5 -> global G2',
  'factors':[1,2,3],
  'factor_stability':stability,
  'evidence_rule':'alignment>=.75 AND residual CAL CV R2>0 AND |TEST residual corr|>=.25 AND conditional permutation p<=.05 AND delta TEST R2>0',
  'mechanistic_status':statuses,
  'all_other_unique_summary':unique_summary,
  'n_conditional_tests':int(len(df)),
  'n_pass':int(df.pass_unique_driver.sum()),
  'caveat':'Conditional predictive identification is not causal identification; TEST is an internal Helena replication split.'
 }
 with open(out/'F1_5_CONDITIONAL_IDENTIFICATION.json','w') as f: json.dump(summary,f,indent=2)
 print('F1_5_CONDITIONAL_IDENTIFICATION',json.dumps(summary),flush=True)

if __name__=='__main__': main()
