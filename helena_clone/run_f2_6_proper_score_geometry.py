import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageE_cluster_g2 import cluster_partition
import run_latent_factor_identification_v2 as F
import run_f2_margin_residual_intervention as F2
import run_f2_5_post_intervention_calibration as F25

LAMBDA_STAR = -0.05
OUTER_SEED = 20260910
BOOT_SEED = 20260926
N_BOOT = 1000


def py_of(p, y):
    return p[np.arange(len(y)), y]


def sample_terms(p3, pr, y):
    py3 = py_of(p3, y).astype(np.float64)
    pyr = py_of(pr, y).astype(np.float64)
    dnll = -np.log(np.maximum(py3, 1e-15)) + np.log(np.maximum(pyr, 1e-15))
    b3 = np.sum((p3 - np.eye(p3.shape[1], dtype=np.float64)[y])**2, axis=1)
    br = np.sum((pr - np.eye(pr.shape[1], dtype=np.float64)[y])**2, axis=1)
    dbrier = b3 - br
    dpy = py3 - pyr
    dnorm2 = np.sum(p3*p3, axis=1) - np.sum(pr*pr, axis=1)
    minus2dpy = -2.0 * dpy
    dtrue = (1.0-py3)**2 - (1.0-pyr)**2
    wrong3 = np.sum(p3*p3, axis=1) - py3*py3
    wrongr = np.sum(pr*pr, axis=1) - pyr*pyr
    dwrong = wrong3 - wrongr
    return pd.DataFrame({
        'py_ref':pyr, 'py_c3':py3, 'delta_py':dpy,
        'delta_nll':dnll, 'delta_brier':dbrier,
        'minus2_delta_py':minus2dpy, 'delta_norm2':dnorm2,
        'delta_true_sq':dtrue, 'delta_wrong_sq':dwrong,
    })


def aggregate_terms(df):
    keys=['delta_nll','delta_brier','delta_py','minus2_delta_py','delta_norm2','delta_true_sq','delta_wrong_sq']
    out={k:float(df[k].mean()) for k in keys}
    out['frac_py_increase']=float((df['delta_py']>0).mean())
    out['identity_err_brier_norm']=float(np.max(np.abs(df['delta_brier']-(df['minus2_delta_py']+df['delta_norm2']))))
    out['identity_err_brier_wrong']=float(np.max(np.abs(df['delta_brier']-(df['delta_true_sq']+df['delta_wrong_sq']))))
    return out


def conflict_state(df):
    nimp=df['delta_nll']<0; bimp=df['delta_brier']<0
    return np.select([nimp & bimp, nimp & ~bimp, ~nimp & bimp],
                     ['BOTH_IMPROVE','LOG_ONLY','BRIER_ONLY'], default='BOTH_WORSEN')


def cuts_from_cal(x):
    return np.quantile(np.asarray(x,dtype=np.float64), [0.2,0.4,0.6,0.8]).astype(np.float64)


def assign_bins(x,cuts):
    return np.searchsorted(np.asarray(cuts), np.asarray(x), side='right') + 1


def signed_share(x, mask):
    den=float(np.sum(x)); num=float(np.sum(np.asarray(x)[mask]))
    return float(num/den) if abs(den)>1e-15 else np.nan


def grouped_rows(df, axis, split):
    rows=[]
    for val, sub in df.groupby(axis, observed=False):
        mask=(df[axis].to_numpy()==val)
        rows.append({
            'split':split,'axis':axis,'stratum':str(val),'n':int(len(sub)),'fraction':float(len(sub)/len(df)),
            **{f'mean_{k}':float(sub[k].mean()) for k in ['delta_nll','delta_brier','delta_py','minus2_delta_py','delta_norm2','delta_true_sq','delta_wrong_sq']},
            'share_total_delta_nll':signed_share(df['delta_nll'].to_numpy(),mask),
            'share_total_delta_brier':signed_share(df['delta_brier'].to_numpy(),mask),
        })
    return rows


def bootstrap_means(df, seed):
    rng=np.random.default_rng(seed); n=len(df)
    keys=['delta_nll','delta_brier','delta_py','delta_true_sq','delta_wrong_sq','delta_norm2']
    vals={k:np.empty(N_BOOT,dtype=np.float64) for k in keys}
    a={k:df[k].to_numpy(dtype=np.float64) for k in keys}
    for b in range(N_BOOT):
        ix=rng.integers(0,n,n)
        for k in keys: vals[k][b]=a[k][ix].mean()
    return {k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in vals.items()}


def mechanism(agg):
    if agg['delta_py']>0 and agg['delta_wrong_sq']>0 and agg['delta_nll']<0 and agg['delta_brier']>0:
        return 'TRUE_PROBABILITY_GAIN_WITH_WRONG_CONCENTRATION_COST'
    if agg['delta_nll']<0 and agg['delta_brier']>0 and agg['delta_py']<=0:
        return 'LOG_TAIL_GAIN_WITH_TRUE_PROBABILITY_TRADEOFF'
    return 'MIXED_PROPER_SCORE_GEOMETRY'


def enrich_primary(df, y, pref, dose, groups, cuts):
    df=df.copy()
    df['confidence_ref']=np.max(pref,axis=1)
    df['dose']=dose
    df['py_bin']=assign_bins(df['py_ref'],cuts['py'])
    df['confidence_bin']=assign_bins(df['confidence_ref'],cuts['confidence'])
    df['dose_bin']=assign_bins(df['dose'],cuts['dose'])
    df['prevalence_group']=groups
    df['conflict_state']=conflict_state(df)
    df['true_class']=y
    return df


def tail_diag(df, cuts):
    # CAL-frozen py quantiles correspond exactly to first 1 or 2 quintiles.
    out={}
    for name,maxbin in [('bottom20',1),('bottom40',2)]:
        m=df['py_bin'].to_numpy()<=maxbin
        out[name]={
            'n':int(m.sum()),
            'mean_delta_nll':float(df.loc[m,'delta_nll'].mean()),
            'mean_delta_brier':float(df.loc[m,'delta_brier'].mean()),
            'share_total_delta_nll':signed_share(df['delta_nll'].to_numpy(),m),
            'share_total_delta_brier':signed_share(df['delta_brier'].to_numpy(),m),
        }
    return out


def correlations(df):
    pairs={
        'pyref_vs_delta_nll':('py_ref','delta_nll'),
        'pyref_vs_delta_brier':('py_ref','delta_brier'),
        'dose_vs_delta_nll':('dose','delta_nll'),
        'dose_vs_delta_brier':('dose','delta_brier'),
        'dose_vs_delta_py':('dose','delta_py'),
        'dose_vs_delta_wrong_sq':('dose','delta_wrong_sq'),
    }
    out={}
    for name,(a,b) in pairs.items():
        r,p=spearmanr(df[a],df[b])
        out[name]={'rho':float(r),'p':float(p)}
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)

    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(np.float64)
    clusters,head,mid,tail=cluster_partition(support)
    class_group=np.array(['Mid']*K,dtype=object); class_group[head]='Head'; class_group[tail]='Tail'
    rawgeo=F.raw_geometry_features(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir); zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]

    # Reconstruct frozen F2.5 OOF probabilities exactly.
    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED)
    p0=np.zeros((len(ycal),K),np.float64); p1=np.zeros_like(p0); p3=np.zeros_like(p0); dose=np.zeros(len(ycal),np.float64)
    for fold,(fi,vi) in enumerate(outer.split(np.zeros(len(ycal)),ycal),1):
        fit_idx=cal[fi]; zfit=zcal[fi]; zval=zcal[vi]; zsfit=zsc[:,fi,:]
        pfit_pre=F2.global_probs(zfit)
        d,ds,dh,ddiag=F2.derive_direction(X,y,tr,fit_idx,zfit,zsfit,pfit_pre,K,support,rawgeo)
        scale=F2.gate_scale(zfit)
        zfit_i,afit=F2.intervene(zfit,d,scale,LAMBDA_STAR); zval_i,aval=F2.intervene(zval,d,scale,LAMBDA_STAR)
        p0[vi]=F2.global_probs(zval); p1[vi]=F2.global_probs(zval_i); dose[vi]=aval
        Tfit=F2.global_bases_fast(zfit_i); Tval=F2.global_bases_fast(zval_i)
        a2,opt2=F25.fit_theta(zfit_i,Tfit,ycal[fi],F25.CANON_ALPHA)
        qfit,qval,qmu,qsd=F25.standardize_gate(afit,aval)
        a3,opt3=F25.fit_theta(zfit_i,F25.dose_bases(Tfit,qfit),ycal[fi],np.array([a2[0],a2[1],0.0,0.0]))
        p3[vi]=F25.apply_bases(zval_i,F25.dose_bases(Tval,qval),a3)

    # Strict replay of F2.5 OOF metrics.
    moof={
        'C0':F2.metrics(ycal,p0,support),
        'C1':F2.metrics(ycal,p1,support),
        'C3':F2.metrics(ycal,p3,support),
    }
    expected={
        'C1':{'nll':2.741716146469116,'brier':0.7825546554981124,'macro_f1':0.19490055879898846},
        'C3':{'nll':2.7397491931915283,'brier':0.7826232437924868,'ece':0.028646533298620415},
    }
    gaps={m:{k:float(moof[m][k]-v) for k,v in vals.items()} for m,vals in expected.items()}
    if max(abs(v) for vals in gaps.values() for v in vals.values())>2e-6:
        raise RuntimeError(f'F2.5 replay mismatch: {gaps}')
    print('F2_6_OOF_REPLAY',json.dumps({'metrics':moof,'gaps':gaps}),flush=True)

    # Full CAL C3 fit and TEST probabilities, exact F2.5 procedure.
    pcal_pre=F2.global_probs(zcal)
    dfull,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,K,support,rawgeo)
    scale=F2.gate_scale(zcal)
    zcal_i,acal=F2.intervene(zcal,dfull,scale,LAMBDA_STAR); ztest_i,atest=F2.intervene(ztest,dfull,scale,LAMBDA_STAR)
    p0t=F2.global_probs(ztest); p1t=F2.global_probs(ztest_i)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,opt2=F25.fit_theta(zcal_i,Tcal,ycal,F25.CANON_ALPHA)
    qcal,qtest,qmu,qsd=F25.standardize_gate(acal,atest)
    a3,opt3=F25.fit_theta(zcal_i,F25.dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0.0,0.0]))
    p3t=F25.apply_bases(ztest_i,F25.dose_bases(Ttest,qtest),a3)

    mtest={'C0':F2.metrics(ytest,p0t,support),'C1':F2.metrics(ytest,p1t,support),'C3':F2.metrics(ytest,p3t,support)}
    expected_test_c3={'nll':2.762006998062134,'brier':0.7782723678335616,'ece':0.022731231811983943}
    tg={k:float(mtest['C3'][k]-v) for k,v in expected_test_c3.items()}
    if max(abs(v) for v in tg.values())>2e-6:
        raise RuntimeError(f'F2.5 TEST replay mismatch: {tg}')

    # CAL-frozen continuous bins from the C1 reference.
    cal_primary=sample_terms(p3,p1,ycal)
    cuts={'py':cuts_from_cal(cal_primary['py_ref']),
          'confidence':cuts_from_cal(np.max(p1,axis=1)),
          'dose':cuts_from_cal(dose)}
    test_primary=sample_terms(p3t,p1t,ytest)
    cal_primary=enrich_primary(cal_primary,ycal,p1,dose,class_group[ycal],cuts)
    test_primary=enrich_primary(test_primary,ytest,p1t,atest,class_group[ytest],cuts)

    # Secondary C3 vs C0 exact geometry.
    cal_secondary=sample_terms(p3,p0,ycal); test_secondary=sample_terms(p3t,p0t,ytest)

    aggs={
        'CAL_C3_vs_C1':aggregate_terms(cal_primary),
        'TEST_C3_vs_C1':aggregate_terms(test_primary),
        'CAL_C3_vs_C0':aggregate_terms(cal_secondary),
        'TEST_C3_vs_C0':aggregate_terms(test_secondary),
    }
    labels={'CAL':mechanism(aggs['CAL_C3_vs_C1']),'TEST':mechanism(aggs['TEST_C3_vs_C1'])}
    replicated=labels['CAL']==labels['TEST']

    rows=[]
    for axis in ['py_bin','confidence_bin','dose_bin','prevalence_group','conflict_state']:
        rows += grouped_rows(cal_primary,axis,'CAL')
        rows += grouped_rows(test_primary,axis,'TEST')
    pd.DataFrame(rows).to_csv(out/'f2_6_stratified_geometry.csv',index=False)

    cal_primary.to_csv(out/'f2_6_cal_sample_geometry.csv',index=False)
    test_primary.to_csv(out/'f2_6_test_sample_geometry.csv',index=False)
    pd.DataFrame([{'split':k.split('_')[0],'comparison':'_'.join(k.split('_')[1:]),**v} for k,v in aggs.items()]).to_csv(out/'f2_6_global_geometry.csv',index=False)

    diagnostic={
        'protocol':'F2.6 diagnostic only',
        'oof_replay':{'metrics':moof,'gaps':gaps},
        'test_replay':{'metrics':mtest,'c3_gap':tg},
        'bin_cuts':{k:[float(x) for x in v] for k,v in cuts.items()},
        'global_geometry':aggs,
        'mechanism_label':labels,
        'mechanism_replicated':replicated,
        'conflict_fractions':{
            'CAL':cal_primary['conflict_state'].value_counts(normalize=True).to_dict(),
            'TEST':test_primary['conflict_state'].value_counts(normalize=True).to_dict(),
        },
        'tail_sensitivity':{'CAL':tail_diag(cal_primary,cuts),'TEST':tail_diag(test_primary,cuts)},
        'correlations':{'CAL':correlations(cal_primary),'TEST':correlations(test_primary)},
        'bootstrap':{'CAL':bootstrap_means(cal_primary,BOOT_SEED),'TEST':bootstrap_means(test_primary,BOOT_SEED+1)},
        'full_cal_c3_coefficients':[float(x) for x in a3],
        'full_cal_direction_diag':ddiag,
        'interpretation':'Exact proper-score decomposition; no model or parameter is selected in F2.6.'
    }
    with open(out/'F2_6_PROPER_SCORE_GEOMETRY.json','w') as f: json.dump(diagnostic,f,indent=2)
    with open(out/'f2_6_summary.json','w') as f: json.dump({
        'mechanism_label':labels,'mechanism_replicated':replicated,
        'CAL_C3_vs_C1':aggs['CAL_C3_vs_C1'],'TEST_C3_vs_C1':aggs['TEST_C3_vs_C1'],
        'tail_sensitivity':diagnostic['tail_sensitivity'],'correlations':diagnostic['correlations'],
        'bootstrap':diagnostic['bootstrap'],'conflict_fractions':diagnostic['conflict_fractions']},f,indent=2)

    print('F2_6_MECHANISM',json.dumps({'labels':labels,'replicated':replicated}),flush=True)
    print('F2_6_CAL_PRIMARY',json.dumps(aggs['CAL_C3_vs_C1']),flush=True)
    print('F2_6_TEST_PRIMARY',json.dumps(aggs['TEST_C3_vs_C1']),flush=True)
    print('F2_6_TAIL',json.dumps(diagnostic['tail_sensitivity']),flush=True)
    print('F2_6_CORR',json.dumps(diagnostic['correlations']),flush=True)
    print('F2_6_CONFLICT',json.dumps(diagnostic['conflict_fractions']),flush=True)

if __name__=='__main__':
    main()
