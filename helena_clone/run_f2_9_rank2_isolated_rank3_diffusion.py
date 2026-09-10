import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import run_stageB1_clone as B
from numeric_split import load_split_numeric
import run_latent_factor_identification_v2 as F
import run_f2_margin_residual_intervention as F2
import run_f2_5_post_intervention_calibration as F25
import run_f2_7_wrong_class_concentration_control as F27

ETA_GRID=np.array([0.0,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40],dtype=np.float64)
LAMBDA_STAR=-0.05
BOOT_SEED=20260931
N_BOOT=1000
TOL=1e-12
PROTECT_TOL=2e-15
RANK_TOL=64*np.finfo(np.float64).eps


def r2i_r3d(p,a,eta):
    p=np.asarray(p,dtype=np.float64)
    a=np.asarray(a,dtype=np.float64)
    if float(eta)==0.0:
        return p.copy()
    n,K=p.shape
    order=np.argsort(-p,axis=1,kind='stable')
    out=p.copy()
    gamma=1.0-float(eta)*a
    if np.any(gamma<=0):
        raise RuntimeError('Non-positive rank3+ exponent')
    for i in range(n):
        idx=order[i,2:]
        mass=float(np.sum(p[i,idx],dtype=np.float64))
        if mass<=1e-300:
            continue
        x=gamma[i]*np.log(np.maximum(p[i,idx],1e-300))
        x-=x.max()
        w=np.exp(x); w/=w.sum()
        out[i,idx]=mass*w
        # rank 1 and rank 2 are protected bitwise by starting from p.copy().
    rowsum=out.sum(axis=1)
    if np.max(np.abs(rowsum-1.0))>TOL:
        raise RuntimeError(f'R2I-R3D row-sum error {np.max(np.abs(rowsum-1.0))}')
    return out


def invariants(p0,p1):
    o=np.argsort(-p0,axis=1,kind='stable')
    o1=np.argsort(-p1,axis=1,kind='stable')
    rows=np.arange(len(p0))
    c1=o[:,0]; c2=o[:,1]
    if not np.array_equal(np.argmax(p0,axis=1),np.argmax(p1,axis=1)):
        return False,{'reason':'argmax_changed'}
    if not np.array_equal(o1[:,1],c2):
        return False,{'reason':'rank2_identity_changed'}
    d1=float(np.max(np.abs(p1[rows,c1]-p0[rows,c1])))
    d2=float(np.max(np.abs(p1[rows,c2]-p0[rows,c2])))
    if d1>PROTECT_TOL or d2>PROTECT_TOL:
        return False,{'reason':'protected_probability_changed','max_top1_delta':d1,'max_top2_delta':d2}
    vals=np.take_along_axis(p1,o,axis=1)
    maxrev=float(np.max(vals[:,1:]-vals[:,:-1]))
    if maxrev>RANK_TOL:
        return False,{'reason':'strict_order_reversal','max_reversal':maxrev}
    return True,{'max_top1_delta':d1,'max_top2_delta':d2,'max_reversal':maxrev}


def geometry(y,p):
    base=F27.geometry(y,p)
    order=np.argsort(-p,axis=1,kind='stable')
    rows=np.arange(len(p))[:,None]
    vals=p[rows,order]
    r3=vals[:,2:]
    r3sq=np.sum(r3*r3,axis=1)
    mass=np.sum(r3,axis=1)
    ent=np.zeros(len(p),dtype=np.float64)
    good=mass>1e-300
    if np.any(good):
        rr=r3[good]/mass[good,None]
        ent[good]=-np.sum(rr*np.log(np.maximum(rr,1e-300)),axis=1)
    return {**base,'rank3plus_sq':r3sq,'rank3plus_entropy':ent}


def summarize(y,p,support):
    m=F2.metrics(y,p,support); g=geometry(y,p)
    return {**m,
            'mean_py':float(g['py'].mean()),
            'mean_true_sq':float(g['true_sq'].mean()),
            'mean_wrong_sq':float(g['wrong_sq'].mean()),
            'mean_rank3plus_sq':float(g['rank3plus_sq'].mean()),
            'mean_rank3plus_entropy':float(g['rank3plus_entropy'].mean())}


def eligible(mm,c0,c1,c3):
    rankkeys=['accuracy','macro_f1','balanced_accuracy','tail20_f1']
    return bool(
        mm['mean_wrong_sq'] < c3['mean_wrong_sq'] and
        mm['mean_rank3plus_sq'] < c3['mean_rank3plus_sq'] and
        mm['brier'] < c3['brier'] and
        mm['brier'] <= c1['brier'] and
        mm['nll'] < c1['nll'] and
        mm['ece'] <= c1['ece'] and
        mm['mean_true_sq'] <= c1['mean_true_sq'] and
        all(abs(mm[k]-c3[k])<=TOL for k in rankkeys)
    )


def paired_ci(x1,x0,rng,Bn=N_BOOT):
    d=np.asarray(x1,dtype=np.float64)-np.asarray(x0,dtype=np.float64)
    n=len(d); vals=np.empty(Bn,dtype=np.float64)
    for b in range(Bn):
        ix=rng.integers(0,n,n); vals[b]=d[ix].mean()
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def rank_strata_rows(y,p0,peta):
    tr=F27.true_ranks(p0,y)
    groups={'rank1':tr==1,'rank2':tr==2,'rank3_5':(tr>=3)&(tr<=5),'rank6_plus':tr>=6}
    g0=geometry(y,p0); nll0=-np.log(np.maximum(g0['py'],1e-15)); b0=g0['true_sq']+g0['wrong_sq']
    rows=[]
    for eta,p in peta.items():
        if eta==0.0: continue
        ge=geometry(y,p); nlle=-np.log(np.maximum(ge['py'],1e-15)); be=ge['true_sq']+ge['wrong_sq']
        for name,mask in groups.items():
            if not np.any(mask): continue
            rows.append({'eta':float(eta),'true_rank_group':name,'n':int(mask.sum()),'share':float(mask.mean()),
                         'mean_delta_nll':float(np.mean(nlle[mask]-nll0[mask])),
                         'mean_delta_brier':float(np.mean(be[mask]-b0[mask])),
                         'mean_delta_py':float(np.mean(ge['py'][mask]-g0['py'][mask])),
                         'mean_delta_true_sq':float(np.mean(ge['true_sq'][mask]-g0['true_sq'][mask])),
                         'mean_delta_wrong_sq':float(np.mean(ge['wrong_sq'][mask]-g0['wrong_sq'][mask])),
                         'mean_delta_rank3plus_sq':float(np.mean(ge['rank3plus_sq'][mask]-g0['rank3plus_sq'][mask]))})
    return pd.DataFrame(rows),{
        'rank1_share':float(np.mean(tr==1)),
        'rank2_share':float(np.mean(tr==2)),
        'rank3_5_share':float(np.mean((tr>=3)&(tr<=5))),
        'rank6_plus_share':float(np.mean(tr>=6)),
        'misclassified_true_is_rank2_fraction':float(np.mean(tr[tr>1]==2)) if np.any(tr>1) else 0.0}


def reconstruct_test(X,y,tr,cal,te,zcal,ztest,zsc,support,rawgeo):
    ycal=y[cal]
    pcal_pre=F2.global_probs(zcal)
    dfull,ds,dh,ddiag=F2.derive_direction(X,y,tr,cal,zcal,zsc,pcal_pre,zcal.shape[1],support,rawgeo)
    scale=F2.gate_scale(zcal)
    zcal_i,acal=F2.intervene(zcal,dfull,scale,LAMBDA_STAR)
    ztest_i,atest=F2.intervene(ztest,dfull,scale,LAMBDA_STAR)
    Tcal=F2.global_bases_fast(zcal_i); Ttest=F2.global_bases_fast(ztest_i)
    a2,_=F25.fit_theta(zcal_i,Tcal,ycal,F25.CANON_ALPHA)
    qcal,qtest,qmu,qsd=F25.standardize_gate(acal,atest)
    a3,opt3=F25.fit_theta(zcal_i,F25.dose_bases(Tcal,qcal),ycal,np.array([a2[0],a2[1],0.0,0.0]))
    pc3=F25.apply_bases(ztest_i,F25.dose_bases(Ttest,qtest),a3)
    return {'C0':F2.global_probs(ztest),'C1':F2.global_probs(ztest_i),'C3':pc3},atest,a3,ddiag


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); support=np.bincount(y[tr],minlength=K).astype(np.float64)
    rawgeo=F.raw_geometry_features(X,y,tr,K)
    zsc,zst=F2.load_seed_logits(args.artifact_dir); zcal=zsc.mean(axis=0); ztest=zst.mean(axis=0)
    ycal=y[cal]; ytest=y[te]

    probs,doses,fold_df,coef_df=F27.build_oof(X,y,tr,cal,zcal,zsc,support,rawgeo)
    fold_df.to_csv(out/'f2_9_fold_diagnostics.csv',index=False)
    coef_df.to_csv(out/'f2_9_c3_cv_coefficients.csv',index=False)
    base={k:summarize(ycal,p,support) for k,p in probs.items()}
    expected={'nll':2.7397491931915283,'brier':0.7826232437924868,'ece':0.028646533298620415,
              'macro_f1':0.19490055879898846}
    gaps={k:float(base['C3'][k]-v) for k,v in expected.items()}
    if max(abs(v) for v in gaps.values())>2e-6:
        raise RuntimeError(f'C3 replay mismatch {gaps}')

    rows=[]; peta={}; invrows=[]
    for eta in ETA_GRID:
        p=r2i_r3d(probs['C3'],doses,float(eta)); peta[float(eta)]=p
        ok,diag=invariants(probs['C3'],p)
        invrows.append({'eta':float(eta),'valid':ok,**diag})
        if not ok: raise RuntimeError(f'Invariant failure eta={eta}: {diag}')
        mm=summarize(ycal,p,support)
        rows.append({'eta':float(eta),**mm,
                     'delta_nll_vs_c3':mm['nll']-base['C3']['nll'],
                     'delta_brier_vs_c3':mm['brier']-base['C3']['brier'],
                     'delta_ece_vs_c3':mm['ece']-base['C3']['ece'],
                     'delta_wrong_sq_vs_c3':mm['mean_wrong_sq']-base['C3']['mean_wrong_sq'],
                     'delta_true_sq_vs_c3':mm['mean_true_sq']-base['C3']['mean_true_sq'],
                     'delta_rank3plus_sq_vs_c3':mm['mean_rank3plus_sq']-base['C3']['mean_rank3plus_sq'],
                     'delta_rank3plus_entropy_vs_c3':mm['mean_rank3plus_entropy']-base['C3']['mean_rank3plus_entropy'],
                     'eligible':False if eta==0 else eligible(mm,base['C0'],base['C1'],base['C3'])})
    pd.DataFrame(rows).to_csv(out/'f2_9_oof_eta_metrics.csv',index=False)
    pd.DataFrame(invrows).to_csv(out/'f2_9_invariants.csv',index=False)
    rank_df,rank_summary=rank_strata_rows(ycal,probs['C3'],peta)
    rank_df.to_csv(out/'f2_9_rank_strata_diagnostics.csv',index=False)

    candidates=[r for r in rows if r['eligible']]
    if candidates:
        win=sorted(candidates,key=lambda r:(r['brier'],r['nll'],r['ece'],r['eta']))[0]
        eta_star=float(win['eta'])
        if win['brier']<=base['C0']['brier'] and win['nll']<=base['C0']['nll'] and win['ece']<=base['C0']['ece']:
            status='FULL_PARETO_RECOVERY'
        else: status='PARTIAL_CONTROL'
    else:
        eta_star=0.0; win=next(r for r in rows if r['eta']==0.0); status='NO_CONTROL'
    decision={'status':status,'eta_star':eta_star,'n_eligible':len(candidates),'eta_grid':ETA_GRID.tolist(),
              'c0':base['C0'],'c1':base['C1'],'c3':base['C3'],'winner':win,'c3_replay_gap':gaps,
              'rank_summary':rank_summary,'test_policy':'eta/status frozen on OOF-CAL before TEST; TEST cannot change closure'}
    with open(out/'f2_9_cal_decision.json','w') as f: json.dump(decision,f,indent=2)
    print('F2_9_CAL_DECISION',json.dumps(decision),flush=True)

    ptest_base,atest,a3,ddiag=reconstruct_test(X,y,tr,cal,te,zcal,ztest,zsc,support,rawgeo)
    pf=r2i_r3d(ptest_base['C3'],atest,eta_star)
    ok,tdiag=invariants(ptest_base['C3'],pf)
    if not ok: raise RuntimeError(f'TEST invariant failure: {tdiag}')
    ptest={**ptest_base,'F2_9':pf}
    tmetrics={k:summarize(ytest,p,support) for k,p in ptest.items()}
    pd.DataFrame([{'model':k,**v} for k,v in tmetrics.items()]).to_csv(out/'f2_9_test_metrics.csv',index=False)

    rng=np.random.default_rng(BOOT_SEED); boot={}; gw=geometry(ytest,pf)
    nw=-np.log(np.maximum(gw['py'],1e-15)); bw=gw['true_sq']+gw['wrong_sq']
    for ref in ['C3','C0']:
        gr=geometry(ytest,ptest_base[ref]); nr=-np.log(np.maximum(gr['py'],1e-15)); br=gr['true_sq']+gr['wrong_sq']
        boot[ref]={'nll_delta_ci95':paired_ci(nw,nr,rng),'brier_delta_ci95':paired_ci(bw,br,rng),
                   'true_sq_delta_ci95':paired_ci(gw['true_sq'],gr['true_sq'],rng),
                   'wrong_sq_delta_ci95':paired_ci(gw['wrong_sq'],gr['wrong_sq'],rng),
                   'rank3plus_sq_delta_ci95':paired_ci(gw['rank3plus_sq'],gr['rank3plus_sq'],rng)}
    with open(out/'f2_9_test_bootstrap.json','w') as f: json.dump(boot,f,indent=2)
    final={'status':status,'eta_star':eta_star,'oof_decision':decision,'test':tmetrics,'bootstrap':boot,
           'fullcal_c3_coefficients':a3.tolist(),'full_direction_diag':ddiag,'test_invariants':tdiag,
           'interpretation':'Rank-2 protected; only rank-3+ simplex is diffused. TEST cannot change OOF closure.'}
    with open(out/'F2_9_RANK2_ISOLATED_RANK3_DIFFUSION.json','w') as f: json.dump(final,f,indent=2)
    print('F2_9_TEST_OPENED_AFTER_CAL_FREEZE',flush=True)
    print('F2_9_FINAL',json.dumps(final),flush=True)

if __name__=='__main__': main()
