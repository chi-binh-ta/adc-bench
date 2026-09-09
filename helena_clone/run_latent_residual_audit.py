import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import run_stageB1_clone as B
from numeric_split import load_split_numeric
from run_stageE_cluster_g2 import build_potential_bases, global_bases, corrected_probs, cluster_partition

SEEDS = [42,123,456,789,2026]
ALPHA_G2 = np.array([-0.9026209634632217, 0.765345197612019], dtype=np.float64)
RANKS = [1,2,3,5,10,20]
N_NULL = 300
RNG_SEED = 20260909


def load_logits(artifact_dir):
    cal=[]; test=[]
    for seed in SEEDS:
        d=np.load(artifact_dir/f'stageD_seed{seed}_predictions.npz')
        zc=d['cal_logits'].astype(np.float64); zt=d['test_logits'].astype(np.float64)
        zc-=zc.mean(axis=1,keepdims=True); zt-=zt.mean(axis=1,keepdims=True)
        cal.append(zc); test.append(zt)
    return np.mean(cal,axis=0), np.mean(test,axis=0)


def final_probs(z, clusters):
    T=global_bases(build_potential_bases(z,clusters))
    return corrected_probs(z,T,ALPHA_G2)


def class_matrices(y,p,K):
    pred=p.argmax(axis=1)
    support=np.bincount(y,minlength=K).astype(np.float64)
    soft=np.zeros((K,K),dtype=np.float64)
    hard=np.zeros((K,K),dtype=np.float64)
    for k in range(K):
        ix=(y==k)
        if ix.sum()==0: continue
        soft[k]=p[ix].mean(axis=0)
        hard[k]=np.bincount(pred[ix],minlength=K)/ix.sum()
    # leakage only: remove self-coordinate so low-rank evidence cannot be a diagonal artifact.
    np.fill_diagonal(soft,0.0)
    np.fill_diagonal(hard,0.0)
    return soft,hard,support


def row_weight(M,support,mode):
    X=M.copy()
    if mode=='support':
        w=np.sqrt(support/np.mean(support))
        X*=w[:,None]
    elif mode=='row_unit':
        n=np.linalg.norm(X,axis=1,keepdims=True)
        X/=np.maximum(n,1e-15)
    return X


def svd_stats(A,B,ranks):
    Ua,sa,Vta=np.linalg.svd(A,full_matrices=False)
    Ub,sb,Vtb=np.linalg.svd(B,full_matrices=False)
    ea=sa**2; eb=sb**2
    out={}
    for r in ranks:
        Va=Vta[:r].T; Vb=Vtb[:r].T
        cal_energy=float(ea[:r].sum()/max(ea.sum(),1e-30))
        test_intrinsic=float(eb[:r].sum()/max(eb.sum(),1e-30))
        proj=B@Va@Va.T
        transfer=float(np.sum(proj*proj)/max(np.sum(B*B),1e-30))
        overlap=float(np.linalg.norm(Va.T@Vb,'fro')**2/r)
        out[r]={'cal_energy':cal_energy,'test_intrinsic_energy':test_intrinsic,
                'cal_to_test_transfer_energy':transfer,'subspace_overlap':overlap}
    erank_cal=float(np.exp(-np.sum((ea/ea.sum())*np.log(np.maximum(ea/ea.sum(),1e-30)))))
    erank_test=float(np.exp(-np.sum((eb/eb.sum())*np.log(np.maximum(eb/eb.sum(),1e-30)))))
    return out,erank_cal,erank_test


def permute_rows_offdiag(M,rng):
    K=M.shape[0]; X=M.copy()
    for k in range(K):
        idx=np.r_[0:k,k+1:K]
        vals=X[k,idx].copy()
        rng.shuffle(vals)
        X[k,idx]=vals
        X[k,k]=0.0
    return X


def audit_matrix(name,A0,B0,supA,supB,mode,rng):
    A=row_weight(A0,supA,mode); Bm=row_weight(B0,supB,mode)
    obs,erA,erB=svd_stats(A,Bm,RANKS)
    null={r:{'cal_energy':[],'transfer':[],'overlap':[]} for r in RANKS}
    for _ in range(N_NULL):
        Ap=permute_rows_offdiag(A0,rng); Bp=permute_rows_offdiag(B0,rng)
        Ap=row_weight(Ap,supA,mode); Bp=row_weight(Bp,supB,mode)
        st,_,_=svd_stats(Ap,Bp,RANKS)
        for r in RANKS:
            null[r]['cal_energy'].append(st[r]['cal_energy'])
            null[r]['transfer'].append(st[r]['cal_to_test_transfer_energy'])
            null[r]['overlap'].append(st[r]['subspace_overlap'])
    rows=[]
    for r in RANKS:
        d=obs[r]; nd=null[r]
        row={'matrix':name,'weighting':mode,'rank':r,'effective_rank_cal':erA,'effective_rank_test':erB,**d}
        for key,obskey in [('cal_energy','cal_energy'),('transfer','cal_to_test_transfer_energy'),('overlap','subspace_overlap')]:
            arr=np.asarray(nd[key])
            row[f'null_{key}_q95']=float(np.quantile(arr,.95))
            row[f'{key}_perm_p']=float((1+np.sum(arr>=d[obskey]))/(N_NULL+1))
            row[f'{key}_z']=float((d[obskey]-arr.mean())/(arr.std(ddof=1)+1e-15))
        rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); args=ap.parse_args()
    out=Path(__file__).with_name('outputs'); out.mkdir(exist_ok=True)
    X,y,tr,meta,cal,te=load_split_numeric(B.ROOT,B.CFG['dataset']['path'])
    K=int(y.max()+1); train_support=np.bincount(y[tr],minlength=K).astype(np.float64)
    clusters,_,_,_=cluster_partition(train_support)
    zc,zt=load_logits(args.artifact_dir)
    pc=final_probs(zc,clusters); pt=final_probs(zt,clusters)
    softC,hardC,supC=class_matrices(y[cal],pc,K); softT,hardT,supT=class_matrices(y[te],pt,K)
    rng=np.random.default_rng(RNG_SEED)
    rows=[]
    for name,A,Bm in [('soft_leakage',softC,softT),('hard_confusion',hardC,hardT)]:
        for mode in ['none','support','row_unit']:
            rows.extend(audit_matrix(name,A,Bm,supC,supT,mode,rng))
    df=pd.DataFrame(rows); df.to_csv(out/'latent_residual_lowrank_audit.csv',index=False)

    # Predeclared evidence rule: there must exist r<=10 such that BOTH independent matrices,
    # under at least the neutral unweighted analysis, beat the 95% null on CAL energy,
    # CAL->TEST transfer, and CAL/TEST subspace overlap. We also report support/row-unit robustness.
    neutral=df[df.weighting.eq('none') & df['rank'].le(10)]
    pass_by={}
    for name in ['soft_leakage','hard_confusion']:
        d=neutral[neutral.matrix.eq(name)].copy()
        ok=d[(d.cal_energy_perm_p<=.05)&(d.transfer_perm_p<=.05)&(d.overlap_perm_p<=.05)]
        pass_by[name]=[int(x) for x in ok['rank'].tolist()]
    overall=bool(pass_by['soft_leakage'] and pass_by['hard_confusion'])

    best={}
    for name in ['soft_leakage','hard_confusion']:
        d=neutral[neutral.matrix.eq(name)].sort_values('rank')
        # smallest rank passing all three; else rank with strongest transfer z
        ok=d[(d.cal_energy_perm_p<=.05)&(d.transfer_perm_p<=.05)&(d.overlap_perm_p<=.05)]
        sel=ok.iloc[0] if len(ok) else d.iloc[d.transfer_z.values.argmax()]
        best[name]={k:(int(sel[k]) if k=='rank' else float(sel[k])) for k in [
            'rank','effective_rank_cal','effective_rank_test','cal_energy','test_intrinsic_energy',
            'cal_to_test_transfer_energy','subspace_overlap','null_cal_energy_q95','null_transfer_q95','null_overlap_q95',
            'cal_energy_perm_p','transfer_perm_p','overlap_perm_p','cal_energy_z','transfer_z','overlap_z']}
    summary={'status':'LATENT_RESIDUAL_LOW_RANK_AUDIT','system':'canonical coherent S5 -> global G2',
             'definition':{'soft_leakage':'E[p_j|Y=k], j!=k, diagonal zero',
                           'hard_confusion':'P(argmax p=j|Y=k), j!=k, diagonal zero'},
             'null':'independent within-row permutation of off-diagonal destination labels; preserves each class row distribution/norm but destroys shared cross-class destination geometry',
             'ranks':RANKS,'n_null':N_NULL,'evidence_rule':'both soft_leakage and hard_confusion must have some r<=10 with permutation p<=.05 for CAL spectral concentration, CAL->TEST transfer energy, and CAL/TEST right-subspace overlap',
             'pass_ranks_unweighted':pass_by,'overall_low_dimensional_evidence':overall,'best_unweighted':best,
             'caveat':'CAL and TEST are random splits from the same Helena distribution; this establishes internal replicated low-rank residual structure, not external-domain invariance.'}
    with open(out/'LATENT_RESIDUAL_AUDIT.json','w') as f: json.dump(summary,f,indent=2)
    print('LATENT_RESIDUAL_AUDIT',json.dumps(summary),flush=True)

if __name__=='__main__': main()
