import os, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(Path(__file__).with_name('config_stageB1.json'), 'r'))
OUT = ROOT / 'helena_clone' / 'outputs'
CACHE = ROOT / 'helena_clone' / 'cache'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SEED = int(CFG['stageB1']['seed'])
M = int(CFG['representation']['m'])
MMAX = int(CFG['representation']['mmax_nested_pool'])
GAMMA = float(CFG['representation']['gamma'])
FBLOCK = int(CFG['representation']['feature_block'])
R = int(CFG['rsp']['rank'])
LSPEC = float(CFG['rsp']['lambda_spec'])
SROWS = int(CFG['rsp']['sketch_rows'])
EPS = float(CFG['training']['eps'])
ALPHA = float(CFG['training']['alpha_wce'])
LAM_F = float(CFG['training']['lambda_soft_f1'])
WCE_STEPS = int(CFG['training']['wce_steps'])
SOFT_STEPS = int(CFG['training']['soft_f1_steps'])
ADAPT_STEPS = int(CFG['training']['adaptive_steps'])
LR0 = float(CFG['training']['bb_lr_init'])
LRMIN = float(CFG['training']['bb_lr_min'])
LRMAX = float(CFG['training']['bb_lr_max'])
CLIP_LO, CLIP_HI = map(float, CFG['training']['adaptive_factor_clip'])
SAMPLE_BLOCK = 2048


def load_split():
    data, _ = arff.loadarff(ROOT / CFG['dataset']['path'])
    df = pd.DataFrame(data)
    raw_y = df['class'].apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else str(v)).to_numpy()
    classes = np.unique(raw_y)
    class_to_i = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_i[v] for v in raw_y], dtype=np.int64)
    X = df[[f'V{i}' for i in range(1, 28)]].to_numpy(np.float32)
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42, stratify=y[tmp])
    meta, cal = train_test_split(va, test_size=0.50, random_state=42, stratify=y[va])
    sc = StandardScaler().fit(X[tr])
    X = sc.transform(X).astype(np.float32)
    print('SPLIT', len(tr), len(meta), len(cal), len(te), flush=True)
    assert (len(tr), len(meta), len(cal), len(te)) == (45637, 4889, 4890, 9780)
    return X, y, tr, meta, cal, te


def rbf(A, C):
    aa = np.einsum('ij,ij->i', A, A)[:, None]
    cc = np.einsum('ij,ij->i', C, C)[None, :]
    d = np.maximum(aa + cc - 2.0 * (A @ C.T), 0.0)
    return np.exp(-GAMMA * d).astype(np.float32)


def build_landmarks(Xt, yt):
    ss = StratifiedShuffleSplit(n_splits=1, train_size=MMAX, random_state=SEED)
    pool, _ = next(ss.split(np.zeros(len(yt)), yt))
    li = pool[:M]
    np.save(OUT / f'landmark_indices_seed{SEED}_m{M}.npy', li)
    return li, Xt[li]


def build_feature_cache(X, splits, C):
    names = ['train', 'meta', 'cal', 'test']
    tr = splits[0]
    Xt = X[tr]
    mean_path = CACHE / f'mean_seed{SEED}_m{M}.npy'
    std_path = CACHE / f'std_seed{SEED}_m{M}.npy'
    mean = np.empty(M, np.float32)
    std = np.empty(M, np.float32)
    print('FEATURE_STATS_START', flush=True)
    for j in range(0, M, FBLOCK):
        jj = min(j + FBLOCK, M)
        K = rbf(Xt, C[j:jj])
        mean[j:jj] = K.mean(axis=0)
        std[j:jj] = K.std(axis=0)
        print('FEATURE_STATS', j, jj, flush=True)
    std = np.maximum(std, 1e-6)
    np.save(mean_path, mean); np.save(std_path, std)

    paths = {}
    for name, ix in zip(names, splits):
        path = CACHE / f'phi_{name}_seed{SEED}_m{M}_corder.dat'
        F = np.memmap(path, dtype='float32', mode='w+', shape=(len(ix), M), order='C')
        for j in range(0, M, FBLOCK):
            jj = min(j + FBLOCK, M)
            F[:, j:jj] = (rbf(X[ix], C[j:jj]) - mean[j:jj]) / std[j:jj]
            F.flush()
            print('FEATURE_WRITE', name, j, jj, flush=True)
        del F
        paths[name] = str(path)
    return paths, mean, std


def open_features(paths, sizes):
    return {k: np.memmap(paths[k], dtype='float32', mode='r', shape=(sizes[k], M), order='C') for k in paths}


def build_rsp(Ftr, ytr):
    rr = StratifiedShuffleSplit(n_splits=1, train_size=SROWS, random_state=SEED)
    si, _ = next(rr.split(np.zeros(len(ytr)), ytr))
    sketch = np.asarray(Ftr[si], dtype=np.float32)
    U, sv, Vt = randomized_svd(sketch / np.sqrt(float(SROWS)), n_components=R,
                                n_iter=int(CFG['rsp']['randomized_svd_n_iter']),
                                random_state=int(CFG['rsp']['randomized_svd_random_state']),
                                flip_sign=False)
    mu = (sv.astype(np.float64) ** 2)
    V = Vt.T.astype(np.float32)
    q = np.sqrt((mu[-1] + LSPEC) / (mu + LSPEC)).astype(np.float32)
    qminus = q - 1.0
    fp = dict(mu1=float(mu[0]), mur=float(mu[-1]), qmin=float(q.min()), sketch_rows=SROWS,
              sketch_seed=SEED, svd_rs=int(CFG['rsp']['randomized_svd_random_state']))
    print('RSP_FINGERPRINT', json.dumps(fp), flush=True)
    json.dump(fp, open(OUT / f'rsp_fingerprint_seed{SEED}_m{M}.json', 'w'), indent=2)
    np.savez_compressed(OUT / f'rsp_seed{SEED}_m{M}.npz', V=V, q=q, mu=mu)
    return V, qminus, fp


def apply_P(A, V, qminus):
    return A + V @ (qminus[:, None] * (V.T @ A))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def class_priors(y, K):
    n = np.bincount(y, minlength=K).astype(np.float64)
    return n / n.sum(), n


def normalized_class_weights(cw, y):
    cw = np.asarray(cw, dtype=np.float64)
    cw = cw / np.mean(cw[y])
    return cw.astype(np.float32)


def logits_block(Xb, theta, bias, V, qminus):
    W = apply_P(theta, V, qminus)
    return np.asarray(Xb) @ W + bias


def wce_loss_grad(F, y, theta, bias, V, qminus, cw):
    K = bias.shape[0]
    W = apply_P(theta, V, qminus)
    gW = np.zeros_like(W, dtype=np.float64)
    gb = np.zeros(K, np.float64)
    loss = 0.0
    n = len(y)
    for s in range(0, n, SAMPLE_BLOCK):
        e = min(s + SAMPLE_BLOCK, n)
        Xb = np.asarray(F[s:e], dtype=np.float32)
        yy = y[s:e]
        z = Xb @ W + bias
        p = softmax(z.astype(np.float64))
        sw = cw[yy].astype(np.float64)
        loss += np.sum(-sw * np.log(np.maximum(p[np.arange(len(yy)), yy], 1e-12)))
        dz = p
        dz[np.arange(len(yy)), yy] -= 1.0
        dz *= sw[:, None]
        gW += Xb.T.astype(np.float64) @ dz
        gb += dz.sum(axis=0)
    loss /= n
    gW /= n; gb /= n
    gtheta = apply_P(gW.astype(np.float32), V, qminus).astype(np.float64)
    return float(loss), gtheta, gb


def soft_counts(F, y, theta, bias, V, qminus, K):
    W = apply_P(theta, V, qminus)
    tp = np.zeros(K, np.float64)
    sump = np.zeros(K, np.float64)
    nll = 0.0
    n = len(y)
    for s in range(0, n, SAMPLE_BLOCK):
        e = min(s + SAMPLE_BLOCK, n)
        Xb = np.asarray(F[s:e], dtype=np.float32); yy = y[s:e]
        p = softmax((Xb @ W + bias).astype(np.float64))
        sump += p.sum(axis=0)
        np.add.at(tp, yy, p[np.arange(len(yy)), yy])
        nll += np.sum(-np.log(np.maximum(p[np.arange(len(yy)), yy], 1e-12)))
    nk = np.bincount(y, minlength=K).astype(np.float64)
    den = sump + nk + 1e-12
    f1 = 2.0 * tp / den
    return tp, sump, nk, den, f1, nll / n


def softf1_loss_grad(F, y, theta, bias, V, qminus, cw, lam_f):
    K = bias.shape[0]
    tp, sump, nk, den, f1, _ = soft_counts(F, y, theta, bias, V, qminus, K)
    soft_loss = 1.0 - f1.mean()
    W = apply_P(theta, V, qminus)
    gW = np.zeros_like(W, dtype=np.float64); gb = np.zeros(K, np.float64)
    ce_loss = 0.0; n = len(y)
    neg_const = (2.0 * tp / (den ** 2)) / K
    pos_extra = (2.0 / den) / K
    for s in range(0, n, SAMPLE_BLOCK):
        e = min(s + SAMPLE_BLOCK, n)
        Xb = np.asarray(F[s:e], dtype=np.float32); yy = y[s:e]
        p = softmax((Xb @ W + bias).astype(np.float64))
        sw = cw[yy].astype(np.float64)
        ce_loss += np.sum(-sw * np.log(np.maximum(p[np.arange(len(yy)), yy], 1e-12)))
        dz_ce = p.copy(); dz_ce[np.arange(len(yy)), yy] -= 1.0; dz_ce *= sw[:, None]
        # d L_softF1 / dp: +2TP/(K*D^2) for all samples, minus 2/(K*D) for true class.
        gp = np.broadcast_to(neg_const[None, :], p.shape).copy()
        gp[np.arange(len(yy)), yy] -= pos_extra[yy]
        # Jacobian-vector product through softmax.
        dz_f = p * (gp - np.sum(gp * p, axis=1, keepdims=True))
        dz = dz_ce + lam_f * dz_f
        gW += Xb.T.astype(np.float64) @ dz
        gb += dz.sum(axis=0)
    ce_loss /= n; gW /= n; gb /= n
    gtheta = apply_P(gW.astype(np.float32), V, qminus).astype(np.float64)
    return float(ce_loss + lam_f * soft_loss), gtheta, gb, dict(ce=float(ce_loss), soft=float(soft_loss), mean_soft_f1=float(f1.mean()))


class BBState:
    def __init__(self):
        self.prev_theta = None; self.prev_bias = None
        self.prev_gtheta = None; self.prev_gbias = None
        self.lr = LR0
    def choose_lr(self, theta, bias, gtheta, gbias):
        if self.prev_theta is not None:
            s1 = (theta.astype(np.float64) - self.prev_theta).ravel()
            s2 = (bias.astype(np.float64) - self.prev_bias).ravel()
            y1 = (gtheta - self.prev_gtheta).ravel()
            y2 = (gbias - self.prev_gbias).ravel()
            sy = float(np.dot(s1, y1) + np.dot(s2, y2))
            ss = float(np.dot(s1, s1) + np.dot(s2, s2))
            if sy > 1e-18 and np.isfinite(sy) and np.isfinite(ss):
                self.lr = float(np.clip(ss / sy, LRMIN, LRMAX))
        return self.lr
    def update_memory(self, theta, bias, gtheta, gbias):
        self.prev_theta = theta.astype(np.float64).copy(); self.prev_bias = bias.astype(np.float64).copy()
        self.prev_gtheta = gtheta.copy(); self.prev_gbias = gbias.copy()


def train_steps(name, F, y, theta, bias, V, qminus, steps, grad_fn):
    bb = BBState()
    history = []
    for t in range(1, steps + 1):
        t0 = time.time()
        out = grad_fn(theta, bias)
        loss, gt, gb = out[:3]
        extra = out[3] if len(out) > 3 else {}
        lr = bb.choose_lr(theta, bias, gt, gb)
        bb.update_memory(theta, bias, gt, gb)
        theta = (theta.astype(np.float64) - lr * gt).astype(np.float32)
        bias = (bias.astype(np.float64) - lr * gb).astype(np.float32)
        rec = dict(stage=name, step=t, loss=float(loss), lr=float(lr), seconds=float(time.time()-t0), **extra)
        history.append(rec); print('TRAIN', json.dumps(rec), flush=True)
        if not np.isfinite(loss) or not np.all(np.isfinite(theta)):
            raise RuntimeError(f'non-finite state in {name} step {t}')
    return theta, bias, history


def predict_proba(F, theta, bias, V, qminus):
    W = apply_P(theta, V, qminus)
    out = np.empty((len(F), bias.shape[0]), np.float32)
    for s in range(0, len(F), SAMPLE_BLOCK):
        e = min(s + SAMPLE_BLOCK, len(F))
        out[s:e] = softmax((np.asarray(F[s:e]) @ W + bias).astype(np.float64)).astype(np.float32)
    return out


def pr_from_probs(y, probs, K):
    pred = probs.argmax(axis=1)
    P, Rr, F, support = precision_recall_fscore_support(y, pred, labels=np.arange(K), zero_division=0)
    cm_tp = np.bincount(y[(pred == y)], minlength=K).astype(np.float64)
    pred_n = np.bincount(pred, minlength=K).astype(np.float64)
    true_n = np.bincount(y, minlength=K).astype(np.float64)
    fp = pred_n - cm_tp; fn = true_n - cm_tp
    return P, Rr, F, support, cm_tp, fp, fn


def ece_score(y, p, bins=15):
    conf = p.max(axis=1); pred = p.argmax(axis=1); correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for b in range(bins):
        mask = (conf >= edges[b]) & ((conf < edges[b+1]) if b < bins-1 else (conf <= edges[b+1]))
        if mask.any(): ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def metrics(y, p, train_support):
    pred = p.argmax(axis=1); K = p.shape[1]
    P, Rr, F, support = precision_recall_fscore_support(y, pred, labels=np.arange(K), zero_division=0)
    tail = np.argsort(train_support)[:20]
    onehot = np.eye(K, dtype=np.float64)[y]
    return dict(
        nll=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
        brier=float(np.mean(np.sum((p.astype(np.float64)-onehot)**2, axis=1))),
        ece=ece_score(y,p),
        accuracy=float(accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
        macro_precision=float(P.mean()), macro_recall=float(Rr.mean()),
        tail20_f1=float(F[tail].mean())
    )


def stageB_weights(pi, Praw, Rraw, tp, fp, fn, tau0, rho, lam):
    Pbar = float(np.mean(Praw)); Rbar = float(np.mean(Rraw))
    Ptilde = (tp + lam * Pbar) / (tp + fp + lam)
    Rtilde = (tp + lam * Rbar) / (tp + fn + lam)
    tauk = np.clip(tau0 + rho * np.log((Ptilde + EPS) / (Rtilde + EPS)), 0.0, 1.0)
    etaR = 0.05 + 0.15 * tauk
    etaP = 0.20 - 0.15 * tauk
    factor = ((Rbar + EPS) / (Rraw + EPS)) ** etaR * ((Praw + EPS) / (Pbar + EPS)) ** etaP
    factor = np.clip(factor, CLIP_LO, CLIP_HI)
    cw = (pi ** (-ALPHA)) * factor
    return tauk, normalized_class_weights(cw, np.repeat(np.arange(len(pi)), np.maximum(1,(pi*100000).astype(int)))) if False else cw


def main():
    print('STATUS RECONSTRUCTED_CLONE', flush=True)
    X, y, tr, meta, cal, te = load_split()
    Xt, yt = X[tr], y[tr]
    li, C = build_landmarks(Xt, yt)
    paths, mean, std = build_feature_cache(X, (tr,meta,cal,te), C)
    sizes = dict(train=len(tr),meta=len(meta),cal=len(cal),test=len(te))
    F = open_features(paths, sizes)
    V, qminus, fp_rsp = build_rsp(F['train'], yt)
    K = int(y.max()+1)
    pi, train_support = class_priors(yt, K)
    cw_base = normalized_class_weights(pi ** (-ALPHA), yt)
    theta = np.zeros((M,K), np.float32); bias = np.zeros(K,np.float32)

    theta, bias, hw = train_steps('WCE', F['train'], yt, theta, bias, V, qminus, WCE_STEPS,
        lambda th,bi: wce_loss_grad(F['train'], yt, th, bi, V, qminus, cw_base))
    np.savez_compressed(OUT/f'checkpoint_wce_seed{SEED}_m{M}.npz', theta=theta,bias=bias)

    theta, bias, hs = train_steps('SoftMacroF1', F['train'], yt, theta, bias, V, qminus, SOFT_STEPS,
        lambda th,bi: softf1_loss_grad(F['train'], yt, th, bi, V, qminus, cw_base, LAM_F))
    np.savez_compressed(OUT/f'checkpoint_soft_seed{SEED}_m{M}.npz', theta=theta,bias=bias)
    json.dump(hw+hs, open(OUT/'training_history_base.json','w'), indent=2)

    pmeta = predict_proba(F['meta'], theta,bias,V,qminus)
    Praw,Rraw,Fraw,supp,tp,fp,fn = pr_from_probs(y[meta],pmeta,K)
    pd.DataFrame(dict(cls=np.arange(K),precision=Praw,recall=Rraw,f1=Fraw,support=supp,tp=tp,fp=fp,fn=fn,pi=pi)).to_csv(OUT/'meta_pr_soft_checkpoint.csv',index=False)

    ptest_soft = predict_proba(F['test'], theta,bias,V,qminus)
    soft_metrics = metrics(y[te],ptest_soft,train_support)
    print('SOFT_CHECKPOINT_METRICS', json.dumps(soft_metrics), flush=True)
    json.dump(soft_metrics, open(OUT/'soft_checkpoint_test_metrics.json','w'), indent=2)

    rows=[]; unique_cache={}
    for tau0 in CFG['stageB1']['tau0']:
        for rho in CFG['stageB1']['rho']:
            for lam in CFG['stageB1']['lambda_shrink']:
                # rho=0 is identical across lambda; train once, duplicate audit row.
                key=(float(tau0),float(rho), float(lam) if float(rho)!=0.0 else 0.0)
                if key not in unique_cache:
                    Pbar=float(np.mean(Praw)); Rbar=float(np.mean(Rraw)); lam=float(lam); rho=float(rho); tau0=float(tau0)
                    Ptilde=(tp+lam*Pbar)/(tp+fp+lam); Rtilde=(tp+lam*Rbar)/(tp+fn+lam)
                    tauk=np.clip(tau0+rho*np.log((Ptilde+EPS)/(Rtilde+EPS)),0,1)
                    etaR=0.05+0.15*tauk; etaP=0.20-0.15*tauk
                    factor=((Rbar+EPS)/(Rraw+EPS))**etaR*((Praw+EPS)/(Pbar+EPS))**etaP
                    factor=np.clip(factor,CLIP_LO,CLIP_HI)
                    cw=normalized_class_weights((pi**(-ALPHA))*factor,yt)
                    th=theta.copy(); bi=bias.copy()
                    th,bi,h=train_steps(f'B_tau{tau0}_rho{rho}_lam{lam}',F['train'],yt,th,bi,V,qminus,ADAPT_STEPS,
                        lambda tt,bb,cw=cw: wce_loss_grad(F['train'],yt,tt,bb,V,qminus,cw))
                    ptest=predict_proba(F['test'],th,bi,V,qminus); met=metrics(y[te],ptest,train_support)
                    unique_cache[key]=(met,tauk,etaR,etaP,h,th,bi)
                    np.savez_compressed(OUT/f'checkpoint_B_tau{tau0}_rho{rho}_lam{lam}_seed{SEED}.npz',theta=th,bias=bi,tauk=tauk,etaR=etaR,etaP=etaP)
                met,tauk,etaR,etaP,h,th,bi=unique_cache[key]
                rec=dict(seed=SEED,m=M,tau0=float(tau0),rho=float(rho),lambda_shrink=float(lam),
                         tauk_mean=float(np.mean(tauk)),tauk_min=float(np.min(tauk)),tauk_max=float(np.max(tauk)),
                         etaR_mean=float(np.mean(etaR)),etaP_mean=float(np.mean(etaP)),**met)
                rows.append(rec); print('B1_RESULT',json.dumps(rec),flush=True)

    df=pd.DataFrame(rows)
    # Paired deltas versus rho=0 matched tau0. Use lambda-specific duplicate controls for audit readability.
    metric_cols=['accuracy','macro_f1','balanced_accuracy','tail20_f1','nll','brier','ece','weighted_f1']
    for mc in metric_cols: df['delta_'+mc]=np.nan
    for i,r in df.iterrows():
        ctrl=df[(df.tau0==r.tau0)&(df.rho==0.0)&(df.lambda_shrink==r.lambda_shrink)].iloc[0]
        for mc in metric_cols: df.loc[i,'delta_'+mc]=r[mc]-ctrl[mc]
    # Pareto gain flag: at least one of Accuracy/Macro/Tail improves and none of those three decreases more than 5e-4.
    df['pareto_gain_vs_scalar']=(
        ((df.delta_accuracy>0)|(df.delta_macro_f1>0)|(df.delta_tail20_f1>0)) &
        (df.delta_accuracy>=-5e-4)&(df.delta_macro_f1>=-5e-4)&(df.delta_tail20_f1>=-5e-4)
    )
    df.to_csv(OUT/f'stageB1_seed{SEED}_m{M}_18rows.csv',index=False)
    best=df.sort_values(['pareto_gain_vs_scalar','delta_macro_f1','delta_tail20_f1','delta_accuracy'],ascending=[False,False,False,False])
    best.to_csv(OUT/f'stageB1_seed{SEED}_m{M}_ranked.csv',index=False)
    summary={'status':'RECONSTRUCTED_CLONE','rsp_fingerprint':fp_rsp,'soft_checkpoint':soft_metrics,
             'n_reporting_rows':int(len(df)),'n_unique_models':int(len(unique_cache)),
             'n_pareto_gain_rows':int(df.pareto_gain_vs_scalar.sum()),
             'top5':best.head(5).to_dict(orient='records')}
    json.dump(summary,open(OUT/f'stageB1_seed{SEED}_m{M}_summary.json','w'),indent=2)
    print('FINAL_SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
