import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260910
TOPN_CHART = 40
TOPN_LAB = 40
WINDOW_HOURS = 48.0

ORGAN_TARGETS = [
    "Respiratory failure; insufficiency; arrest (adult)",
    "Acute and unspecified renal failure",
    "Shock",
    "Septicemia (except in labor)",
    "Fluid and electrolyte disorders",
    "Other liver diseases",
]

AGGS = ["count", "mean", "std", "min", "max", "last"]


def read_csv(path, **kwargs):
    return pd.read_csv(path, compression="infer", low_memory=False, **kwargs)


def load_defs(path):
    with open(path, "r", encoding="utf-8") as f:
        defs = yaml.safe_load(f)
    bench = {k: v for k, v in defs.items() if bool(v.get("use_in_benchmark", False))}
    code_to_group = {}
    for g, spec in defs.items():
        for c in spec.get("codes", []):
            code_to_group.setdefault(str(c), g)
    return defs, bench, code_to_group


def build_labels(root, defs_path):
    icu = read_csv(root / "icu" / "icustays.csv.gz")
    dx = read_csv(root / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    _, bench, code_to_group = load_defs(defs_path)
    bench_names = sorted(bench.keys())

    dx9 = dx.loc[dx["icd_version"].astype(str) == "9", ["subject_id", "hadm_id", "icd_code"]].copy()
    dx9["icd_code"] = dx9["icd_code"].astype(str).str.strip()
    hadm_with_icd9 = set(dx9["hadm_id"].dropna().astype(int))
    icu = icu.loc[icu["hadm_id"].astype(int).isin(hadm_with_icd9)].copy()
    if len(icu) == 0:
        raise RuntimeError("No ICU stays with ICD-9 diagnoses in public demo")

    dx9["phenotype"] = dx9["icd_code"].map(code_to_group)
    dx9 = dx9.loc[dx9["phenotype"].isin(bench_names)]
    lab = pd.crosstab(dx9["hadm_id"], dx9["phenotype"]).clip(upper=1)
    lab = lab.reindex(columns=bench_names, fill_value=0)
    y = icu[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]].copy()
    y = y.merge(lab, how="left", left_on="hadm_id", right_index=True)
    y[bench_names] = y[bench_names].fillna(0).astype(int)
    y["intime"] = pd.to_datetime(y["intime"])
    y["outtime"] = pd.to_datetime(y["outtime"])
    return y.reset_index(drop=True), bench_names


def filter_first48(df, stays, time_col, has_stay_id):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[df["valuenum"].notna() & df[time_col].notna()].copy()
    if has_stay_id:
        base = stays[["stay_id", "intime"]]
        z = df.merge(base, on="stay_id", how="inner")
    else:
        # Assign hospital labs to any ICU stay of the same admission if the lab time falls in its first 48h.
        base = stays[["hadm_id", "stay_id", "intime"]]
        z = df.merge(base, on="hadm_id", how="inner")
    hours = (z[time_col] - z["intime"]).dt.total_seconds() / 3600.0
    z = z.loc[(hours >= 0.0) & (hours <= WINDOW_HOURS)].copy()
    z["hours"] = hours.loc[z.index]
    return z


def aggregate_modality(z, prefix, topn):
    if len(z) == 0:
        return pd.DataFrame(index=pd.Index([], name="stay_id")), []
    freq = z.groupby("itemid").size().sort_values(ascending=False)
    items = list(freq.head(topn).index)
    q = z.loc[z["itemid"].isin(items), ["stay_id", "itemid", "hours", "valuenum"]].copy()
    q = q.sort_values(["stay_id", "itemid", "hours"])
    grp = q.groupby(["stay_id", "itemid"])["valuenum"]
    parts = {
        "count": grp.count(),
        "mean": grp.mean(),
        "std": grp.std(),
        "min": grp.min(),
        "max": grp.max(),
        "last": grp.last(),
    }
    wide = []
    for agg, s in parts.items():
        p = s.unstack("itemid")
        p.columns = [f"{prefix}_{int(c)}_{agg}" for c in p.columns]
        wide.append(p)
    X = pd.concat(wide, axis=1).sort_index(axis=1)
    return X, items


def build_features(root, stays):
    chart_cols = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
    lab_cols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
    chart = read_csv(root / "icu" / "chartevents.csv.gz", usecols=chart_cols)
    labs = read_csv(root / "hosp" / "labevents.csv.gz", usecols=lab_cols)
    c48 = filter_first48(chart, stays, "charttime", True)
    l48 = filter_first48(labs, stays, "charttime", False)
    Xc, chart_items = aggregate_modality(c48, "chart", TOPN_CHART)
    Xl, lab_items = aggregate_modality(l48, "lab", TOPN_LAB)
    X = Xc.join(Xl, how="outer")
    X = X.reindex(stays["stay_id"].values)
    X.index.name = "stay_id"
    return X, {"chart_items": [int(x) for x in chart_items], "lab_items": [int(x) for x in lab_items],
               "n_chart_rows_48h": int(len(c48)), "n_lab_rows_48h": int(len(l48))}


def choose_valid_cv(y, groups):
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    npos, nneg = int(y.sum()), int((1-y).sum())
    for n_splits in range(min(5, len(np.unique(groups))), 1, -1):
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            folds = list(cv.split(np.zeros(len(y)), y, groups))
        except Exception:
            continue
        ok = True
        for tr, va in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                ok = False; break
        if ok:
            return folds
    return None


def make_model(class_weight):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=3000,
                                   class_weight=class_weight, random_state=SEED)),
    ])


def evaluate_label(X, y, groups, class_weight):
    y = np.asarray(y, dtype=int)
    npos, nneg = int(y.sum()), int((1-y).sum())
    base = {"n": int(len(y)), "positives": npos, "negatives": nneg,
            "prevalence": float(y.mean()) if len(y) else float("nan")}
    if npos < 5 or nneg < 5:
        return {**base, "status": "INSUFFICIENT_SUPPORT"}, None
    folds = choose_valid_cv(y, groups)
    if folds is None:
        return {**base, "status": "INSUFFICIENT_VALID_GROUP_CV"}, None
    pred = np.full(len(y), np.nan)
    pbase = np.full(len(y), np.nan)
    for tr, va in folds:
        model = make_model(class_weight)
        model.fit(X.iloc[tr], y[tr])
        pred[va] = model.predict_proba(X.iloc[va])[:, 1]
        pbase[va] = float(y[tr].mean())
    if np.isnan(pred).any():
        raise RuntimeError("OOF predictions incomplete")
    prev = float(y.mean())
    ap = float(average_precision_score(y, pred))
    out = {**base, "status": "EVALUABLE", "n_splits": int(len(folds)),
           "auroc": float(roc_auc_score(y, pred)), "auprc": ap,
           "auprc_lift": float(ap / max(prev, 1e-15)),
           "brier": float(brier_score_loss(y, pred)),
           "baseline_brier_oof": float(np.mean((pbase-y)**2))}
    return out, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--defs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("mimic_transfer/outputs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels, bench_names = build_labels(args.mimic_root, args.defs)
    X, feat_diag = build_features(args.mimic_root, labels)
    if len(X) != len(labels):
        raise RuntimeError("Feature/label row mismatch")
    groups = labels["subject_id"].to_numpy()

    support = []
    for name in bench_names:
        y = labels[name].to_numpy(dtype=int)
        support.append({"label": name, "positives": int(y.sum()), "negatives": int((1-y).sum()),
                        "prevalence": float(y.mean()), "organ_target": name in ORGAN_TARGETS})
    support_df = pd.DataFrame(support).sort_values(["organ_target", "positives"], ascending=[False, True])
    support_df.to_csv(args.out / "mimic_demo_label_support.csv", index=False)

    all_results = []
    primary_preds = {}
    for name in bench_names:
        y = labels[name].to_numpy(dtype=int)
        for model_name, cw in [("logit_l2", None), ("logit_l2_balanced", "balanced")]:
            res, pred = evaluate_label(X, y, groups, cw)
            res.update({"label": name, "model": model_name, "organ_target": name in ORGAN_TARGETS})
            all_results.append(res)
            if pred is not None and model_name == "logit_l2_balanced":
                primary_preds[name] = pred
    results = pd.DataFrame(all_results)
    results.to_csv(args.out / "mimic_demo_phenotyping_metrics.csv", index=False)

    organ = results[(results["organ_target"] == True) & (results["model"] == "logit_l2_balanced")].copy()
    organ.to_csv(args.out / "mimic_demo_organ_failure_metrics.csv", index=False)
    evaluable_organ = organ[organ["status"] == "EVALUABLE"]

    # Descriptive multi-organ proxy from the six prespecified labels.
    present_targets = [x for x in ORGAN_TARGETS if x in labels.columns]
    multi_y = (labels[present_targets].sum(axis=1) >= 2).astype(int).to_numpy() if present_targets else np.zeros(len(labels), int)
    multi_res, _ = evaluate_label(X, multi_y, groups, "balanced")
    multi_res.update({"label": "multi_organ_proxy_ge2", "model": "logit_l2_balanced", "organ_target": True,
                      "descriptive_proxy": True})
    pd.DataFrame([multi_res]).to_csv(args.out / "mimic_demo_multi_organ_proxy.csv", index=False)

    eval_all = results[(results["model"] == "logit_l2_balanced") & (results["status"] == "EVALUABLE")]
    macro = {}
    for col in ["auroc", "auprc", "auprc_lift", "brier"]:
        macro[col] = float(eval_all[col].mean()) if len(eval_all) else None
    macro["n_evaluable_labels"] = int(len(eval_all))
    macro["n_total_labels"] = int(len(bench_names))
    macro["n_evaluable_organ_targets"] = int(len(evaluable_organ))
    macro["n_prespecified_organ_targets"] = int(len(ORGAN_TARGETS))

    verdict = "DEMO_RARE_LABEL_EVALUABLE" if len(evaluable_organ) >= 3 else "DEMO_SMOKE_ONLY"
    summary = {
        "status": verdict,
        "scope": "public MIMIC-IV demo v2.2 only; no clinical/generalization claim",
        "task": "48h structured-EHR -> 25 HCUP CCS multi-label phenotypes",
        "label_mapping": "ICD-9 only in demo smoke run",
        "n_icu_stays": int(len(labels)),
        "n_subjects": int(labels["subject_id"].nunique()),
        "n_features": int(X.shape[1]),
        "feature_diagnostics": feat_diag,
        "macro_primary": macro,
        "organ_targets": ORGAN_TARGETS,
        "multi_organ_proxy": multi_res,
    }
    with open(args.out / "MIMIC_DEMO_PHENOTYPING_SMOKE.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("MIMIC_DEMO_SMOKE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
