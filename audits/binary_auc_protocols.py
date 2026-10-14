import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
for _sub in ("adapters", "extractors", "runners", "analysis",
             "figures", "audits", "diagnostics", "verification",
             "helpers"):
    _p = _os.path.join(_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ------------------------------------------------------------------


# binary_auc_protocols.py
"""
Binary HTN AUC under three evaluation protocols, on Aurora.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor

np.random.seed(SEED)


def label_binary(sbp, dbp):
    """ACC/AHA 2017 binary: HTN if SBP >= 130 or DBP >= 80."""
    return ((sbp >= 130) | (dbp >= 80)).astype(int)


def segment_level_cv(df, CANDIDATE):
    """Return (subjects, y_true, y_prob), all three aligned in the same order."""
    y = label_binary(df.sbp.values, df.dbp.values)
    g = df.subject_id.astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=SEED)
    subj_all, y_true, y_prob = [], [], []
    for tr, te in sgkf.split(df[CANDIDATE].values, y, groups=g):
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                          objective="binary:logistic", tree_method="hist",
                          random_state=SEED)
        m.fit(sc.transform(X_tr), y[tr])
        p = m.predict_proba(sc.transform(X_te))[:, 1]
        subj_all.extend(g[te].tolist())
        y_true.extend(y[te].tolist())
        y_prob.extend(p.tolist())
    auc = roc_auc_score(y_true, y_prob)
    return auc, np.array(subj_all), np.array(y_true), np.array(y_prob)


def subject_aggregated_auc(subj_seg, y_seg, p_seg, df):
    """Average segment probabilities per subject, then AUC on subject level."""
    # Subject-level label from median SBP/DBP (fixed for all segments of a subject)
    subj_meta = df.groupby("subject_id").agg(
        sbp=("sbp", "median"), dbp=("dbp", "median"))
    subj_label = pd.Series(
        label_binary(subj_meta.sbp.values, subj_meta.dbp.values),
        index=subj_meta.index.astype(str))

    # Aggregate: mean of segment probabilities per subject
    agg = pd.DataFrame({"subject": subj_seg, "y_prob": p_seg}) \
            .groupby("subject")["y_prob"].mean()

    y_true = subj_label.loc[agg.index].values
    return roc_auc_score(y_true, agg.values)


def subject_mean_feature_cv(df, CANDIDATE):
    """Aggregate features per subject, subject-level label from median SBP/DBP."""
    subj_meta = df.groupby("subject_id").agg(
        sbp=("sbp", "median"), dbp=("dbp", "median"))
    y_subj = label_binary(subj_meta.sbp.values, subj_meta.dbp.values)

    grouped = df.groupby("subject_id")[CANDIDATE].mean()
    X_agg_df = grouped.copy()
    g_agg = grouped.index.astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=SEED)
    y_true, y_prob = [], []
    for tr, te in sgkf.split(X_agg_df.values, y_subj, groups=g_agg):
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            X_agg_df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(X_agg_df.iloc[tr])
        X_te = pre.transform(X_agg_df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                          objective="binary:logistic", tree_method="hist",
                          random_state=SEED)
        m.fit(sc.transform(X_tr), y_subj[tr])
        p = m.predict_proba(sc.transform(X_te))[:, 1]
        y_true.extend(y_subj[te].tolist())
        y_prob.extend(p.tolist())
    return roc_auc_score(y_true, y_prob)


def main():
    print("=" * 72)
    print("Binary HTN AUC — three evaluation protocols on Aurora  (v3)")
    print("=" * 72)

    df, CANDIDATE = load_aurora_raw()
    print(f"[data] {len(df)} segments / {df.subject_id.nunique()} subjects")

    print("\n[1] segment-level ...")
    auc_a, subj_seg, y_seg, p_seg = segment_level_cv(df, CANDIDATE)
    print(f"    AUC = {auc_a:.4f}")

    print("\n[2] subject-aggregated (aligned aggregation) ...")
    auc_b = subject_aggregated_auc(subj_seg, y_seg, p_seg, df)
    print(f"    AUC = {auc_b:.4f}")

    print("\n[3] subject-mean-feature ...")
    auc_c = subject_mean_feature_cv(df, CANDIDATE)
    print(f"    AUC = {auc_c:.4f}")

    out = pd.DataFrame([
        {"protocol": "segment_level",        "auc": auc_a, "n_rows": len(df)},
        {"protocol": "subject_aggregated",   "auc": auc_b, "n_rows": df.subject_id.nunique()},
        {"protocol": "subject_mean_feature", "auc": auc_c, "n_rows": df.subject_id.nunique()},
    ])
    csv = os.path.join(RESULTS, "v2_binary_auc_protocols.csv")
    out.to_csv(csv, index=False)
    print(f"\n[saved] {csv}")
    print("\n" + out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()