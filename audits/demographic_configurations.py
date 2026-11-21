# audits/demographic_configurations.py
"""
Three-way demographics comparison across all four cohorts.

For three clearly defined configurations
under identical splits:
  - demographics only
  - strictly PPG-derived features only
  - combined

This is evaluated on each cohort independently (within-cohort 5-fold CV),
not transferred from Aurora, so the "identical splits" requirement is
satisfied per cohort.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor


XGB_CFG = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
    objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", n_jobs=-1,
    random_state=SEED, verbosity=0)


DEMO_FEATURES = ["age", "demo_age", "demo_bmi", "demo_gender",
                 "demo_height", "demo_weight"]
PPG_ONLY_EXCLUSIONS = {
    "age", "demo_age", "demo_bmi", "demo_gender", "demo_height",
    "demo_weight", "htn_stiffness_index_proxy",
    "lasi_ms", "period_cv__lasi_ms", "period_max__lasi_ms",
    "period_mean__lasi_ms", "period_median__lasi_ms",
    "period_min__lasi_ms", "period_std__lasi_ms",
    "period_cv__stiffness_index", "period_max__stiffness_index",
    "period_mean__stiffness_index", "period_median__stiffness_index",
    "period_min__stiffness_index", "period_std__stiffness_index",
    "step_Psi_age_adjusted", "step_vascular_compliance_index",
    "stiffness_index",
}

def cv_three_way(df, features, y, g, label):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED)
    aucs, f1s, baccs = [], [], []
    for tr, te in sgkf.split(df[features].values, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], features)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(**XGB_CFG)
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        pred = prob.argmax(1)
        aucs.append(roc_auc_score(y[te], prob, multi_class="ovr",
                                    average="macro"))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
    if not aucs:
        return None
    return {
        "cohort": label[0], "configuration": label[1],
        "n_features": len(features),
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs, ddof=1)),
        "f1_mean":  float(np.mean(f1s)),
        "bacc_mean": float(np.mean(baccs)),
        "n_folds": len(aucs),
    }


def main():
    df_aur, CANDIDATE = load_aurora_raw()
    medians = aurora_medians(df_aur, CANDIDATE)

    demo_feats = [c for c in CANDIDATE if c in DEMO_FEATURES]
    ppg_feats  = [c for c in CANDIDATE if c not in PPG_ONLY_EXCLUSIONS]
    full_feats = list(CANDIDATE)
    # Added by me:
    combined_420_feats = list(set(ppg_feats + demo_feats))
    print(f"Feature counts: demo={len(demo_feats)}, ppg-only={len(ppg_feats)}, "
          f"combined={len(full_feats)}")

    rows = []

    # Aurora
    y = df_aur.htn3.values
    g = df_aur.subject_id.astype(str).values
    for cfg, feats in [("demographics only", demo_feats),
                        ("PPG-only", ppg_feats),
                       ("combined 420", combined_420_feats),  # <-- Added by me
                       ("combined 437", full_feats)]: # <-- renamed by me
        print(f"\nAurora-BP — {cfg} ({len(feats)} features)")
        r = cv_three_way(df_aur, feats, y, g, ("Aurora-BP", cfg))
        if r:
            rows.append(r)
            print(f"  AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")

    # External cohorts (load via Aurora feature list + medians)
    for name, key in [("PPG-BP", "ppgbp"),
                       ("PulseDB-Vital", "pulsedb_vital"),
                       ("SWCS", "ds4")]:
        path = CSV.get(key)
        if not path or not os.path.exists(path):
            print(f"\n[skip] {name}")
            continue
        tg = load_target_csv(path, full_feats, medians)
        if tg is None or len(tg) == 0:
            print(f"\n[skip] {name}: load failed")
            continue
        y_t = tg.htn3.values.astype(int)
        g_t = tg.subject_id.astype(str).values
        for cfg, feats in [("demographics only", demo_feats),
                            ("PPG-only", ppg_feats),
                           ("combined 420", combined_420_feats),  # <-- Added by me
                           ("combined 437", full_feats)]:  # <-- renamed by me
            print(f"\n{name} — {cfg} ({len(feats)} features)")
            r = cv_three_way(tg, feats, y_t, g_t, (name, cfg))
            if r:
                rows.append(r)
                print(f"  AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "v2_demographic_configurations.csv")
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")

    # Summary
    pivot = df.pivot(index="cohort", columns="configuration",
                      values="auc_mean").round(4)
    print("\n" + "=" * 72)
    print("THREE-WAY DEMOGRAPHICS COMPARISON — AUC by cohort")
    print("=" * 72)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
