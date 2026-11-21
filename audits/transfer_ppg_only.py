# audits/transfer_ppg_only.py
"""
Transfer experiments with the PPG-only feature set (414 features).

Same five strategies, same folds, same targets as v2_transfer_strategies.py,
but the source model is trained on the 414 PPG-only features (no
demographics). This isolates whether cross-cohort degradation is driven
by demographic availability or by physiological domain shift.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier
from scipy.linalg import sqrtm

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor

from config import XGB_BEST, XGB_FINETUNE

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


def coral_align(X_s, X_t, lam=0.1):
    d = X_s.shape[1]
    C_s = np.cov(X_s, rowvar=False) + np.eye(d) * lam
    C_t = np.cov(X_t, rowvar=False) + np.eye(d) * lam
    A = np.linalg.inv(sqrtm(C_s).real) @ sqrtm(C_t).real
    return (X_s - X_s.mean(0)) @ A + X_t.mean(0)


def evaluate(name, csv_path, df_aur, FEATURES, medians):
    tg = load_target_csv(csv_path, FEATURES, medians)
    X = tg[FEATURES].values.astype(np.float64)
    y = tg.htn3.values.astype(np.int64)
    g = tg.subject_id.astype(str).values

    print(f"\n{'='*72}\n{name}: {len(tg)} seg / {len(set(g))} subj\n{'='*72}")

    X_src = df_aur[FEATURES].values.astype(np.float64)
    y_src = df_aur.htn3.values.astype(np.int64)
    scaler_src = StandardScaler().fit(X_src)
    X_src_scaled = scaler_src.transform(X_src)

    src_model = XGBClassifier(**XGB_BEST)
    src_model.fit(X_src_scaled, y_src)

    STRATEGIES = ["zero_shot", "from_scratch", "warm_start",
                   "target_scaler", "coral"]
    fold_results = {s: [] for s in STRATEGIES}

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=g), 1):
        if len(np.unique(y[te])) < 3 or len(np.unique(y[tr])) < 3:
            continue
        X_tr = X[tr]; X_te = X[te]; y_tr = y[tr]; y_te = y[te]

        sc = StandardScaler().fit(X_tr)
        X_tr_t = sc.transform(X_tr); X_te_t = sc.transform(X_te)

        def record(strategy, prob):
            fold_results[strategy].append({
                "auc": roc_auc_score(y_te, prob, multi_class="ovr",
                                       average="macro"),
                "f1": f1_score(y_te, prob.argmax(1),
                                average="macro", zero_division=0),
                "bacc": balanced_accuracy_score(y_te, prob.argmax(1)),
            })

        # Zero-shot
        try:
            record("zero_shot",
                   src_model.predict_proba(scaler_src.transform(X_te)))
        except Exception as e:
            print(f"  fold {fold}: zero_shot failed: {e}")

        # From-scratch
        try:
            m = XGBClassifier(**XGB_BEST)
            m.fit(X_tr_t, y_tr)
            record("from_scratch", m.predict_proba(X_te_t))
        except Exception as e:
            print(f"  fold {fold}: from_scratch failed: {e}")

        # Warm-start
        try:
            X_tr_w = scaler_src.transform(X_tr)
            X_te_w = scaler_src.transform(X_te)
            m = XGBClassifier(**XGB_FINETUNE)
            try:
                m.fit(X_tr_w, y_tr, xgb_model=src_model.get_booster())
            except Exception:
                m.fit(X_tr_w, y_tr)
            record("warm_start", m.predict_proba(X_te_w))
        except Exception as e:
            print(f"  fold {fold}: warm_start failed: {e}")

        # Target-scaler
        try:
            record("target_scaler", src_model.predict_proba(X_te_t))
        except Exception as e:
            print(f"  fold {fold}: target_scaler failed: {e}")

        # CORAL
        try:
            X_src_al = coral_align(X_src_scaled, X_tr_t, lam=0.1)
            m = XGBClassifier(**XGB_BEST)
            m.fit(X_src_al, y_src)
            record("coral", m.predict_proba(X_te_t))
        except Exception as e:
            print(f"  fold {fold}: coral failed: {e}")

    rows = []
    for s in STRATEGIES:
        if not fold_results[s]:
            continue
        df_s = pd.DataFrame(fold_results[s])
        rows.append({
            "dataset": name, "strategy": s,
            "auc_mean": df_s.auc.mean(), "auc_std": df_s.auc.std(),
            "f1_mean": df_s.f1.mean(), "bacc_mean": df_s.bacc.mean(),
            "n_folds": len(df_s),
        })
        print(f"  {s:15s}  AUC={rows[-1]['auc_mean']:.4f}±{rows[-1]['auc_std']:.4f}  "
              f"F1={rows[-1]['f1_mean']:.4f}")
    return rows


def main():
    df_aur, CANDIDATE = load_aurora_raw()
    pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(df_aur, CANDIDATE)
    FEATURES_ALL = pre.feature_list_
    FEATURES_PPG = [c for c in FEATURES_ALL if c not in PPG_ONLY_EXCLUSIONS]
    medians = aurora_medians(df_aur, FEATURES_ALL)

    # Transform Aurora using the fitted preprocessor
    X_aur = pre.transform(df_aur)
    df_aur_feat = df_aur.copy()
    for i, c in enumerate(FEATURES_ALL):
        df_aur_feat[c] = X_aur[:, i]

    print(f"Aurora: {len(df_aur_feat)} seg, {len(FEATURES_PPG)} PPG-only features")

    all_rows = []
    for name, key in [("PPG-BP", "ppgbp"),
                       ("PulseDB-Vital", "pulsedb_vital"),
                       ("SWCS", "ds4")]:
        path = CSV.get(key)
        if not path or not os.path.exists(path):
            print(f"\n[skip] {name}")
            continue
        # Use the PPG-only feature list AND recompute medians
        med_ppg = aurora_medians(df_aur_feat, FEATURES_PPG)
        all_rows.extend(evaluate(name, path, df_aur_feat,
                                  FEATURES_PPG, med_ppg))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS, "v2_transfer_ppg_only.csv"), index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
