# audits/hrv_sensitivity.py
"""
HRV sensitivity analysis.

This script runs the principal three-class analysis under three conditions and
reports the AUC difference between them:

  A  Full features             (baseline; should match Table 2, 0.778)
  B  HRV features excluded     (does HRV matter at all?)
  C  HRV zeros treated as NaN  (alternative imputation; failed values -> median)

If A ≈ B ≈ C, the zero-imputation convention does not affect the
reported metrics. If B or C differ materially from A, the paper's
HRV-related claims need revision.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score)
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor
from feature_families import family_of

# =========================================================================
# XGBoost config matching the paper's Table 2
# =========================================================================
def make_xgb():
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=-1,
        random_state=SEED, verbosity=0)


# =========================================================================
# CV runner
# =========================================================================
def cv_xgb(df, CANDIDATE, y, g, label):
    """5-fold StratifiedGroupKFold with FoldAwarePreprocessor."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED)
    aucs, f1s, baccs, accs = [], [], [], []
    n_feats_per_fold = []

    for tr, te in sgkf.split(df[CANDIDATE].values, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = make_xgb()
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        pred = prob.argmax(1)

        aucs.append(roc_auc_score(y[te], prob, multi_class="ovr",
                                   average="macro"))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
        accs.append(accuracy_score(y[te], pred))
        n_feats_per_fold.append(X_tr.shape[1])

    return {
        "condition": label,
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs, ddof=1)),
        "f1_mean":  float(np.mean(f1s)),
        "bacc_mean": float(np.mean(baccs)),
        "acc_mean": float(np.mean(accs)),
        "n_features_median": int(np.median(n_feats_per_fold)),
        "n_folds": len(aucs),
    }


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 72)
    print("HRV sensitivity analysis — Aurora-BP three-class JNC7")
    print("=" * 72)

    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values
    print(f"\nData: {len(df)} segments, {df.subject_id.nunique()} subjects, "
          f"{len(CANDIDATE)} candidate features")

    # Identify HRV features by family
    hrv_features = [c for c in CANDIDATE if family_of(c) == "HRV"]
    non_hrv_features = [c for c in CANDIDATE if family_of(c) != "HRV"]
    print(f"HRV features: {len(hrv_features)} -> {hrv_features}")
    print(f"Non-HRV features: {len(non_hrv_features)}")

    # Sanity: confirm the HRV features actually exist and contain zeros
    for c in hrv_features:
        v = df[c].values
        n_zero = int((v == 0).sum())
        frac = n_zero / len(v)
        print(f"  {c:25s}  zeros: {n_zero:5d} ({100*frac:.1f}%)")

    results = []

    # ============================================================
    # A — Full features (baseline)
    # ============================================================
    print("\n[A] Full features (baseline; should match Table 2)")
    r_a = cv_xgb(df, CANDIDATE, y, g, "A: full features")
    results.append(r_a)
    print(f"    AUC = {r_a['auc_mean']:.4f} ± {r_a['auc_std']:.4f}  "
          f"F1 = {r_a['f1_mean']:.4f}  BAcc = {r_a['bacc_mean']:.4f}  "
          f"n_feat = {r_a['n_features_median']}")

    # ============================================================
    # B — HRV excluded entirely
    # ============================================================
    print("\n[B] HRV features excluded")
    r_b = cv_xgb(df, non_hrv_features, y, g, "B: HRV excluded")
    results.append(r_b)
    print(f"    AUC = {r_b['auc_mean']:.4f} ± {r_b['auc_std']:.4f}  "
          f"F1 = {r_b['f1_mean']:.4f}  BAcc = {r_b['bacc_mean']:.4f}  "
          f"n_feat = {r_b['n_features_median']}")

    # ============================================================
    # C — HRV zeros treated as NaN
    # ============================================================
    print("\n[C] HRV zeros treated as NaN (alternative imputation)")
    df_alt = df.copy()
    n_zero_total = 0
    n_hrv_total = 0
    for c in hrv_features:
        v = df_alt[c].values
        mask = (v == 0)
        n_zero_total += int(mask.sum())
        n_hrv_total += int(len(v))
        df_alt.loc[mask, c] = np.nan
    pct = 100 * n_zero_total / max(1, n_hrv_total)
    print(f"    Masked {n_zero_total} zero HRV values ({pct:.1f}% of HRV cells) -> NaN")

    r_c = cv_xgb(df_alt, CANDIDATE, y, g, "C: HRV zeros as NaN")
    results.append(r_c)
    print(f"    AUC = {r_c['auc_mean']:.4f} ± {r_c['auc_std']:.4f}  "
          f"F1 = {r_c['f1_mean']:.4f}  BAcc = {r_c['bacc_mean']:.4f}  "
          f"n_feat = {r_c['n_features_median']}")

    # ============================================================
    # Summary
    # ============================================================
    df_out = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS, "v2_hrv_sensitivity.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    print("\n" + "=" * 72)
    print("SUMMARY — HRV sensitivity")
    print("=" * 72)
    print(df_out.round(4).to_string(index=False))

    # ============================================================
    # Paired differences
    # ============================================================
    print("\n" + "=" * 72)
    print("PAIRED DIFFERENCES vs. full-feature baseline")
    print("=" * 72)
    for r in results[1:]:
        delta_auc = r["auc_mean"] - r_a["auc_mean"]
        delta_f1 = r["f1_mean"] - r_a["f1_mean"]
        delta_bacc = r["bacc_mean"] - r_a["bacc_mean"]
        print(f"\n{r['condition']}:")
        print(f"  Δ AUC    = {delta_auc:+.4f}")
        print(f"  Δ F1     = {delta_f1:+.4f}")
        print(f"  Δ BAcc   = {delta_bacc:+.4f}")

    # ============================================================
    # Verdict
    # ============================================================
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    delta_b = abs(r_b["auc_mean"] - r_a["auc_mean"])
    delta_c = abs(r_c["auc_mean"] - r_a["auc_mean"])
    max_delta = max(delta_b, delta_c)
    if max_delta < 0.003:
        print(f"  ✓ PASS — largest ΔAUC is {max_delta:.4f} (< 0.003)")
        print(f"    The zero-imputation convention has no material effect on AUC.")
        print(f"    The paper's claim is supported.")
    elif max_delta < 0.010:
        print(f"  ~ MARGINAL — largest ΔAUC is {max_delta:.4f} (0.003–0.010)")
        print(f"    Effect is small but detectable. Consider reporting the")
        print(f"    sensitivity range explicitly in the supplementary.")
    else:
        print(f"  ✗ FAIL — largest ΔAUC is {max_delta:.4f} (> 0.010)")
        print(f"    The zero-imputation convention materially affects AUC.")
        print(f"    The paper's HRV treatment needs revision.")


if __name__ == "__main__":
    main()