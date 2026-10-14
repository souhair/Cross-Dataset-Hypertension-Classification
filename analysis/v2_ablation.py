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


# v2_ablation.py
"""
Feature-family ablation with fold-aware preprocessing.

Uses the mutually-exclusive family_of() map from feature_families.
Every family (including 'other') is evaluated both alone and leave-one-out.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH
from v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor
from feature_families import (
    family_of, all_families, features_by_family, count_by_family,
)


def cv_evaluate(df, y, g, cols, label):
    X_raw = df[cols].values
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=SEED)
    aucs, f1s, baccs = [], [], []
    for tr, te in sgkf.split(X_raw, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], cols)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(**XGB_3CLASS)
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        pred = prob.argmax(1)
        aucs.append(roc_auc_score(y[te], prob, multi_class="ovr",
                                  average="macro"))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        baccs.append(balanced_accuracy_score(y[te], pred))
    if not aucs:
        return None
    out = {"config": label, "n_features": len(cols),
           "auc_mean": float(np.mean(aucs)),
           "auc_std": float(np.std(aucs)),
           "f1_mean": float(np.mean(f1s)),
           "bacc_mean": float(np.mean(baccs))}
    print(f"  {label:45s}  n={len(cols):3d}  "
          f"AUC={out['auc_mean']:.4f}±{out['auc_std']:.4f}  "
          f"F1={out['f1_mean']:.4f}  BAcc={out['bacc_mean']:.4f}")
    return out


def main():
    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values

    # ---- Sanity: print family counts ------------------------------------
    counts = count_by_family(CANDIDATE)
    total = sum(counts.values())
    print("Feature counts by family (mutually exclusive):")
    for fam in all_families():
        print(f"  {fam:25s}  {counts[fam]:>4d}")
    print(f"  {'TOTAL':25s}  {total:>4d}")
    if total != len(CANDIDATE):
        raise SystemExit(
            f"Family partition does not cover all features: "
            f"{total} vs {len(CANDIDATE)}"
        )
    print()

    results = []

    print("--- Baseline ---")
    results.append(cv_evaluate(df, y, g, CANDIDATE, "ALL"))

    # ---- Family-alone: iterate ALL 9 families via features_by_family ----
    print("\n--- Each family alone ---")
    fam_features = features_by_family(CANDIDATE)
    for fam in all_families():
        cols = fam_features[fam]
        if not cols:
            print(f"  (skip) {fam} — 0 features")
            continue
        r = cv_evaluate(df, y, g, cols, f"{fam} only")
        if r:
            results.append(r)

    # ---- Leave-one-family-out: also all 9 families ----------------------
    print("\n--- Leave-one-family-out ---")
    for fam in all_families():
        cols = fam_features[fam]
        if not cols:
            print(f"  (skip) {fam} — 0 features")
            continue
        remaining = [c for c in CANDIDATE if family_of(c) != fam]
        r = cv_evaluate(df, y, g, remaining, f"ALL minus {fam}")
        if r:
            results.append(r)

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(RESULTS, "v2_ablation_corrected.csv"),
                  index=False)
    print("\n" + df_out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()