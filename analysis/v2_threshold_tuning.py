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


# v2_threshold_tuning.py
"""Nested-CV threshold tuning with fold-aware preprocessing.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH
from v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor


def tune_weights(y, prob, grid=(0.5, 2.0, 0.15)):
    lo, hi, step = grid
    w_range = np.arange(lo, hi + 1e-9, step)
    best_f1, best_w = 0.0, np.array([1.0, 1.0, 1.0])
    for w0 in w_range:
        for w1 in w_range:
            for w2 in w_range:
                ww = np.array([w0, w1, w2])
                f1 = f1_score(y, (prob * ww).argmax(1),
                              average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_w = f1, ww
    return best_w, best_f1


def evaluate(name, df, y, g, CANDIDATE):
    """5-fold outer + 5-fold inner; preprocessor refit per fold.

    df : DataFrame containing candidate features
    y  : int labels
    g  : subject IDs (grouping variable)
    """
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=SEED)
    f1_a, f1_t, b_a, b_t, aucs = [], [], [], [], []

    for tr, te in sgkf.split(df[CANDIDATE].values, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue

        # -------- Outer: fit preprocessor on OUTER-TRAIN only --------
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])

        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(**XGB_3CLASS)
        m.fit(sc.transform(X_tr), y[tr])
        pt = m.predict_proba(sc.transform(X_te))

        # -------- Inner: nested GroupKFold, refit per inner-train --------
        gkf = GroupKFold(n_splits=5)
        votes = []
        for itr, iva in gkf.split(X_tr, y[tr], groups=g[tr]):
            pre_i = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
                df.iloc[tr].iloc[itr], CANDIDATE)
            X_itr = pre_i.transform(df.iloc[tr].iloc[itr])
            X_iva = pre_i.transform(df.iloc[tr].iloc[iva])
            sc_i = StandardScaler().fit(X_itr)
            mi = XGBClassifier(**XGB_3CLASS)
            mi.fit(sc_i.transform(X_itr), y[tr][itr])
            pv = mi.predict_proba(sc_i.transform(X_iva))
            w, _ = tune_weights(y[tr][iva], pv)
            votes.append(w)
        w_mean = np.mean(votes, axis=0)

        pred_a = pt.argmax(1)
        pred_t = (pt * w_mean).argmax(1)

        f1_a.append(f1_score(y[te], pred_a, average="macro", zero_division=0))
        f1_t.append(f1_score(y[te], pred_t, average="macro", zero_division=0))
        b_a.append(balanced_accuracy_score(y[te], pred_a))
        b_t.append(balanced_accuracy_score(y[te], pred_t))
        aucs.append(roc_auc_score(y[te], pt, multi_class="ovr",
                                  average="macro"))

    if not aucs:
        print("  (no usable folds)")
        return None

    out = {
        "dataset": name,
        "auc": float(np.mean(aucs)),
        "f1_argmax": float(np.mean(f1_a)),
        "f1_nested": float(np.mean(f1_t)),
        "bacc_argmax": float(np.mean(b_a)),
        "bacc_nested": float(np.mean(b_t)),
        "delta_f1": float(np.mean(f1_t) - np.mean(f1_a)),
        "delta_bacc": float(np.mean(b_t) - np.mean(b_a)),
    }
    print(f"  AUC={out['auc']:.4f}  "
          f"F1 argmax={out['f1_argmax']:.4f}  nested={out['f1_nested']:.4f}  "
          f"Δ={out['delta_f1']:+.4f}  "
          f"BAcc argmax={out['bacc_argmax']:.4f}  "
          f"nested={out['bacc_nested']:.4f}  Δ={out['delta_bacc']:+.4f}")
    return out


def main():
    df_aur, CANDIDATE = load_aurora_raw()
    results = []
    results.append(evaluate("Aurora", df_aur, df_aur.htn3.values,
                            df_aur.subject_id.astype(str).values,
                            CANDIDATE))

    # Target datasets: use Aurora's feature list and medians as the fixed
    # preprocessing spec (never touching target labels before CV).
    pre_aur = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
        df_aur, CANDIDATE)
    FEATURES = pre_aur.feature_list_
    medians = aurora_medians(df_aur, FEATURES)

    for name, key in [("PPG-BP", "ppgbp"),
                      ("PulseDB-Vital", "pulsedb_vital"),
                      ("DS4", "ds4")]:
        tg = load_target_csv(CSV[key], FEATURES, medians)
        r = evaluate(name, tg, tg.htn3.values,
                     tg.subject_id.astype(str).values, FEATURES)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS, "v2_threshold_tuning_corrected.csv"),
              index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()