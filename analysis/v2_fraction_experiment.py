# ===== Reproducibility seed — auto-injected by patch_analysis_seeds.py =====
import os as _os, random as _random, numpy as _np
import logging
_SEED = 42
_os.environ["PYTHONHASHSEED"] = str(_SEED)
_random.seed(_SEED)
_np.random.seed(_SEED)
try:
    import torch as _torch
    _torch.manual_seed(_SEED)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(_SEED)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
except ImportError:
    pass
# ==========================================================================

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



# v2_fraction_experiment.py
"""Fraction sweep with FIXED outer test set."""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                              roc_auc_score, accuracy_score)
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_BEST, XGB_FINETUNE, NAN_THRESH, CORR_THRESH
from v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor

FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def run_fraction(name, csv_path, df_aur, FEATURES, medians):
    tg = load_target_csv(csv_path, FEATURES, medians)
    X = tg[FEATURES].values.astype(np.float64)
    y = tg.htn3.values.astype(np.int64)
    g = tg.subject_id.astype(str).values

    print(f"\n{'='*72}\n{name}: {len(tg)} seg / {len(set(g))} subj\n{'='*72}")

    # FIXED test set — same across fractions
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30,
                              random_state=SEED)
    ft_pool_idx, test_idx = next(gss.split(X, y, groups=g))

    X_src = df_aur[FEATURES].values.astype(np.float64)
    y_src = df_aur.htn3.values.astype(np.int64)
    scaler_src = StandardScaler().fit(X_src)
    src = XGBClassifier(**XGB_BEST)
    src.fit(scaler_src.transform(X_src), y_src)

    all_subj = np.unique(g[ft_pool_idx])
    rows = []
    rng = np.random.default_rng(SEED)

    for frac in FRACTIONS:
        n_sub = max(3, int(len(all_subj) * frac))
        sel_subj = rng.choice(all_subj, size=n_sub, replace=False)
        train_mask = np.isin(g[ft_pool_idx], sel_subj)
        train_idx = ft_pool_idx[train_mask]

        X_tr = X[train_idx]; X_te = X[test_idx]
        y_tr = y[train_idx]; y_te = y[test_idx]

        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)

        # Pretrained + fine-tune (source-scaler coordinates — see fix #3)
        # [warm_fix_v1] transform target features with scaler_src
        X_tr_warm = scaler_src.transform(X_tr)
        X_te_warm = scaler_src.transform(X_te)
        m1 = XGBClassifier(**XGB_FINETUNE)
        continuation_ok = True
        try:
            m1.fit(X_tr_warm, y_tr, xgb_model=src.get_booster())
        except Exception as e:
            continuation_ok = False
            logging.warning(
                f"frac={frac:.2f}: continuation failed ({e}); "
                f"falling back to from-scratch on source-scaled features"
            )
            m1.fit(X_tr_warm, y_tr)
        p1 = m1.predict_proba(X_te_warm);
        pred1 = p1.argmax(1)

        # From-scratch
        m2 = XGBClassifier(**XGB_BEST)
        m2.fit(X_tr_s, y_tr)
        p2 = m2.predict_proba(X_te_s); pred2 = p2.argmax(1)

        auc1 = roc_auc_score(y_te, p1, multi_class="ovr", average="macro")
        auc2 = roc_auc_score(y_te, p2, multi_class="ovr", average="macro")
        f1_1 = f1_score(y_te, pred1, average="macro", zero_division=0)
        f1_2 = f1_score(y_te, pred2, average="macro", zero_division=0)
        b1 = balanced_accuracy_score(y_te, pred1)
        b2 = balanced_accuracy_score(y_te, pred2)

        print(f"  frac={frac:.0%}  n_subj={n_sub}  "
              f"pretrained AUC={auc1:.4f}  scratch AUC={auc2:.4f}  "
              f"delta={auc1-auc2:+.4f}")

        rows.append({
            "dataset": name, "fraction": frac,
            "n_train_subjects": n_sub,
            "n_train_segments": len(train_idx),
            "pretrained_auc": auc1, "pretrained_f1": f1_1,
            "pretrained_bacc": b1,
            "scratch_auc": auc2, "scratch_f1": f1_2, "scratch_bacc": b2,
            "delta_auc": auc1 - auc2,
            "continuation_ok": continuation_ok,
        })

    return rows


def main():
    df_aur, CANDIDATE = load_aurora_raw()
    pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(df_aur, CANDIDATE)
    FEATURES = pre.feature_list_
    medians = aurora_medians(df_aur, FEATURES)
    X = pre.transform(df_aur)
    df_aur_feat = df_aur.copy()
    for i, c in enumerate(FEATURES):
        df_aur_feat[c] = X[:, i]

    all_rows = []
    for name, key in [("PPG-BP", "ppgbp"),
                       ("PulseDB-Vital", "pulsedb_vital"),
                       ("DS4", "ds4")]:
        all_rows.extend(run_fraction(name, CSV[key], df_aur_feat,
                                       FEATURES, medians))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS, "v2_fraction_experiment_corrected.csv"),
               index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()