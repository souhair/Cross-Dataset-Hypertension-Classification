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



# v2_external_cv_stratified.py
"""5-fold Stratified GroupKFold external validation."""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                              roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

from config import (SEED, N_FOLDS, XGB_BEST, XGB_FINETUNE,
                    XGB_FINETUNE_BINARY, XGB_BINARY,
                    NAN_THRESH, CORR_THRESH)
from v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor


def evaluate_cv(name, csv_path, df_aur, FEATURES, medians):
    tg = load_target_csv(csv_path, FEATURES, medians)
    X = tg[FEATURES].values.astype(np.float64)
    y = tg.htn3.values.astype(np.int64)
    g = tg.subject_id.astype(str).values

    print(f"\n{'='*72}\n{name}: {len(tg)} seg / {len(set(g))} subj\n{'='*72}")

    X_src = df_aur[FEATURES].values.astype(np.float64)
    y_src = df_aur.htn3.values.astype(np.int64)
    scaler_src = StandardScaler().fit(X_src)
    X_src_s = scaler_src.transform(X_src)

    # Source flat model
    flat_src = XGBClassifier(**XGB_BEST)
    flat_src.fit(X_src_s, y_src)

    # Source hierarchical (binary objectives)
    y1_src = (y_src > 0).astype(int)
    s1_src = XGBClassifier(**XGB_BINARY)
    s1_src.fit(X_src_s, y1_src)
    abn = y_src > 0
    s2_src = XGBClassifier(**XGB_BINARY)
    s2_src.fit(X_src_s[abn], (y_src[abn] == 2).astype(int))

    # Sanity: source stage predictions are binary
    assert s1_src.predict_proba(X_src_s[:10]).shape[1] == 2
    assert s2_src.predict_proba(X_src_s[abn][:10]).shape[1] == 2

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)
    flat_aucs, flat_f1s, flat_baccs = [], [], []
    hier_aucs, hier_f1s, hier_baccs = [], [], []
    flat_cont_ok_list, hier_s1_ok_list, hier_s2_ok_list = [], [], []

    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=g), 1):
        if len(np.unique(y[te])) < 3 or len(np.unique(y[tr])) < 3:
            continue
        X_tr = X[tr]; X_te = X[te]; y_tr = y[tr]; y_te = y[te]

        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)

        # Source-scaler transformed inputs for warm-start continuation
        # [warm_fix_v1] target features must be in source coordinates
        X_tr_warm = scaler_src.transform(X_tr)
        X_te_warm = scaler_src.transform(X_te)

        # Flat (warm-start in source coordinate system)
        # [tier2b] track continuation status
        m_flat = XGBClassifier(**XGB_FINETUNE)
        flat_continuation_ok = True
        try:
            m_flat.fit(X_tr_warm, y_tr, xgb_model=flat_src.get_booster())
        except Exception as e:
            flat_continuation_ok = False
            logging.warning(
                f"{name} fold {fold}: flat continuation failed ({e}); "
                f"falling back to from-scratch on source-scaled features"
            )
            m_flat.fit(X_tr_warm, y_tr)
        prob = m_flat.predict_proba(X_te_warm); pred = prob.argmax(1)
        flat_aucs.append(roc_auc_score(y_te, prob, multi_class="ovr",
                                         average="macro"))
        flat_f1s.append(f1_score(y_te, pred, average="macro", zero_division=0))
        flat_baccs.append(balanced_accuracy_score(y_te, pred))
        flat_cont_ok_list.append(flat_continuation_ok)

        # Hierarchical (both stages warm-start in source coordinates)
        y1_tr = (y_tr > 0).astype(int)
        s1_ft = XGBClassifier(**XGB_FINETUNE_BINARY)
        hier_s1_ok = True
        try:
            s1_ft.fit(X_tr_warm, y1_tr, xgb_model=s1_src.get_booster())
        except Exception as e:
            hier_s1_ok = False
            logging.warning(f"{name} fold {fold}: hier stage-1 continuation failed ({e})")
            s1_ft.fit(X_tr_warm, y1_tr)
        abn_tr = y_tr > 0
        s2_ft = XGBClassifier(**XGB_FINETUNE_BINARY)
        hier_s2_ok = True
        try:
            s2_ft.fit(X_tr_warm[abn_tr], (y_tr[abn_tr] == 2).astype(int),
                      xgb_model=s2_src.get_booster())
        except Exception as e:
            hier_s2_ok = False
            logging.warning(f"{name} fold {fold}: hier stage-2 continuation failed ({e})")
            s2_ft.fit(X_tr_warm[abn_tr], (y_tr[abn_tr] == 2).astype(int))
        pa = s1_ft.predict_proba(X_te_warm)[:, 1]
        ph = s2_ft.predict_proba(X_te_warm)[:, 1]
        prob_h = np.vstack([1 - pa, pa * (1 - ph), pa * ph]).T
        pred_h = prob_h.argmax(1)
        hier_aucs.append(roc_auc_score(y_te, prob_h, multi_class="ovr",
                                         average="macro"))
        hier_f1s.append(f1_score(y_te, pred_h, average="macro",
                                   zero_division=0))
        hier_baccs.append(balanced_accuracy_score(y_te, pred_h))
        hier_s1_ok_list.append(hier_s1_ok)
        hier_s2_ok_list.append(hier_s2_ok)

        print(f"  fold {fold}: flat AUC={flat_aucs[-1]:.4f}  "
              f"hier AUC={hier_aucs[-1]:.4f}")

    print(f"\n  Flat warm-start: AUC={np.mean(flat_aucs):.4f}±{np.std(flat_aucs):.4f}  "
          f"F1={np.mean(flat_f1s):.4f}  BAcc={np.mean(flat_baccs):.4f}")
    print(f"  Hier warm-start: AUC={np.mean(hier_aucs):.4f}±{np.std(hier_aucs):.4f}  "
          f"F1={np.mean(hier_f1s):.4f}  BAcc={np.mean(hier_baccs):.4f}")

    return {
        "dataset": name, "n_segments": len(tg), "n_subjects": len(set(g)),
        "flat_cv_auc_mean": float(np.mean(flat_aucs)),
        "flat_cv_auc_std": float(np.std(flat_aucs)),
        "flat_cv_f1": float(np.mean(flat_f1s)),
        "flat_cv_bacc": float(np.mean(flat_baccs)),
        "hier_cv_auc_mean": float(np.mean(hier_aucs)),
        "hier_cv_auc_std": float(np.std(hier_aucs)),
        "hier_cv_f1": float(np.mean(hier_f1s)),
        "hier_cv_bacc": float(np.mean(hier_baccs)),
        "flat_continuation_ok_rate":
            float(np.mean(flat_cont_ok_list)) if flat_cont_ok_list else None,
        "hier_s1_continuation_ok_rate":
            float(np.mean(hier_s1_ok_list)) if hier_s1_ok_list else None,
        "hier_s2_continuation_ok_rate":
            float(np.mean(hier_s2_ok_list)) if hier_s2_ok_list else None,
    }


def main():
    df_aur, CANDIDATE = load_aurora_raw()
    pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(df_aur, CANDIDATE)
    FEATURES = pre.feature_list_
    medians = aurora_medians(df_aur, FEATURES)
    X = pre.transform(df_aur)
    df_aur_feat = df_aur.copy()
    for i, c in enumerate(FEATURES):
        df_aur_feat[c] = X[:, i]

    print(f"Aurora source: {len(df_aur)} seg / "
          f"{df_aur.subject_id.nunique()} subj / {len(FEATURES)} features")

    results = []
    for name, key in [("PPG-BP", "ppgbp"),
                       ("PulseDB-Vital", "pulsedb_vital"),
                       ("DS4", "ds4")]:
        results.append(evaluate_cv(name, CSV[key], df_aur_feat, FEATURES, medians))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS, "v2_external_cv_stratified_corrected.csv"),
               index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()