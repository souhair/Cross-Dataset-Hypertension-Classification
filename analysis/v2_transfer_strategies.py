# ===== Reproducibility seed — auto-injected by patch_analysis_seeds.py =====
import os as _os, random as _random, numpy as _np
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



# v2_transfer_strategies.py
"""
Transfer strategies with CORRECTED CORAL.
CORAL aligns SOURCE features to TARGET UDA covariance, trains on
SOURCE labels, evaluates on TARGET test. Target labels are NEVER used in
the CORAL branch.
"""
import numpy as np
import pandas as pd
import os
import logging
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                              roc_auc_score, accuracy_score)
from xgboost import XGBClassifier
from scipy.linalg import sqrtm

from config import (SEED, N_FOLDS, XGB_BEST, XGB_FINETUNE,
                    NAN_THRESH, CORR_THRESH)
from v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)s | %(message)s")

STRATEGIES = ["zero_shot", "from_scratch", "warm_start",
               "target_scaler", "coral"]


def coral_align(X_s, X_t, lam=1.0):
    """CORAL: align source features X_s to target covariance of X_t."""
    d = X_s.shape[1]
    C_s = np.cov(X_s, rowvar=False) + np.eye(d) * lam
    C_t = np.cov(X_t, rowvar=False) + np.eye(d) * lam
    A = np.linalg.inv(sqrtm(C_s).real) @ sqrtm(C_t).real
    return (X_s - X_s.mean(0)) @ A + X_t.mean(0)


def evaluate_strategies(name, csv_path, df_aur, FEATURES, medians):
    """Run all 5 strategies with 5-fold StratifiedGroupKFold on target."""
    tg = load_target_csv(csv_path, FEATURES, medians)
    X = tg[FEATURES].values.astype(np.float64)
    y = tg.htn3.values.astype(np.int64)
    g = tg.subject_id.astype(str).values

    print(f"\n{'='*72}\n{name}: {len(tg)} seg / {len(set(g))} subj\n{'='*72}")

    # Source features from Aurora (FEATURES list is same as Aurora's)
    X_src = df_aur[FEATURES].values.astype(np.float64)
    y_src = df_aur.htn3.values.astype(np.int64)
    scaler_src = StandardScaler().fit(X_src)
    X_src_scaled = scaler_src.transform(X_src)

    # Source model
    src_model = XGBClassifier(**XGB_BEST)
    src_model.fit(X_src_scaled, y_src)

    fold_results = {s: [] for s in STRATEGIES}
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=g), 1):
        if len(np.unique(y[te])) < 3 or len(np.unique(y[tr])) < 3:
            continue
        X_tr = X[tr]; X_te = X[te]; y_tr = y[tr]; y_te = y[te]

        scaler_uda = StandardScaler().fit(X_tr)
        X_tr_s = scaler_uda.transform(X_tr)
        X_te_s = scaler_uda.transform(X_te)

        def _record(strategy, prob, **extra):
            fold_results[strategy].append({
                "auc": roc_auc_score(y_te, prob, multi_class="ovr",
                                     average="macro"),
                "macro_f1": f1_score(y_te, prob.argmax(1),
                                     average="macro", zero_division=0),
                "bacc": balanced_accuracy_score(y_te, prob.argmax(1)),
                "acc": accuracy_score(y_te, prob.argmax(1)),
                **extra,
            })

        # --- Zero-shot ---
        try:
            prob = src_model.predict_proba(scaler_src.transform(X_te))
            _record("zero_shot", prob)
        except Exception as e:
            logging.error(f"zero_shot failed: {e}")

        # --- From-scratch ---
        try:
            m = XGBClassifier(**XGB_BEST)
            m.fit(X_tr_s, y_tr)
            _record("from_scratch", m.predict_proba(X_te_s))
        except Exception as e:
            logging.error(f"from_scratch failed: {e}")

        # --- Warm-start (source-scaler coordinates — see reviewer fix #3) ---
        # [warm_fix_v1] transform target features with scaler_src
        # [tier2b] record whether continuation succeeded
        continuation_ok = True
        try:
            X_tr_warm = scaler_src.transform(X_tr)
            X_te_warm = scaler_src.transform(X_te)
            m = XGBClassifier(**XGB_FINETUNE)
            n_before = src_model.get_booster().num_boosted_rounds()
            try:
                m.fit(X_tr_warm, y_tr, xgb_model=src_model.get_booster())
            except Exception as e:
                continuation_ok = False
                logging.warning(
                    f"fold {fold}: continuation failed ({e}); "
                    f"falling back to from-scratch on source-scaled features"
                )
                m.fit(X_tr_warm, y_tr)
            n_after = m.get_booster().num_boosted_rounds()
            if n_after <= n_before:
                continuation_ok = False
                logging.warning(f"fold {fold}: continuation did not add rounds")
            _record("warm_start", m.predict_proba(X_te_warm),
                    continuation_ok=continuation_ok)
        except Exception as e:
            logging.error(f"warm_start failed: {e}")

        # --- Target-scaler (no labels, freeze source model) ---
        try:
            prob = src_model.predict_proba(X_te_s)
            _record("target_scaler", prob)
        except Exception as e:
            logging.error(f"target_scaler failed: {e}")

        # --- CORAL (no target labels used) ---
        try:
            X_src_aligned = coral_align(X_src_scaled, X_tr_s, lam=0.1)
            m = XGBClassifier(**XGB_BEST)
            m.fit(X_src_aligned, y_src)         # <-- source labels
            prob = m.predict_proba(X_te_s)
            _record("coral", prob)
        except Exception as e:
            logging.error(f"coral failed: {e}")

    # Summary
    rows = []
    for s in STRATEGIES:
        if not fold_results[s]:
            continue
        df_s = pd.DataFrame(fold_results[s])
        row = {"dataset": name, "strategy": s,
               "auc_mean": df_s.auc.mean(), "auc_std": df_s.auc.std(),
               "f1_mean": df_s.macro_f1.mean(),
               "bacc_mean": df_s.bacc.mean(),
               "acc_mean": df_s.acc.mean(),
               "n_folds": len(df_s),
               "continuation_ok_rate":
                   float(df_s["continuation_ok"].mean())
                   if "continuation_ok" in df_s.columns else None}
        rows.append(row)
        print(f"  {s:15s}  AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"F1={row['f1_mean']:.4f}  BAcc={row['bacc_mean']:.4f}  "
              f"(n={row['n_folds']})")
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

    print(f"Aurora source: {len(df_aur)} seg / "
          f"{df_aur.subject_id.nunique()} subj / {len(FEATURES)} features")

    all_rows = []
    for name, key in [("PPG-BP", "ppgbp"),
                       ("PulseDB-Vital", "pulsedb_vital"),
                       ("DS4", "ds4")]:
        all_rows.extend(evaluate_strategies(
            name, CSV[key], df_aur_feat, FEATURES, medians))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS, "v2_transfer_strategies_corrected.csv"),
               index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()