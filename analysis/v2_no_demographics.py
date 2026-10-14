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



# v2_no_demographics.py
"""PPG-only sensitivity with fold-aware preprocessing and dependency-aware exclusion."""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH
from v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor
from feature_dependencies import is_demographic_dependent
from v2_bootstrap import bootstrap_ci, format_ci


def cv_eval(df, y, g, cols, label):
    X_raw = df[cols].values
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)
    aucs, f1s, baccs, accs = [], [], [], []
    all_y, all_p, all_prob = [], [], []
    all_subj = []
    for tr, te in sgkf.split(X_raw, y, groups=g):
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
        accs.append((pred == y[te]).mean())
        all_y.extend(y[te].tolist())
        all_p.extend(pred.tolist())
        all_prob.extend(prob.tolist())
        all_subj.extend(g[te].tolist())
    print(f"  {label:35s}  AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"F1={np.mean(f1s):.4f}  BAcc={np.mean(baccs):.4f}")
    ci = bootstrap_ci(np.array(all_y), np.array(all_p), np.array(all_prob),
                       stratify_by=np.array(all_subj), n_boot=1000, seed=SEED)
    return {
        "label": label, "n_features": len(cols),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)), "bacc_mean": float(np.mean(baccs)),
        "acc_mean": float(np.mean(accs)),
        "pooled_auc_ci": format_ci(ci, "auc"),
        "pooled_f1_ci": format_ci(ci, "macro_f1"),
        "pooled_bacc_ci": format_ci(ci, "bacc"),
    }


def main():
    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values

    ppg_cols = [c for c in CANDIDATE if not is_demographic_dependent(c)]
    print(f"Full candidate features: {len(CANDIDATE)}")
    print(f"PPG-only features: {len(ppg_cols)}")
    excluded = sorted(set(CANDIDATE) - set(ppg_cols))
    print(f"Excluded (demographic-dependent): {excluded}\n")

    results = []
    results.append(cv_eval(df, y, g, CANDIDATE, "ALL features"))
    results.append(cv_eval(df, y, g, ppg_cols, "PPG-only"))

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(RESULTS, "v2_no_demographics_corrected.csv"),
                   index=False)
    print("\n" + df_out.to_string(index=False))


if __name__ == "__main__":
    main()