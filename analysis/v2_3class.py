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



# v2_3class.py
"""3-class JNC7 baseline with fold-aware preprocessing."""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              balanced_accuracy_score, confusion_matrix,
                              classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (SEED, N_FOLDS, XGB_3CLASS, RF_PARAMS, LR_PARAMS,
                    NAN_THRESH, CORR_THRESH)
from v2_utils import load_aurora_raw, RESULTS, save_oof_predictions
from fold_preprocessor import FoldAwarePreprocessor
from v2_bootstrap import bootstrap_ci, format_ci


def main():
    df, CANDIDATE = load_aurora_raw()
    X_raw = df[CANDIDATE].values
    y = df.htn3.values
    g = df.subject_id.astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)

    models = {
        "LogReg": lambda: LogisticRegression(**LR_PARAMS),
        "RF": lambda: RandomForestClassifier(**RF_PARAMS),
        "XGB": lambda: XGBClassifier(**XGB_3CLASS),
    }

    results = {name: {"y_true": [], "y_pred": [], "y_prob": [],
                       "fold_metrics": [], "oof_records": []}
               for name in models}

    for fold, (tr, te) in enumerate(sgkf.split(X_raw, y, groups=g), 1):
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        y_tr, y_te = y[tr], y[te]

        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        for name, fn in models.items():
            m = fn()
            m.fit(X_tr_s, y_tr)
            prob = m.predict_proba(X_te_s)
            pred = prob.argmax(1)
            results[name]["fold_metrics"].append({
                "acc": accuracy_score(y_te, pred),
                "macro_f1": f1_score(y_te, pred, average="macro",
                                      zero_division=0),
                "auc": roc_auc_score(y_te, prob, multi_class="ovr",
                                      average="macro"),
                "bal_acc": balanced_accuracy_score(y_te, pred),
            })
            results[name]["y_true"].extend(y_te.tolist())
            results[name]["y_pred"].extend(pred.tolist())
            results[name]["y_prob"].extend(prob.tolist())
            for i, idx in enumerate(te):
                results[name]["oof_records"].append({
                    "segment_id": df.iloc[idx].get("segment_id", f"seg_{idx}"),
                    "subject_id": str(g[idx]),
                    "fold": fold,
                    "y_true": int(y_te[i]),
                    "y_prob": prob[i].tolist(),
                })

    summary = []
    for name in models:
        df_f = pd.DataFrame(results[name]["fold_metrics"])
        yt = np.array(results[name]["y_true"])
        yp = np.array(results[name]["y_pred"])
        yprob = np.array(results[name]["y_prob"])

        ci = bootstrap_ci(yt, yp, yprob,
                          stratify_by=np.array(
                              [r["subject_id"] for r in
                               results[name]["oof_records"]]),
                          n_boot=1000, seed=SEED)

        print(f"\n== {name} ==")
        print(f"  fold AUC: {df_f.auc.mean():.4f} ± {df_f.auc.std():.4f}")
        print(f"  fold F1 : {df_f.macro_f1.mean():.4f} ± {df_f.macro_f1.std():.4f}")
        print(f"  fold BAcc: {df_f.bal_acc.mean():.4f} ± {df_f.bal_acc.std():.4f}")
        print(f"  pooled AUC CI: {format_ci(ci, 'auc')}")
        print(f"  pooled F1 CI : {format_ci(ci, 'macro_f1')}")
        print(f"  pooled BAcc CI: {format_ci(ci, 'bacc')}")

        cm = confusion_matrix(yt, yp)
        print("  pooled confusion matrix:")
        print(f"  {cm.tolist()}")

        save_oof_predictions(
            results[name]["oof_records"],
            os.path.join(RESULTS, f"v2_oof_3class_{name}.csv"))

        summary.append({
            "model": name,
            "auc_mean": df_f.auc.mean(), "auc_std": df_f.auc.std(),
            "f1_mean": df_f.macro_f1.mean(), "f1_std": df_f.macro_f1.std(),
            "bacc_mean": df_f.bal_acc.mean(), "bacc_std": df_f.bal_acc.std(),
            "acc_mean": df_f.acc.mean(),
            "bootstrap_auc_mean": ci["auc"]["mean"],
            "bootstrap_auc_mean_lo": ci["auc"]["lo"],
            "bootstrap_auc_mean_hi": ci["auc"]["hi"],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(RESULTS, "v2_3class_summary.csv"),
                       index=False)
    print("\n" + df_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()