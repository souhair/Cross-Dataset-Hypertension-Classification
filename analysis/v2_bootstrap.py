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


# v2_bootstrap.py
"""Bootstrap confidence intervals for all reported metrics."""
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                              accuracy_score)


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, alpha=0.95,
                  seed=42, stratify_by=None):
    """Stratified bootstrap for AUC, macro F1, BalAcc, Acc.

    Handles both binary and multiclass.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    metrics = {"auc": [], "macro_f1": [], "bacc": [], "acc": []}

    groups = (np.asarray(stratify_by) if stratify_by is not None
              else np.arange(n))
    unique = np.unique(groups)

    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in chosen])
        if len(idx) < 5:
            continue
        n_classes = len(np.unique(y_true[idx]))
        if n_classes < 2:
            continue
        try:
            if n_classes == 2:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    prob_pos = y_prob[idx, 1]
                else:
                    prob_pos = y_prob[idx]
                metrics["auc"].append(roc_auc_score(y_true[idx], prob_pos))
            else:
                metrics["auc"].append(roc_auc_score(
                    y_true[idx], y_prob[idx],
                    multi_class="ovr", average="macro"))
            metrics["macro_f1"].append(f1_score(
                y_true[idx], y_pred[idx], average="macro", zero_division=0))
            metrics["bacc"].append(balanced_accuracy_score(
                y_true[idx], y_pred[idx]))
            metrics["acc"].append(accuracy_score(y_true[idx], y_pred[idx]))
        except Exception:
            continue

    out = {}
    for k, v in metrics.items():
        if not v:
            out[k] = {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n_valid": 0}
            continue
        arr = np.array(v)
        lo = np.percentile(arr, 100 * (1 - alpha) / 2)
        hi = np.percentile(arr, 100 * (1 + alpha) / 2)
        out[k] = {"mean": float(arr.mean()),
                  "lo": float(lo),
                  "hi": float(hi),
                  "std": float(arr.std()),
                  "n_valid": int(len(arr))}
    return out


def format_ci(d, key):
    if key not in d or np.isnan(d[key]["mean"]):
        return "n/a"
    return (f"{d[key]['mean']:.4f} "
            f"[{d[key]['lo']:.4f}, {d[key]['hi']:.4f}] "
            f"(n={d[key]['n_valid']})")