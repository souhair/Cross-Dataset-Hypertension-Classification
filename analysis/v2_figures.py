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



# v2_figures.py
"""Generate all main figures with fold-aware preprocessing."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              roc_auc_score, f1_score,
                              balanced_accuracy_score,
                              classification_report)
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH
from v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor

FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 14, "figure.dpi": 130})


def main():
    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)
    all_y, all_p, all_prob = [], [], []
    for tr, te in sgkf.split(df[CANDIDATE].values, y, groups=g):
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = XGBClassifier(**XGB_3CLASS)
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        all_y.extend(y[te].tolist())
        all_p.extend(prob.argmax(1).tolist())
        all_prob.extend(prob.tolist())

    all_y = np.array(all_y); all_p = np.array(all_p)
    all_prob = np.array(all_prob)

    # Fig 1 — Confusion matrix
    cm = confusion_matrix(all_y, all_p)
    fig, ax = plt.subplots(figsize=(5, 4.2)) #5, 4.2
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Normal", "PreHTN", "HTN"])
    ax.set_yticklabels(["Normal", "PreHTN", "HTN"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    auc_ = roc_auc_score(all_y, all_prob, multi_class="ovr", average="macro")
    f1_ = f1_score(all_y, all_p, average="macro", zero_division=0)
    # ax.set_title(f"Aurora-BP 3-class confusion matrix\n"
    #              f"Macro AUC={auc_:.2f}  Macro F1={f1_:.2f}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_confusion_aurora.png"),
                bbox_inches="tight")
    plt.close()

    # Fig 2 — Per-class bars
    cr = classification_report(all_y, all_p,
                                 target_names=["Normal", "PreHTN", "HTN"],
                                 output_dict=True, zero_division=0)
    classes = ["Normal", "PreHTN", "HTN"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); w = 0.27
    ax.bar(x - w, [cr[c]["precision"] for c in classes], w, label="Precision")
    ax.bar(x, [cr[c]["recall"] for c in classes], w, label="Recall")
    ax.bar(x + w, [cr[c]["f1-score"] for c in classes], w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class performance — Aurora-BP", fontsize=14)
    ax.legend(loc='upper center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_per_class_aurora.png"),
                bbox_inches="tight")
    plt.close()

    # Fig 3 — ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, c in enumerate(classes):
        yb = (all_y == k).astype(int)
        fpr, tpr, _ = roc_curve(yb, all_prob[:, k])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False Positive Rate", fontsize=14); ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("One-vs-rest ROC — Aurora-BP", fontsize=14)
    ax.legend(loc="lower right", fontsize=12); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_roc_aurora.png"),
                bbox_inches="tight")
    plt.close()

    print(f"Saved figures. Macro AUC = {auc_:.4f}")


if __name__ == "__main__":
    main()