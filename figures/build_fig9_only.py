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


# build_fig9_only.py
"""
Build Fig 9 — per-class ROC for the 9 neural architectures.

Correctly reads the per-architecture key layout in v2_nn_sweep_predictions.npz:
    <arch>_y_true, <arch>_y_pred, <arch>_y_prob, <arch>_y_subj
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

RES = r"add_your_root_folder\out_v2\results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

CLASS_NAMES = ["Normal", "PreHTN", "HTN"]


def fig9_per_class_roc():
    npz_path = os.path.join(RES, "v2_nn_sweep_predictions.npz")
    data = np.load(npz_path, allow_pickle=True)

    # Discover architectures from the key pattern "<name>_y_true"
    arch_names = sorted({
        k.replace("_y_true", "")
        for k in data.files if k.endswith("_y_true")
    })
    print(f"[fig9] architectures found: {arch_names}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5))#4.6

    for class_idx, (ax, cname) in enumerate(zip(axes, CLASS_NAMES)):
        for arch in arch_names:
            y_true = data[f"{arch}_y_true"]
            y_prob = data[f"{arch}_y_prob"]

            # Guard: probability matrix must have 3 columns
            if y_prob.ndim != 2 or y_prob.shape[1] < 3:
                print(f"[fig9] skipping {arch}: y_prob shape {y_prob.shape}")
                continue

            y_bin = (y_true == class_idx).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue

            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, class_idx])
            a = auc(fpr, tpr)

            is_xgb = (arch.lower() == "xgboost")
            color = "#c53030" if is_xgb else None
            lw = 1.2 if is_xgb else 1.0 # 2.2
            alpha = 0.8 if is_xgb else 0.75 #1.0
            ax.plot(fpr, tpr, color=color, lw=lw, alpha=alpha,
                    label=f"{arch} ({a:.3f})", zorder=5 if is_xgb else 2)

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.7)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate" if class_idx == 0 else "")
        ax.set_title(f"{cname} (one-vs-rest)", fontsize=16)
        ax.legend(loc="lower right", fontsize=9.5, frameon=False)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

    fig.suptitle(
        "Per-class ROC — neural network architectures vs. XGBoost (5-fold CV pooled)",
        fontsize=16, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG, "fig9_per_class_roc_nn.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig9] saved {out}")


if __name__ == "__main__":
    fig9_per_class_roc()