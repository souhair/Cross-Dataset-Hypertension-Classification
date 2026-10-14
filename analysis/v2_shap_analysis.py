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



# v2_shap_analysis.py
"""SHAP interpretability with unified family definitions."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH
from v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor
from feature_families import family_of

FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 14, "figure.dpi": 130})


def main():
    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values

    # Use a fixed 80/20 split for SHAP (single model for interpretability)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr, te = next(gss.split(df[CANDIDATE].values, y, groups=g))

    pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
        df.iloc[tr], CANDIDATE)
    X_tr = pre.transform(df.iloc[tr])
    X_te = pre.transform(df.iloc[te])
    FEATURES = pre.feature_list_

    sc = StandardScaler().fit(X_tr)
    m = XGBClassifier(**XGB_3CLASS)
    m.fit(sc.transform(X_tr), y[tr])

    print(f"Computing SHAP values for {len(FEATURES)} features...")
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(sc.transform(X_te))

    if isinstance(shap_values, list):
        shap_mean_abs = np.mean([np.abs(s).mean(0) for s in shap_values], axis=0)
        per_class = [np.abs(s).mean(0) for s in shap_values]
    elif shap_values.ndim == 3:
        shap_mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        per_class = [np.abs(shap_values[:, :, c]).mean(0)
                     for c in range(shap_values.shape[2])]
    else:
        shap_mean_abs = np.abs(shap_values).mean(0)
        per_class = None

    np.save(os.path.join(RESULTS, "v2_shap_mean_abs.npy"), shap_mean_abs)
    if per_class:
        np.save(os.path.join(RESULTS, "v2_shap_per_class.npy"),
                np.array(per_class))
    with open(os.path.join(RESULTS, "v2_feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURES))

    # Figure 5 — top-25 global
    order = np.argsort(shap_mean_abs)[::-1][:25]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh([FEATURES[i] for i in order][::-1],
            [shap_mean_abs[i] for i in order][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 25 features by SHAP — Aurora-BP v2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_shap_top25.png"),
                bbox_inches="tight")
    plt.close()

    # Figure 7 — family contribution (uses all 9 families)
    from feature_families import all_families, features_by_family
    fam_features = features_by_family(FEATURES)
    rows = []
    for fam in all_families():
        members = fam_features[fam]
        if not members:
            continue
        total = 0.0
        for f in members:
            i = FEATURES.index(f)
            total += float(shap_mean_abs[i])
        rows.append({
            "family": fam,
            "total_shap": total,
            "n_features": len(members),
            "mean_shap_per_feature": total / len(members) if members else 0.0,
        })
    fam_df = pd.DataFrame(rows).sort_values("total_shap", ascending=False)

    fam_df.to_csv(os.path.join(RESULTS, "v2_shap_family_summary.csv"),
                   index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["coral" if f == "SMD" else "steelblue" for f in fam_df.family]
    ax.bar(fam_df.family, fam_df.total_shap, color=colors)
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.set_title("Feature-family contribution by SHAP")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_shap_families.png"),
                bbox_inches="tight")
    plt.close()

    print("\nFamily summary:")
    print(fam_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()