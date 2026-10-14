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



# v2_optional_analyses.py
"""Calibration, DCA, per-class ROC with corrected definitions."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, brier_score_loss,
                              roc_auc_score)
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, XGB_3CLASS, NAN_THRESH, CORR_THRESH
from v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor

FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with inclusive first bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        if i == 0:
            m = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        else:
            m = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        w = m.sum() / len(y_prob)
        err = abs(acc - conf)
        ece += w * err
        mce = max(mce, err)
    return ece, mce


def reliability_diagram(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        if i == 0:
            m = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        else:
            m = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if m.sum() == 0:
            continue
        centers.append(float(y_prob[m].mean()))   # actual mean prob
        accs.append(float(y_true[m].mean()))
        counts.append(int(m.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    nb_model, nb_all, nb_none = [], [], []
    prev = y_true.mean()
    for pt in thresholds:
        pred = (y_prob >= pt).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        nb_model.append((tp / n) - (fp / n) * (pt / (1 - pt)))
        nb_all.append(prev - (1 - prev) * (pt / (1 - pt)))
        nb_none.append(0.0)
    return np.array(nb_model), np.array(nb_all), np.array(nb_none)


def main():
    df, CANDIDATE = load_aurora_raw()
    y3 = df.htn3.values
    y_bin = df.htn_bin.values
    g = df.subject_id.astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=SEED)
    all_y3, all_p3, all_ybin, all_pbin = [], [], [], []
    for tr, te in sgkf.split(df[CANDIDATE].values, y3, groups=g):
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)

        m3 = XGBClassifier(**XGB_3CLASS)
        m3.fit(sc.transform(X_tr), y3[tr])
        p3 = m3.predict_proba(sc.transform(X_te))
        all_y3.extend(y3[te].tolist()); all_p3.extend(p3.tolist())

        mb = XGBClassifier(**{**XGB_3CLASS,
                                "objective": "binary:logistic",
                                "eval_metric": "logloss",
                                "num_class": None})
        mb.fit(sc.transform(X_tr), y_bin[tr])
        pb = mb.predict_proba(sc.transform(X_te))[:, 1]
        all_ybin.extend(y_bin[te].tolist()); all_pbin.extend(pb.tolist())

    all_y3 = np.array(all_y3); all_p3 = np.array(all_p3)
    all_ybin = np.array(all_ybin); all_pbin = np.array(all_pbin)

    # --- Calibration ---
    prevalence = all_ybin.mean()
    brier = brier_score_loss(all_ybin, all_pbin)
    brier_noskill = prevalence * (1 - prevalence)
    ece_bin, mce_bin = expected_calibration_error(all_ybin, all_pbin)
    print(f"Binary HTN: Brier={brier:.4f}  (no-skill={brier_noskill:.4f})  "
          f"ECE={ece_bin:.4f}  MCE={mce_bin:.4f}")

    rows = [{"task": "binary", "class": "HTN", "brier": brier,
             "brier_noskill": brier_noskill, "ece": ece_bin, "mce": mce_bin}]
    for k, label in enumerate(["Normal", "PreHTN", "HTN"]):
        yk = (all_y3 == k).astype(int)
        pk = all_p3[:, k]
        b = brier_score_loss(yk, pk)
        e, m = expected_calibration_error(yk, pk)
        prev = yk.mean()
        rows.append({"task": "3class", "class": label, "brier": b,
                     "brier_noskill": prev * (1 - prev), "ece": e, "mce": m})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS, "v2_calibration_corrected.csv"), index=False)

    # Figure 9 — reliability diagrams
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bc, ba, bn = reliability_diagram(all_ybin, all_pbin)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    ax.plot(bc, ba, "o-", color="steelblue", label="Binary HTN")
    ax.set_xlabel("Mean predicted prob", fontsize=14); ax.set_ylabel("Observed freq", fontsize=14)
    ax.set_title(f"Binary calibration (Brier={brier:.4f}, ECE={ece_bin:.4f})", fontsize=14)
    ax.legend(loc="upper left", fontsize=14); ax.grid(alpha=0.3)

    ax = axes[1]
    for k, (label, c) in enumerate(zip(["Normal", "PreHTN", "HTN"],
                                          ["green", "orange", "red"])):
        yk = (all_y3 == k).astype(int)
        pk = all_p3[:, k]
        bc, ba, _ = reliability_diagram(yk, pk)
        ax.plot(bc, ba, "o-", color=c, label=label, alpha=0.85)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted prob", fontsize=14); ax.set_ylabel("Observed freq", fontsize=14)
    ax.set_title("3-class calibration (one-vs-rest)", fontsize=14)
    ax.legend(loc="upper left", fontsize=14); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_calibration.png"),
                bbox_inches="tight")
    plt.close()

    # --- DCA ---
    thresholds = np.linspace(0.01, 0.99, 99)
    nb_model, nb_all, nb_none = net_benefit(all_ybin, all_pbin, thresholds)

    # Correct winning conditions
    beat_all = nb_model > nb_all
    beat_none = nb_model > 0
    beat_both = beat_all & beat_none
    print(f"Model beats treat-all: "
          f"[{thresholds[beat_all].min():.2f}, "
          f"{thresholds[beat_all].max():.2f}]")
    print(f"Model beats treat-none: "
          f"[{thresholds[beat_none].min():.2f}, "
          f"{thresholds[beat_none].max():.2f}]")
    print(f"Model beats both: "
          f"[{thresholds[beat_both].min():.2f}, "
          f"{thresholds[beat_both].max():.2f}]")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, nb_model, color="steelblue", lw=2, label="XGBoost")
    ax.plot(thresholds, nb_all, "r--", lw=1, label="Treat all")
    ax.plot(thresholds, nb_none, "k:", lw=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=16); ax.set_ylabel("Net benefit", fontsize=16)
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlim(0, 0.9)
    ax.set_title("Decision curve — binary HTN", fontsize=16)
    ax.legend(fontsize=16); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig10_decision_curve.png"),
                bbox_inches="tight")
    plt.close()

    print("Saved fig9_calibration.png, fig10_decision_curve.png")


if __name__ == "__main__":
    main()