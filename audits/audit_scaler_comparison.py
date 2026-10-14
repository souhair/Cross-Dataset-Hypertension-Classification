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


# audit_scaler_comparison.py
"""
Verify the scaler-sensitivity claim in Supplementary S3:

  "The scaler-sensitivity comparison considered RobustScaler,
   QuantileTransformer, MinMax followed by standardization, and
   percentile clipping followed by standardization. AUCs differed
   by no more than 0.002."

Runs 5-fold StratifiedGroupKFold on Aurora-BP with XGBoost, comparing
five preprocessing variants. Reports per-scaler mean AUC and the
maximum spread across scalers.

Output: out_v2/results/v2_scaler_comparison.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import (StandardScaler, RobustScaler,
                                   QuantileTransformer, MinMaxScaler)
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor

np.random.seed(SEED)


# ---- Scaler variants -----------------------------------------------------
class MinMaxThenStandard:
    def __init__(self):
        self.mm = MinMaxScaler()
        self.ss = StandardScaler()
    def fit(self, X, y=None):
        self.mm.fit(X); self.ss.fit(self.mm.transform(X)); return self
    def transform(self, X):
        return self.ss.transform(self.mm.transform(X))


class ClipThenStandard:
    """Clip each feature to [1st, 99th] percentile, then standardize."""
    def __init__(self, lo=1.0, hi=99.0):
        self.lo, self.hi = lo, hi
    def fit(self, X, y=None):
        self.lower_ = np.percentile(X, self.lo, axis=0)
        self.upper_ = np.percentile(X, self.hi, axis=0)
        clipped = np.clip(X, self.lower_, self.upper_)
        self.ss = StandardScaler().fit(clipped)
        return self
    def transform(self, X):
        return self.ss.transform(np.clip(X, self.lower_, self.upper_))


SCALERS = {
    "StandardScaler":       StandardScaler,
    "RobustScaler":         RobustScaler,
    "QuantileTransformer":  lambda: QuantileTransformer(
                                n_quantiles=1000, output_distribution="normal",
                                random_state=SEED),
    "MinMax+Standard":      MinMaxThenStandard,
    "Clip1-99%+Standard":   ClipThenStandard,
}


# ---- XGBoost config (matches §3.6c of the paper) ------------------------
def make_xgb():
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=-1,
        random_state=SEED, verbosity=0,
    )


def main():
    print("=" * 72)
    print("Scaler sensitivity audit — Aurora-BP, 5-fold StratifiedGroupKFold")
    print("=" * 72)

    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values
    print(f"Data: {len(df)} segments / {df.subject_id.nunique()} subjects / "
          f"{len(CANDIDATE)} candidate features\n")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED)
    results = {name: [] for name in SCALERS}

    for fold, (tr, te) in enumerate(sgkf.split(df[CANDIDATE].values, y,
                                                 groups=g), 1):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue

        # Fit preprocessor on this fold's training split
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])

        print(f"Fold {fold}: {X_tr.shape[1]} retained features")

        for name, scaler_fn in SCALERS.items():
            sc = scaler_fn()
            try:
                sc.fit(X_tr)
                Xtr_s = sc.transform(X_tr)
                Xte_s = sc.transform(X_te)

                # Sanity: no NaN/inf after scaling
                if not (np.isfinite(Xtr_s).all() and np.isfinite(Xte_s).all()):
                    print(f"  {name:24s}  SKIP — non-finite after scaling")
                    continue

                m = make_xgb()
                m.fit(Xtr_s, y[tr])
                prob = m.predict_proba(Xte_s)
                auc = roc_auc_score(y[te], prob, multi_class="ovr",
                                     average="macro")
                results[name].append(auc)
                print(f"  {name:24s}  AUC = {auc:.4f}")
            except Exception as e:
                print(f"  {name:24s}  FAILED — {type(e).__name__}: {e}")

    # ---- Summary --------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY — scaler sensitivity")
    print("=" * 72)

    rows = []
    for name, aucs in results.items():
        if not aucs:
            rows.append({"scaler": name, "n_folds": 0,
                         "auc_mean": np.nan, "auc_std": np.nan})
            continue
        rows.append({
            "scaler":   name,
            "n_folds":  len(aucs),
            "auc_mean": float(np.mean(aucs)),
            "auc_std":  float(np.std(aucs, ddof=1)),
            "auc_min":  float(np.min(aucs)),
            "auc_max":  float(np.max(aucs)),
        })

    summary = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    print(summary.round(4).to_string(index=False))

    # ---- Spread ---------------------------------------------------------
    means = [r["auc_mean"] for r in rows if not np.isnan(r["auc_mean"])]
    if means:
        spread = max(means) - min(means)
        print()
        print(f"  Max mean AUC:   {max(means):.4f}")
        print(f"  Min mean AUC:   {min(means):.4f}")
        print(f"  Spread:         {spread:.4f}")
        print()
        if spread <= 0.002:
            print(f"  ✓ PASS — spread {spread:.4f} ≤ 0.002")
            print(f"    The supplementary claim is confirmed.")
        elif spread <= 0.005:
            print(f"  ~ MARGINAL — spread {spread:.4f} is slightly above 0.002")
            print(f"    Update the supplementary text to reflect the observed spread.")
        else:
            print(f"  ✗ FAIL — spread {spread:.4f} > 0.002")
            print(f"    The supplementary claim understates scaler sensitivity.")
            print(f"    Rewrite the sentence with the actual numbers.")

    # ---- Save -----------------------------------------------------------
    out_csv = os.path.join(RESULTS, "v2_scaler_comparison.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")


if __name__ == "__main__":
    main()