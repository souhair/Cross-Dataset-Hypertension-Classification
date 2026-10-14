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


# v2_leakage.py
"""Leakage controls on Aurora v2.

Produces Table S2:
  Test 1 — subject overlap
  Test 2 — random-label AUC (20 shuffles)
  Test 3 — single-feature AUC scan
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from analysis.v2_utils import load_aurora_raw, RESULTS

# ---- Load Aurora ---------------------------------------------------------
aur, FEATURES = load_aurora_raw()
print(f"Aurora v2: {len(aur)} seg / {aur.subject_id.nunique()} subj / "
      f"{len(FEATURES)} features")

X = aur[FEATURES].values
y = aur.htn3.values
g = aur.subject_id.astype(str).values

# =========================================================================
# TEST 1 — subject overlap between train and test
# =========================================================================
print("\n" + "=" * 60)
print("Test 1 — subject overlap")
print("=" * 60)

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr, te = next(gss.split(X, y, groups=g))
overlap = set(g[tr]) & set(g[te])
n_overlap = len(overlap)
print(f"  Overlap count: {n_overlap}")
print(f"  {'✓ PASS' if n_overlap == 0 else '✗ FAIL — leakage detected'}")

# =========================================================================
# TEST 2 — random-label control (20 shuffles)
# =========================================================================
print("\n" + "=" * 60)
print("Test 2 — random-label control (20 shuffles)")
print("=" * 60)

aucs = []
for seed in range(20):
    rng = np.random.default_rng(seed)

    # Subject-level consensus label, then shuffle the assignment across
    # subjects so that each subject receives another subject's label.
    subj_lbl = (aur.groupby("subject_id").htn3.mean().round().astype(int))
    shuffled = pd.Series(rng.permutation(subj_lbl.values),
                          index=subj_lbl.index)
    y_r = aur.subject_id.map(shuffled).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr2, te2 = next(gss.split(X, y_r, groups=g))

    sc = StandardScaler().fit(X[tr2])
    m = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=-1,
        random_state=seed, verbosity=0,
    )
    m.fit(sc.transform(X[tr2]), y_r[tr2])
    p = m.predict_proba(sc.transform(X[te2]))
    aucs.append(roc_auc_score(y_r[te2], p, multi_class="ovr",
                               average="macro"))

    if (seed + 1) % 5 == 0:
        print(f"  [{seed+1}/20]  running mean AUC = {np.mean(aucs):.4f}")

mean_auc = float(np.mean(aucs))
std_auc = float(np.std(aucs))
print(f"\n  Random-label AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  {'✓ PASS — near chance' if mean_auc < 0.55 else '✗ FAIL'}")

# =========================================================================
# TEST 3 — single-feature AUC scan
# =========================================================================
print("\n" + "=" * 60)
print("Test 3 — single-feature AUC scan")
print("=" * 60)

single = []
for c in FEATURES:
    v = aur[c].fillna(aur[c].median()).values
    if np.std(v) < 1e-12:
        continue
    best = 0.0
    for k in [0, 1, 2]:
        y_bin = (y == k).astype(int)
        try:
            a = roc_auc_score(y_bin, v)
            a = max(a, 1 - a)   # orient so AUC >= 0.5
            best = max(best, a)
        except ValueError:
            continue
    single.append((c, best))

single.sort(key=lambda kv: -kv[1])
n_high = sum(1 for _, a in single if a > 0.85)

print(f"  Features with per-class AUC > 0.85: {n_high}")
print(f"  {'✓ PASS' if n_high == 0 else '✗ FAIL'}")
print("\n  Top-10 single features by per-class max AUC:")
for c, a in single[:10]:
    print(f"    {c:50s}  {a:.4f}")

# =========================================================================
# Save
# =========================================================================
out_path = os.path.join(RESULTS, "v2_leakage_single_feature.csv")
pd.DataFrame(single, columns=["feature", "auc"]).to_csv(out_path, index=False)
print(f"\nSaved per-feature AUC scan to:\n  {out_path}")

# =========================================================================
# Summary — Table S2 ready
# =========================================================================
print("\n" + "=" * 60)
print("TABLE S2 (ready to paste)")
print("=" * 60)
print(f"""
Control                          Result                Interpretation
Subject overlap                  {n_overlap:<21} {'No overlap reported' if n_overlap==0 else 'Leakage detected'}
Random-label AUC (20 seeds)      {mean_auc:.3f} ± {std_auc:.3f}        {'Near-chance discrimination' if mean_auc < 0.55 else 'Above chance'}
Single features > 0.85 AUC       {n_high:<21} {'No tested feature above cutoff' if n_high==0 else 'Features above cutoff'}
""")