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


# diagnose_pulsedb_discrepancy.py  (v2)
"""
Compare PulseDB-Vital from-scratch AUC between the two scripts.

After investigation, the expected difference is ~0.03-0.05 AUC because the
two scripts use different split protocols:
  - v2_transfer_strategies.py : 5-fold StratifiedGroupKFold
  - v2_fraction_experiment.py : single 70/30 stratified subject split

Small differences (< 0.06) are expected. Differences > 0.10 would indicate
a real bug (mismatched features, different label definitions, etc.).
"""
import os
import pandas as pd

RES = r"add_your_root_folder\aurora021020206\out_v2\results"

EXPECTED_DIFF_MAX = 0.06   # upper bound of the expected protocol difference
BUG_THRESHOLD = 0.10       # above this, look for a real bug

print("=" * 72)
print("PulseDB-Vital from-scratch AUC — protocol-difference check")
print("=" * 72)

# --- Transfer strategies (5-fold CV) ---
t = pd.read_csv(os.path.join(RES, "v2_transfer_strategies_corrected.csv"))
t_ps = t[(t["dataset"] == "PulseDB-Vital") & (t["strategy"] == "from_scratch")]
t_auc = t_ps["auc_mean"].iloc[0]
t_std = t_ps["auc_std"].iloc[0]
print(f"\n[transfer_strategies.py]  5-fold StratifiedGroupKFold")
print(f"  from_scratch AUC = {t_auc:.4f}  ± {t_std:.4f}")

# --- Fraction experiment (single split) ---
f = pd.read_csv(os.path.join(RES, "v2_fraction_experiment_corrected.csv"))
f_ps = f[(f["dataset"] == "PulseDB-Vital") & (f["train_pool_fraction"] == 1.0)]
f_auc = f_ps["scratch_auc"].iloc[0]
f_n = int(f_ps["n_train_subjects"].iloc[0])
print(f"\n[fraction_experiment.py]  single 70/30 stratified split")
print(f"  scratch @ 100% AUC = {f_auc:.4f}  ({f_n} training subjects)")

# --- Comparison ---
diff = abs(t_auc - f_auc)
print(f"\n[Δ] |AUC_transfer − AUC_fraction| = {diff:.4f}")
print()

if diff <= EXPECTED_DIFF_MAX:
    print(f"✓ EXPECTED — protocol difference ({diff:.4f} ≤ {EXPECTED_DIFF_MAX:.2f}).")
    print("  This is 5-fold CV vs single 70/30 split. Not a bug.")
    print("  Document both numbers in the paper with an explicit footnote.")
elif diff <= BUG_THRESHOLD:
    print(f"⚠ BORDERLINE — {diff:.4f} is larger than expected ({EXPECTED_DIFF_MAX:.2f})")
    print("  but below the bug threshold ({BUG_THRESHOLD:.2f}).")
    print("  Likely still a protocol difference but check fold assignments.")
else:
    print(f"✗ LIKELY BUG — {diff:.4f} > {BUG_THRESHOLD:.2f}")
    print("  Inspect feature ordering, label definitions, and class balance")
    print("  between the two scripts before proceeding.")

print("\n" + "=" * 72)
print("Protocol summary:")
print("  transfer_strategies.py → 5-fold CV, ~80% train / ~20% test per fold")
print("  fraction_experiment.py → single 70/30 split, fraction = % of train pool")
print("=" * 72)
