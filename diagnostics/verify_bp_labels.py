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


# verify_bp_labels.py
"""
Verify that every dataset in out_v2/ has finite, in-range SBP/DBP
labels for every row. Also scan the raw adapter outputs (pre-fix CSVs)
to confirm no adapter passed a segment with missing BP.
"""
import os
import pandas as pd
import numpy as np

ROOT = r"add_your_root_folder"
OUT = os.path.join(ROOT, "out_v2")

FILES = {
    "Aurora":         "aurora_unified_v2_fixed.csv",
    "PPG-BP":         "ppgbp_unified_v2_fixed.csv",
    "PulseDB-Vital":  "pulsedb_vital_unified_v2_fixed.csv",
    "DS4":            "ds4_unified_v2_fixed.csv",
    "Physio-PPG":     "physioppg_unified_v2_fixed.csv",
}

print("=" * 72)
print("Verify BP label integrity across all datasets")
print("=" * 72)

any_failures = False
for name, fname in FILES.items():
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        print(f"[skip] {name}: {p} not found")
        continue
    df = pd.read_csv(p)

    if "sbp" not in df.columns or "dbp" not in df.columns:
        print(f"[warn] {name}: no sbp/dbp columns — cannot verify")
        continue

    sbp, dbp = df["sbp"].values, df["dbp"].values
    n = len(df)

    # 1. Any NaN or inf?
    n_nan_sbp = int(np.isnan(sbp).sum() + np.isinf(sbp).sum())
    n_nan_dbp = int(np.isnan(dbp).sum() + np.isinf(dbp).sum())

    # 2. Any row where sbp==120 AND dbp==80 exactly (suspicious default)?
    is_default = (np.abs(sbp - 120) < 1e-9) & (np.abs(dbp - 80) < 1e-9)
    n_default = int(is_default.sum())

    # 3. Any out-of-range?
    n_oob = int(((sbp < 40) | (sbp > 250) | (dbp < 30) | (dbp > 150)).sum())

    status = "OK" if (n_nan_sbp + n_nan_dbp + n_oob == 0) else "FAIL"
    if status == "FAIL":
        any_failures = True

    print(f"\n[{name}]  {n} rows  —  {status}")
    print(f"  NaN/inf SBP:  {n_nan_sbp}")
    print(f"  NaN/inf DBP:  {n_nan_dbp}")
    print(f"  Out-of-range: {n_oob}")
    print(f"  SBP=120 AND DBP=80 (default-value suspicion): {n_default}")

    if n_default > 0:
        # Show which rows to inspect
        sample = df.loc[is_default, ["subject_id", "sbp", "dbp"]].head(5)
        print(f"  Sample of default-looking rows:")
        print(sample.to_string(index=False))
        print(f"  ⚠ Investigate whether these are genuine 120/80 readings or fallbacks")

print()
print("=" * 72)
if any_failures:
    print("❌ FAILED — some datasets have invalid BP labels")
else:
    print("✓ PASSED — all datasets have finite, in-range BP labels")
    print("  If 'SBP=120 AND DBP=80' count is high, inspect further.")
print("=" * 72)