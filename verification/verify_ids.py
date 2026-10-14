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


# verify_ids.py
"""Verify subject IDs are preserved after extraction."""
import os
import pandas as pd

OUT_DIR = r"add_your_root_folder\out_v2"
FILES = {
    "Aurora": "aurora_unified_v2_fixed.csv",
    "PPG-BP": "ppgbp_unified_v2_fixed.csv",
    "PulseDB-Vital": "pulsedb_vital_unified_v2_fixed.csv",
    "DS4": "ds4_unified_v2_fixed.csv",
    "Physio-PPG": "physioppg_unified_v2_fixed.csv",
}

for name, fname in FILES.items():
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"[skip] {name}: {fname} not found")
        continue
    df = pd.read_csv(path)
    n_subj = df.subject_id.nunique()
    n_zero = (df.subject_id.astype(str) == "0").sum()
    print(f"{name:15s}  rows={len(df):5d}  subjects={n_subj:4d}  "
          f"zero_id_rows={n_zero}")
    assert n_zero == 0, f"{name}: {n_zero} rows have subject_id == '0'"
    assert n_subj > 1, f"{name}: only {n_subj} unique subjects"

print("\n✓ All subject IDs preserved.")