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


# v2_fix_subject_ids.py
"""Recover subject_id from source_file if the extractor lost it.
(Only needed if safe_extract_patient_id was not applied.)"""
import os
import re
import pandas as pd

OUT_DIR = r"add_your_root_folder\out_v2"

JOBS = [
    ("aurora_unified_v2.csv", "aurora_unified_v2_fixed.csv",
        r"^([a-z]\d+)"),
    ("ppgbp_unified_v2.csv", "ppgbp_unified_v2_fixed.csv",
        r"^ppgbp_(.+?)_seg\d+$"),
    ("pulsedb_vital_unified_v2.csv", "pulsedb_vital_unified_v2_fixed.csv",
        r"^p?(\d+)\.mat$"),
    ("ds4_unified_v2.csv", "ds4_unified_v2_fixed.csv",
        r"^ds4_([A-Za-z0-9]+)_w\d+$"),
    ("physioppg_unified_v2.csv", "physioppg_unified_v2_fixed.csv",
        r"^physio_(\d+)_sig\d+$"),
]


for src, dst, pattern in JOBS:
    src_path = os.path.join(OUT_DIR, src)
    if not os.path.exists(src_path):
        print(f"[skip] {src} not found")
        continue
    df = pd.read_csv(src_path)
    df["subject_id_old"] = df["subject_id"]
    extracted = df["source_file"].astype(str).str.extract(pattern,
                                                            expand=True).iloc[:, 0]
    n_missing = int(extracted.isna().sum())
    if n_missing > 0:
        print(f"[warn] {src}: {n_missing} unmatched rows")
    df["subject_id"] = extracted.fillna(df["subject_id_old"]).astype(str)
    df.to_csv(os.path.join(OUT_DIR, dst), index=False)
    print(f"[fixed] {src} -> {dst}  "
          f"(subjects={df['subject_id'].nunique()}, rows={len(df)})")