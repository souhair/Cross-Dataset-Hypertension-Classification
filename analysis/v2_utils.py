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


# v2_utils.py
"""Shared helpers for all v2 downstream scripts."""
import os
import numpy as np
import pandas as pd

OUT_V2 = r"add_your_root_folder\out_v2"
RESULTS = os.path.join(OUT_V2, "results")
os.makedirs(RESULTS, exist_ok=True)

CSV = {
    "aurora": os.path.join(OUT_V2, "aurora_unified_v2_fixed.csv"),
    "ppgbp": os.path.join(OUT_V2, "ppgbp_unified_v2_fixed.csv"),
    "pulsedb_vital": os.path.join(OUT_V2, "pulsedb_vital_unified_v2_fixed.csv"),
    "ds4": os.path.join(OUT_V2, "ds4_unified_v2_fixed.csv"),
    "physioppg": os.path.join(OUT_V2, "physioppg_unified_v2_fixed.csv"),
}

LEAK = {
    "subject_id", "source_file", "segment_id", "subject_id_old",
    "sbp", "dbp", "htn", "htn_bin", "htn3",
    "ptt_r_to_foot_ms", "ptt_r_to_peak_ms", "ptt_btb_std_ms", "ptt_btb_cv",
    "pep_ms", "lvet_ms", "pep_lvet_ratio", "pep_variability", "lvet_variability",
    "pwv_surrogate", "morph_perfusion_index", "htn_perfusion_index",
}


def label_3c(s, d):
    if s < 120 and d < 80:
        return 0
    if s >= 140 or d >= 90:
        return 2
    return 1


def label_bin(s, d):
    return int((s >= 130) or (d >= 80))


def load_aurora_raw():
    """Load Aurora without any imputation or feature selection."""
    df = pd.read_csv(CSV["aurora"])
    df["htn3"] = [label_3c(s, d) for s, d in zip(df.sbp, df.dbp)]
    df["htn_bin"] = [label_bin(s, d) for s, d in zip(df.sbp, df.dbp)]
    candidate_features = [
        c for c in df.columns
        if c not in LEAK and df[c].dtype != object
    ]
    return df, candidate_features


def load_target_csv(path, FEATURES, medians_from_aurora):
    """Load external CSV, align to FEATURES, impute with Aurora medians."""
    df = pd.read_csv(path)
    df["htn3"] = [label_3c(s, d) for s, d in zip(df.sbp, df.dbp)]
    df["htn_bin"] = [label_bin(s, d) for s, d in zip(df.sbp, df.dbp)]
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = medians_from_aurora.get(c, np.nan)
        if df[c].isna().any():
            df[c] = df[c].fillna(medians_from_aurora.get(c, np.nan))
    return df


def aurora_medians(df, FEATURES):
    return {c: float(df[c].median()) if df[c].notna().any() else 0.0
            for c in FEATURES}


def save_oof_predictions(records, output_path):
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Saved {len(records)} OOF predictions to {output_path}")