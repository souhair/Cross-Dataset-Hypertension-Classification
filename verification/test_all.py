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


# test_all.py
"""Run all smoke tests after corrections."""
import numpy as np
import pandas as pd

print("=" * 72)
print("TEST 1 — Adapters")
print("=" * 72)

from adapters.aurora_tsv_adapter_fixed import load_aurora_segments
segs = load_aurora_segments(r"C:\A- third year\Aurora",
                              max_segments=20, target_fs=125)
ids = {s["SubjectID"] for s in segs}
assert all(isinstance(i, str) for i in ids)
assert all(s["SegSBP"] is not None and not np.isnan(s["SegSBP"]) for s in segs)
print(f"✓ Aurora: {len(segs)} seg, {len(ids)} subj: {sorted(ids)[:5]}")

from adapters.ppgbp_adapter import load_ppgbp_segments
pp = load_ppgbp_segments(
    r"add_your_root_folder\5459299\PPG-BP Database\Data File"
)
print(f"✓ PPG-BP: {len(pp)} seg")

from adapters.dataset4_adapter import load_dataset4_segments
ds4 = load_dataset4_segments(max_segments=10)
print(f"✓ DS4: {len(ds4)} seg")

print("\n" + "=" * 72)
print("TEST 2 — Extractor")
print("=" * 72)

from extractors.ppg_feature_extractor import (
    process_single_segment_unified, PERIOD_P_DEFAULT)

for seg in segs[:3]:
    out = process_single_segment_unified(seg, seg["source_file"],
                                          fs=125, period_p=PERIOD_P_DEFAULT)
    assert out is not None
    flat = out["flat"]
    assert isinstance(flat["subject_id"], str)
    assert flat["subject_id"] == seg["SubjectID"]
    print(f"✓ Extracted {seg['SubjectID']} → "
          f"gender={flat.get('demo_gender')}, "
          f"hrv_lf={flat.get('hrv_lf_power', 0):.4f}")

print("\n" + "=" * 72)
print("TEST 3 — Fold-aware preprocessor")
print("=" * 72)

from analysis.v2_utils import load_aurora_raw
from fold_preprocessor import FoldAwarePreprocessor
from sklearn.model_selection import StratifiedGroupKFold

# Only run if the CSV exists
import os
if os.path.exists(r"add_your_root_folder\out_v2"
                   r"\aurora_unified_v2_fixed.csv"):
    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    feature_lists = []
    for tr, te in sgkf.split(df[CANDIDATE].values, y, groups=g):
        pre = FoldAwarePreprocessor().fit(df.iloc[tr], CANDIDATE)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        assert X_tr.shape[1] == X_te.shape[1]
        assert not np.isnan(X_tr).any()
        assert not np.isnan(X_te).any()
        feature_lists.append(set(pre.feature_list_))
    common = feature_lists[0]
    for fl in feature_lists[1:]:
        common &= fl
    print(f"✓ Fold features: {[len(fl) for fl in feature_lists]} "
          f"(common={len(common)})")
else:
    print("SKIP — run extraction first")

print("\n" + "=" * 72)
print("ALL TESTS PASSED")
print("=" * 72)