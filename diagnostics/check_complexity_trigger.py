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


# check_complexity_trigger.py
"""Verify whether the complexity decimation branch fires for any
production Aurora segment (>5000 samples at 125 Hz)."""
import h5py
import numpy as np
import os

H5 = r"add_your_root_folder\out_v2\aurora_unified_v2_sequences.h5"

if not os.path.exists(H5):
    raise SystemExit(f"Missing: {H5}")

lengths = []
with h5py.File(H5, "r") as f:
    seg_keys = sorted(k for k in f.keys() if k.startswith("segment_"))
    for k in seg_keys:
        # Try to read the raw signal length from the segment
        try:
            # The HDF5 stores beat_features / period_features, not the raw PPG.
            # But metadata has 'n_periods' and seq_length. We can use seq_length
            # as a proxy for the number of periods, not samples.
            pass
        except Exception:
            continue
    print(f"Scanned {len(seg_keys)} segments in {H5}")
    print("HDF5 does not store raw signal length directly.")
    print()

# Check from the flat CSV: count segments with n_periods > some threshold,
# but that doesn't tell us samples either.
# The definitive check: read one Aurora TSV, downsample, look at length.

print("Checking Aurora raw TSV lengths directly...")
import pandas as pd
from scipy.signal import resample_poly

AUR_ROOT = r"Aurora\measurements_auscultatory\measurements_auscultatory"
n_long = 0
n_short = 0
max_len = 0
sample_lens = []

subjects = [d for d in os.listdir(AUR_ROOT)
            if os.path.isdir(os.path.join(AUR_ROOT, d))][:50]  # sample 50 subjects

for subj in subjects:
    subj_dir = os.path.join(AUR_ROOT, subj)
    for fname in os.listdir(subj_dir):
        if not fname.endswith(".tsv"):
            continue
        try:
            raw = pd.read_csv(os.path.join(subj_dir, fname), sep="\t")["optical"].to_numpy(float)
            # Downsample 500 -> 125
            ds = resample_poly(raw, 1, 4)
            L = len(ds)
            sample_lens.append(L)
            if L > 5000:
                n_long += 1
            else:
                n_short += 1
            max_len = max(max_len, L)
        except Exception:
            continue
    if len(sample_lens) > 500:  # stop after enough samples
        break

print(f"Sampled {len(sample_lens)} Aurora segments (at 125 Hz):")
print(f"  Lengths > 5000 (branch fires):  {n_long}")
print(f"  Lengths <= 5000 (no decimation): {n_short}")
print(f"  Max length:  {max_len} samples")
print(f"  Median length: {int(np.median(sample_lens))} samples")