# physioppg_adapter.py
"""Physio-PPG adapter — corrected."""
import os
import numpy as np
import pandas as pd
from math import gcd
from scipy.signal import resample_poly

from missing_sample_utils import handle_missing_samples


ROOT = r"Physio-PPG"
FS_IN = 100.0
FS_OUT = 125.0


def _resample(x, fs_in=FS_IN, fs_out=FS_OUT):
    g = gcd(int(round(fs_in)), int(fs_out))
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    return resample_poly(x, int(fs_out) // g, int(round(fs_in)) // g)


def load_physioppg_segments(root=ROOT):
    signals = np.load(os.path.join(root, "signals.npy"))
    labels = pd.read_excel(os.path.join(root, "label.xlsx"))

    n_signals = len(signals)
    n_subjects = len(labels)
    per_subj = n_signals // n_subjects
    print(f"[adapter] {n_signals} signals, {n_subjects} subjects, "
          f"{per_subj} signals/subject")

    segments = []
    n_rejected = 0

    for i, row in labels.iterrows():
        sid = int(row["id"])
        height = float(row["height"])
        weight = float(row["weight (kg)"])
        bmi = float(row["BMI"])
        sbp = float(row["SBP"])
        dbp = float(row["DBP"])
        height_cm = height * 100 if height < 3 else height

        if not (50 <= sbp <= 260 and 30 <= dbp <= 150 and sbp > dbp):
            n_rejected += 1
            continue

        s_idx = i * per_subj
        e_idx = min(s_idx + per_subj, n_signals)

        for j in range(s_idx, e_idx):
            sig = signals[j]
            sig = handle_missing_samples(sig, max_gap_fraction=0.02)
            if sig is None:
                n_rejected += 1
                continue

            ppg = _resample(sig, FS_IN, FS_OUT)

            segments.append(dict(
                SubjectID=f"physio_{sid}",
                Age=30.0,
                SegSBP=sbp,
                SegDBP=dbp,
                PPG_F=ppg,
                ECG_F=None,
                BMI=bmi,
                Height=height_cm,
                Weight=weight,
                Gender="M",
                source_file=f"physio_{sid}_sig{j}",
            ))

    print(f"[adapter] kept={len(segments)}  rejected={n_rejected}")
    return segments