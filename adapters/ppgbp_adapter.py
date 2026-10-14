# ppgbp_adapter.py
"""
PPG-BP (Guilin) adapter — corrected.
- Processes segments independently (no artificial concatenation)
- Uses handle_missing_samples
- Resamples 1000 -> 125 Hz
- Preserves subject ID as string
"""
import os
import re
import numpy as np
import pandas as pd
from math import gcd
from scipy.signal import resample_poly

from missing_sample_utils import handle_missing_samples


FNAME_RE = re.compile(r"^(\d+)_(\d+)\.txt$")


def _downsample(x, fs_in=1000, fs_out=125):
    if fs_in == fs_out:
        return x
    g = gcd(int(fs_in), int(fs_out))
    x = np.asarray(x, dtype=float)
    dc = float(np.mean(x))
    return resample_poly(x - dc, int(fs_out) // g, int(fs_in) // g)


def _load_waveform(path):
    try:
        sig = np.loadtxt(path)
    except Exception:
        sig = pd.read_csv(path, sep=r"\s+", header=None).values.astype(float)
    return np.asarray(sig, dtype=float).flatten()


def _load_metadata(xlsx_path):
    df = pd.read_excel(xlsx_path, header=1)
    df = df.rename(columns={
        "subject_ID": "subject_id",
        "Sex(M/F)": "gender",
        "Age(year)": "age",
        "Height(cm)": "height",
        "Weight(kg)": "weight",
        "Systolic Blood Pressure(mmHg)": "sbp",
        "Diastolic Blood Pressure(mmHg)": "dbp",
        "Heart Rate(b/m)": "heart_rate",
        "BMI(kg/m^2)": "bmi",
        "Hypertension": "htn_label_str",
    })
    df = df[df["subject_id"].notna()].copy()
    df["subject_id"] = df["subject_id"].astype(int)
    return df.set_index("subject_id")


def _decode_gender(g):
    if isinstance(g, str):
        g = g.strip().lower()
        if g.startswith("m"):
            return "M"
        if g.startswith("f"):
            return "F"
    return "M"


def load_ppgbp_segments(root, target_fs=125, concat=True,
                          join_boundary_margin_sec=0.5):
    """
    Load PPG-BP segments.

    Each subject has 3 short clips (~2.1 s each). They are concatenated
    into a single ~6.3 s recording to provide enough beats for the extractor.
    The `join_boundary_margin_sec` parameter defines a margin around each
    join where beats are dropped, so that artificial transitions do not
    contaminate detected peaks.
    """
    xlsx = os.path.join(root, "PPG-BP dataset.xlsx")
    wf_dir = os.path.join(root, "0_subject")
    if not os.path.exists(xlsx):
        raise FileNotFoundError(xlsx)
    if not os.path.isdir(wf_dir):
        raise FileNotFoundError(wf_dir)

    meta = _load_metadata(xlsx)
    print(f"[adapter] metadata subjects: {len(meta)}")

    by_subj = {}
    for fname in sorted(os.listdir(wf_dir)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        sid = int(m.group(1))
        seg = int(m.group(2))
        by_subj.setdefault(sid, []).append(
            (seg, os.path.join(wf_dir, fname)))

    print(f"[adapter] subject folders with waveforms: {len(by_subj)}")

    segments = []
    n_missing_meta = 0
    n_rejected = 0
    n_gap = 0

    for sid, items in sorted(by_subj.items()):
        if sid not in meta.index:
            n_missing_meta += 1
            continue
        row = meta.loc[sid]

        # Sort clips by segment number
        items.sort(key=lambda x: x[0])
        raw_clips = []
        for _, path in items:
            sig = _load_waveform(path)
            if len(sig) < 50:
                continue
            sig = handle_missing_samples(sig, max_gap_fraction=0.05)
            if sig is None:
                continue
            raw_clips.append(sig)

        if len(raw_clips) == 0:
            n_rejected += 1
            continue

        if concat and len(raw_clips) > 1:
            # Record the join indices in raw samples
            join_indices_raw = []
            offset = 0
            for clip in raw_clips[:-1]:
                offset += len(clip)
                join_indices_raw.append(offset)
            # Concatenate
            sig_combined = np.concatenate(raw_clips)
        else:
            sig_combined = raw_clips[0]
            join_indices_raw = []

        # Downsample 1000 -> 125 Hz
        ppg = _downsample(sig_combined, fs_in=1000, fs_out=target_fs)

        # Convert join indices to the target rate
        if join_indices_raw:
            ratio = target_fs / 1000.0
            join_indices = [int(round(j * ratio))
                            for j in join_indices_raw]
        else:
            join_indices = []

        # Build a rejection mask around joins
        n = len(ppg)
        reject = np.zeros(n, dtype=bool)
        margin = int(round(join_boundary_margin_sec * target_fs))
        for j in join_indices:
            lo = max(0, j - margin)
            hi = min(n, j + margin)
            reject[lo:hi] = True

        # Metadata
        height_cm = (float(row.get("height", 170))
                     if pd.notna(row.get("height")) else 170.0)
        weight_kg = (float(row.get("weight", 70))
                     if pd.notna(row.get("weight")) else 70.0)
        bmi_val = row.get("bmi")
        bmi = (float(bmi_val) if pd.notna(bmi_val) else
                (weight_kg / (height_cm / 100.0) ** 2
                 if height_cm > 0 else 25.0))

        sbp = float(row.get("sbp", np.nan))
        dbp = float(row.get("dbp", np.nan))
        if np.isnan(sbp) or np.isnan(dbp):
            n_rejected += 1
            continue
        if not (60 <= sbp <= 260 and 30 <= dbp <= 150):
            n_rejected += 1
            continue

        segments.append(dict(
            SubjectID=f"ppgbp_{sid}",
            Age=(float(row.get("age", 30))
                 if pd.notna(row.get("age")) else 30.0),
            SegSBP=sbp,
            SegDBP=dbp,
            PPG_F=ppg,
            ECG_F=None,
            BMI=bmi,
            Height=height_cm,
            Weight=weight_kg,
            Gender=_decode_gender(row.get("gender", "M")),
            source_file=f"ppgbp_{sid}",
            htn_label_str=str(row.get("htn_label_str", "")),
            join_boundary_mask=reject,   # Passed to extractor if it supports it
            join_indices=join_indices,
        ))

    print(f"[adapter] kept={len(segments)}  "
          f"missing_meta={n_missing_meta}  rejected={n_rejected}")
    return segments