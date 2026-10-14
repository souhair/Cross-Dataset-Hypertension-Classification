# dataset4_adapter.py
"""
DS4 adapter (SWCS)
- Separate DC paths: PPG DC-removed, BP preserved
- Uses handle_missing_samples
- Rejects missing BP (no fallback invented labels)
- Preserves subject ID as string
"""
import os
import numpy as np
import pandas as pd
from math import gcd
from scipy.signal import resample_poly, find_peaks

from missing_sample_utils import handle_missing_samples


ROOT = r"Dataset 4"
FS_IN = 50.0
FS_OUT = 125.0
WINDOW_SEC = 30.0
PPG_COL = "PPG (mV)"
BP_COL = "Reference Blood Pressure (mmHg)"


def _resample_ppg(x, fs_in=FS_IN, fs_out=FS_OUT):
    g = gcd(int(round(fs_in)), int(fs_out))
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    return resample_poly(x, int(fs_out) // g, int(round(fs_in)) // g)


def _resample_bp(x, fs_in=FS_IN, fs_out=FS_OUT):
    g = gcd(int(round(fs_in)), int(fs_out))
    x = np.asarray(x, dtype=float)
    return resample_poly(x, int(fs_out) // g, int(round(fs_in)) // g)


def _load_demo():
    p = os.path.join(ROOT, "Participant_Health_Data.xlsx")
    if not os.path.exists(p):
        return {}
    df = pd.read_excel(p)
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str).str.strip()
    return df.set_index(id_col).to_dict(orient="index")


def _decode_gender(g):
    if isinstance(g, str):
        g = g.strip().upper()
        if g.startswith("M"):
            return "M"
        if g.startswith("F"):
            return "F"
    return "M"


def _pick(info, *keys, default=None):
    for k in keys:
        for actual in info:
            if actual.lower() == k.lower():
                return info[actual]
    return default


def load_dataset4_segments(root=ROOT, max_segments=None, min_samples=300):
    demo = _load_demo()
    print(f"[adapter] demographics loaded for {len(demo)} subjects")

    files = sorted([f for f in os.listdir(root) if f.endswith(".csv")])
    segments, n_rejected, n_gap, n_bp_missing = [], 0, 0, 0

    for fname in files:
        sid = fname.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(root, fname))
        except Exception:
            n_rejected += 1
            continue

        if PPG_COL not in df.columns or BP_COL not in df.columns:
            n_rejected += 1
            continue

        ppg = df[PPG_COL].to_numpy(dtype=float)
        bp = df[BP_COL].to_numpy(dtype=float)

        ppg = handle_missing_samples(ppg, max_gap_fraction=0.02)
        bp = handle_missing_samples(bp, max_gap_fraction=0.02)
        if ppg is None or bp is None:
            n_gap += 1
            continue
        if len(ppg) < min_samples:
            n_rejected += 1
            continue

        ppg_125 = _resample_ppg(ppg, FS_IN, FS_OUT)
        bp_125 = _resample_bp(bp, FS_IN, FS_OUT)

        win = int(WINDOW_SEC * FS_OUT)
        n_win = max(1, len(ppg_125) // win)

        info = demo.get(sid, {}) if demo else {}
        age = float(_pick(info, "age", default=30))
        gender = _decode_gender(_pick(info, "gender", "sex", default="M"))
        bmi = float(_pick(info, "bmi", default=25.0))
        height = 170.0
        weight = bmi * (height / 100.0) ** 2

        for w in range(n_win):
            s, e = w * win, (w + 1) * win
            seg_ppg = ppg_125[s:e]
            seg_bp = bp_125[s:e]
            if len(seg_ppg) < min_samples:
                continue

            try:
                peaks, _ = find_peaks(seg_bp, distance=int(0.4 * FS_OUT),
                                       prominence=1.0)
                troughs, _ = find_peaks(-seg_bp, distance=int(0.4 * FS_OUT),
                                         prominence=1.0)
            except Exception:
                peaks, troughs = np.array([]), np.array([])

            # Reject if peak detection fails — do NOT invent labels
            if len(peaks) < 2 or len(troughs) < 2:
                n_bp_missing += 1
                continue

            sbp = float(np.median(seg_bp[peaks]))
            dbp = float(np.median(seg_bp[troughs]))

            if not (50 <= sbp <= 260 and 30 <= dbp <= 150 and sbp > dbp):
                n_rejected += 1
                continue

            segments.append(dict(
                SubjectID=sid,
                Age=age,
                SegSBP=sbp,
                SegDBP=dbp,
                PPG_F=seg_ppg,
                ECG_F=None,
                BMI=bmi,
                Height=height,
                Weight=weight,
                Gender=gender,
                source_file=f"ds4_{sid}_w{w}",
            ))

            if max_segments and len(segments) >= max_segments:
                break
        if max_segments and len(segments) >= max_segments:
            break

    print(f"[adapter] kept={len(segments)}  "
          f"rejected={n_rejected}  gap_rejected={n_gap}  "
          f"bp_peak_failed={n_bp_missing}")
    return segments