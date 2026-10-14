# aurora_tsv_adapter_fixed.py
"""
Aurora-BP TSV adapter with all corrections applied.
- Uses released sbp/dbp averages (not primary observer)
- Rejects missing BP (no artificial 120/80 default)
- Preserves subject ID as string
- Uses handle_missing_samples for gap interpolation
- DC removal before AND after resampling
"""
from math import gcd
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from missing_sample_utils import handle_missing_samples


FNAME_RE = re.compile(
    r"^(?P<pid>[a-z]\d+)\.(?P<phase>[^.]+)\.(?P<segname>.+)\.tsv$",
    re.IGNORECASE,
)


def _waveform_root(root, protocol):
    outer = os.path.join(root, f"measurements_{protocol}")
    nested = os.path.join(outer, f"measurements_{protocol}")
    for cand in (nested, outer):
        if not os.path.isdir(cand):
            continue
        subs = [s for s in os.listdir(cand)
                if os.path.isdir(os.path.join(cand, s))]
        if subs:
            return cand
    return outer


def _resolve_waveform_path(root, rel_path, protocol):
    if not isinstance(rel_path, str) or not rel_path:
        return None
    p = os.path.join(root, rel_path.replace("/", os.sep))
    if os.path.exists(p):
        return p
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        pid = parts[-2]
        fname = parts[-1]
        p2 = os.path.join(root, f"measurements_{protocol}",
                          f"measurements_{protocol}", pid, fname)
        if os.path.exists(p2):
            return p2
    return None


def _to_float(x, default=np.nan):
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except Exception:
        return default


def _decode_gender(g):
    if isinstance(g, str):
        g = g.strip().upper()
        if g.startswith("M"):
            return "M"
        if g.startswith("F"):
            return "F"
    return "M"


def _resample(x, fs_in=500, fs_out=125, dc_remove=True):
    x = np.asarray(x, dtype=float)
    if dc_remove:
        x = x - np.mean(x)
    if fs_out is None or fs_in == fs_out:
        return x
    g = gcd(int(fs_in), int(fs_out))
    y = resample_poly(x, int(fs_out) // g, int(fs_in) // g)
    if dc_remove:
        y = y - np.mean(y)
    return y


def load_aurora_segments(root,
                          protocol="auscultatory",
                          exclude_calibration=True,
                          quality_min=0.5,
                          min_duration_s=10.0,
                          max_segments=None,
                          source_fs=500,
                          target_fs=125,
                          flip_ppg=False,
                          min_samples=200):
    participants = pd.read_csv(
        os.path.join(root, "participants.tsv"), sep="\t")
    participants = participants.set_index("pid")

    meas = pd.read_csv(
        os.path.join(root, f"measurements_{protocol}.tsv"), sep="\t")

    meas = meas[meas["waveforms_generated"] == 1]
    meas = meas[meas["waveform_file_path"].notna()]
    if "optical_quality" in meas.columns:
        meas = meas[meas["optical_quality"] >= quality_min]
    if "duration" in meas.columns:
        meas = meas[meas["duration"] >= min_duration_s]
    if exclude_calibration:
        meas = meas[~meas["measurement"].str.contains(
            "calibration", case=False, na=False)]

    segments = []
    n_rejected = 0
    n_missing_bp = 0
    n_oor = 0
    n_gap = 0

    for _, row in meas.iterrows():
        wf_path = _resolve_waveform_path(
            root, row["waveform_file_path"], protocol)
        if wf_path is None:
            n_rejected += 1
            continue
        try:
            wf = pd.read_csv(wf_path, sep="\t")
        except Exception:
            n_rejected += 1
            continue
        if "optical" not in wf.columns:
            n_rejected += 1
            continue

        ppg = wf["optical"].to_numpy(dtype=float)
        if len(ppg) < min_samples:
            n_rejected += 1
            continue

        # Handle missing samples (preserves time)
        ppg = handle_missing_samples(ppg, max_gap_fraction=0.02)
        if ppg is None:
            n_gap += 1
            continue

        if flip_ppg:
            ppg = -ppg

        ppg = _resample(ppg, fs_in=source_fs, fs_out=target_fs,
                        dc_remove=True)

        ecg = None
        if "ekg" in wf.columns:
            e = wf["ekg"].to_numpy(dtype=float)
            if len(e) == len(wf["optical"]):
                e = handle_missing_samples(e, max_gap_fraction=0.02)
                if e is not None:
                    ecg = _resample(e, fs_in=source_fs, fs_out=target_fs,
                                    dc_remove=True)

        pid = str(row["pid"])
        meta = participants.loc[pid] if pid in participants.index else None

        age = _to_float(meta["age"], 30.0) if meta is not None else 30.0
        height_in = _to_float(meta["height"], 68.0) if meta is not None else 68.0
        weight_lb = _to_float(meta["weight"], 150.0) if meta is not None else 150.0
        gender = (_decode_gender(meta["gender"])
                  if (meta is not None and "gender" in meta) else "M")

        height_cm = height_in * 2.54
        weight_kg = weight_lb * 0.453592
        bmi = (weight_kg / (height_cm / 100.0) ** 2) if height_cm > 0 else 25.0

        # BP labels: use released averages (sbp/dbp), not primary observer
        if protocol == "auscultatory":
            sbp = _to_float(row.get("sbp"), np.nan)
            dbp = _to_float(row.get("dbp"), np.nan)
        else:
            sbp = _to_float(row.get("sbp"), np.nan)
            dbp = _to_float(row.get("dbp"), np.nan)

        if np.isnan(sbp) or np.isnan(dbp):
            n_missing_bp += 1
            continue
        if not (60 <= sbp <= 260 and 30 <= dbp <= 150):
            n_oor += 1
            continue

        segments.append(dict(
            SubjectID=pid,
            Age=age,
            SegSBP=sbp,
            SegDBP=dbp,
            PPG_F=ppg,
            ECG_F=ecg,
            BMI=bmi,
            Height=height_cm,
            Weight=weight_kg,
            Gender=gender,
            protocol=protocol,
            phase=str(row.get("phase", "")),
            measurement=str(row.get("measurement", "")),
            optical_quality=float(row.get("optical_quality", 0.0)),
            source_file=os.path.basename(wf_path),
        ))

        if max_segments and len(segments) >= max_segments:
            break

    print(f"[adapter] protocol={protocol}  source_fs={source_fs}  "
          f"target_fs={target_fs}  kept={len(segments)}  "
          f"rejected={n_rejected}  missing_bp={n_missing_bp}  "
          f"oor_bp={n_oor}  gap_rejected={n_gap}")
    return segments