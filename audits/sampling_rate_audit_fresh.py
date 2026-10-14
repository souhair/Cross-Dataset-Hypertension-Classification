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


# sampling_rate_audit_fresh.py
"""
Sampling-rate audit — fresh extraction with the canonicalizing extractor.

Loads raw Aurora TSVs at native 500 Hz. For each segment:
  - Extract with the patched extractor at fs=500 (canonicalization resamples
    internally to 125 Hz)
  - Downsample the SAME raw segment ourselves via resample_poly(up=1, down=4)
    and extract at fs=125 (canonicalization is a no-op since fs==125)

Since both paths feed identical 125 Hz data into the feature computation,
every feature should be identical to floating-point precision.

Writes: out_v2/results/v2_sampling_rate_audit_fresh.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from config import SEED
from adapters.aurora_tsv_adapter_fixed import load_aurora_segments
from extractors.ppg_feature_extractor import process_single_segment_unified

np.random.seed(SEED)

ROOT = r"C:\A- third year\Aurora"
RESULTS = r"C:\A- third year\FinalProj\aurora021020206\out_v2\results"
os.makedirs(RESULTS, exist_ok=True)

AUDIT_FEATURES = [
    "step_F_mean", "step_F_std", "step_Phi_mean", "step_Phi_std",
    "step_Psi_mean", "step_Psi_std",
    "period_mean__compliance_index", "period_mean__stiffness_index",
    "period_median__compliance_index", "period_median__stiffness_index",
    "deriv_max_velocity", "deriv_min_velocity",
    "deriv_max_acceleration", "deriv_min_acceleration",
    "spec_fundamental_freq", "spec_centroid", "spec_bandwidth",
    "spec_entropy", "spec_harmonic_h2_h1",
    "spec_harmonic_h3_h1", "spec_harmonic_h4_h1",
    "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
    "morph_skewness", "morph_kurtosis", "morph_perfusion_index",
    "timing_systolic_ms", "timing_diastolic_ms", "timing_ts_td_ratio",
    "wavelet_energy_1", "wavelet_energy_2",
]


def downsample_500_to_125(sig):
    """Match the canonicalization's resample_poly(up=1, down=4)."""
    return resample_poly(sig, up=1, down=4)


def main():
    print("=" * 72)
    print("FRESH sampling-rate audit — 500 Hz vs 125 Hz (canonicalized)")
    print("=" * 72)

    # 1. Load Aurora at native 500 Hz (adapter with target_fs=500 = no resample)
    print(f"[load] Aurora raw from {ROOT} at 500 Hz...")
    segs = load_aurora_segments(
        ROOT, protocol="auscultatory", exclude_calibration=True,
        quality_min=0.5, min_duration_s=10.0, target_fs=500)
    print(f"[load] {len(segs)} segments")

    # 2. One segment per subject, up to 460 subjects
    by_subj = {}
    for s in segs:
        pid = str(s.get("SubjectID", ""))
        if pid and pid not in by_subj:
            by_subj[pid] = s
    pids = list(by_subj.keys())
    rng = np.random.default_rng(SEED)
    n_take = min(460, len(pids))
    sel = rng.choice(pids, size=n_take, replace=False)
    sampled = [by_subj[p] for p in sel]
    print(f"[audit] {n_take} subjects (one segment each)")

    # 3. Extract at both rates
    rows = []
    ok, fail = 0, 0
    for i, seg in enumerate(sampled, 1):
        ppg_500 = np.asarray(seg["PPG_F"], dtype=float)

        # Path A: raw 500 Hz -> extractor at fs=500 (canonicalization fires)
        seg_a = dict(seg)
        seg_a["PPG_F"] = ppg_500
        out_a = process_single_segment_unified(
            seg_a, seg.get("source_file", f"hi_{i}"), fs=500)

        # Path B: pre-downsample -> extractor at fs=125 (no-op)
        ppg_125 = downsample_500_to_125(ppg_500)
        seg_b = dict(seg)
        seg_b["PPG_F"] = ppg_125
        out_b = process_single_segment_unified(
            seg_b, seg.get("source_file", f"lo_{i}"), fs=125)

        if out_a is None or out_b is None:
            fail += 1
            continue

        row = {"subject_id": seg.get("SubjectID", f"seg_{i}")}
        for f in AUDIT_FEATURES:
            v_hi = float(out_a["flat"].get(f, np.nan))
            v_lo = float(out_b["flat"].get(f, np.nan))
            if np.isfinite(v_hi) and np.isfinite(v_lo):
                if abs(v_hi) > 1e-12:
                    row[f + "__ratio"] = v_lo / v_hi
                else:
                    row[f + "__ratio"] = 1.0 if abs(v_lo) < 1e-12 else np.nan
                row[f + "__absdiff"] = abs(v_lo - v_hi)
            else:
                row[f + "__ratio"] = np.nan
                row[f + "__absdiff"] = np.nan
        rows.append(row)
        ok += 1
        if i % 50 == 0:
            print(f"  [{i:>4}/{n_take}] ok={ok} fail={fail}")

    if not rows:
        raise SystemExit("no usable segments")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "v2_sampling_rate_audit_fresh.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}  ({len(df)} rows, ok={ok}, fail={fail})")

    # 4. Summary
    print("\n" + "=" * 72)
    print("Per-feature ratio (125 Hz / 500 Hz)")
    print("=" * 72)
    summary = []
    for f in AUDIT_FEATURES:
        rc = f + "__ratio"
        if rc not in df.columns:
            summary.append({"feature": f, "n": 0, "ratio_mean": np.nan,
                            "ratio_std": np.nan, "max_absdiff": np.nan})
            continue
        v = df[rc].dropna().values
        d = df[f + "__absdiff"].dropna().values if (f + "__absdiff") in df.columns else np.array([])
        summary.append({
            "feature": f,
            "n": len(v),
            "ratio_mean": float(np.mean(v)) if len(v) else np.nan,
            "ratio_std":  float(np.std(v))  if len(v) else np.nan,
            "max_absdiff": float(np.max(d)) if len(d) else np.nan,
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(RESULTS, "v2_sampling_rate_audit_fresh_summary.csv"),
               index=False)
    print(sdf.round(8).to_string(index=False))

    # 5. Headline
    all_ratios, all_diffs = [], []
    for f in AUDIT_FEATURES:
        rc = f + "__ratio"
        dc = f + "__absdiff"
        if rc in df.columns:
            all_ratios.extend(df[rc].dropna().tolist())
        if dc in df.columns:
            all_diffs.extend(df[dc].dropna().tolist())
    all_ratios = np.array(all_ratios)
    all_diffs = np.array(all_diffs)
    print(f"\nOVERALL across {len(all_ratios)} feature×subject pairs:")
    print(f"  ratio mean:      {all_ratios.mean():.10f}")
    print(f"  ratio std:       {all_ratios.std():.10f}")
    print(f"  ratio min/max:   {all_ratios.min():.10f} / {all_ratios.max():.10f}")
    print(f"  max |Δ| feature: {all_diffs.max():.10e}")

    exact = int(np.sum(np.abs(all_ratios - 1.0) < 1e-9))
    print(f"\nPairs with ratio == 1.0 (|Δ|<1e-9): {exact}/{len(all_ratios)}")


if __name__ == "__main__":
    main()