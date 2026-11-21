# audits/cohort_breakdown.py
"""
Cohort-level breakdown for cross-dataset comparability.
  - cohort-specific class distributions
  - feature missingness
  - extent of demographic imputation

This script reports those three things for all four cohorts, plus the
recording and segment-duration context that motivates the preprocessing
differences described in §3.2.

Outputs:
  out_v2/results/v2_cohort_breakdown.csv          — per-cohort summary
  out_v2/results/v2_cohort_breakdown_missing.csv  — per-family missingness
  out_v2/results/v2_cohort_breakdown_demographic.csv — per-demographic provenance
"""
import os
import numpy as np
import pandas as pd

from analysis.v2_utils import CSV, LEAK, RESULTS
from feature_families import family_of


# ============================================================================
# Cohort registry
# ============================================================================
COHORTS = [
    ("Aurora-BP",     "aurora",        500,  125, "auscultatory laboratory"),
    ("PPG-BP",        "ppgbp",        1000,  125, "Omron HEM-7201 cuff"),
    ("PulseDB-Vital", "pulsedb_vital", 125,  125, "invasive arterial pressure"),
    ("SWCS",          "ds4",            50,  125, "CNAP Monitor 500"),
]

# Feature families for the missingness breakdown
FAMILIES = [
    "period morphology", "SMD", "morphology", "spectral+nonlinear",
    "HRV", "ppg variability", "HTN proxies", "demographics", "other",
]

# ---- What we know about demographic imputation per cohort ----------------
# This is documented from the adapters in the shipped code. Update if the
# adapters change.
DEMOGRAPHIC_PROVENANCE = {
    "Aurora-BP": {
        "age":    "original (participants.tsv)",
        "gender": "original (participants.tsv)",
        "height": "imputed to 170 cm when missing",
        "weight": "imputed to 70 kg when missing",
        "bmi":    "computed from weight / (height/100)^2",
    },
    "PPG-BP": {
        "age":    "original (PPG-BP dataset.xlsx)",
        "gender": "original",
        "height": "original",
        "weight": "original",
        "bmi":    "original",
    },
    "PulseDB-Vital": {
        "age":    "original (per-subject)",
        "gender": "original (per-subject)",
        "height": "original",
        "weight": "original",
        "bmi":    "original",
    },
    "SWCS": {
        "age":    "original (per-subject)",
        "gender": "original (per-subject)",
        "height": "assumed 170 cm for all subjects (no height in source)",
        "weight": "reconstructed from BMI × (170/100)^2",
        "bmi":    "original (per-subject)",
    },
}


# ============================================================================
# Loading
# ============================================================================
def _safe_load(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    subj = "subject_id" if "subject_id" in df.columns else "SubjectID"
    return df, subj


def _derive_htn3(df):
    """Derive JNC7 3-class label from sbp/dbp."""
    if "sbp" not in df.columns or "dbp" not in df.columns:
        return None
    sbp = df["sbp"].values
    dbp = df["dbp"].values
    out = np.full(len(df), np.nan)
    valid = np.isfinite(sbp) & np.isfinite(dbp)
    # Normal
    out[valid & (sbp < 120) & (dbp < 80)] = 0
    # HTN
    out[valid & ((sbp >= 140) | (dbp >= 90))] = 2
    # PreHTN (everything else in-range)
    remaining = valid & np.isnan(out)
    out[remaining] = 1
    return out


def _class_distribution(df):
    htn3 = df["htn3"].values if "htn3" in df.columns else _derive_htn3(df)
    if htn3 is None:
        return {"Normal": 0, "PreHTN": 0, "HTN": 0}
    counts = pd.Series(htn3).value_counts().sort_index()
    return {
        "Normal": int(counts.get(0.0, 0)),
        "PreHTN": int(counts.get(1.0, 0)),
        "HTN":    int(counts.get(2.0, 0)),
    }


def _subjects_per_class(df, subj_col):
    htn3 = df["htn3"].values if "htn3" in df.columns else _derive_htn3(df)
    if htn3 is None:
        return {"Normal": 0, "PreHTN": 0, "HTN": 0}
    df = df.copy()
    df["_htn3"] = htn3
    df = df.dropna(subset=["_htn3"])
    subj_label = df.groupby(subj_col)["_htn3"].apply(
        lambda x: int(np.bincount(x.astype(int), minlength=3).argmax()))
    counts = subj_label.value_counts().sort_index()
    return {
        "Normal": int(counts.get(0, 0)),
        "PreHTN": int(counts.get(1, 0)),
        "HTN":    int(counts.get(2, 0)),
    }


def _segment_duration_stats(df):
    """If 'n_periods' or 'pulse_rate' available, derive duration proxies."""
    out = {}
    if "n_periods" in df.columns:
        out["n_periods_median"] = float(df["n_periods"].median())
        out["n_periods_p10"] = float(df["n_periods"].quantile(0.10))
        out["n_periods_p90"] = float(df["n_periods"].quantile(0.90))
    return out


# ============================================================================
# Missingness by family
# ============================================================================
def _family_missingness(df, subj_col):
    """Return per-family missingness and zero-fraction."""
    meta = {"subject_id", "SubjectID", "segment_id", "source_file",
            "sbp", "dbp", "htn3", "htn_bin", "htn",
            "pulse_rate", "signal_quality", "n_beats", "n_periods", "age",
            "seg_index", "quality_score"} | set(LEAK)
    feats = [c for c in df.columns
             if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

    fam_stats = {f: {"n": 0, "n_nan": 0, "n_zero": 0, "n_total": 0}
                 for f in FAMILIES}

    for c in feats:
        fam = family_of(c)
        if fam not in fam_stats:
            continue
        v = df[c].values
        fam_stats[fam]["n"] += 1
        fam_stats[fam]["n_nan"] += int(np.isnan(v).sum())
        fam_stats[fam]["n_zero"] += int((v == 0).sum())
        fam_stats[fam]["n_total"] += int(len(v))

    rows = []
    for fam in FAMILIES:
        s = fam_stats[fam]
        if s["n"] == 0:
            continue
        rows.append({
            "family": fam,
            "n_features": s["n"],
            "nan_fraction": s["n_nan"] / max(1, s["n_total"]),
            "zero_fraction": s["n_zero"] / max(1, s["n_total"]),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 78)
    print("Cohort breakdown — for cross-dataset comparability (Reviewer 1, point 3)")
    print("=" * 78)

    cohort_rows = []
    missingness_frames = []

    for name, key, fs_native, fs_target, ref in COHORTS:
        path = CSV.get(key)
        loaded = _safe_load(path)
        if loaded is None:
            print(f"[skip] {name}: {path} not found")
            continue
        df, subj = loaded

        # Class distribution
        seg_counts = _class_distribution(df)
        subj_counts = _subjects_per_class(df, subj)

        # Duration stats
        dur_stats = _segment_duration_stats(df)

        # Missingness by family
        miss = _family_missingness(df, subj)
        miss.insert(0, "cohort", name)
        missingness_frames.append(miss)

        # Aggregate
        row = {
            "cohort": name,
            "subjects": int(df[subj].nunique()),
            "segments": int(len(df)),
            "median_segments_per_subject": float(
                df.groupby(subj).size().median()),
            "native_fs_hz": fs_native,
            "working_fs_hz": fs_target,
            "reference_bp": ref,
            "segments_Normal": seg_counts["Normal"],
            "segments_PreHTN": seg_counts["PreHTN"],
            "segments_HTN":    seg_counts["HTN"],
            "subjects_Normal": subj_counts["Normal"],
            "subjects_PreHTN": subj_counts["PreHTN"],
            "subjects_HTN":    subj_counts["HTN"],
        }
        row.update(dur_stats)
        cohort_rows.append(row)

        print(f"\n[{name}]")
        print(f"  subjects: {row['subjects']:4d}    segments: {row['segments']:5d}")
        print(f"  native fs: {fs_native:4d} Hz    working fs: {fs_target} Hz")
        print(f"  class distribution (subjects):  "
              f"Normal={subj_counts['Normal']:3d}  "
              f"PreHTN={subj_counts['PreHTN']:3d}  "
              f"HTN={subj_counts['HTN']:3d}")
        print(f"  class distribution (segments):  "
              f"Normal={seg_counts['Normal']:4d}  "
              f"PreHTN={seg_counts['PreHTN']:4d}  "
              f"HTN={seg_counts['HTN']:4d}")
        if "n_periods_median" in row:
            print(f"  periods per segment:  median={row['n_periods_median']:.1f}  "
                  f"p10={row['n_periods_p10']:.1f}  p90={row['n_periods_p90']:.1f}")

    # ---- Save the cohort summary ----------------------------------------
    df_cohorts = pd.DataFrame(cohort_rows)
    out_cohorts = os.path.join(RESULTS, "v2_cohort_breakdown.csv")
    df_cohorts.to_csv(out_cohorts, index=False)
    print(f"\n[saved] {out_cohorts}")

    # ---- Save the missingness breakdown ---------------------------------
    if missingness_frames:
        df_miss = pd.concat(missingness_frames, ignore_index=True)
        out_miss = os.path.join(RESULTS, "v2_cohort_breakdown_missing.csv")
        df_miss.to_csv(out_miss, index=False)
        print(f"[saved] {out_miss}")

        print("\n" + "=" * 78)
        print("Missingness by family (NaN fraction, zero fraction)")
        print("=" * 78)
        pivot_nan = df_miss.pivot(index="family", columns="cohort",
                                    values="nan_fraction").round(3)
        print("\nNaN fraction:")
        print(pivot_nan.to_string())

        pivot_zero = df_miss.pivot(index="family", columns="cohort",
                                     values="zero_fraction").round(3)
        print("\nZero-value fraction:")
        print(pivot_zero.to_string())

    # ---- Save demographic provenance -----------------------------------
    prov_rows = []
    for cohort, features in DEMOGRAPHIC_PROVENANCE.items():
        for feat, status in features.items():
            prov_rows.append({"cohort": cohort,
                              "feature": feat,
                              "provenance": status})
    df_prov = pd.DataFrame(prov_rows)
    out_prov = os.path.join(RESULTS, "v2_cohort_breakdown_demographic.csv")
    df_prov.to_csv(out_prov, index=False)
    print(f"\n[saved] {out_prov}")

    # ---- Print a paper-ready summary -----------------------------------
    print("\n" + "=" * 78)
    print("PAPER-READY SUPPLEMENTARY TABLE")
    print("=" * 78)
    print(df_cohorts.to_string(index=False))

    # ---- Notes for the paper text --------------------------------------
    print("\n" + "=" * 78)
    print("HIGHLIGHTS FOR THE PAPER")
    print("=" * 78)
    for _, row in df_cohorts.iterrows():
        name = row["cohort"]
        n_htn = row["subjects_HTN"]
        n_pre = row["subjects_PreHTN"]
        n_norm = row["subjects_Normal"]
        total = row["subjects"]
        print(f"\n{name}:")
        print(f"  Class balance (subjects): Normal {100*n_norm/total:.0f}% / "
              f"PreHTN {100*n_pre/total:.0f}% / HTN {100*n_htn/total:.0f}%")
        print(f"  Native-to-working rate change: {row['native_fs_hz']} → "
              f"{row['working_fs_hz']} Hz "
              f"({'×' + str(row['working_fs_hz']/row['native_fs_hz']) if row['working_fs_hz'] != row['native_fs_hz'] else 'no change'})")


if __name__ == "__main__":
    main()
