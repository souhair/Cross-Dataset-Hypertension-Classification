# audits/feature_stability.py
"""
Cross-cohort feature stability analysis.

This script answers the second question directly and provides evidence
on the first:

  1. Loads the four fixed CSVs (Aurora, PPG-BP, PulseDB-Vital, SWCS).
  2. Aggregates each feature to subject-level medians (one row per subject).
  3. For each feature present in all four cohorts, computes three
     pairwise distances across all 6 cohort pairs:
         - KS statistic      (max CDF difference, robust, 0-1)
         - Cohen's d         (standardized mean difference)
         - Std. Wasserstein  (earth-mover distance, scale-normalized)
  4. Ranks features by mean KS (lower = more stable).
  5. Aggregates by feature family.
  6. Cross-references with SHAP importance to estimate each feature's
     contribution to cross-dataset degradation (KS × SHAP).
  7. Also reports the source-to-target subset (Aurora ↔ each target),
     since that is what transfer depends on.

Outputs:
  out_v2/results/v2_feature_stability.csv
  out_v2/results/v2_feature_stability_summary.csv
  out_v2/results/figures/fig15_feature_stability.png
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from itertools import combinations

from analysis.v2_utils import CSV, LEAK, RESULTS
from feature_families import family_of

FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

# -- Cohort registry --------------------------------------------------------
COHORTS = [
    ("Aurora-BP",     CSV.get("aurora")),
    ("PPG-BP",        CSV.get("ppgbp")),
    ("PulseDB-Vital", CSV.get("pulsedb_vital")),
    ("SWCS",          CSV.get("ds4")),
]

META_COLS = {
    "subject_id", "SubjectID", "segment_id", "source_file",
    "sbp", "dbp", "htn3", "htn_bin", "htn",
    "pulse_rate", "signal_quality",  "n_periods", # "n_beats", "age"
} | set(LEAK)


# -- Helpers ----------------------------------------------------------------
def load_subject_medians(path):
    """Return DataFrame: rows = subjects, columns = numeric features."""
    df = pd.read_csv(path)
    subj = "subject_id" if "subject_id" in df.columns else "SubjectID"
    feats = [c for c in df.columns
             if c not in META_COLS
             and pd.api.types.is_numeric_dtype(df[c])]
    df_feat = df[[subj] + feats].copy()
    df_feat[feats] = df_feat[feats].replace([np.inf, -np.inf], np.nan)
    return df_feat.groupby(subj)[feats].median()


def pairwise_metrics(x, y, min_n=10):
    """Return (KS, |Cohen's d|, std. Wasserstein) between two 1-D arrays."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if len(x) < min_n or len(y) < min_n:
        return np.nan, np.nan, np.nan

    ks = float(ks_2samp(x, y).statistic)

    mx, my = float(np.mean(x)), float(np.mean(y))
    sx = float(np.std(x, ddof=1)); sy = float(np.std(y, ddof=1))
    pooled = np.sqrt((sx ** 2 + sy ** 2) / 2.0)
    if pooled < 1e-10:
        d = 0.0 if abs(mx - my) < 1e-10 else 10.0
    else:
        d = abs(mx - my) / pooled

    all_v = np.concatenate([x, y])
    mu, sd = float(np.mean(all_v)), float(np.std(all_v))
    if sd < 1e-10:
        wass = 0.0
    else:
        wass = float(wasserstein_distance((x - mu) / sd, (y - mu) / sd))

    return ks, d, wass


# -- Main -------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Cross-cohort feature stability analysis")
    print("=" * 72)

    # 1. Load subject-level medians
    cohort_data = {}
    for name, path in COHORTS:
        if path is None or not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        med = load_subject_medians(path)
        cohort_data[name] = med
        print(f"[load] {name:15s}  {med.shape[0]:4d} subjects  "
              f"{med.shape[1]:3d} features")

    if len(cohort_data) < 2:
        raise SystemExit("[error] need at least 2 cohorts")

    # 2. Intersect on common feature columns
    common = set(next(iter(cohort_data.values())).columns)
    for med in cohort_data.values():
        common &= set(med.columns)
    common = sorted(common)
    print(f"\n[intersect] {len(common)} features present in all cohorts\n")

    # 3. Pairwise metrics
    cohort_names = list(cohort_data.keys())
    pairs = list(combinations(cohort_names, 2))
    source = "Aurora-BP"

    rows = []
    for feat in common:
        row = {"feature": feat}
        ks_vals, d_vals, wass_vals = [], [], []
        ks_st = []

        for a, b in pairs:
            ks, d, wass = pairwise_metrics(
                cohort_data[a][feat].values, cohort_data[b][feat].values)
            key = f"{a}__vs__{b}"
            row[f"ks_{key}"] = ks
            row[f"d_{key}"] = d
            row[f"wass_{key}"] = wass
            if np.isfinite(ks):
                ks_vals.append(ks)
                if source in (a, b):
                    ks_st.append(ks)
            if np.isfinite(d):
                d_vals.append(d)
            if np.isfinite(wass):
                wass_vals.append(wass)

        row["ks_mean"] = float(np.mean(ks_vals)) if ks_vals else np.nan
        row["abs_d_mean"] = float(np.mean(d_vals)) if d_vals else np.nan
        row["wass_mean"] = float(np.mean(wass_vals)) if wass_vals else np.nan
        row["ks_source_target"] = float(np.mean(ks_st)) if ks_st else np.nan
        row["n_pairs"] = len(ks_vals)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 4. Family assignment
    df["family"] = df["feature"].map(family_of)

    # 5. Cross-reference SHAP
    shap_npy = os.path.join(RESULTS, "v2_shap_mean_abs.npy")
    shap_txt = os.path.join(RESULTS, "v2_feature_names.txt")
    if os.path.exists(shap_npy) and os.path.exists(shap_txt):
        shap = np.load(shap_npy)
        with open(shap_txt, "r", encoding="utf-8") as f:
            shap_names = [l.strip() for l in f if l.strip()]
        shap_map = dict(zip(shap_names, shap))
        df["shap_mean_abs"] = df["feature"].map(shap_map)
        df["degradation_contribution"] = (
            df["ks_mean"] * df["shap_mean_abs"].fillna(0.0))
        n_shap_matched = int(df["shap_mean_abs"].notna().sum())
        print(f"[shap] matched {n_shap_matched} features to SHAP importance")
    else:
        df["shap_mean_abs"] = np.nan
        df["degradation_contribution"] = np.nan
        print("[shap] not found — skipping SHAP cross-reference")

    # 6. Sort and save
    df = df.sort_values("ks_mean", na_position="last").reset_index(drop=True)
    out_csv = os.path.join(RESULTS, "v2_feature_stability.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}  ({len(df)} features)\n")

    # 7. Family summary
    fam = df.groupby("family").agg(
        n=("feature", "size"),
        ks_mean=("ks_mean", "mean"),
        abs_d_mean=("abs_d_mean", "mean"),
        wass_mean=("wass_mean", "mean"),
        ks_source_target=("ks_source_target", "mean"),
        shap_sum=("shap_mean_abs", "sum"),
    ).sort_values("ks_mean")
    fam.to_csv(os.path.join(RESULTS, "v2_feature_stability_summary.csv"))
    print("Family stability summary (lower KS = more stable across cohorts):")
    print(fam.round(3).to_string())
    print()

    # 8. Top / bottom features
    print("=" * 72)
    print("Top 20 most stable features")
    print("=" * 72)
    for _, r in df.head(20).iterrows():
        shap = "" if pd.isna(r["shap_mean_abs"]) else f"  SHAP={r['shap_mean_abs']:.3f}"
        print(f"  {r['feature']:45s}  KS={r['ks_mean']:.3f}  "
              f"family={r['family']}{shap}")

    print("\n" + "=" * 72)
    print("Top 20 least stable features")
    print("=" * 72)
    for _, r in df.tail(20).iloc[::-1].iterrows():
        shap = "" if pd.isna(r["shap_mean_abs"]) else f"  SHAP={r['shap_mean_abs']:.3f}"
        print(f"  {r['feature']:45s}  KS={r['ks_mean']:.3f}  "
              f"family={r['family']}{shap}")

    # 9. PPG-only interpretation
    ppg_families = ["period morphology", "SMD", "morphology",
                    "spectral+nonlinear", "HRV", "ppg variability",
                    "HTN proxies", "other"]

    ppg_df = df[df["family"].isin(ppg_families)]
    print("\n" + "=" * 72)
    print("Most stable PPG-only features (top 10)")
    print("=" * 72)
    for _, r in ppg_df.head(10).iterrows():
        shap = "" if pd.isna(r["shap_mean_abs"]) else f"  SHAP={r['shap_mean_abs']:.3f}"
        print(f"  {r['feature']:45s}  KS={r['ks_mean']:.3f}  "
              f"family={r['family']}{shap}")

    print("\nFamily stability ranking (PPG families only):")
    ppg_fam = fam[fam.index.isin(ppg_families)]
    print(ppg_fam[["n", "ks_mean", "abs_d_mean",
                    "ks_source_target"]].round(3).to_string())

    # 10. SHAP × stability cross-reference
    if df["shap_mean_abs"].notna().any():
        print("\n" + "=" * 72)
        print("Top 20 SHAP features — stability and degradation contribution")
        print("=" * 72)
        print("(degradation_contribution = KS × SHAP — a proxy for how much")
        print(" a feature's instability affects the model's cross-dataset drop)")
        top_shap = (df.dropna(subset=["shap_mean_abs"])
                      .nlargest(20, "shap_mean_abs"))
        for _, r in top_shap.iterrows():
            print(f"  {r['feature']:45s}  SHAP={r['shap_mean_abs']:.3f}  "
                  f"KS={r['ks_mean']:.3f}  "
                  f"contrib={r['degradation_contribution']:.3f}  "
                  f"family={r['family']}")

        top_contrib = (df.dropna(subset=["degradation_contribution"])
                         .nlargest(15, "degradation_contribution"))
        print("\nTop 15 features by degradation contribution (KS × SHAP):")
        for _, r in top_contrib.iterrows():
            print(f"  {r['feature']:45s}  contrib={r['degradation_contribution']:.3f}  "
                  f"KS={r['ks_mean']:.3f}  SHAP={r['shap_mean_abs']:.3f}  "
                  f"family={r['family']}")

    # 11. Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 14, "figure.dpi": 130,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25,
        })
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Distribution of KS
        ax = axes[0, 0]
        ax.hist(df["ks_mean"].dropna(), bins=40,
                color="#2b6cb0", alpha=0.75, edgecolor="white")
        ax.set_xlabel("Mean KS across 6 cohort pairs")
        ax.set_ylabel("Number of features")
        ax.set_title("(a) Distribution of cross-cohort instability")

        # (b) Family stability vs. importance
        ax = axes[0, 1]
        for name, r in fam.iterrows():
            if pd.notna(r["shap_sum"]) and r["shap_sum"] > 0:
                ax.scatter(r["ks_mean"], r["shap_sum"],
                           s=100, alpha=0.8)
                ax.annotate(name, (r["ks_mean"], r["shap_sum"]),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=12)
        ax.set_xlabel("Mean KS (lower = more stable)")
        ax.set_ylabel("Total family SHAP")
        ax.set_title("(b) Family-level stability vs. attribution")

        # (c) 20 most stable
        ax = axes[1, 0]
        top = df.head(20).iloc[::-1]
        fam_colors = {"period morphology": "#2b6cb0", "SMD": "#dd6b20",
                      "morphology": "#38a169", "spectral+nonlinear": "#805ad5",
                      "demographics": "#c53030", "HRV": "#d69e2e",
                      "ppg variability": "#dd6b20", "HTN proxies": "#2c5282",
                      "other": "#718096"}
        colors = [fam_colors.get(f, "#4a5568") for f in top["family"]]
        ax.barh(range(len(top)), top["ks_mean"], color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=12)
        ax.set_xlabel("Mean KS")
        ax.set_title("(c) 20 most stable features")

        # (d) 20 least stable
        ax = axes[1, 1]
        bot = df.tail(20)
        colors = [fam_colors.get(f, "#4a5568") for f in bot["family"]]
        ax.barh(range(len(bot)), bot["ks_mean"], color=colors)
        ax.set_yticks(range(len(bot)))
        ax.set_yticklabels(bot["feature"], fontsize=12)
        ax.set_xlabel("Mean KS")
        ax.set_title("(d) 20 least stable features")

        plt.tight_layout()
        fig_path = os.path.join(FIG, "fig15_feature_stability.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"\n[saved] {fig_path}")
    except Exception as e:
        print(f"\n[figure] failed: {e}")


if __name__ == "__main__":
    main()

##
#  python -c "
# >> import pandas as pd
# >> df = pd.read_csv(r'out_v2\results\v2_feature_stability.csv')
# >> informative = df[(df['shap_mean_abs'].fillna(0) > 0.001) & df['ks_mean'].notna()]
# >> top20 = informative.nsmallest(20, 'ks_mean')
# >> print('20 most stable features with non-zero SHAP:')
# >> for _, r in top20.iterrows():
# >>         print('  {:42s}  KS={:.3f}  SHAP={:.3f}  family={}'.format(
# >>         r['feature'], r['ks_mean'], r['shap_mean_abs'], r['family']))
# >> "
