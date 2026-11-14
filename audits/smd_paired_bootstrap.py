# audits/smd_paired_bootstrap.py
"""
SMD paired bootstrap analysis.

A paired subject-level bootstrap CI on the ΔAUC from removing the SMD family,
and for a matched morphology-only comparison.

Three comparisons are run under identical folds:

  [1] ALL features vs ALL minus SMD          (main claim)
  [2] PPG-only vs PPG-only minus SMD         (PPG-only regime)
  [3] SMD-alone vs Morphology-alone          (matched family comparison)

Method:
  1. Run 5-fold StratifiedGroupKFold with fold-aware preprocessing.
  2. Save out-of-fold (OOF) predictions for both conditions in each pair.
  3. Bootstrap 1,000 times: resample subjects with replacement (all
     segments belonging to a subject move together), recompute AUC for
     both models on the resampled set, compute ΔAUC.
  4. Report mean, SD, 95% CI, and two-sided p-value of the ΔAUC.

"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor
from feature_families import family_of

# ============================================================================
# Model
# ============================================================================
def make_xgb():
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=-1,
        random_state=SEED, verbosity=0)


# ============================================================================
# OOF prediction collection
# ============================================================================
def cv_oof(df, features, y, g):
    """Run 5-fold CV, return OOF probability matrix and validity mask."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED)
    oof_prob = np.zeros((len(df), 3), dtype=np.float32)
    oof_valid = np.zeros(len(df), dtype=bool)

    for tr, te in sgkf.split(df[features].values, y, groups=g):
        if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
            continue
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[tr], features)
        X_tr = pre.transform(df.iloc[tr])
        X_te = pre.transform(df.iloc[te])
        sc = StandardScaler().fit(X_tr)
        m = make_xgb()
        m.fit(sc.transform(X_tr), y[tr])
        prob = m.predict_proba(sc.transform(X_te))
        oof_prob[te] = prob
        oof_valid[te] = True

    return oof_prob, oof_valid


# ============================================================================
# Paired subject-cluster bootstrap
# ============================================================================
def paired_bootstrap(y, prob_a, prob_b, subj_ids, valid,
                     n_boot=1000, seed=SEED):
    """
    Paired subject-cluster bootstrap.

    For each iteration:
      1. Resample N subjects with replacement (all segments per subject).
      2. Concatenate predictions from drawn subjects (a subject drawn
         k times contributes its segments k times — standard bootstrap).
      3. Compute AUC_a, AUC_b on the concatenated set.
      4. Record ΔAUC = AUC_a − AUC_b.
    """
    rng = np.random.default_rng(seed)

    # Restrict to segments with valid OOF predictions from both models
    valid_idx = np.where(valid)[0]
    vy = y[valid_idx]
    va = prob_a[valid_idx]
    vb = prob_b[valid_idx]
    vs = subj_ids[valid_idx]

    unique_subj = np.unique(vs)
    n_subj = len(unique_subj)
    subj_to_idx = {s: np.where(vs == s)[0] for s in unique_subj}

    auc_a_arr, auc_b_arr, delta_arr = [], [], []

    for _ in range(n_boot):
        drawn = rng.choice(unique_subj, size=n_subj, replace=True)
        concat = np.concatenate([subj_to_idx[s] for s in drawn])

        y_b = vy[concat]
        if len(np.unique(y_b)) < 3:
            continue

        try:
            auc_a = roc_auc_score(y_b, va[concat],
                                    multi_class="ovr", average="macro")
            auc_b = roc_auc_score(y_b, vb[concat],
                                    multi_class="ovr", average="macro")
        except Exception:
            continue

        auc_a_arr.append(auc_a)
        auc_b_arr.append(auc_b)
        delta_arr.append(auc_a - auc_b)

    a = np.array(auc_a_arr)
    b = np.array(auc_b_arr)
    d = np.array(delta_arr)

    def _s(arr, name):
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std":  float(np.std(arr, ddof=1)),
            f"{name}_lo":   float(np.percentile(arr, 2.5)),
            f"{name}_hi":   float(np.percentile(arr, 97.5)),
        }

    stats = {}
    stats.update(_s(a, "auc_a"))
    stats.update(_s(b, "auc_b"))
    stats.update(_s(d, "delta"))
    p_neg = float(np.mean(d < 0))
    stats["p_delta_negative"] = p_neg
    stats["p_two_sided"] = min(2 * min(p_neg, 1 - p_neg), 1.0)
    stats["n_boot_valid"] = len(d)
    return stats


# ============================================================================
# Family / feature helpers
# ============================================================================
def ppg_only_exclusions():
    """The 23-feature exclusion list used by v2_no_demographics.py."""
    return {
        "age", "demo_age", "demo_bmi", "demo_gender", "demo_height",
        "demo_weight", "htn_stiffness_index_proxy",
        "lasi_ms", "period_cv__lasi_ms", "period_max__lasi_ms",
        "period_mean__lasi_ms", "period_median__lasi_ms",
        "period_min__lasi_ms", "period_std__lasi_ms",
        "period_cv__stiffness_index", "period_max__stiffness_index",
        "period_mean__stiffness_index", "period_median__stiffness_index",
        "period_min__stiffness_index", "period_std__stiffness_index",
        "step_Psi_age_adjusted", "step_vascular_compliance_index",
        "stiffness_index",
    }


def report(label, stats, auc_a_point, auc_b_point):
    delta = stats["delta_mean"]
    lo, hi = stats["delta_lo"], stats["delta_hi"]
    crosses = (lo < 0) and (hi > 0)
    print(f"\n  {label}")
    print(f"    Point AUCs:  A = {auc_a_point:.4f}   B = {auc_b_point:.4f}")
    print(f"    ΔAUC:        {delta:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    Two-sided p: {stats['p_two_sided']:.3f}")
    if crosses:
        print(f"    → CI contains zero — no significant difference")
    elif delta > 0:
        print(f"    → A significantly higher than B (p={stats['p_two_sided']:.3f})")
    else:
        print(f"    → A significantly lower than B (p={stats['p_two_sided']:.3f})")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("SMD paired bootstrap analysis")
    print("=" * 72)

    df, CANDIDATE = load_aurora_raw()
    y = df.htn3.values
    g = df.subject_id.astype(str).values
    subj_ids = df.subject_id.values

    smd_features = [c for c in CANDIDATE if family_of(c) == "SMD"]
    morphology_features = [c for c in CANDIDATE if family_of(c) == "morphology"]
    exclusions = ppg_only_exclusions()

    all_features = list(CANDIDATE)
    all_minus_smd = [c for c in CANDIDATE if c not in smd_features]

    ppg_features = [c for c in CANDIDATE if c not in exclusions]
    ppg_minus_smd = [c for c in ppg_features if c not in smd_features]

    print(f"\nData: {len(df)} segments, {df.subject_id.nunique()} subjects")
    print(f"  ALL features:              {len(all_features):>3d}")
    print(f"  ALL minus SMD:             {len(all_minus_smd):>3d}  (SMD n={len(smd_features)})")
    print(f"  PPG-only:                  {len(ppg_features):>3d}")
    print(f"  PPG-only minus SMD:        {len(ppg_minus_smd):>3d}")
    print(f"  SMD alone:                 {len(smd_features):>3d}")
    print(f"  Morphology alone:          {len(morphology_features):>3d}")

    rows = []

    # ------------------------------------------------------------------
    # [1] ALL vs ALL-minus-SMD
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[1/3] ALL features vs ALL minus SMD")
    print("=" * 72)

    print("  Computing OOF predictions for ALL features...")
    prob_all, valid_all = cv_oof(df, all_features, y, g)
    auc_all = roc_auc_score(y[valid_all], prob_all[valid_all],
                              multi_class="ovr", average="macro")
    print(f"    point AUC = {auc_all:.4f}")

    print("  Computing OOF predictions for ALL minus SMD...")
    prob_minus, valid_minus = cv_oof(df, all_minus_smd, y, g)
    auc_minus = roc_auc_score(y[valid_minus], prob_minus[valid_minus],
                                multi_class="ovr", average="macro")
    print(f"    point AUC = {auc_minus:.4f}")

    valid_both = valid_all & valid_minus
    print(f"  Bootstrapping (1,000 resamples over {valid_both.sum()} segments)...")
    stats1 = paired_bootstrap(y, prob_all, prob_minus, subj_ids, valid_both)
    report("ALL vs ALL-minus-SMD", stats1, auc_all, auc_minus)

    rows.append({
        "comparison": "ALL vs ALL-minus-SMD",
        "n_features_a": len(all_features),
        "n_features_b": len(all_minus_smd),
        "auc_a_point": auc_all,
        "auc_b_point": auc_minus,
        **stats1,
    })

    # ------------------------------------------------------------------
    # [2] PPG-only vs PPG-only-minus-SMD
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[2/3] PPG-only vs PPG-only minus SMD")
    print("=" * 72)

    print("  Computing OOF predictions for PPG-only...")
    prob_ppg, valid_ppg = cv_oof(df, ppg_features, y, g)
    auc_ppg = roc_auc_score(y[valid_ppg], prob_ppg[valid_ppg],
                             multi_class="ovr", average="macro")
    print(f"    point AUC = {auc_ppg:.4f}")

    print("  Computing OOF predictions for PPG-only minus SMD...")
    prob_ppg_minus, valid_ppg_minus = cv_oof(df, ppg_minus_smd, y, g)
    auc_ppg_minus = roc_auc_score(y[valid_ppg_minus],
                                   prob_ppg_minus[valid_ppg_minus],
                                   multi_class="ovr", average="macro")
    print(f"    point AUC = {auc_ppg_minus:.4f}")

    valid_both2 = valid_ppg & valid_ppg_minus
    print(f"  Bootstrapping (1,000 resamples)...")
    stats2 = paired_bootstrap(y, prob_ppg, prob_ppg_minus, subj_ids, valid_both2)
    report("PPG-only vs PPG-only-minus-SMD", stats2, auc_ppg, auc_ppg_minus)

    rows.append({
        "comparison": "PPG-only vs PPG-only-minus-SMD",
        "n_features_a": len(ppg_features),
        "n_features_b": len(ppg_minus_smd),
        "auc_a_point": auc_ppg,
        "auc_b_point": auc_ppg_minus,
        **stats2,
    })

    # ------------------------------------------------------------------
    # [3] SMD-alone vs Morphology-alone
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[3/3] SMD-alone vs Morphology-alone")
    print("=" * 72)

    print("  Computing OOF predictions for SMD-only...")
    prob_smd, valid_smd = cv_oof(df, smd_features, y, g)
    auc_smd = roc_auc_score(y[valid_smd], prob_smd[valid_smd],
                              multi_class="ovr", average="macro")
    print(f"    point AUC = {auc_smd:.4f}")

    print("  Computing OOF predictions for Morphology-only...")
    prob_mor, valid_mor = cv_oof(df, morphology_features, y, g)
    auc_mor = roc_auc_score(y[valid_mor], prob_mor[valid_mor],
                              multi_class="ovr", average="macro")
    print(f"    point AUC = {auc_mor:.4f}")

    valid_both3 = valid_smd & valid_mor
    print(f"  Bootstrapping (1,000 resamples)...")
    stats3 = paired_bootstrap(y, prob_smd, prob_mor, subj_ids, valid_both3)
    report("SMD-alone vs Morphology-alone", stats3, auc_smd, auc_mor)

    rows.append({
        "comparison": "SMD-alone vs Morphology-alone",
        "n_features_a": len(smd_features),
        "n_features_b": len(morphology_features),
        "auc_a_point": auc_smd,
        "auc_b_point": auc_mor,
        **stats3,
    })

    # ------------------------------------------------------------------
    # Save and summarize
    # ------------------------------------------------------------------
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "v2_smd_paired_bootstrap.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    print("\n" + "=" * 72)
    print("SUMMARY — paired subject-cluster bootstrap (1,000 resamples)")
    print("=" * 72)
    print(df_out[["comparison", "auc_a_point", "auc_b_point",
                    "delta_mean", "delta_lo", "delta_hi",
                    "p_two_sided"]].round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    def interpret(r, a_label, b_label):
        lo, hi = r["delta_lo"], r["delta_hi"]
        d = r["delta_mean"]
        p = r["p_two_sided"]
        crosses = (lo < 0) and (hi > 0)
        if crosses:
            print(f"  {r['comparison']}:  Δ = {d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  p = {p:.3f}")
            print(f"      → No significant difference between "
                  f"{a_label} and {b_label}")
        else:
            direction = "higher" if d > 0 else "lower"
            print(f"  {r['comparison']}:  Δ = {d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  p = {p:.3f}")
            print(f"      → {a_label} significantly {direction} than "
                  f"{b_label}")

    for _, r in df_out.iterrows():
        parts = r["comparison"].split(" vs ")
        interpret(r, parts[0], parts[1])


if __name__ == "__main__":
    main()