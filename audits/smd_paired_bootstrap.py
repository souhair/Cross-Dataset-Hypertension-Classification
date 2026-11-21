"""Corrected paired subject-cluster bootstrap analysis for SMD ablations.

Three paired comparisons are evaluated using ONE shared set of
StratifiedGroupKFold splits:

1. ALL features vs ALL minus the COMPLETE SMD family (94 candidates).
2. PPG-only vs PPG-only minus the SMD candidates retained in PPG-only
   (92 candidates; the two age-dependent SMD descriptors are already
   excluded from the PPG-only set).
3. Non-age-dependent SMD-only (92 candidates) vs Morphology-only
   (65 candidates).

Important implementation details
---------------------------------
- The complete SMD family (94) and standalone SMD set (92) are kept as
  separate lists. This prevents the age-dependent SMD features from
  accidentally remaining in the full-model ablation.
- One set of fold indices is generated once and reused for all six models.
- The same eligible folds and the same valid OOF observations are required
  within each paired comparison. Point AUCs and bootstrap CIs therefore use
  the same evaluation observations.
- Missingness screening, median imputation, correlation filtering, and
  scaling are fitted on training-fold data only.
- The XGBoost settings are imported from config.XGB_3CLASS to avoid drift
  between this script and the project's centralized configuration.
- The two-sided empirical bootstrap p-value follows the project's stated
  convention: q = proportion of paired bootstrap DeltaAUC values <= 0;
  p = min(1, 2 * min(q, 1-q)).

Run from the project root, for example:
    python audits/smd_paired_bootstrap_corrected.py

Expected candidate counts for the current Aurora feature inventory:
    all=437; complete SMD=94; non-age-dependent SMD=92;
    all-minus-SMD=343; PPG-only=414; PPG-only-minus-SMD=322;
    morphology=65.

Outputs are written under config/analysis.v2_utils RESULTS:
- v2_smd_paired_bootstrap_corrected.csv
- v2_smd_paired_bootstrap_feature_audit.csv
- v2_smd_paired_bootstrap_fold_audit.csv
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    SEED,
    N_FOLDS,
    NAN_THRESH,
    CORR_THRESH,
    XGB_3CLASS,
)
from analysis.v2_utils import load_aurora_raw, RESULTS
from fold_preprocessor import FoldAwarePreprocessor
from feature_families import family_of


# ---------------------------------------------------------------------------
# Analysis definitions
# ---------------------------------------------------------------------------
AGE_DEPENDENT_SMD = {
    "step_Psi_age_adjusted",
    "step_vascular_compliance_index",
}

EXPECTED_COUNTS = {
    "all": 437,
    "complete_smd": 94,
    "non_age_smd": 92,
    "all_minus_smd": 343,
    "ppg_only": 414,
    "ppg_only_minus_smd": 322,
    "morphology": 65,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_xgb() -> XGBClassifier:
    """Build the three-class XGBoost model from the central project config."""
    return XGBClassifier(**dict(XGB_3CLASS))


# ---------------------------------------------------------------------------
# Feature definitions and inventory checks
# ---------------------------------------------------------------------------
def ppg_only_exclusions() -> set[str]:
    """The 23-feature exclusion list used by the project's PPG-only analysis."""
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


def build_feature_sets(candidate_features: Sequence[str]) -> Dict[str, List[str]]:
    """Build and validate all feature sets used in the three comparisons."""
    candidates = list(candidate_features)

    if len(candidates) != len(set(candidates)):
        duplicates = pd.Series(candidates)[pd.Series(candidates).duplicated()].tolist()
        raise RuntimeError(f"Duplicate candidate feature names detected: {duplicates}")

    candidate_set = set(candidates)
    all_smd = [c for c in candidates if family_of(c) == "SMD"]

    missing_age_smd = AGE_DEPENDENT_SMD - set(all_smd)
    if missing_age_smd:
        raise RuntimeError(
            "The expected age-dependent SMD descriptors are not both present "
            f"in the SMD family: {sorted(missing_age_smd)}"
        )

    # Standalone SMD model excludes the two age-dependent descriptors.
    smd_standalone = [c for c in all_smd if c not in AGE_DEPENDENT_SMD]

    morphology = [c for c in candidates if family_of(c) == "morphology"]
    exclusions = ppg_only_exclusions()

    missing_exclusions = exclusions - candidate_set
    if missing_exclusions:
        raise RuntimeError(
            "The PPG-only exclusion list contains features not found in the "
            f"candidate inventory: {sorted(missing_exclusions)}"
        )

    all_features = list(candidates)

    # Critical correction: remove the COMPLETE 94-feature SMD family here.
    all_minus_smd = [c for c in candidates if c not in set(all_smd)]

    ppg_only = [c for c in candidates if c not in exclusions]

    # Remove all SMD candidates still present in PPG-only. The two age-
    # dependent descriptors are already absent from ppg_only, so this removes
    # the 92 remaining SMD candidates.
    ppg_only_minus_smd = [c for c in ppg_only if c not in set(all_smd)]

    sets = {
        "all": all_features,
        "all_minus_smd": all_minus_smd,
        "ppg_only": ppg_only,
        "ppg_only_minus_smd": ppg_only_minus_smd,
        "smd_standalone": smd_standalone,
        "morphology": morphology,
        "complete_smd": all_smd,
    }

    observed = {
        "all": len(all_features),
        "complete_smd": len(all_smd),
        "non_age_smd": len(smd_standalone),
        "all_minus_smd": len(all_minus_smd),
        "ppg_only": len(ppg_only),
        "ppg_only_minus_smd": len(ppg_only_minus_smd),
        "morphology": len(morphology),
    }

    for key, expected in EXPECTED_COUNTS.items():
        actual = observed[key]
        if actual != expected:
            raise RuntimeError(
                f"Feature inventory check failed for '{key}': expected "
                f"{expected}, found {actual}. Do not proceed until the "
                "candidate inventory or expected counts are reconciled."
            )

    # Ensure the two age-dependent SMD descriptors are absent from PPG-only.
    if AGE_DEPENDENT_SMD.intersection(ppg_only):
        raise RuntimeError(
            "Age-dependent SMD descriptors unexpectedly remain in PPG-only."
        )

    print("\nFEATURE INVENTORY AUDIT")
    print("-----------------------")
    print(f"All candidate features:                    {len(all_features)}")
    print(f"Complete SMD family:                       {len(all_smd)}")
    print(f"Age-dependent SMD descriptors excluded:    {len(AGE_DEPENDENT_SMD)}")
    print(f"Standalone non-age-dependent SMD:          {len(smd_standalone)}")
    print(f"ALL minus complete SMD:                    {len(all_minus_smd)}")
    print(f"PPG-only:                                  {len(ppg_only)}")
    print(f"PPG-only minus remaining SMD:              {len(ppg_only_minus_smd)}")
    print(f"Morphology-only:                           {len(morphology)}")
    print("\nExcluded from standalone SMD:")
    for name in sorted(AGE_DEPENDENT_SMD):
        print(f"  - {name}")

    return sets


# ---------------------------------------------------------------------------
# Shared subject-grouped folds
# ---------------------------------------------------------------------------
def build_shared_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_rows: int,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    """Generate folds once and reuse them for every feature-set condition.

    Folds that fail the original script's three-class train/test condition are
    marked ineligible once here, so exactly the same eligible folds apply to
    all models. No feature set is allowed to create its own fold assignment.
    """
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    X_index = np.arange(n_rows).reshape(-1, 1)
    raw_splits = list(splitter.split(X_index, y, groups=groups))

    if len(raw_splits) != N_FOLDS:
        raise RuntimeError(
            f"Expected {N_FOLDS} folds, generated {len(raw_splits)}."
        )

    expected_classes = set(np.unique(y).tolist())
    if len(expected_classes) != 3:
        raise RuntimeError(
            f"Expected three outcome classes in the full data; found {expected_classes}."
        )

    eligible_splits: List[Tuple[int, np.ndarray, np.ndarray]] = []
    fold_id = np.full(n_rows, -1, dtype=int)
    audit_rows = []

    print("\nSHARED STRATIFIED GROUP FOLD AUDIT")
    print("----------------------------------")

    for fold_number, (train_idx, test_idx) in enumerate(raw_splits, start=1):
        train_subjects = set(groups[train_idx].tolist())
        test_subjects = set(groups[test_idx].tolist())
        overlap = train_subjects.intersection(test_subjects)

        if overlap:
            raise RuntimeError(
                f"Subject leakage in fold {fold_number}: {len(overlap)} "
                "subjects appear in both train and test."
            )

        train_classes = set(np.unique(y[train_idx]).tolist())
        test_classes = set(np.unique(y[test_idx]).tolist())
        eligible = (train_classes == expected_classes and test_classes == expected_classes)

        fold_id[test_idx] = fold_number if eligible else -fold_number

        audit_rows.append({
            "fold": fold_number,
            "n_train_segments": len(train_idx),
            "n_test_segments": len(test_idx),
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subjects),
            "train_classes": json.dumps(sorted(train_classes)),
            "test_classes": json.dumps(sorted(test_classes)),
            "subject_overlap_count": len(overlap),
            "eligible_for_oof": eligible,
        })

        status = "USED" if eligible else "SKIPPED (not all classes in train/test)"
        print(
            f"Fold {fold_number}: train={len(train_idx)} segments/"
            f"{len(train_subjects)} subjects; test={len(test_idx)} segments/"
            f"{len(test_subjects)} subjects; {status}"
        )

        if eligible:
            eligible_splits.append((fold_number, train_idx, test_idx))

    if not eligible_splits:
        raise RuntimeError("No eligible folds contain all three classes in train and test.")

    if len(eligible_splits) != N_FOLDS:
        print(
            f"WARNING: {N_FOLDS - len(eligible_splits)} fold(s) are ineligible. "
            "The same eligible folds will be used for every model, matching "
            "the skip condition in the original script."
        )

    return eligible_splits, fold_id, pd.DataFrame(audit_rows)


# ---------------------------------------------------------------------------
# OOF predictions
# ---------------------------------------------------------------------------
def cv_oof(
    df: pd.DataFrame,
    features: Sequence[str],
    y: np.ndarray,
    splits: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    model_label: str,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Fit fold-aware preprocessing and XGBoost using shared fold indices."""
    if not features:
        raise RuntimeError(f"Model '{model_label}' has an empty candidate feature set.")

    oof_prob = np.full((len(df), 3), np.nan, dtype=np.float64)
    oof_valid = np.zeros(len(df), dtype=bool)
    feature_audit_rows: List[dict] = []

    for fold_number, train_idx, test_idx in splits:
        # All screening, imputation, correlation filtering and scaling are
        # learned using training-fold observations only.
        pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(
            df.iloc[train_idx], list(features)
        )

        selected_features = list(pre.feature_list_)
        if not selected_features:
            raise RuntimeError(
                f"All features were removed by fold preprocessing for "
                f"{model_label}, fold {fold_number}."
            )

        X_train = pre.transform(df.iloc[train_idx])
        X_test = pre.transform(df.iloc[test_idx])

        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise RuntimeError(
                f"Non-finite transformed values found for {model_label}, "
                f"fold {fold_number}."
            )

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = make_xgb()
        model.fit(X_train_scaled, y[train_idx])
        probabilities = model.predict_proba(X_test_scaled)

        if probabilities.shape != (len(test_idx), 3):
            raise RuntimeError(
                f"Unexpected probability shape for {model_label}, fold "
                f"{fold_number}: {probabilities.shape}. Expected "
                f"({len(test_idx)}, 3)."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                f"Non-finite OOF probabilities for {model_label}, fold {fold_number}."
            )

        oof_prob[test_idx] = probabilities
        oof_valid[test_idx] = True

        feature_audit_rows.append({
            "model": model_label,
            "fold": fold_number,
            "n_candidate_features": len(features),
            "n_after_missingness_screen": len(pre.medians_),
            "n_after_correlation_filter": len(selected_features),
            "selected_features": json.dumps(selected_features),
        })

        print(
            f"  {model_label}, fold {fold_number}: "
            f"candidate={len(features)}, "
            f"after missingness={len(pre.medians_)}, "
            f"after correlation={len(selected_features)}"
        )

    if not oof_valid.any():
        raise RuntimeError(f"No valid OOF predictions produced for {model_label}.")

    if np.isnan(oof_prob[oof_valid]).any():
        raise RuntimeError(f"NaN OOF probabilities remain for {model_label}.")

    return oof_prob, oof_valid, feature_audit_rows


# ---------------------------------------------------------------------------
# Paired evaluation checks and point AUC
# ---------------------------------------------------------------------------
def assert_identical_valid_masks(
    comparison_label: str,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """Require both paired models to use exactly the same OOF observations."""
    if not np.array_equal(valid_a, valid_b):
        only_a = int(np.sum(valid_a & ~valid_b))
        only_b = int(np.sum(valid_b & ~valid_a))
        raise RuntimeError(
            f"{comparison_label}: paired validity masks differ. "
            f"A-only valid rows={only_a}; B-only valid rows={only_b}. "
            "Do not compare point AUCs or bootstrap CIs until resolved."
        )

    valid = valid_a & valid_b
    n_segments = int(valid.sum())
    n_subjects = int(np.unique(subject_ids[valid]).size)
    print(f"\nPAIRED EVALUATION AUDIT — {comparison_label}")
    print(f"  Shared valid segments: {n_segments}")
    print(f"  Shared valid subjects: {n_subjects}")
    print("  Identical OOF validity masks: YES")

    if n_segments == 0 or n_subjects < 2:
        raise RuntimeError(f"Insufficient paired data for {comparison_label}.")

    return valid


def macro_ovr_auc(y: np.ndarray, probabilities: np.ndarray, valid: np.ndarray) -> float:
    """Compute macro one-vs-rest AUC on the supplied common valid subset."""
    labels = np.unique(y[valid])
    if len(labels) != 3:
        raise RuntimeError(
            f"Expected all three classes in the OOF evaluation subset; found {labels}."
        )
    return float(roc_auc_score(
        y[valid],
        probabilities[valid],
        multi_class="ovr",
        average="macro",
    ))


# ---------------------------------------------------------------------------
# Paired subject-cluster bootstrap
# ---------------------------------------------------------------------------
def paired_bootstrap(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    subject_ids: np.ndarray,
    valid: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """Paired subject-cluster bootstrap of macro OVR AUC differences."""
    valid_idx = np.flatnonzero(valid)
    y_valid = y[valid_idx]
    prob_a_valid = prob_a[valid_idx]
    prob_b_valid = prob_b[valid_idx]
    subjects_valid = subject_ids[valid_idx]

    unique_subjects = np.unique(subjects_valid)
    if len(unique_subjects) < 2:
        raise RuntimeError("At least two subjects are required for cluster bootstrap.")

    subject_to_rows = {
        subject: np.flatnonzero(subjects_valid == subject)
        for subject in unique_subjects
    }

    rng = np.random.default_rng(seed)
    auc_a_values: List[float] = []
    auc_b_values: List[float] = []
    delta_values: List[float] = []

    for _ in range(n_boot):
        # Draw subjects, not individual segments. If a subject is drawn more
        # than once, all of that subject's segments are repeated accordingly.
        sampled_subjects = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        sampled_rows = np.concatenate([
            subject_to_rows[subject] for subject in sampled_subjects
        ])
        y_boot = y_valid[sampled_rows]

        if len(np.unique(y_boot)) != 3:
            continue

        try:
            auc_a = roc_auc_score(
                y_boot,
                prob_a_valid[sampled_rows],
                multi_class="ovr",
                average="macro",
            )
            auc_b = roc_auc_score(
                y_boot,
                prob_b_valid[sampled_rows],
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            continue

        auc_a_values.append(float(auc_a))
        auc_b_values.append(float(auc_b))
        delta_values.append(float(auc_a - auc_b))

    a = np.asarray(auc_a_values, dtype=float)
    b = np.asarray(auc_b_values, dtype=float)
    d = np.asarray(delta_values, dtype=float)

    if len(d) < 2:
        raise RuntimeError(
            f"Only {len(d)} valid bootstrap resamples were obtained; cannot "
            "estimate the bootstrap SD and interval."
        )

    def summarize(values: np.ndarray, prefix: str) -> dict:
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values, ddof=1)),
            f"{prefix}_lo": float(np.percentile(values, 2.5)),
            f"{prefix}_hi": float(np.percentile(values, 97.5)),
        }

    result = {}
    result.update(summarize(a, "auc_a_boot"))
    result.update(summarize(b, "auc_b_boot"))
    result.update(summarize(d, "delta"))

    # Empirical two-sided p-value convention used in the manuscript workflow.
    q = float(np.mean(d <= 0))
    result["q_delta_nonpositive"] = q
    result["p_two_sided"] = float(min(2.0 * min(q, 1.0 - q), 1.0))
    result["n_boot_requested"] = int(n_boot)
    result["n_boot_valid"] = int(len(d))
    result["n_boot_skipped"] = int(n_boot - len(d))

    return result


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------
def run_comparison(
    label: str,
    model_a_label: str,
    model_b_label: str,
    prob_a: np.ndarray,
    valid_a: np.ndarray,
    prob_b: np.ndarray,
    valid_b: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_features_a: int,
    n_features_b: int,
) -> dict:
    """Calculate paired point estimates and bootstrap statistics."""
    valid = assert_identical_valid_masks(
        label, valid_a, valid_b, subject_ids
    )

    # Both point AUCs are deliberately computed on the exact same rows.
    auc_a_point = macro_ovr_auc(y, prob_a, valid)
    auc_b_point = macro_ovr_auc(y, prob_b, valid)
    delta_point = auc_a_point - auc_b_point

    stats = paired_bootstrap(
        y,
        prob_a,
        prob_b,
        subject_ids,
        valid,
        n_boot=1000,
        seed=SEED,
    )

    print(f"\n{label}")
    print(f"  {model_a_label}: AUC = {auc_a_point:.4f} ({n_features_a} candidates)")
    print(f"  {model_b_label}: AUC = {auc_b_point:.4f} ({n_features_b} candidates)")
    print(f"  Point DeltaAUC (A-B): {delta_point:+.4f}")
    print(
        f"  Bootstrap mean DeltaAUC: {stats['delta_mean']:+.4f}; "
        f"95% percentile CI [{stats['delta_lo']:+.4f}, "
        f"{stats['delta_hi']:+.4f}]"
    )
    print(
        f"  Two-sided empirical p = {stats['p_two_sided']:.3f}; "
        f"valid resamples = {stats['n_boot_valid']}/{stats['n_boot_requested']}"
    )

    if stats["n_boot_valid"] < stats["n_boot_requested"]:
        print(
            "  WARNING: Some bootstrap resamples were skipped because they "
            "did not contain all three classes."
        )

    row = {
        "comparison": label,
        "model_a": model_a_label,
        "model_b": model_b_label,
        "n_features_a": int(n_features_a),
        "n_features_b": int(n_features_b),
        "n_segments_paired": int(valid.sum()),
        "n_subjects_paired": int(np.unique(subject_ids[valid]).size),
        "auc_a_point": auc_a_point,
        "auc_b_point": auc_b_point,
        "delta_point": delta_point,
        **stats,
    }
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("CORRECTED SMD PAIRED SUBJECT-CLUSTER BOOTSTRAP ANALYSIS")
    print("=" * 78)

    df, candidate_features = load_aurora_raw()
    candidate_features = list(candidate_features)

    required_columns = {"htn3", "subject_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise RuntimeError(f"Required data columns are missing: {sorted(missing_columns)}")

    y = df["htn3"].to_numpy()
    subject_ids = df["subject_id"].astype(str).to_numpy()
    groups = subject_ids.copy()

    feature_sets = build_feature_sets(candidate_features)

    print(f"\nData: {len(df)} segments, {pd.Series(subject_ids).nunique()} subjects")

    # Generate folds once. Every model receives exactly these same indices.
    shared_splits, fold_id, fold_audit = build_shared_folds(
        y=y,
        groups=groups,
        n_rows=len(df),
    )

    feature_audit_rows: List[dict] = []
    prediction_store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    model_definitions = [
        ("ALL", feature_sets["all"]),
        ("ALL-minus-complete-SMD", feature_sets["all_minus_smd"]),
        ("PPG-only", feature_sets["ppg_only"]),
        ("PPG-only-minus-SMD", feature_sets["ppg_only_minus_smd"]),
        ("SMD-only-92-non-age-dependent", feature_sets["smd_standalone"]),
        ("Morphology-only", feature_sets["morphology"]),
    ]

    for model_label, features in model_definitions:
        print("\n" + "-" * 78)
        print(f"FITTING: {model_label} ({len(features)} candidate features)")
        print("-" * 78)

        probabilities, valid, model_audit = cv_oof(
            df=df,
            features=features,
            y=y,
            splits=shared_splits,
            model_label=model_label,
        )

        # The valid masks must equal the common mask implied by the shared
        # eligible folds. A mismatch is a hard error rather than silently
        # changing the paired evaluation subset.
        expected_valid = np.zeros(len(df), dtype=bool)
        for _, _, test_idx in shared_splits:
            expected_valid[test_idx] = True

        if not np.array_equal(valid, expected_valid):
            raise RuntimeError(
                f"{model_label}: OOF validity mask does not match the shared folds."
            )

        prediction_store[model_label] = (probabilities, valid)
        feature_audit_rows.extend(model_audit)

    # Run the three paired comparisons.
    rows = []

    # [1] Full feature set vs removal of all 94 SMD candidates.
    rows.append(run_comparison(
        label="ALL vs ALL-minus-complete-SMD",
        model_a_label="ALL",
        model_b_label="ALL-minus-complete-SMD",
        prob_a=prediction_store["ALL"][0],
        valid_a=prediction_store["ALL"][1],
        prob_b=prediction_store["ALL-minus-complete-SMD"][0],
        valid_b=prediction_store["ALL-minus-complete-SMD"][1],
        y=y,
        subject_ids=subject_ids,
        n_features_a=len(feature_sets["all"]),
        n_features_b=len(feature_sets["all_minus_smd"]),
    ))

    # [2] PPG-only vs removal of all SMD candidates present in PPG-only.
    rows.append(run_comparison(
        label="PPG-only vs PPG-only-minus-SMD",
        model_a_label="PPG-only",
        model_b_label="PPG-only-minus-SMD",
        prob_a=prediction_store["PPG-only"][0],
        valid_a=prediction_store["PPG-only"][1],
        prob_b=prediction_store["PPG-only-minus-SMD"][0],
        valid_b=prediction_store["PPG-only-minus-SMD"][1],
        y=y,
        subject_ids=subject_ids,
        n_features_a=len(feature_sets["ppg_only"]),
        n_features_b=len(feature_sets["ppg_only_minus_smd"]),
    ))

    # [3] Standalone non-age-dependent SMD (92) vs morphology (65).
    rows.append(run_comparison(
        label="SMD-only-92-non-age-dependent vs Morphology-only",
        model_a_label="SMD-only (92 non-age-dependent candidates)",
        model_b_label="Morphology-only",
        prob_a=prediction_store["SMD-only-92-non-age-dependent"][0],
        valid_a=prediction_store["SMD-only-92-non-age-dependent"][1],
        prob_b=prediction_store["Morphology-only"][0],
        valid_b=prediction_store["Morphology-only"][1],
        y=y,
        subject_ids=subject_ids,
        n_features_a=len(feature_sets["smd_standalone"]),
        n_features_b=len(feature_sets["morphology"]),
    ))

    # ------------------------------------------------------------------
    # Save results and audit files
    # ------------------------------------------------------------------
    os.makedirs(RESULTS, exist_ok=True)

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(
        RESULTS, "v2_smd_paired_bootstrap_corrected.csv"
    )
    results_df.to_csv(results_path, index=False)

    feature_audit_df = pd.DataFrame(feature_audit_rows)
    feature_audit_path = os.path.join(
        RESULTS, "v2_smd_paired_bootstrap_feature_audit.csv"
    )
    feature_audit_df.to_csv(feature_audit_path, index=False)

    fold_audit["seed"] = SEED
    fold_audit["n_splits"] = N_FOLDS
    fold_audit_path = os.path.join(
        RESULTS, "v2_smd_paired_bootstrap_fold_audit.csv"
    )
    fold_audit.to_csv(fold_audit_path, index=False)

    # Row-level fold membership enables independent verification of subject
    # grouping and the exact shared OOF evaluation subset.
    row_fold_audit = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "subject_id": subject_ids,
        "target": y,
        "fold_assignment": fold_id,
    })
    row_fold_audit["oof_used"] = row_fold_audit["fold_assignment"].gt(0)
    row_fold_path = os.path.join(
        RESULTS, "v2_smd_paired_bootstrap_row_fold_assignments.csv"
    )
    row_fold_audit.to_csv(row_fold_path, index=False)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    display_columns = [
        "comparison",
        "n_features_a",
        "n_features_b",
        "auc_a_point",
        "auc_b_point",
        "delta_point",
        "delta_mean",
        "delta_lo",
        "delta_hi",
        "p_two_sided",
        "n_subjects_paired",
        "n_boot_valid",
    ]
    print(results_df[display_columns].round(4).to_string(index=False))

    print("\nSAVED FILES")
    print(f"  Results:       {results_path}")
    print(f"  Feature audit: {feature_audit_path}")
    print(f"  Fold audit:    {fold_audit_path}")
    print(f"  Row/fold map:  {row_fold_path}")

    print("\nINTERPRETATION NOTE")
    print(
        "DeltaAUC is AUC(model A) minus AUC(model B). The bootstrap CI is "
        "the 2.5th–97.5th percentile interval of paired subject-cluster "
        "bootstrap differences. The empirical p-value uses the <= 0 "
        "proportion convention documented in this script."
    )


if __name__ == "__main__":
    main()
