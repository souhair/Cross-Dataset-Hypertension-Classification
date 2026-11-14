# audits/controlled_adaptation.py
"""
Controlled adaptation experiment.

This script isolates the warm-start benefit from two confounds:
  (1) Boost count — warm-start effectively has ~400 total rounds
      (200 source + 200 fine-tune) while from-scratch has 200.
  (2) Scaler coordinates — warm-start uses source-scaled features,
      from-scratch uses target-scaled features.

Five configurations under identical subject-grouped 5-fold splits:

  A  Warm-start: source booster, source scaler, 200 FT rounds  (current)
  B  From-scratch, target scaler, 200 rounds                    (current baseline)
  C  From-scratch, target scaler, 400 rounds                    (matched boost count)
  D  From-scratch, source scaler, 400 rounds                    (matched boost AND scaler)
  E  From-scratch, target scaler, 500 rounds                    (matched to internal)

Key comparisons:
  A vs C  — isolates the value of source initialization from extra boosting
  A vs D  — isolates the value of source initialization from boosting + scaler
  If A >> C or A >> D → warm-start initialization genuinely helps
  If A ≈ C ≈ D → earlier "warm-start advantage" was extra rounds + scaler

"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                              balanced_accuracy_score)
from xgboost import XGBClassifier
from config import XGB_BEST, XGB_FINETUNE
from config import SEED, N_FOLDS, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)

# =========================================================================
# Hyperparameter constants (must match v2_transfer_strategies.py)
# =========================================================================
def xgb_with(**overrides):
    """Build an XGBClassifier with XGB_BEST + overrides."""
    cfg = {**XGB_BEST, **overrides}
    return XGBClassifier(**cfg)

# =========================================================================
# Target dataset loader
# =========================================================================
def load_target(path, features, medians):
    """Load a target CSV using the Aurora feature list + medians."""
    tg = load_target_csv(path, features, medians)
    if tg is None or len(tg) == 0:
        return None
    return tg

# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 78)
    print("Controlled adaptation — isolating warm-start from boost-count confounds")
    print("=" * 78)

    # ---- Load Aurora and fit the source model once ----------------------
    from fold_preprocessor import FoldAwarePreprocessor
    from config import NAN_THRESH, CORR_THRESH

    df_aur, CANDIDATE = load_aurora_raw()
    pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(df_aur, CANDIDATE)
    FEATURES = pre.feature_list_

    # Transform Aurora with the fitted preprocessor
    X_src_transformed = pre.transform(df_aur)
    df_aur_feat = df_aur.copy()
    for i, c in enumerate(FEATURES):
        df_aur_feat[c] = X_src_transformed[:, i]

    y_src = df_aur.htn3.values
    X_src_raw = df_aur_feat[FEATURES].values.astype(np.float64)
    print(f"\nAurora: {len(df_aur)} segments, "
          f"{df_aur.subject_id.nunique()} subjects, "
          f"{len(FEATURES)} candidates")

    # Source scaler — fit on full Aurora (matches v2_transfer_strategies.py)
    scaler_src = StandardScaler().fit(X_src_raw)
    X_src_scaled = scaler_src.transform(X_src_raw)

    print("\nFitting source booster (XGB_BEST, 200 rounds)...")
    src_model = xgb_with()
    src_model.fit(X_src_scaled, y_src)
    print(f"  source booster trained")

    # =========================================================================
    # Per-target loops
    # =========================================================================
    all_results = []

    for target_name, csv_key in [("PPG-BP", "ppgbp"),
                                  ("PulseDB-Vital", "pulsedb_vital"),
                                  ("SWCS", "ds4")]:
        print(f"\n{'=' * 78}")
        print(f"{target_name}")
        print(f"{'=' * 78}")

        csv_path = CSV.get(csv_key)
        if csv_path is None or not os.path.exists(csv_path):
            print(f"  [skip] not found: {csv_path}")
            continue

        medians = aurora_medians(df_aur, FEATURES)
        tg = load_target(csv_path, FEATURES, medians)
        if tg is None:
            print(f"  [skip] load failed")
            continue

        X = tg[FEATURES].values.astype(np.float64)
        y = tg.htn3.values.astype(np.int64)
        g = tg.subject_id.astype(str).values
        print(f"  {len(tg)} segments / {len(set(g))} subjects")

        # Bucket per-configuration metrics
        metrics = {label: {"auc": [], "f1": [], "bacc": []}
                   for label in ["A_warm", "B_scratch_200",
                                   "C_scratch_400", "D_scratch_src_400",
                                   "E_scratch_500"]}

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                     random_state=SEED)
        for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=g), 1):
            if len(np.unique(y[tr])) < 3 or len(np.unique(y[te])) < 3:
                continue

            X_tr = X[tr]; X_te = X[te]
            y_tr = y[tr]; y_te = y[te]

            # Two scalers — target (fold-fitted) and source (Aurora-fitted)
            sc_t = StandardScaler().fit(X_tr)
            X_tr_t = sc_t.transform(X_tr)
            X_te_t = sc_t.transform(X_te)
            X_tr_src = scaler_src.transform(X_tr)
            X_te_src = scaler_src.transform(X_te)

            # -------------------------------------------------------------
            # A — Warm-start (source booster, source scaler, +200 FT rounds)
            # -------------------------------------------------------------
            m = XGBClassifier(**XGB_FINETUNE)
            try:
                m.fit(X_tr_src, y_tr,
                      xgb_model=src_model.get_booster())
                prob = m.predict_proba(X_te_src)
                _record(metrics["A_warm"], y_te, prob)
            except Exception as e:
                print(f"    fold {fold} A_warm failed: {e}")

            # -------------------------------------------------------------
            # B — From-scratch, target scaler, 200 rounds (current baseline)
            # -------------------------------------------------------------
            m = xgb_with(n_estimators=200)
            m.fit(X_tr_t, y_tr)
            _record(metrics["B_scratch_200"], y_te, m.predict_proba(X_te_t))

            # -------------------------------------------------------------
            # C — From-scratch, target scaler, 400 rounds (matched boost)
            # -------------------------------------------------------------
            m = xgb_with(n_estimators=400)
            m.fit(X_tr_t, y_tr)
            _record(metrics["C_scratch_400"], y_te, m.predict_proba(X_te_t))

            # -------------------------------------------------------------
            # D — From-scratch, source scaler, 400 rounds (matched boost+scaler)
            # -------------------------------------------------------------
            m = xgb_with(n_estimators=400)
            m.fit(X_tr_src, y_tr)
            _record(metrics["D_scratch_src_400"], y_te, m.predict_proba(X_te_src))

            # -------------------------------------------------------------
            # E — From-scratch, target scaler, 500 rounds (matched to internal)
            # -------------------------------------------------------------
            m = xgb_with(n_estimators=500)
            m.fit(X_tr_t, y_tr)
            _record(metrics["E_scratch_500"], y_te, m.predict_proba(X_te_t))

        # ---- Summarize this target -------------------------------------
        print(f"\n  {target_name} — summary across {N_FOLDS} folds:")
        print(f"  {'config':24s} {'AUC':>16s} {'Macro F1':>10s} {'BAcc':>10s}")
        print(f"  {'-' * 60}")
        for label, m in metrics.items():
            if not m["auc"]:
                continue
            auc_m = np.mean(m["auc"])
            auc_s = np.std(m["auc"], ddof=1)
            f1_m = np.mean(m["f1"])
            bacc_m = np.mean(m["bacc"])
            print(f"  {label:24s} {auc_m:.4f} ± {auc_s:.4f}   "
                  f"{f1_m:.4f}    {bacc_m:.4f}")

            all_results.append({
                "dataset": target_name,
                "configuration": label,
                "auc_mean": float(auc_m),
                "auc_std": float(auc_s),
                "f1_mean": float(f1_m),
                "bacc_mean": float(bacc_m),
                "n_folds": len(m["auc"]),
            })

    # =========================================================================
    # Save and summarize
    # =========================================================================
    df_out = pd.DataFrame(all_results)
    out_csv = os.path.join(RESULTS, "v2_controlled_adaptation.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")

    print("\n" + "=" * 78)
    print("KEY COMPARISONS")
    print("=" * 78)

    # Group by dataset and extract A, C, D
    for dataset in df_out["dataset"].unique():
        sub = df_out[df_out["dataset"] == dataset].set_index("configuration")
        a = sub.loc["A_warm", "auc_mean"]
        b = sub.loc["B_scratch_200", "auc_mean"]
        c = sub.loc["C_scratch_400", "auc_mean"]
        d = sub.loc["D_scratch_src_400", "auc_mean"]
        e = sub.loc["E_scratch_500", "auc_mean"]

        print(f"\n{dataset}:")
        print(f"  A (warm-start, current):        {a:.4f}")
        print(f"  B (from-scratch, 200, current): {b:.4f}")
        print(f"  C (from-scratch, 400, target):  {c:.4f}")
        print(f"  D (from-scratch, 400, source):  {d:.4f}")
        print(f"  E (from-scratch, 500, target):  {e:.4f}")
        print(f"\n  Δ (A − B) = {a - b:+.4f}   ← original warm-start advantage")
        print(f"  Δ (A − C) = {a - c:+.4f}   ← after matching boost count")
        print(f"  Δ (A − D) = {a - d:+.4f}   ← after matching boost AND scaler")
        print(f"  Δ (C − B) = {c - b:+.4f}   ← effect of extra rounds alone")
        print(f"  Δ (D − C) = {d - c:+.4f}   ← effect of source scaler alone")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    for dataset in df_out["dataset"].unique():
        sub = df_out[df_out["dataset"] == dataset].set_index("configuration")
        a = sub.loc["A_warm", "auc_mean"]
        c = sub.loc["C_scratch_400", "auc_mean"]
        d = sub.loc["D_scratch_src_400", "auc_mean"]

        delta_c = a - c
        delta_d = a - d

        print(f"\n{dataset}:")
        if delta_d < 0.005:
            print(f"  Source initialization does NOT meaningfully help")
            print(f"  (Δ after matching boost+scaler = {delta_d:+.4f})")
            print(f"  The earlier warm-start 'advantage' was largely extra")
            print(f"  boosting rounds and source-scaler coordinate alignment.")
        elif delta_d < 0.020:
            print(f"  Modest residual benefit from source initialization")
            print(f"  (Δ after matching = {delta_d:+.4f})")
            print(f"  Some warm-start advantage is genuine, but smaller than")
            print(f"  the original comparison suggested.")
        else:
            print(f"  Substantial residual benefit from source initialization")
            print(f"  (Δ after matching = {delta_d:+.4f})")
            print(f"  Warm-start provides genuine gain beyond boost count and scaler.")


def _record(bucket, y_true, prob):
    """Append metrics for one fold into the bucket."""
    pred = prob.argmax(1)
    bucket["auc"].append(roc_auc_score(y_true, prob, multi_class="ovr",
                                         average="macro"))
    bucket["f1"].append(f1_score(y_true, pred, average="macro",
                                   zero_division=0))
    bucket["bacc"].append(balanced_accuracy_score(y_true, pred))


if __name__ == "__main__":
    main()