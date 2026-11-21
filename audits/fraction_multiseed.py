import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import SEED, XGB_BEST, XGB_FINETUNE, NAN_THRESH, CORR_THRESH
from analysis.v2_utils import (CSV, load_aurora_raw, load_target_csv,
                       aurora_medians, RESULTS)
from fold_preprocessor import FoldAwarePreprocessor

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
N_SPLITS = 5


def run_fraction_seed(name, csv_path, df_aur, FEATURES, medians, seed):
    tg = load_target_csv(csv_path, FEATURES, medians)
    X = tg[FEATURES].values.astype(np.float64)
    y = tg.htn3.values.astype(np.int64)
    g = tg.subject_id.astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    ft_pool, test_idx = next(gss.split(X, y, groups=g))

    X_src = df_aur[FEATURES].values.astype(np.float64)
    y_src = df_aur.htn3.values.astype(np.int64)
    scaler_src = StandardScaler().fit(X_src)
    src = XGBClassifier(**XGB_BEST)
    src.fit(scaler_src.transform(X_src), y_src)

    all_subj = np.unique(g[ft_pool])
    rng = np.random.default_rng(seed)
    rows = []
    for frac in FRACTIONS:
        n_sub = max(3, int(len(all_subj) * frac))
        sel = rng.choice(all_subj, size=n_sub, replace=False)
        train_mask = np.isin(g[ft_pool], sel)
        train_idx = ft_pool[train_mask]

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        sc = StandardScaler().fit(X_tr)
        X_tr_t = sc.transform(X_tr); X_te_t = sc.transform(X_te)
        X_tr_w = scaler_src.transform(X_tr); X_te_w = scaler_src.transform(X_te)

        # Warm-start
        try:
            m1 = XGBClassifier(**XGB_FINETUNE)
            try:
                m1.fit(X_tr_w, y_tr, xgb_model=src.get_booster())
            except Exception:
                m1.fit(X_tr_w, y_tr)
            p1 = m1.predict_proba(X_te_w)
            auc1 = roc_auc_score(y_te, p1, multi_class="ovr", average="macro")
        except Exception:
            auc1 = np.nan

        # From-scratch
        try:
            m2 = XGBClassifier(**XGB_BEST)
            m2.fit(X_tr_t, y_tr)
            p2 = m2.predict_proba(X_te_t)
            auc2 = roc_auc_score(y_te, p2, multi_class="ovr", average="macro")
        except Exception:
            auc2 = np.nan

        rows.append({
            "dataset": name, "seed": seed, "fraction": frac,
            "n_train_subjects": n_sub, "n_train_segments": len(train_idx),
            "pretrained_auc": auc1, "scratch_auc": auc2,
            "delta": auc1 - auc2 if np.isfinite(auc1) and np.isfinite(auc2) else np.nan,
        })
        print(f"  seed={seed} frac={frac:.0%} n={n_sub}  "
              f"pretrained={auc1:.4f}  scratch={auc2:.4f}  delta={auc1-auc2:+.4f}")

    return rows


def main():
    df_aur, CANDIDATE = load_aurora_raw()
    pre = FoldAwarePreprocessor(NAN_THRESH, CORR_THRESH).fit(df_aur, CANDIDATE)
    FEATURES = pre.feature_list_
    medians = aurora_medians(df_aur, FEATURES)
    X_aur = pre.transform(df_aur)
    df_aur_feat = df_aur.copy()
    for i, c in enumerate(FEATURES):
        df_aur_feat[c] = X_aur[:, i]

    print(f"Aurora: {len(df_aur_feat)} seg, {len(FEATURES)} features")

    all_rows = []
    for name, key in [("PPG-BP", "ppgbp"),
                       ("PulseDB-Vital", "pulsedb_vital"),
                       ("SWCS", "ds4")]:
        path = CSV.get(key)
        if not path or not os.path.exists(path):
            print(f"\n[skip] {name}")
            continue
        print(f"\n{'='*72}\n{name} — {N_SPLITS} seeds\n{'='*72}")
        for seed_i in range(N_SPLITS):
            seed = SEED + seed_i * 100
            print(f"\n  --- seed {seed} ---")
            all_rows.extend(run_fraction_seed(
                name, path, df_aur_feat, FEATURES, medians, seed))

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS, "v2_fraction_multiseed_raw.csv"), index=False)

    # Aggregate
    summary = df.groupby(["dataset", "fraction"]).agg(
        n_seeds=("seed", "count"),
        pretrained_mean=("pretrained_auc", "mean"),
        pretrained_std=("pretrained_auc", "std"),
        scratch_mean=("scratch_auc", "mean"),
        scratch_std=("scratch_auc", "std"),
        delta_mean=("delta", "mean"),
        delta_std=("delta", "std"),
    ).reset_index()

    # summary["delta_ci_lo"] = summary["delta_mean"] - 1.96 * summary["delta_std"] / np.sqrt(summary["n_seeds"])
    # summary["delta_ci_hi"] = summary["delta_mean"] + 1.96 * summary["delta_std"] / np.sqrt(summary["n_seeds"])
    # Use t-distribution critical value for df=4 (5 seeds)
    t_crit = 2.776
    summary["delta_ci_lo"] = summary["delta_mean"] - t_crit * summary["delta_std"] / np.sqrt(summary["n_seeds"])
    summary["delta_ci_hi"] = summary["delta_mean"] + t_crit * summary["delta_std"] / np.sqrt(summary["n_seeds"])
    summary.to_csv(os.path.join(RESULTS, "v2_fraction_multiseed_summary.csv"), index=False)

    print("\n" + "=" * 72)
    print("MULTI-SEED FRACTION SWEEP — SUMMARY")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
