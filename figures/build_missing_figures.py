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


# build_missing_figures.py
"""
Build/rebuild all analysis figures that are missing from out_v2/results/figures/.

Reads (must exist):
  out_v2/results/v2_nn_sweep_summary.csv
  out_v2/results/v2_ablation_corrected.csv
  out_v2/results/v2_transfer_strategies_corrected.csv
  out_v2/results/v2_fraction_experiment_corrected.csv

Writes to out_v2/results/figures/:
  fig8_nn_comparison.png
  fig11_ablation_delta.png
  fig12_transfer_heatmap.png
  fig13_fraction_curves.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = r"add_your_root_folder"
RES = os.path.join(ROOT, "out_v2", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Fig 8 — Neural network sweep comparison (bar chart, AUC + F1)
# ---------------------------------------------------------------------------
def fig8_nn_comparison():
    df = _load("v2_nn_sweep_summary.csv").copy()
    # Normalise column names (tolerate variants)
    colmap = {
        "architecture": "architecture",
        "AUC_mean": "auc", "auc_mean": "auc",
        "AUC_std": "auc_std", "auc_std": "auc_std",
        "MacroF1_mean": "f1", "f1_mean": "f1",
        "MacroF1_std": "f1_std", "f1_std": "f1_std",
        "BalAcc_mean": "bacc", "bacc_mean": "bacc",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    order = df.sort_values("auc", ascending=False)["architecture"].tolist()
    df = df.set_index("architecture").loc[order].reset_index()

    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 7.5))# 9, 4.4
    b1 = ax.bar(x - w/2, df["auc"], w,
                yerr=df.get("auc_std"), capsize=3,
                color="#2b6cb0", label="AUC")
    b2 = ax.bar(x + w/2, df["f1"], w,
                yerr=df.get("f1_std"), capsize=3,
                color="#dd6b20", label="Macro F1")

    # Highlight best AUC
    best = df["auc"].idxmax()
    b1[best].set_color("#1a365d")

    ax.axhline(0.7777, ls="--", lw=1, color="gray",
               label="XGBoost AUC reference")
    ax.set_xticks(x)
    ax.set_xticklabels(df["architecture"], rotation=25, ha="right")
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("Score")
    ax.set_title("Neural-network sweep vs XGBoost reference — Aurora-BP")
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    plt.tight_layout()
    out = os.path.join(FIG, "fig8_nn_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig8] saved {out}")


# ---------------------------------------------------------------------------
# Fig 11 — Ablation: leave-one-family-out ΔAUC
# ---------------------------------------------------------------------------
def fig11_ablation():
    df = _load("v2_ablation_corrected.csv").copy()
    base = df.loc[df["config"] == "ALL", "auc_mean"].iloc[0]

    # Leave-one-out: rows whose config begins with "ALL minus "
    loo = df[df["config"].str.startswith("ALL minus ", na=False)].copy()
    loo["family"] = (loo["config"]
                     .str.replace("ALL minus ", "", regex=False)
                     .str.replace(" only", "", regex=False))
    loo["delta"] = loo["auc_mean"] - base
    loo = loo.sort_values("delta")

    # Family-alone: rows whose config ends with " only" AND does NOT start with "ALL minus "
    alone = df[(df["config"].str.endswith(" only", na=False)) &
               (~df["config"].str.startswith("ALL minus ", na=False))].copy()
    alone["family"] = alone["config"].str.replace(" only", "", regex=False)
    alone = alone.sort_values("auc_mean")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: leave-one-out Δ ---
    ax = axes[0]
    colors = ["#c53030" if d < -0.01 else "#4a5568" for d in loo["delta"]]
    ax.barh(loo["family"], loo["delta"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔAUC vs. all-features baseline")
    ax.set_title(f"Leave-one-family-out  (baseline AUC = {base:.3f})")
    for i, v in enumerate(loo["delta"]):
        ax.text(v - 0.001, i, f"{v:+.3f}", va="center", ha="right",
                fontsize=12, color="white" if v < -0.01 else "black")

    # --- right: family-alone AUC ---
    ax = axes[1]
    ax.barh(alone["family"], alone["auc_mean"], color="#2b6cb0")
    ax.axvline(base, ls="--", lw=1, color="gray",
               label=f"All features ({base:.3f})")
    ax.set_xlim(0.5, 0.82)
    ax.set_xlabel("AUC (family alone)")
    ax.set_title("Discriminative power per family")
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    for i, v in enumerate(alone["auc_mean"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=12)

    plt.tight_layout()
    out = os.path.join(FIG, "fig11_ablation_delta.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig11] saved {out}")

# ---------------------------------------------------------------------------
# Fig 12 — Transfer strategy heatmap
# ---------------------------------------------------------------------------
def fig12_transfer():
    df = _load("v2_transfer_strategies_corrected.csv").copy()

    # [tier3] DS4 -> SWCS rename
    df["dataset"] = df["dataset"].replace({"DS4": "SWCS"})
    pivot = df.pivot(index="strategy", columns="dataset", values="auc_mean")

    # Order rows and columns logically
    strat_order = ["zero_shot", "from_scratch", "warm_start",
                   "target_scaler", "coral"]
    ds_order = ["PPG-BP", "PulseDB-Vital", "SWCS"]
    pivot = pivot.reindex(index=[s for s in strat_order if s in pivot.index],
                          columns=[d for d in ds_order if d in pivot.columns])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0.45, vmax=0.78, cbar_kws={"label": "AUC"},
                linewidths=0.5, ax=ax)

    # Bold the winning strategy per dataset
    for j, ds in enumerate(pivot.columns):
        i = int(np.argmax(pivot[ds].values))
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor="crimson", lw=2.2))

    ax.set_xlabel("")
    ax.set_ylabel("Transfer strategy")
    ax.set_title("Cross-dataset transfer — mean AUC "
                 "(red box = best per dataset)")
    plt.tight_layout()
    out = os.path.join(FIG, "fig12_transfer_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig12] saved {out}")


# ---------------------------------------------------------------------------
# Fig 13 — Few-shot learning curves (pretrained vs scratch)
# ---------------------------------------------------------------------------
def fig13_fraction():
    df = _load("v2_fraction_experiment_corrected.csv").copy()
    # ds_title= ["PPG-BP", "PulseDB-Vital", "SWCS"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
    for ax, ds in zip(axes, ["PPG-BP", "PulseDB-Vital", "DS4"]):
        frac_col = "train_pool_fraction" if "train_pool_fraction" in df.columns else "fraction"
        sub = df[df["dataset"] == ds].sort_values(frac_col)
        if sub.empty:
            ax.set_visible(False)
            continue
        frac = sub[frac_col] * 100
        ax.plot(frac, sub["pretrained_auc"], "o-",
                color="#2b6cb0", label="Pretrained (Aurora)")
        ax.plot(frac, sub["scratch_auc"], "s--",
                color="#dd6b20", label="From scratch")
        ax.fill_between(frac, sub["pretrained_auc"], sub["scratch_auc"],
                        where=sub["pretrained_auc"] >= sub["scratch_auc"],
                        color="#2b6cb0", alpha=0.10,
                        label=None)
        ax.fill_between(frac, sub["pretrained_auc"], sub["scratch_auc"],
                        where=sub["pretrained_auc"] < sub["scratch_auc"],
                        color="#dd6b20", alpha=0.10,
                        label=None)
        if ds == "DS4":
            ax.set_title("SWCS")
        else:
            ax.set_title(ds)
        # ax.set_title(ds_title)
        ax.set_xlabel("Training fraction (%)")
        ax.set_xticks([10, 25, 50, 100])
        ax.set_ylim(0.30, 0.95)
        ax.axhline(0.5, lw=0.7, ls=":", color="gray")

    axes[0].set_ylabel("AUC")
    axes[0].legend(loc="lower right", fontsize=14, frameon=False)
    # fig.suptitle("Few-shot transfer — pretrained vs from-scratch",
    #              fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG, "fig13_fraction_curves.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig13] saved {out}")


if __name__ == "__main__":
    print(f"[figures] output dir: {FIG}")
    try:
        fig8_nn_comparison()
    except Exception as e:
        print(f"[fig8] FAILED: {e}")
    try:
        fig11_ablation()
    except Exception as e:
        print(f"[fig11] FAILED: {e}")
    try:
        fig12_transfer()
    except Exception as e:
        print(f"[fig12] FAILED: {e}")
    try:
        fig13_fraction()
    except Exception as e:
        print(f"[fig13] FAILED: {e}")
    print("Done.")