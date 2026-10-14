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


# rebuild_fig6_and_fig11.py
"""
Regenerate Fig 6 (SHAP family contribution) and Fig 11 (ablation) with the
corrected mutually-exclusive family definitions.

Fig 6 — two-panel: total SHAP sum + mean SHAP per feature
Fig 11 — two-panel: leave-one-out ΔAUC + family-alone AUC (9 families each)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"add_your_root_folder"
RES  = os.path.join(ROOT, "out_v2", "results")
FIG  = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


# ---------------------------------------------------------------------------
# Fig 6 — SHAP family contribution (total + per-feature)
# ---------------------------------------------------------------------------
def fig6_shap_families():
    df = pd.read_csv(os.path.join(RES, "v2_shap_family_summary.csv"))

    # Prettier family labels
    label_map = {
        "period morphology": "Period\nmorphology",
        "SMD": "SMD\n(F, Φ, Ψ)",
        "morphology": "Morphology",
        "spectral+nonlinear": "Spectral +\nnon-linear",
        "other": "Other",
        "HTN proxies": "HTN\nproxies",
        "HRV": "HRV",
        "ppg variability": "PPG\nvariability",
        "demographics": "Demo-\ngraphics",
    }
    df["label"] = df["family"].map(label_map).fillna(df["family"])

    df_total = df.sort_values("total_shap", ascending=True)
    df_mean  = df.sort_values("mean_shap_per_feature", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: total SHAP sum ---
    ax = axes[0]
    colors = ["#c53030" if f == "period morphology" else "#2b6cb0"
              for f in df_total["family"]]
    ax.barh(df_total["label"], df_total["total_shap"], color=colors)
    ax.set_xlabel("Sum of mean |SHAP|")
    ax.set_title("(a) Total family attribution")
    for i, (v, n) in enumerate(zip(df_total["total_shap"], df_total["n_features"])):
        ax.text(v + 0.01, i, f"{v:.2f}  (n={n})", va="center", fontsize=8)
    ax.set_xlim(0, df_total["total_shap"].max() * 1.30)

    # --- Right: mean SHAP per feature ---
    ax = axes[1]
    colors = ["#c53030" if f == "demographics" else "#2b6cb0"
              for f in df_mean["family"]]
    ax.barh(df_mean["label"], df_mean["mean_shap_per_feature"], color=colors)
    ax.set_xlabel("Mean |SHAP| per feature")
    ax.set_title("(b) Per-feature leverage")
    for i, v in enumerate(df_mean["mean_shap_per_feature"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, df_mean["mean_shap_per_feature"].max() * 1.25)

    fig.suptitle("Feature-family contribution by SHAP — Aurora-BP v2",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = os.path.join(FIG, "fig6_feature_family_shap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig6] saved {out}")


# ---------------------------------------------------------------------------
# Fig 11 — Ablation (leave-one-out + family-alone, 9 families)
# ---------------------------------------------------------------------------
def fig11_ablation():
    df = pd.read_csv(os.path.join(RES, "v2_ablation_corrected.csv"))
    base = df.loc[df["config"] == "ALL", "auc_mean"].iloc[0]

    # Leave-one-out rows
    loo = df[df["config"].str.startswith("ALL minus ", na=False)].copy()
    loo["family"] = loo["config"].str.replace("ALL minus ", "", regex=False)
    loo["delta"] = loo["auc_mean"] - base
    loo = loo.sort_values("delta")

    # Family-alone rows
    alone = df[df["config"].str.endswith(" only", na=False) &
               ~df["config"].str.startswith("ALL minus ", na=False)].copy()
    alone["family"] = alone["config"].str.replace(" only", "", regex=False)
    alone = alone.sort_values("auc_mean")

    label_map = {
        "period morphology": "period morphology",
        "SMD": "SMD",
        "morphology": "morphology",
        "spectral+nonlinear": "spectral+nonlinear",
        "other": "other",
        "HTN proxies": "HTN proxies",
        "HRV": "HRV",
        "ppg variability": "ppg variability",
        "demographics": "demographics",
    }
    loo["label"] = loo["family"].map(label_map)
    alone["label"] = alone["family"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: leave-one-out Δ ---
    ax = axes[0]
    colors = ["#c53030" if d < -0.01 else "#4a5568" for d in loo["delta"]]
    ax.barh(loo["label"], loo["delta"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔAUC vs all-features baseline")
    ax.set_title(f"(a) Leave-one-family-out  (baseline = {base:.3f})")
    for i, v in enumerate(loo["delta"]):
        ax.text(v - 0.001, i, f"{v:+.3f}", va="center", ha="right",
                fontsize=8, color="white" if v < -0.01 else "black")

    # --- Right: family-alone ---
    ax = axes[1]
    ax.barh(alone["label"], alone["auc_mean"], color="#2b6cb0")
    ax.axvline(base, ls="--", lw=1, color="gray",
               label=f"All features ({base:.3f})")
    ax.set_xlim(0.5, 0.82)
    ax.set_xlabel("AUC (family alone)")
    ax.set_title("(b) Discriminative power per family")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for i, v in enumerate(alone["auc_mean"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG, "fig11_ablation_delta.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig11] saved {out}")


if __name__ == "__main__":
    print(f"[build] output dir: {FIG}")
    fig6_shap_families()
    fig11_ablation()
    print("Done.")