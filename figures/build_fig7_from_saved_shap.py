# figures/build_fig7_from_saved_shap.py
"""
Rebuild fig7_per_class_shap.png from the SAVED outputs of v2_shap_analysis.py.

"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"add_your_root_folder"
RES = os.path.join(ROOT, "out_v2", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

CLASS_NAMES = ["Normal", "PreHTN", "HTN"]


def main():
    npz = os.path.join(RES, "v2_shap_per_class.npy")
    names_txt = os.path.join(RES, "v2_feature_names.txt")

    if not os.path.exists(npz):
        raise SystemExit(
            f"[error] {npz} not found. Run analysis/v2_shap_analysis.py first."
        )
    if not os.path.exists(names_txt):
        raise SystemExit(
            f"[error] {names_txt} not found. Run analysis/v2_shap_analysis.py first."
        )

    per_class = np.load(npz)   # shape (3, n_features) or (n_features, 3)
    with open(names_txt, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    print(f"[fig7] per_class shape: {per_class.shape}")
    print(f"[fig7] features: {len(features)}")

    # Ensure shape is (3, n_features)
    if per_class.shape[0] != 3 and per_class.shape[1] == 3:
        per_class = per_class.T
    assert per_class.shape[0] == 3, (
        f"Expected 3 classes, got shape {per_class.shape}"
    )
    assert per_class.shape[1] == len(features), (
        f"Feature axis mismatch: {per_class.shape[1]} vs {len(features)}"
    )

    top_n = 15
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for ax, mean_abs, cname in zip(axes, per_class, CLASS_NAMES):
        order = np.argsort(mean_abs)[::-1][:top_n]
        names = [features[i] for i in order]
        vals = mean_abs[order]
        ax.barh(range(top_n), vals[::-1], color="#2b6cb0")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names[::-1], fontsize=7.5)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(cname, fontsize=11, weight="bold")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Top {top_n} features per class by SHAP — Aurora-BP (held-out split)",
        fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG, "fig7_per_class_shap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig7] saved {out}")


if __name__ == "__main__":
    main()