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


# build_fig4_and_fig6.py
"""
Build Fig 4 (pipeline diagram) and Fig 6 (SHAP family contribution)
with the exact titles used in the paper.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"add_your_root_folder"
RES  = os.path.join(ROOT, "out_v2", "results")
FIG  = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})


# ---------------------------------------------------------------------------
# Fig 4 — Feature extraction and classification pipeline
# ---------------------------------------------------------------------------
def fig4_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, facecolor, edgecolor, fontsize=12, weight="normal"):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05,rounding_size=0.10",
                           linewidth=1.1, facecolor=facecolor,
                           edgecolor=edgecolor)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color="#4a5568"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.2, color=color))

    # ---- Row 1: Inputs (four datasets) ----
    datasets = [("Aurora-BP\n(500 Hz, N=656)", "#bee3f8"),
                ("PPG-BP\n(1000 Hz, N=219)", "#bee3f8"),
                ("PulseDB-Vital\n(125 Hz, N=250)", "#bee3f8"),
                ("SWCS / DS4\n(50 Hz, N=115)", "#bee3f8")]
    for i, (t, c) in enumerate(datasets):
        box(0.25 + i*2.65, 3.75, 2.35, 0.65, t, c, "#2b6cb0", 8.5, "bold")

    # ---- Row 2: Preprocessing ----
    box(0.25, 2.70, 10.5, 0.65,
        "Preprocessing:  DC removal  →  polyphase resample to canonical 125 Hz  →  "
        "band-pass filter (0.5–8 Hz)  →  NeuroKit2 ppg_clean  →  30-s windowing",
        "#fefcbf", "#b7791f", 9)

    # ---- Row 3: Feature families ----
    box(0.25, 1.85, 1.6, 0.55, "Peak\nDetection", "#c6f6d5", "#276749", 8.5)
    families = [("Period\nMorphology", "#c6f6d5"),
                ("SMD\n(F, Φ, Ψ)", "#c6f6d5"),
                ("Morphology\n(dicrotic, AIx)", "#c6f6d5"),
                ("Spectral +\nNon-linear", "#c6f6d5"),
                ("HRV /\nVariability", "#c6f6d5"),
                ("HTN\nProxies", "#c6f6d5"),
                ("Demo-\ngraphics", "#c6f6d5")]
    for i, (t, c) in enumerate(families):
        box(2.05 + i*1.20, 1.85, 1.15, 0.55, t, c, "#276749", 8)

    ax.text(5.5, 2.55,
            "437 candidate features  →  213 after correlation filter (|r| > 0.95)",
            ha="center", va="center", fontsize=12, style="italic",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="none", alpha=0.95))

    # ---- Row 4: Models ----
    box(0.25, 0.95, 3.4, 0.60,
        "Classification\nXGBoost (primary)  ·  LogReg  ·  RF  ·  9 NNs",
        "#e9d8fd", "#553c9a", 8.5)
    box(3.85, 0.95, 3.4, 0.60,
        "Evaluation\nSubject-level 5-fold CV  ·  Bootstrap 95% CI",
        "#e9d8fd", "#553c9a", 8.5)
    box(7.45, 0.95, 3.3, 0.60,
        "Transfer\nWarm-start  ·  CORAL  ·  Target-scaler  ·  Zero-shot",
        "#e9d8fd", "#553c9a", 8.5)

    # ---- Row 5: Outputs ----
    box(0.25, 0.10, 3.4, 0.55, "SHAP interpretability\n(feature + family)", "#fed7d7", "#9b2c2c", 8.5)
    box(3.85, 0.10, 3.4, 0.55, "Calibration + DCA\n(Brier, ECE, net benefit)", "#fed7d7", "#9b2c2c", 8.5)
    box(7.45, 0.10, 3.3, 0.55, "Few-shot analysis\n(fraction sweep, 10–100%)", "#fed7d7", "#9b2c2c", 8.5)

    # ---- Vertical arrows between rows ----
    for x in [3.4, 6.3, 9.2]:
        arrow(x, 3.70, x, 3.40)
        arrow(x, 2.65, x, 2.45)
        arrow(x, 1.80, x, 1.60)
        arrow(x, 0.90, x, 0.70)

    ax.set_title("Feature extraction and classification pipeline", fontsize=14, pad=6)
    plt.tight_layout()
    out = os.path.join(FIG, "fig4_pipeline_diagram.png")
    plt.savefig(out); plt.close()
    print(f"[fig4] saved {out}")


# ---------------------------------------------------------------------------
# Fig 6 — Feature-family contribution by SHAP
# ---------------------------------------------------------------------------
def fig6_shap_families():
    # Numbers come from v2_shap_analysis.py stdout (sum of mean |SHAP|, normalized to 1.0)
    data = pd.DataFrame({
        "family": ["period morphology", "demographics", "SMD",
                   "spectral+nonlinear", "morphology", "other",
                   "HTN proxies", "HRV", "ppg variability"],
        "shap":   [1.0000, 0.9087, 0.6530, 0.5215, 0.4660,
                   0.2910, 0.1172, 0.0950, 0.0440],
    }).sort_values("shap", ascending=False).reset_index(drop=True)

    colors = ["#2b6cb0"] * len(data)
    # highlight the three most important families
    for i in range(3):
        colors[i] = "#dd6b20" if i == 0 else "#2b6cb0"
    colors[0] = "#dd6b20"

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bars = ax.bar(range(len(data)), data["shap"], color=colors)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["family"], rotation=22, ha="right")
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.set_title("Feature-family contribution by SHAP", fontsize=12)
    ax.set_ylim(0, 1.10)

    for i, v in enumerate(data["shap"]):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=12)

    # Highlight the top family with a red bar for visual salience
    bars[0].set_color("#c53030")
    ax.legend([bars[0], bars[1]],
              ["dominant family", "other families"],
              loc="upper right", fontsize=12, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIG, "fig6_feature_family_shap.png")
    plt.savefig(out); plt.close()
    print(f"[fig6] saved {out}")


if __name__ == "__main__":
    print(f"[build] output dir: {FIG}")
    fig4_pipeline()
    fig6_shap_families()
    print("Done.")