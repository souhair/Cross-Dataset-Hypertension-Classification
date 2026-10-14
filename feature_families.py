# feature_families.py
"""
Single source of truth for feature-family membership.

Every feature is assigned to exactly ONE family. Both v2_ablation.py and
v2_shap_analysis.py import from here; neither should define its own
overlapping predicates.

Family precedence (checked in this order):
    SMD               → step_* and explicit period_*phi*/psi*/F features
    demographics      → 'age' or 'demo_*'
    spectral+nonlinear→ 'spec_*', 'wavelet_*', 'hilbert_*', 'inst_freq_*'
    morphology        → sdppg_*, tdppg_*, notch_*, timing_*, aix*, ...
    HRV               → hrv_*
    ppg variability   → ppg_* (except ppg_width_* and ppg_shape_* which
                        are morphology — see precedence)
    HTN proxies       → htn_*
    period morphology → period_* not captured by SMD
    other             → everything else
"""
from typing import Dict, List

# ---- Family labels used throughout the codebase -------------------------
FAMILIES: List[str] = [
    "SMD",
    "period morphology",
    "demographics",
    "spectral+nonlinear",
    "morphology",
    "HRV",
    "ppg variability",
    "HTN proxies",
    "other",
]

# ---- SMD period keys (used to route period_* features) ------------------
_SMD_PERIOD_KEYS = (
    "f_mean", "f_std", "f_max",
    "phi_value", "psi_mean", "psi_std", "psi_max",
    "n_local_maxima", "n_local_minima", "pulse_duration_ms",
)


# ---- Predicates (defined for documentation; use family_of in code) ------
def is_smd_strict(name: str) -> bool:
    """Explicit F/Phi/Psi-derived features only."""
    if name.startswith("step_"):
        return True
    if name.startswith("period_phi"):
        return True
    if name.startswith("period_"):
        return any(k in name for k in _SMD_PERIOD_KEYS)
    return False


def is_demographics(name: str) -> bool:
    return name == "age" or name.startswith("demo_")


def is_spectral(name: str) -> bool:
    return name.startswith(("spec_", "wavelet_", "hilbert_", "inst_freq_"))


def is_morphology(name: str) -> bool:
    prefixes = ("sdppg_", "tdppg_", "notch_", "timing_", "aix",
                "diastolic_", "compliance_", "elasticity_",
                "arterial_", "crest_", "shoulder_",
                "peak_bifurcation", "systolic_roundness",
                "inflection_", "augmentation_", "reflection_",
                "pulse_pressure_", "ppg_width_", "ppg_shape_",
                "morph_", "deriv_", "forward_wave_",
                "reflected_wave_", "dicrotic_notch_")
    return name.startswith(prefixes)


def is_hrv(name: str) -> bool:
    return name.startswith("hrv_")


def is_ppg_variability(name: str) -> bool:
    """Variability features.

    Deliberately EXCLUDES ppg_width_* and ppg_shape_* which are counted
    under morphology (see family_of precedence order). This prevents a
    feature from appearing in two families.
    """
    if not name.startswith("ppg_"):
        return False
    if name.startswith(("ppg_width_", "ppg_shape_")):
        return False
    return True


def is_htn_proxy(name: str) -> bool:
    return name.startswith("htn_")


def is_period_morphology(name: str) -> bool:
    """Period-level aggregates that are NOT F/Phi/Psi-derived."""
    if not name.startswith("period_"):
        return False
    return not is_smd_strict(name)


# ---- Authoritative classifier ------------------------------------------
def family_of(name: str) -> str:
    """Return exactly one family for the given feature name.

    Precedence order matters: SMD wins over period morphology, and
    morphology wins over ppg variability, so ppg_width_* is morphology.
    """
    if is_smd_strict(name):
        return "SMD"
    if is_demographics(name):
        return "demographics"
    if is_spectral(name):
        return "spectral+nonlinear"
    if is_morphology(name):
        return "morphology"
    if is_hrv(name):
        return "HRV"
    if is_ppg_variability(name):
        return "ppg variability"
    if is_htn_proxy(name):
        return "HTN proxies"
    if is_period_morphology(name):
        return "period morphology"
    return "other"


# ---- Convenience helpers ------------------------------------------------
def all_families() -> List[str]:
    return list(FAMILIES)


def features_by_family(feature_list: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {f: [] for f in FAMILIES}
    for f in feature_list:
        out[family_of(f)].append(f)
    return out


def count_by_family(feature_list: List[str]) -> Dict[str, int]:
    return {k: len(v) for k, v in features_by_family(feature_list).items()}