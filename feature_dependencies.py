# feature_dependencies.py
"""Features that depend on demographic inputs (age, height, weight, BMI)."""

DEMOGRAPHIC_DESCENDANTS = {
    # Direct demographic columns
    "age", "demo_age", "demo_bmi", "demo_height",
    "demo_weight", "demo_gender",
    # Age-dependent
    "step_Psi_age_adjusted",
    "step_vascular_compliance_index",
    # Height-dependent
    "stiffness_index", "htn_stiffness_index_proxy", "lasi_ms",
    # All period-level summaries of the above
    "period_mean__stiffness_index",
    "period_std__stiffness_index",
    "period_median__stiffness_index",
    "period_min__stiffness_index",
    "period_max__stiffness_index",
    "period_cv__stiffness_index",
    "period_mean__lasi_ms",
    "period_std__lasi_ms",
    "period_median__lasi_ms",
    "period_min__lasi_ms",
    "period_max__lasi_ms",
    "period_cv__lasi_ms",
}


def is_demographic_dependent(name):
    return name in DEMOGRAPHIC_DESCENDANTS