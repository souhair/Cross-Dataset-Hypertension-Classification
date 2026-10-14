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


# run_all_analysis.py
"""Run all downstream analysis scripts in order."""
import subprocess
import sys
import time

SCRIPTS = [
    "v2_3class.py",
    "v2_ablation.py",
    "v2_no_demographics.py",
    "v2_shap_analysis.py",
    "v2_figures.py",
    "v2_nn_sweep.py",
    "v2_transfer_strategies.py",
    "v2_external_cv_stratified.py",
    "v2_fraction_experiment.py",
    "v2_optional_analyses.py",
    "v2_threshold_tuning.py",
]

for s in SCRIPTS:
    print("\n" + "=" * 80)
    print(f"RUNNING: {s}")
    print("=" * 80)
    t0 = time.time()
    r = subprocess.run([sys.executable, s])
    print(f"\n[{s}] exit={r.returncode}  time={(time.time()-t0)/60:.2f} min")
    if r.returncode != 0:
        print(f"ERROR in {s}. Stopping.")
        sys.exit(1)

print("\n✓ All analysis scripts completed.")