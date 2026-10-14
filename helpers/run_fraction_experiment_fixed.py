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


# run_fraction_experiment_fixed.py
"""
Run v2_fraction_experiment.py with the seed fix already applied,
then post-process the output CSV to clarify fraction semantics.
"""
import os, subprocess, sys

ROOT = r"add_your_root_folder"
os.chdir(ROOT)

# 1. Ensure seeds are patched
print("[step 1] Patching seeds...")
subprocess.run([sys.executable, "patch_analysis_seeds.py"], check=True)

# 2. Run the fraction experiment
print("\n[step 2] Running v2_fraction_experiment.py...")
r = subprocess.run([sys.executable, "v2_fraction_experiment.py"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    raise SystemExit(r.returncode)

# 3. Post-process the CSV
print("\n[step 3] Post-processing CSV...")
subprocess.run([sys.executable, "postprocess_fraction_csv.py"], check=True)

print("\nDone. Check out_v2/results/v2_fraction_experiment_corrected.csv")