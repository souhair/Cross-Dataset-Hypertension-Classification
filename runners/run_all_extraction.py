# run_all_extraction.py
"""Run all extraction runners in order.

Runs from any working directory. Set INCLUDE_PHYSIO_PPG=1 to include
the auxiliary Physio-PPG dataset, which is not one of the paper's
four cohorts.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(script_rel_path, label=None, stop_on_error=True):
    """Run a script by absolute path; return exit code."""
    label = label or script_rel_path
    script_abs = os.path.join(ROOT, script_rel_path)
    if not os.path.isfile(script_abs):
        print(f"\n[error] missing script: {script_abs}")
        return 1

    print("\n" + "=" * 80)
    print(f"RUNNING: {script_rel_path}")
    print("=" * 80)
    t0 = time.time()
    r = subprocess.run([PY, script_abs], cwd=ROOT)
    print(f"\n[{script_rel_path}] exit={r.returncode}  "
          f"time={(time.time() - t0) / 60:.2f} min")
    if r.returncode != 0 and stop_on_error:
        print(f"ERROR in {script_rel_path}. Stopping.")
        sys.exit(1)
    return r.returncode


# ---- Main extraction runners (paper cohorts) ----------------------------
MAIN_RUNNERS = [
    "runners/run_aurora_full_v2.py",
    "runners/run_ppgbp_full_v2.py",
    "runners/run_pulsedb_vital_full_v2.py",
    "runners/run_ds4_full_v2.py",
]

for script in MAIN_RUNNERS:
    run(script)

# ---- Optional auxiliary dataset (not a paper cohort) --------------------
if os.environ.get("INCLUDE_PHYSIO_PPG", "0") == "1":
    run("runners/run_physioppg_full_v2.py")
else:
    print("\n[skip] Physio-PPG (auxiliary; set INCLUDE_PHYSIO_PPG=1 to run)")

# ---- Post-processing ----------------------------------------------------
print("\nAll extraction runners completed. Running ID recovery...")
run("runners/v2_fix_subject_ids.py")
run("verification/verify_ids.py")

print("\nDone.")