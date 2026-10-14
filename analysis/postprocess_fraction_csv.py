# postprocess_fraction_csv.py
"""
Add clarifying columns to v2_fraction_experiment_corrected.csv.

Adds:
  train_pool_fraction       — the fraction label, relabeled to be honest
  frac_of_all_subjects      — n_train_subjects / total_subjects

Idempotent: refuses to re-process an already-processed CSV.
"""
import os
import pandas as pd

ROOT = r"add_your_root_folder"
RES = os.path.join(ROOT, "out_v2", "results")
csv_path = os.path.join(RES, "v2_fraction_experiment_corrected.csv")

if not os.path.exists(csv_path):
    raise SystemExit(f"Missing: {csv_path}")

# ---- TIER2B_IDEMPOTENT: refuse double processing -----------------------
_hdr = pd.read_csv(csv_path, nrows=1).columns.tolist()
if "train_pool_fraction" in _hdr:
    print(f"[skip] {csv_path} already contains 'train_pool_fraction'.")
    print(f"       Delete or rename the file to re-process.")
    raise SystemExit(0)

# Total subjects per dataset
TOTAL_SUBJECTS = {
    "PPG-BP": 218,
    "PulseDB-Vital": 250,
    "DS4": 115,
    "SWCS": 115,
}

df = pd.read_csv(csv_path)

if "fraction" not in df.columns or "n_train_subjects" not in df.columns:
    raise SystemExit(f"Unexpected columns: {list(df.columns)}")

# Back up the original
backup = csv_path.replace(".csv", "_orig.csv")
if not os.path.exists(backup):
    df.to_csv(backup, index=False)
    print(f"[backup] {backup}")

# Add clarifying columns
df["train_pool_fraction"] = df["fraction"]
df["frac_of_all_subjects"] = df.apply(
    lambda r: round(
        r["n_train_subjects"] / TOTAL_SUBJECTS.get(r["dataset"], r["n_train_subjects"]),
        3,
    ),
    axis=1,
)

# Reorder: put the clarifying columns at the front
front = ["dataset", "train_pool_fraction", "frac_of_all_subjects",
         "n_train_subjects", "n_train_segments"]
rest = [c for c in df.columns if c not in front + ["fraction"]]
df = df[front + rest]

df.to_csv(csv_path, index=False)
print(f"[post-processed] {csv_path}")
print()
print(df.to_string(index=False))