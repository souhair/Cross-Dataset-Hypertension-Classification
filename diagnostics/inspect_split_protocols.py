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


# inspect_split_protocols.py
"""Print the split logic used by each script to confirm protocol difference."""
import os, re

ROOT = r"add_your_root_folder"
FILES = ["v2_transfer_strategies.py", "v2_fraction_experiment.py"]

PATTERNS = [
    ("StratifiedGroupKFold", r"StratifiedGroupKFold\s*\([^)]*\)"),
    ("GroupShuffleSplit",    r"GroupShuffleSplit\s*\([^)]*\)"),
    ("train_test_split",     r"train_test_split\s*\([^)]*\)"),
    ("n_splits",             r"n_splits\s*=\s*\d+"),
    ("test_size",            r"test_size\s*=\s*[\d.]+"),
    ("random_state",         r"random_state\s*=\s*\d+"),
]

for fname in FILES:
    path = os.path.join(ROOT, fname)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    print("=" * 72)
    print(fname)
    print("=" * 72)
    for label, pat in PATTERNS:
        for m in re.finditer(pat, src):
            line_no = src[:m.start()].count("\n") + 1
            print(f"  [L{line_no:>4}] {label:<22} {m.group(0)[:70]}")
    print()