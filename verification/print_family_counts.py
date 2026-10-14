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


# print_family_counts.py
"""Print the canonical per-family feature counts on the Aurora dataset.
This is the source of truth for Table 1 in the paper."""
from analysis.v2_utils import load_aurora_raw
from feature_families import count_by_family, all_families

df, CANDIDATE = load_aurora_raw()
counts = count_by_family(CANDIDATE)

print("=" * 60)
print("Feature counts per family (mutually exclusive)")
print("=" * 60)
total = 0
for fam in all_families():
    n = counts[fam]
    total += n
    print(f"  {fam:25s}  {n:>5d}")
print("-" * 60)
print(f"  {'TOTAL':25s}  {total:>5d}")
print("=" * 60)