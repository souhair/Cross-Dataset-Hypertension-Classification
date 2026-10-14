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


# polarity_full_test.py
"""
Extend polarity_multi_test to the full Aurora-BP cohort.

Reports:
  - Median rise time (ms) across all files
  - Median fall time (ms) across all files
  - Median rise/fall ratio
  - Mean optical offset of raw signal (before DC removal)
  - Fraction of files where flip helps (fewer peaks detected)
"""
import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from math import gcd
from scipy.signal import resample_poly

ROOT = (r"Aurora\measurements_auscultatory"
        r"\measurements_auscultatory\a000")


def downsample(x, fs_in=500, fs_out=125):
    g = gcd(fs_in, fs_out)
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    return resample_poly(x, fs_out // g, fs_in // g)


def measure(sig, fs=125):
    clean = nk.ppg_clean(sig, sampling_rate=fs)
    peaks_df, _ = nk.ppg_peaks(clean, sampling_rate=fs)
    peaks = np.where(peaks_df["PPG_Peaks"].values == 1)[0]
    if len(peaks) < 4:
        return None, None, len(peaks)
    rise, fall = [], []
    for i in range(1, len(peaks) - 1):
        p, nxt = peaks[i], peaks[i + 1]
        search = max(peaks[i - 1], p - 50)
        foot = search + int(np.argmin(clean[search:p]))
        nf_end = min(nxt, p + 50)
        nf = p + int(np.argmin(clean[p:nf_end]))
        rise.append((p - foot) / fs * 1000)
        fall.append((nf - p) / fs * 1000)
    if not rise:
        return None, None, len(peaks)
    return float(np.median(rise)), float(np.median(fall)), len(peaks)


def main():
    files = sorted([f for f in os.listdir(ROOT) if f.endswith(".tsv")])
    print(f"Testing {len(files)} files in {ROOT}\n")

    raw_means = []
    rise_all = []
    fall_all = []
    ratios_all = []
    better_with_flip = 0
    total = 0

    for f in files:
        try:
            raw = pd.read_csv(os.path.join(ROOT, f), sep="\t")["optical"].to_numpy(float)
        except Exception:
            continue
        raw_means.append(raw.mean())

        sig_flip = downsample(-raw)
        sig_noflip = downsample(raw)

        r_f, fall_f, n_f = measure(sig_flip)
        r_n, fall_n, n_n = measure(sig_noflip)

        # Determine which orientation gives more physiological rise/fall ratio
        # (normal < 1; inverted > 1)
        if r_f and fall_f:
            ratios_all.append(r_f / fall_f)
            rise_all.append(r_f)
            fall_all.append(fall_f)

        if n_f is not None and n_n is not None:
            total += 1
            if n_f > n_n:
                better_with_flip += 1

    print("=" * 60)
    print("RAW SIGNAL MEAN (before DC removal)")
    print("=" * 60)
    rm = np.array(raw_means)
    print(f"  n files:                {len(rm)}")
    print(f"  mean optical offset:    {rm.mean():.1f}")
    print(f"  median optical offset:  {np.median(rm):.1f}")
    print()

    print("=" * 60)
    print("RISE / FALL MEDIAN ACROSS FILES (Flipped polarity)")
    print("=" * 60)
    if rise_all:
        r = float(np.median(rise_all))
        f = float(np.median(fall_all))
        ratio = float(np.median(ratios_all))
        print(f"  median rise time:  {r:.1f} ms")
        print(f"  median fall time:  {f:.1f} ms")
        print(f"  median ratio:      {ratio:.3f}")
    print()

    print(f"  Files where flipped orientation yields more peaks: "
          f"{better_with_flip}/{total}")


if __name__ == "__main__":
    main()