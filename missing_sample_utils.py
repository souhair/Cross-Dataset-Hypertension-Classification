# missing_sample_utils.py
"""Handle missing samples without silently compressing time."""
import numpy as np


def handle_missing_samples(signal, max_gap_fraction=0.02):
    """
    Handle non-finite samples in a signal.

    Short NaN gaps (< max_gap_fraction of signal length) are linearly
    interpolated, preserving the original length.
    Longer gaps cause the signal to be rejected (returns None).

    Parameters
    ----------
    signal : 1D np.array
    max_gap_fraction : float
        Maximum allowed NaN run as a fraction of total length.

    Returns
    -------
    Signal with NaNs interpolated (same length), or None if a gap too long.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return None
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if not finite.any():
        return None

    # Find longest NaN run
    n = len(signal)
    max_run = 0
    current_run = 0
    for f in finite:
        if f:
            current_run = 0
        else:
            current_run += 1
            if current_run > max_run:
                max_run = current_run

    if max_run > max_gap_fraction * n:
        return None   # too many missing samples

    # Interpolate
    idx = np.arange(n)
    signal = signal.copy()
    signal[~finite] = np.interp(idx[~finite], idx[finite], signal[finite])
    return signal