
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PPG Feature Extraction Pipeline

Outputs
-------
- HDF5 sequences : beat_matrix (N_beats x 34) + segment_vector (context)
- Flat CSV       : one row per segment with all features expanded
================================================================================
"""

import os
import sys
import warnings
import logging
import traceback
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch, hilbert
from scipy.stats import skew, kurtosis, iqr
import pandas as pd
import h5py
import neurokit2 as nk
import pywt

warnings.filterwarnings("ignore")


def prepare_output(out_csv, out_h5, overwrite=True):
    """Remove stale output files before a fresh run to avoid duplicates."""
    if not overwrite:
        return
    # [join_mask_v2] include periods CSV so stale rows are not retained
    periods_csv = out_csv.replace(".csv", "_periods.csv")
    for f in [out_csv, out_h5, periods_csv,
              out_csv + ".log", out_h5 + ".log"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[cleanup] removed {f}")
# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = 'pulsedb_vital/pulsedb_vital_source'
OUTPUT_CSV = 'pulsedb_vital_unified.csv'
OUTPUT_H5 = 'pulsedb_vital_unified_sequences.h5'
LOG_FILE = 'processed_files_vital_unified.log'
LOG_FILE_SEQ = 'processed_files_vital_unified_seq.log'
MAX_WORKERS = 6
SAVE_INTERVAL = 2000

# Frozen literature-facing period-morphology configuration.
PERIOD_P_DEFAULT = 2.0
PERIOD_FORMULA_REFERENCE = 'formulas functions F, phi, psi'
PERIOD_PSI_EVALUATION_RULE = 'midpoint_per_decline_step'


# ============================================================
# FEATURE ORDERS
# ============================================================
BEAT_FEATURE_ORDER = [
    'foot_pos', 'sys_pos', 'notch_pos', 'dia_pos',
    'sys_amp_rel', 'notch_amp_rel', 'dia_amp_rel',
    'rise_time_ms', 'pulse_width_ms', 'systolic_time_ms', 'diastolic_time_ms',
    'ts_td_ratio', 'delta_t_ms', 'reflection_index',
    'sys_area', 'dia_area', 'total_area',
    'max_rising_slope', 'upslope_mean', 'downslope_mean',
    'aix_percent', 'augmentation_pressure', 'inflection_point_ratio',
    'stiffness_index', 'lasi_ms', 'diastolic_decay_constant',
    'compliance_index', 'notch_width_ms', 'notch_sharpness',
    'notch_depth_ratio', 'notch_symmetry_index',
    'shoulder_present', 'shoulder_height_ratio', 'crest_time_ratio'
]

# 83 base + unique pipeline additions
SEGMENT_FEATURE_ORDER = [
    'step_Psi_age_adjusted',
    'step_vascular_compliance_index',
    'pep_variability',
    'lvet_variability',
    'pep_ms',
    'pwv_surrogate',
    'pep_lvet_ratio',
    'lvet_ms',
    'tdppg_d',
    'htn_diastolic_decay_tau',
    'compliance_index',
    'timing_diastolic_ms',
    'stiffness_index',
    'deriv_downslope_mean',
    'sdppg_e',
    'tdppg_f',
    'step_function_stability',
    'step_shape_consistency',
    'tdppg_bd_index',
    'step_Phi_short_term_var',
    'step_Phi_cv',
    'step_Phi_total_var',
    'step_Phi_consistency',
    'morph_perfusion_index',
    'htn_perfusion_index',
    'reflection_index_pda',
    'hrv_lf_hf_ratio',
    'augmentation_pressure',
    'step_F_Psi_ratio',
    'step_function_balance',
    'step_Phi_Psi_ratio',
    'sdppg_d',
    'ppg_amp_sdnn',
    'sdppg_aging_index',
    'ppg_area_cv',
    'spec_harmonic_h3_h1',
    'tdppg_e',
    'notch_sharpness',
    'beat_l2_distance_mean',
    'ppg_amp_rmssd',
    'tdppg_cf_index',
    'peak_bifurcation_index',
    'step_F_Phi_ratio',
    'reflection_index_v2',
    'htn_reflection_index_validated',
    'tdppg_c',
    'post_notch_recovery_slope',
    'sdppg_c',
    'tdppg_a',
    'reflected_wave_amplitude',
    'lasi_ms',
    'diastolic_decay_time_ms',
    'tdppg_b',
    'forward_wave_time',
    'forward_wave_amplitude',
    'dicrotic_notch_amplitude_pda',
    'timing_notch_depth_ratio',
    'notch_foot_ratio',
    'hrv_hf_power',
    'deriv_upslope_mean',
    'pulse_pressure_ppg',
    'aix_amplitude_based',
    'timing_reflection_index',
    'hrv_lf_power',
    'hilbert_std',
    'wavelet_energy_0',
    'deriv_max_velocity',
    'hrv_mean_nn',
    'rms',
    'wavelet_energy_1',
    'mean_step_duration_ms',
    'deriv_max_acceleration',
    'pulse_rate',
    'step_Phi_std',
    'timing_deceleration_ms',
    'spec_harmonic_h4_h1',
    'hilbert_mean',
    'htn_stiffness_index_proxy',
    'sdppg_b',
    'timing_ts_td_ratio',
    'sdppg_a',
    'timing_systolic_ms',
    'step_F_cv',
    'step_F_total_var',
    'step_F_consistency',
    'spec_fundamental_freq',
    'spec_harmonic_h2_h1',
    'baseline_wander_std',
    'hrv_rmssd',
    'elasticity_index',
    'spec_bandwidth',
    'morph_kurtosis',
    'spec_peak_skewness',
    'step_duration_std',
    'hrv_sdnn',
    'morph_skewness',
    'multiscale_entropy',
    'shoulder_time_ratio',
    'shoulder_height_ratio',
    'spec_entropy',
    'deriv_min_acceleration',
    'shoulder_present_ratio',
    'aix_percent',
    'htn_augmentation_index_validated',
    'ppg_width_cv',
    'spec_centroid',
    'hilbert_range',
    'hrv_cvnn',
    'htn_systolic_rise_time_pct',
    'signal_quality',
    'notch_symmetry_index',
    'dicrotic_notch_time_pda',
    'clipping_fraction',
    'deriv_min_velocity',
    'spec_very_hf_ratio',
    'step_F_short_term_var',
    'ppg_shape_consistency',
    'spec_hf_power_ratio',
    'spec_centroid_above_fund',
    'step_Psi_consistency',
    'zero_crossing_rate',
    'snr_db',
    'hrv_pnn50',
    'step_Psi_cv',
    'step_Psi_total_var',
    'step_Psi_short_term_var',
    'reflected_wave_time',
    'ppg_amp_cv',
    'wavelet_energy_2',
    'step_valley_count',
    'timing_delta_t_ms',
    'step_F_std',
    'notch_width_ms',
    'arterial_stiffness_proxy',
    'htn_normalized_pulse_volume',
    'diastolic_decay_constant',
    'systolic_roundness',
    'approx_entropy',
    'spec_entropy_highband',
    'spec_slope_1_4_hz',
    'step_Phi_mean',
    'step_Psi_mean',
    'htn_vascular_compliance_score',
    'inst_freq_mean',
    'timing_notch_position_pct',
    'notch_to_end_ratio',
    'htn_pulse_pressure_variability',
    'ppg_sample_entropy',
    'step_Psi_std',
    'step_F_mean',
    'inst_freq_range',
    'crest_factor',
    'inflection_point_ratio',
    'crest_time_ratio',
    'deriv_upslope_downslope_ratio',
    'ppg_lle',
    'inst_freq_std',
    'deriv_norm_max_rising_slope',
    'recurrence_rate',
    'deriv_velocity_skewness',
    'wavelet_energy_4',
    'step_peak_count',
    'wavelet_energy_3',
    'n_beats',
]

# ============================================================
# 1. HELPERS
# ============================================================
def to_scalar(x):
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x.flatten()[0]
    elif isinstance(x, list):
        return x[0] if len(x) == 1 else x
    return x


def safe_get_field(structure, field_name, default_value):
    try:
        if isinstance(structure, dict):
            value = structure.get(field_name, default_value)
        else:
            value = getattr(structure, field_name, default_value)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = default_value
        if isinstance(default_value, (int, float)) and isinstance(value, str):
            try:
                value = float(value)
            except Exception:
                value = default_value
        return value
    except Exception:
        return default_value


def safe_std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def trapezoid_compat(y, x=None, dx=1.0):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, dx=dx)
    return np.trapz(y, x=x, dx=dx)


def get_core_beats(beats):
    if beats is None:
        return []
    if len(beats) > 2:
        return beats[1:-1]
    return beats


def is_valid_fiducials(fid):
    if fid['sys_peak_idx'] <= fid['foot_idx']:
        return False
    if fid['sys_peak_amp'] <= fid['foot_amp']:
        return False
    if fid['notch_idx'] is not None and fid['notch_idx'] <= fid['sys_peak_idx']:
        return False
    if (fid['dia_peak_idx'] is not None and fid['notch_idx'] is not None and
            fid['dia_peak_idx'] <= fid['notch_idx']):
        return False
    return True


def extract_peak_indices(peaks_output):
    if peaks_output is None:
        return np.array([], dtype=int)
    if isinstance(peaks_output, np.ndarray):
        return peaks_output.astype(int)
    if hasattr(peaks_output, 'values'):
        if 'PPG_Peaks' in peaks_output.columns:
            return np.where(peaks_output['PPG_Peaks'].values == 1)[0].astype(int)
        return peaks_output.values.flatten().astype(int)
    return np.array([], dtype=int)


def decode_gender(gender_raw):
    try:
        if isinstance(gender_raw, str):
            g = gender_raw.strip().upper()
            if g in ('M', 'MALE', '1', '77'):
                return 1
            elif g in ('F', 'FEMALE', '0', '70'):
                return 0
            else:
                return 0
        g = int(float(gender_raw))
        if g == 77:
            return 1
        elif g == 70:
            return 0
        elif g in (0, 1):
            return g
        return 0
    except Exception:
        return 0


def safe_extract_patient_id(mat_filepath, subject_id):
    """Preserve the original participant ID as a string.

    Aurora IDs are strings like 'a0001'; PulseDB IDs are like 'p000022'.
    Never collapse unrecognized IDs to a shared default.
    """
    if subject_id is not None:
        sid = str(subject_id).strip()
        if sid and sid.lower() not in ("nan", "none", "unknown", "0", "0.0"):
            return sid
    fname = os.path.basename(mat_filepath)
    m = re.match(r'^([a-zA-Z]\d+)', fname)
    if m:
        return m.group(1)
    m = re.search(r'p(\d+)', fname)
    if m:
        return 'p' + m.group(1)
    return os.path.splitext(fname)[0]

# ============================================================
# 2. STEP REPRESENTATION (from ppg_pipeline.py v5)
# ============================================================
def calculate_age_coefficient(age):
    if age < 30:
        return 2
    elif age < 45:
        return 3
    elif age < 60:
        return 4
    else:
        return 5


def convert_to_step_representation(ppg, pulse_rate, fs):
    Y = ppg / (np.max(ppg) + 1e-10)
    Y = np.round(Y, decimals=4)
    total_time = len(Y) / fs
    dt = total_time / len(Y)
    X = np.arange(1, len(Y) + 1) * dt
    first = 0
    while first < len(Y) - 1 and Y[first + 1] == Y[first]:
        first += 1
    step_amps = [Y[first]]
    step_times = [X[first]]
    step_durations = [(first + 1) / fs]
    for i in range(first + 1, len(Y) - 1):
        if Y[i + 1] != Y[i]:
            step_amps.append(Y[i])
            step_times.append(X[i])
            step_durations.append(1 / fs)
        else:
            step_durations[-1] += 1 / fs
    step_amps.append(Y[-1])
    step_times.append(X[-1])
    return np.array(step_amps), np.array(step_times), np.array(step_durations)


def detect_step_extrema(step_amps):
    peaks = {'locations': [], 'amplitudes': []}
    valleys = {'locations': [], 'amplitudes': []}
    n = len(step_amps)
    for i in range(1, n - 1):
        if step_amps[i] > step_amps[i - 1] and step_amps[i] > step_amps[i + 1]:
            peaks['locations'].append(i)
            peaks['amplitudes'].append(step_amps[i])
        if step_amps[i] < step_amps[i - 1] and step_amps[i] < step_amps[i + 1]:
            valleys['locations'].append(i)
            valleys['amplitudes'].append(step_amps[i])
    peaks['locations'] = np.array(peaks['locations'])
    peaks['amplitudes'] = np.array(peaks['amplitudes'])
    valleys['locations'] = np.array(valleys['locations'])
    valleys['amplitudes'] = np.array(valleys['amplitudes'])
    return peaks, valleys


def calculate_peak_steepness(step_amps, step_durations, peak_loc):
    if peak_loc > 0 and peak_loc < len(step_amps):
        delta_A = step_amps[peak_loc] - step_amps[peak_loc - 1]
        delta_tau = step_durations[peak_loc]
        if delta_tau > 0:
            return abs(delta_A / delta_tau)
    return 0.0


def calculate_rise_phi1(step_amps, step_durations, peak_loc, p):
    """Literature-faithful Eq. (2) with a Python 0-based peak_loc.

    The source paper is 1-based: m_math = peak_loc + 1. The denominator uses
    (m_math - 1) = peak_loc. No artificial min(base, 2.0) cap is applied.
    """
    peak_loc = int(peak_loc)
    if peak_loc <= 0 or peak_loc >= len(step_amps):
        return 0.0
    return float(_paper_phi1_period(
        step_amps, step_durations, m_math=peak_loc + 1, p=p
    ))

def calculate_rise_phi2(step_amps, step_durations, valley_loc, peak_loc, p):
    """Literature-faithful Eq. (3) with Python 0-based step indices."""
    valley_loc = int(valley_loc)
    peak_loc = int(peak_loc)
    if valley_loc < 0 or peak_loc >= len(step_amps) or valley_loc >= peak_loc:
        return 0.0
    return float(_paper_phi2_period(
        step_amps, step_durations,
        k1_math=valley_loc + 1,
        m2_math=peak_loc + 1,
        p=p,
    ))

def calculate_rise_phi3(step_amps, step_durations, valley_loc, peak_loc, p):
    """Literature-faithful Eq. (4) with Python 0-based step indices."""
    valley_loc = int(valley_loc)
    peak_loc = int(peak_loc)
    if valley_loc < 0 or peak_loc >= len(step_amps) or valley_loc >= peak_loc:
        return 0.0
    return float(_paper_phi3_period(
        step_amps, step_durations,
        k2_math=valley_loc + 1,
        m3_math=peak_loc + 1,
        p=p,
    ))

def calculate_decay_function(step_amps, step_times, step_durations, peak_loc, valley_loc, p):
    if peak_loc >= valley_loc or peak_loc < 0 or valley_loc >= len(step_amps):
        return 0.0
    A_m = step_amps[peak_loc]
    n = valley_loc
    m = peak_loc
    psi_values = []
    for k in range(peak_loc + 1, valley_loc + 1):
        t_from_peak = step_times[k] - step_times[peak_loc]
        delta_tau_k = step_durations[k]
        denom = (n - m) * delta_tau_k
        if denom == 0:
            continue
        exponent = (-t_from_peak * p) / denom
        amplitude_diff = step_amps[k - 1] - step_amps[k]
        linear_term = (n - m) / p
        psi = A_m * np.exp(exponent) * amplitude_diff * linear_term
        if np.isfinite(psi) and psi >= 0:
            psi_values.append(psi)
    return float(np.mean(psi_values)) if psi_values else 0.0


def process_cardiac_cycles_per_beat(step_amps, step_times, step_durations, peaks, valleys, p, max_cycles=3):
    cycles = []
    num_peaks = len(peaks['locations'])
    num_valleys = len(valleys['locations'])
    for cycle_idx in range(min(num_peaks, max_cycles)):
        cycle = {'cycle_id': cycle_idx + 1}
        peak_loc = peaks['locations'][cycle_idx]
        cycle['F_value'] = calculate_peak_steepness(step_amps, step_durations, peak_loc)
        if cycle_idx == 0:
            cycle['Phi_value'] = calculate_rise_phi1(step_amps, step_durations, peak_loc, p)
        elif cycle_idx == 1 and num_valleys >= 1:
            v_loc = valleys['locations'][0]
            cycle['Phi_value'] = calculate_rise_phi2(step_amps, step_durations, v_loc, peak_loc, p)
        elif cycle_idx == 2 and num_valleys >= 2:
            v_loc = valleys['locations'][1]
            cycle['Phi_value'] = calculate_rise_phi3(step_amps, step_durations, v_loc, peak_loc, p)
        else:
            cycle['Phi_value'] = 0.0
        if cycle_idx < num_valleys:
            v_loc = valleys['locations'][cycle_idx]
            if peak_loc < v_loc:
                cycle['Psi_value'] = calculate_decay_function(
                    step_amps, step_times, step_durations, peak_loc, v_loc, p
                )
            else:
                cycle['Psi_value'] = 0.0
        else:
            cycle['Psi_value'] = 0.0
        cycles.append(cycle)
    return cycles


# ============================================================
# 3. HDF5 LOADER (MATLAB v7.3)
# ============================================================
def load_v73_mat_to_dict_list(filepath, var_name=None):
    with h5py.File(filepath, 'r') as f:
        if var_name is None:
            exclude = {'#refs#', '#subsystem#', '#attributes#', 'MATLAB_class', 'MATLAB_fields'}
            candidates = [k for k in f.keys() if k not in exclude]
            if not candidates:
                raise ValueError("No suitable variable found.")
            var_name = candidates[0]
        var_obj = f[var_name]

        def dereference_element(element):
            if isinstance(element, h5py.h5r.Reference):
                obj = f[element]
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if data.dtype.kind in {'S', 'U'}:
                        if data.ndim == 0:
                            return data.item().strip()
                        elif data.ndim == 1:
                            return ''.join(data.astype(str)).strip()
                        else:
                            return ''.join(data.astype(str).flatten()).strip()
                    else:
                        return data.item() if data.size == 1 else data
                else:
                    return None
            else:
                return element

        if isinstance(var_obj, h5py.Group):
            field_names = [k for k in var_obj.keys() if isinstance(var_obj[k], h5py.Dataset)]
            if not field_names:
                return []
            first_dset = var_obj[field_names[0]]
            if first_dset.shape is not None and len(first_dset.shape) == 2 and first_dset.shape[0] == 1:
                N = first_dset.shape[1]
                segments = []
                for i in range(N):
                    seg_dict = {}
                    for fname in field_names:
                        dset = var_obj[fname]
                        element = dset[0, i]
                        seg_dict[fname] = dereference_element(element)
                    segments.append(seg_dict)
                return segments
            else:
                seg_dict = {}
                for fname in field_names:
                    seg_dict[fname] = dereference_element(
                        var_obj[fname][0, 0] if var_obj[fname].shape else var_obj[fname][()])
                return [seg_dict]
        else:
            raise TypeError("Unexpected variable type.")

# ============================================================
# 4. CORE SIGNAL PROCESSING (from all_beats base)
# ============================================================
def estimate_pulse_rate(ppg_signal, fs):
    try:
        ppg = ppg_signal - np.mean(ppg_signal)
        b, a = butter(3, [0.5, 4.0], btype='bandpass', fs=fs)
        ppg_filt = filtfilt(b, a, ppg)
        prominence = 0.2 * (np.max(ppg_filt) - np.min(ppg_filt))
        distance = int(0.4 * fs)
        peaks, _ = find_peaks(ppg_filt, distance=distance, prominence=prominence)
        if len(peaks) >= 2:
            rr = np.diff(peaks) / fs
            valid_rr = rr[(rr > 0.4) & (rr < 2.0)]
            if len(valid_rr) >= 1:
                pr = 60.0 / np.mean(valid_rr)
                return float(np.clip(pr, 40, 180))
    except Exception:
        pass
    return 72.0


def segment_beats(ppg_clean, peaks_raw, fs, reject_mask=None):
    """
    Segment beats. If reject_mask is provided, peaks whose surrounding
    window overlaps the mask are skipped (used for join boundaries).
    """
    peaks = extract_peak_indices(peaks_raw)
    beats = []
    foot_indices = []
    min_beat_len = int(0.3 * fs)
    for i in range(len(peaks)):
        # Skip peaks near join boundaries
        if reject_mask is not None:
            lo = max(0, peaks[i] - int(0.3 * fs))
            hi = min(len(reject_mask), peaks[i] + 1)
            if np.any(reject_mask[lo:hi]):
                continue

        search_start = max(0, peaks[i] - int(0.3 * fs))
        if search_start >= peaks[i]:
            continue
        foot = search_start + np.argmin(ppg_clean[search_start:peaks[i]])
        if i + 1 < len(peaks):
            search_end = min(len(ppg_clean), peaks[i + 1] + int(0.1 * fs))
            next_foot = peaks[i] + np.argmin(ppg_clean[peaks[i]:search_end])
        else:
            next_foot = min(len(ppg_clean), peaks[i] + int(0.8 * fs))
        if next_foot - foot < min_beat_len:
            next_foot = min(foot + min_beat_len, len(ppg_clean))
        seg = ppg_clean[foot:next_foot]
        if len(seg) >= min_beat_len:
            beats.append(seg)
            foot_indices.append(foot)
    return beats, foot_indices, peaks

def detect_ppg_fiducials(beat, fs):
    result = {
        'foot_idx': 0, 'foot_amp': beat[0],
        'sys_peak_idx': 0, 'sys_peak_amp': beat[0],
        'notch_idx': None, 'notch_amp': None,
        'dia_peak_idx': None, 'dia_peak_amp': None,
        'inflection_idx': None, 'inflection_amp': None,
    }
    if len(beat) < 10:
        return result

    foot_idx = int(np.argmin(beat[:len(beat) // 3])) if len(beat) >= 6 else 0
    result['foot_idx'] = foot_idx
    result['foot_amp'] = float(beat[foot_idx])

    sys_peak_idx = foot_idx + np.argmax(beat[foot_idx:])
    result['sys_peak_idx'] = sys_peak_idx
    result['sys_peak_amp'] = float(beat[sys_peak_idx])

    if sys_peak_idx < len(beat) - 5:
        descent = beat[sys_peak_idx:]
        min_notch = int(0.12 * fs)
        max_notch = int(0.50 * fs)
        search_start = min_notch
        search_end = min(max_notch, len(descent) - 3)

        if search_end > search_start:
            region = descent[search_start:search_end]
            minima, _ = find_peaks(-region,
                                   distance=max(3, int(0.03 * fs)),
                                   prominence=0.003 * np.ptp(beat))
            if len(minima) > 0:
                notch_rel = search_start + minima[0]
                notch_idx = sys_peak_idx + notch_rel
                result['notch_idx'] = notch_idx
                result['notch_amp'] = float(beat[notch_idx])

        if result['notch_idx'] is None:
            search_end_inf = int(0.6 * (len(beat) - sys_peak_idx))
            if search_end_inf > 5:
                d2 = np.diff(descent[:search_end_inf], 2)
                if len(d2) > 3:
                    sign_changes = np.where((d2[:-1] < 0) & (d2[1:] >= 0))[0]
                    if len(sign_changes) > 0:
                        inf_rel = sign_changes[0] + 1
                        inf_idx = sys_peak_idx + inf_rel
                        result['inflection_idx'] = inf_idx
                        result['inflection_amp'] = float(beat[inf_idx])
                        result['notch_idx'] = inf_idx
                        result['notch_amp'] = float(beat[inf_idx])

        notch_idx = result['notch_idx']
        if notch_idx is not None and notch_idx < len(beat) - 3:
            after_notch = beat[notch_idx:]
            dia_search_end = min(len(after_notch), int(0.35 * fs) + 1)
            dia_region = after_notch[:dia_search_end]
            dia_peaks, _ = find_peaks(dia_region,
                                      distance=max(3, int(0.03 * fs)),
                                      prominence=0.003 * np.ptp(beat))
            if len(dia_peaks) > 0:
                dia_rel = dia_peaks[0]
                result['dia_peak_idx'] = notch_idx + dia_rel
                result['dia_peak_amp'] = float(beat[notch_idx + dia_rel])
    return result


def extract_beat_features_single(beat, fs, height_cm=170):
    n = len(beat)
    if n < 20:
        return None

    fid = detect_ppg_fiducials(beat, fs)
    if not is_valid_fiducials(fid):
        return None

    foot_idx = fid['foot_idx']
    sys_idx  = fid['sys_peak_idx']
    foot_amp = fid['foot_amp']
    sys_amp  = fid['sys_peak_amp']
    pp = sys_amp - foot_amp
    if pp <= 0:
        return None

    feat = {}
    feat['foot_pos']  = foot_idx / n
    feat['sys_pos']   = sys_idx / n
    feat['notch_pos'] = fid['notch_idx'] / n if fid['notch_idx'] is not None else 0.0
    feat['dia_pos']   = fid['dia_peak_idx'] / n if fid['dia_peak_idx'] is not None else 0.0

    feat['sys_amp_rel']   = pp
    feat['notch_amp_rel'] = (fid['notch_amp'] - foot_amp) if fid['notch_amp'] is not None else 0.0
    feat['dia_amp_rel']   = (fid['dia_peak_amp'] - foot_amp) if fid['dia_peak_amp'] is not None else 0.0

    feat['rise_time_ms']      = (sys_idx - foot_idx) / fs * 1000.0
    feat['pulse_width_ms']    = n / fs * 1000.0
    feat['systolic_time_ms']  = feat['rise_time_ms']
    feat['diastolic_time_ms'] = (n - sys_idx) / fs * 1000.0
    feat['ts_td_ratio'] = (feat['systolic_time_ms'] / feat['diastolic_time_ms']
                           if feat['diastolic_time_ms'] > 0 else 0.0)

    if fid['dia_peak_idx'] is not None:
        feat['delta_t_ms'] = (fid['dia_peak_idx'] - sys_idx) / fs * 1000.0
        feat['reflection_index'] = feat['dia_amp_rel'] / pp if pp > 0 else 0.0
    else:
        feat['delta_t_ms'] = 0.0
        feat['reflection_index'] = 0.0

    beat_rel = beat - foot_amp
    notch_idx = fid['notch_idx'] if fid['notch_idx'] is not None else int(0.65 * n)
    notch_idx = max(sys_idx + 1, min(notch_idx, n - 2))
    feat['sys_area']   = float(trapezoid_compat(beat_rel[foot_idx:notch_idx + 1]))
    feat['dia_area']   = float(trapezoid_compat(beat_rel[notch_idx:]))
    feat['total_area'] = feat['sys_area'] + feat['dia_area']

    if sys_idx > foot_idx + 1:
        rise = beat[foot_idx:sys_idx + 1]
        rise_vel = np.diff(rise)
        feat['max_rising_slope'] = float(np.max(rise_vel))
        feat['upslope_mean']     = float(np.mean(rise_vel))
    else:
        feat['max_rising_slope'] = 0.0
        feat['upslope_mean']     = 0.0

    if n - sys_idx > 1:
        fall = beat[sys_idx:]
        fall_vel = np.diff(fall)
        feat['downslope_mean'] = float(np.mean(fall_vel))
    else:
        feat['downslope_mean'] = 0.0

    feat['aix_percent'] = 0.0
    feat['augmentation_pressure'] = 0.0
    feat['inflection_point_ratio'] = 0.0
    if sys_idx < n - 5:
        descent_early = beat[sys_idx:sys_idx + int(0.6 * (n - sys_idx))]
        d2 = np.diff(descent_early, 2)
        if len(d2) > 3:
            sign_changes = np.where((d2[:-1] < 0) & (d2[1:] >= 0))[0]
            if len(sign_changes) > 0:
                inf_idx = sys_idx + sign_changes[0] + 1
                inf_amp = float(beat[inf_idx])
                aug_p = sys_amp - inf_amp
                feat['augmentation_pressure'] = aug_p
                feat['aix_percent'] = aug_p / pp * 100.0 if pp > 0 else 0.0
                feat['inflection_point_ratio'] = inf_idx / n

    feat['stiffness_index'] = 0.0
    feat['lasi_ms'] = 0.0
    feat['diastolic_decay_constant'] = 0.0
    feat['crest_time_ratio'] = (sys_idx - foot_idx) / n if n > 0 else 0.0

    if fid['dia_peak_idx'] is not None and fid['dia_peak_idx'] > foot_idx:
        delta_t = (fid['dia_peak_idx'] - foot_idx) / fs
        if delta_t > 0.05:
            feat['stiffness_index'] = height_cm / 100.0 / delta_t

    if sys_idx < n - 5:
        diastolic = beat[sys_idx:]
        t = np.arange(len(diastolic)) / fs
        y = np.maximum(diastolic - foot_amp, 1e-10)
        log_y = np.log(y)
        if len(t) > 3:
            A = np.vstack([t, np.ones(len(t))]).T
            slope, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
            if slope < 0 and abs(slope) > 1e-6:
                feat['diastolic_decay_constant'] = -slope
                feat['lasi_ms'] = 1.0 / (-slope) * 1000.0

    dc_baseline = abs(foot_amp) if abs(foot_amp) > 1e-10 else abs(np.min(beat))
    feat['compliance_index'] = pp / dc_baseline if dc_baseline > 1e-10 else 0.0

    feat['notch_width_ms'] = 0.0
    feat['notch_sharpness'] = 0.0
    feat['notch_depth_ratio'] = 0.0
    feat['notch_symmetry_index'] = 0.0

    if fid['notch_idx'] is not None and sys_idx < n - 3:
        notch_idx = fid['notch_idx']
        notch_amp = fid['notch_amp']
        search_win = max(3, int(0.05 * fs))
        left_start = max(sys_idx, notch_idx - search_win)
        left_region = beat[left_start:notch_idx + 1]
        right_end = min(n, notch_idx + search_win + 1)
        right_region = beat[notch_idx:right_end]

        left_max = np.max(left_region) if len(left_region) > 0 else notch_amp
        right_max = np.max(right_region) if len(right_region) > 0 else notch_amp
        shoulder_level = min(left_max, right_max)
        depth = shoulder_level - notch_amp

        if depth > 0:
            half_level = notch_amp + 0.5 * depth
            left_cross = notch_idx
            for i in range(notch_idx - 1, left_start - 1, -1):
                if beat[i] >= half_level:
                    left_cross = i
                    break
            right_cross = notch_idx
            for i in range(notch_idx + 1, right_end):
                if beat[i] >= half_level:
                    right_cross = i
                    break
            feat['notch_width_ms'] = (right_cross - left_cross) / fs * 1000.0

        if notch_idx > 1 and notch_idx < n - 2:
            feat['notch_sharpness'] = float(
                beat[notch_idx - 1] - 2 * beat[notch_idx] + beat[notch_idx + 1]
            )

        feat['notch_depth_ratio'] = (pp - (notch_amp - foot_amp)) / pp if pp > 0 else 0.0

        if left_max > notch_amp and right_max > notch_amp:
            left_drop = left_max - notch_amp
            right_drop = right_max - notch_amp
            if (left_drop + right_drop) > 0:
                feat['notch_symmetry_index'] = abs(left_drop - right_drop) / (left_drop + right_drop)

    feat['shoulder_present'] = 0.0
    feat['shoulder_height_ratio'] = 0.0
    if sys_idx > foot_idx + 3:
        upslope = beat[foot_idx:sys_idx + 1]
        if len(upslope) >= 10:
            local_peaks, _ = find_peaks(
                upslope, distance=max(3, int(0.03 * fs)), prominence=0.05 * pp
            )
            local_peaks = local_peaks[local_peaks < sys_idx - foot_idx - 2]
            if len(local_peaks) > 0:
                feat['shoulder_present'] = 1.0
                shoulder_idx_rel = local_peaks[-1]
                feat['shoulder_height_ratio'] = (
                    (upslope[shoulder_idx_rel] - foot_amp) / pp if pp > 0 else 0.0
                )

    return feat

# ============================================================
# 5. BASE SEGMENT EXTRACTORS (from all_beats.py)
# ============================================================
def extract_enhanced_morphology(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'mean_amplitude': 0.0, 'amplitude_std': 0.0,
        'mean_area_systolic': 0.0, 'mean_area_diastolic': 0.0,
        'systolic_area_fraction': 0.0, 'diastolic_area_fraction': 0.0,
        'sys_dia_area_ratio': 0.0,
        'skewness_ppg': 0.0, 'kurtosis_ppg': 0.0,
        'rise_time_ms': 0.0, 'pulse_width_ms': 0.0,
        'perfusion_index': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        sys_areas, dia_areas, amplitudes, rise_times, pulse_widths = [], [], [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 10:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            amp = fid['sys_peak_amp'] - fid['foot_amp']
            amplitudes.append(amp)
            rise_times.append((sys_idx - foot_idx) / fs * 1000.0)
            pulse_widths.append(len(beat) / fs * 1000.0)
            beat_rel = beat - fid['foot_amp']
            notch_idx = fid['notch_idx']
            if notch_idx is None:
                notch_idx = int(0.65 * len(beat))
            notch_idx = max(sys_idx + 1, min(notch_idx, len(beat) - 2))
            sys_areas.append(trapezoid_compat(beat_rel[foot_idx:notch_idx + 1]))
            dia_areas.append(trapezoid_compat(beat_rel[notch_idx:]))
        result = default.copy()
        if amplitudes:
            result['mean_amplitude'] = safe_mean(amplitudes)
            result['amplitude_std'] = safe_std(amplitudes)
        if sys_areas and dia_areas:
            result['mean_area_systolic'] = safe_mean(sys_areas)
            result['mean_area_diastolic'] = safe_mean(dia_areas)
            total_area = result['mean_area_systolic'] + result['mean_area_diastolic']
            if total_area > 0:
                result['systolic_area_fraction'] = result['mean_area_systolic'] / total_area
                result['diastolic_area_fraction'] = result['mean_area_diastolic'] / total_area
            if result['mean_area_diastolic'] > 0:
                result['sys_dia_area_ratio'] = result['mean_area_systolic'] / result['mean_area_diastolic']
        if rise_times: result['rise_time_ms'] = safe_mean(rise_times)
        if pulse_widths: result['pulse_width_ms'] = safe_mean(pulse_widths)
        result['skewness_ppg'] = float(skew(ppg_signal, bias=False))
        result['kurtosis_ppg'] = float(kurtosis(ppg_signal, bias=False))
        ac = ppg_signal - np.mean(ppg_signal)
        result['perfusion_index'] = float(np.std(ac) / (np.mean(ppg_signal) + 1e-10))
        return result
    except Exception:
        return default


def extract_enhanced_timing_features(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'systolic_time_ms': 0.0, 'diastolic_time_ms': 0.0,
        'ts_td_ratio': 0.0, 'delta_t_ms': 0.0,
        'augmentation_time_ms': 0.0, 'deceleration_time_ms': 0.0,
        'notch_depth_ratio': 0.0, 'notch_position_pct': 0.0,
        'reflection_index_p2_p1': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        sys_times, dia_times, ts_td_ratios, delta_ts, aug_times, dec_times = [], [], [], [], [], []
        notch_depths, notch_positions, ref_indices = [], [], []

        for beat in get_core_beats(beats):
            if len(beat) < 10:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            st = (sys_idx - foot_idx) / fs * 1000.0
            sys_times.append(st)
            dt = (len(beat) - sys_idx) / fs * 1000.0
            dia_times.append(dt)
            if dt > 0:
                ts_td_ratios.append(st / dt)
            if fid['notch_idx'] is not None:
                aug_time = (fid['notch_idx'] - foot_idx) / fs * 1000.0
                aug_times.append(aug_time)
            if fid['dia_peak_idx'] is not None:
                delta_t = (fid['dia_peak_idx'] - sys_idx) / fs * 1000.0
                if delta_t < 500:
                    delta_ts.append(delta_t)
                pp = fid['sys_peak_amp'] - fid['foot_amp']
                if pp > 0:
                    ref_indices.append((fid['dia_peak_amp'] - fid['foot_amp']) / pp)
            if fid['notch_idx'] is not None:
                dec_t = (fid['notch_idx'] - sys_idx) / fs * 1000.0
                dec_times.append(dec_t)
                peak_rel = fid['sys_peak_amp'] - fid['foot_amp']
                notch_rel = fid['notch_amp'] - fid['foot_amp'] if fid['notch_amp'] is not None else peak_rel
                if peak_rel > 0:
                    notch_depths.append((peak_rel - notch_rel) / peak_rel)
                notch_positions.append(fid['notch_idx'] / len(beat) * 100.0)

        result = default.copy()
        if sys_times: result['systolic_time_ms'] = safe_mean(sys_times)
        if dia_times: result['diastolic_time_ms'] = safe_mean(dia_times)
        if ts_td_ratios: result['ts_td_ratio'] = safe_mean(ts_td_ratios)
        if delta_ts: result['delta_t_ms'] = safe_mean(delta_ts)
        if aug_times: result['augmentation_time_ms'] = safe_mean(aug_times)
        if dec_times: result['deceleration_time_ms'] = safe_mean(dec_times)
        if notch_depths: result['notch_depth_ratio'] = safe_mean(notch_depths)
        if notch_positions: result['notch_position_pct'] = safe_mean(notch_positions)
        if ref_indices: result['reflection_index_p2_p1'] = safe_mean(ref_indices)
        return result
    except Exception:
        return default


def extract_sdppg_features(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'sdppg_a': 0.0, 'sdppg_b': 0.0, 'sdppg_c': 0.0, 'sdppg_d': 0.0, 'sdppg_e': 0.0,
        'sdppg_aging_index': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        valid_beats = [b for b in get_core_beats(beats) if len(b) >= int(0.4 * fs)]
        if len(valid_beats) < 2:
            return default

        lengths = [len(b) for b in valid_beats]
        med_len = int(np.median(lengths))
        aligned = []
        for b in valid_beats:
            if len(b) == med_len:
                aligned.append(b)
            elif len(b) > med_len:
                aligned.append(b[:med_len])
            else:
                aligned.append(np.concatenate([b, np.full(med_len - len(b), b[-1])]))
        avg_beat = np.mean(aligned, axis=0)
        apg = np.diff(avg_beat, 2)
        if len(apg) < 20:
            return default

        b, a = butter(2, 8.0 / (fs / 2), btype='low')
        apg_filt = filtfilt(b, a, apg)

        pos_peaks, _ = find_peaks(apg_filt, distance=max(3, int(0.03 * fs)), prominence=0.03 * np.std(apg_filt))
        neg_peaks, _ = find_peaks(-apg_filt, distance=max(3, int(0.03 * fs)), prominence=0.03 * np.std(apg_filt))

        if len(pos_peaks) < 1:
            return default

        a_idx = pos_peaks[0]
        a_val = apg_filt[a_idx]
        b_val = c_val = d_val = e_val = 0.0

        neg_after_a = neg_peaks[neg_peaks > a_idx]
        if len(neg_after_a) > 0:
            b_idx = neg_after_a[0]
            b_val = apg_filt[b_idx]
            pos_after_b = pos_peaks[pos_peaks > b_idx]
            if len(pos_after_b) > 0:
                c_idx = pos_after_b[0]
                c_val = apg_filt[c_idx]
                neg_after_c = neg_peaks[neg_peaks > c_idx]
                if len(neg_after_c) > 0:
                    d_idx = neg_after_c[0]
                    d_val = apg_filt[d_idx]
                    pos_after_d = pos_peaks[pos_peaks > d_idx]
                    if len(pos_after_d) > 0:
                        e_idx = pos_after_d[0]
                        e_val = apg_filt[e_idx]

        if abs(a_val) < 1e-10:
            return default

        return {
            'sdppg_a': float(a_val), 'sdppg_b': float(b_val), 'sdppg_c': float(c_val),
            'sdppg_d': float(d_val), 'sdppg_e': float(e_val),
            'sdppg_aging_index': float((b_val - c_val - d_val - e_val) / a_val),
        }
    except Exception:
        return default


def extract_enhanced_spectral_features(ppg_signal, fs, pulse_rate, ppg_clean=None):
    default = {
        'spectral_fundamental_freq': 0.0, 'spectral_centroid': 0.0,
        'spectral_bandwidth': 0.0, 'spectral_entropy': 0.0,
        'harmonic_ratio_h2_h1': 0.0, 'harmonic_ratio_h3_h1': 0.0,
        'harmonic_ratio_h4_h1': 0.0, 'lf_hf_ratio': 0.0,
    }
    try:
        sig = ppg_clean if ppg_clean is not None else ppg_signal
        ppg_detrended = sig - np.mean(sig)
        nperseg = min(256, len(ppg_signal))
        freqs, psd = welch(ppg_detrended, fs=fs, nperseg=nperseg, nfft=max(512, nperseg * 2))
        if len(psd) == 0:
            return default

        total_power = np.sum(psd)
        fundamental_hz = pulse_rate / 60.0

        hr_mask = np.abs(freqs - fundamental_hz) < 0.4
        if np.any(hr_mask):
            local_idx = np.argmax(psd[hr_mask])
            fund_freq = float(freqs[hr_mask][local_idx])
            fund_power = float(psd[hr_mask][local_idx])
        else:
            fund_idx = np.argmax(psd)
            fund_freq = float(freqs[fund_idx])
            fund_power = float(psd[fund_idx])

        h_ratios = []
        for h in range(2, 5):
            hf = h * fundamental_hz
            if hf < freqs[-1]:
                idx = np.argmin(np.abs(freqs - hf))
                h_power = psd[idx]
                h_ratios.append(h_power / fund_power if fund_power > 0 else 0.0)
            else:
                h_ratios.append(0.0)

        centroid = np.sum(freqs * psd) / total_power if total_power > 0 else 0.0
        bandwidth = np.sqrt(np.sum(psd * (freqs - centroid) ** 2) / total_power) if total_power > 0 else 0.0
        norm_psd = psd / total_power
        nz = norm_psd > 0
        entropy = -np.sum(norm_psd[nz] * np.log2(norm_psd[nz]))

        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        lf = np.sum(psd[lf_mask])
        hfp = np.sum(psd[hf_mask])

        return {
            'spectral_fundamental_freq': fund_freq, 'spectral_centroid': float(centroid),
            'spectral_bandwidth': float(bandwidth), 'spectral_entropy': float(entropy),
            'harmonic_ratio_h2_h1': float(h_ratios[0]), 'harmonic_ratio_h3_h1': float(h_ratios[1]),
            'harmonic_ratio_h4_h1': float(h_ratios[2]),
            'lf_hf_ratio': float(lf / hfp) if hfp > 0 else 0.0,
        }
    except Exception:
        return default


def extract_ptt_features(ppg_signal, ecg_signal, fs, ppg_clean=None, ecg_clean=None,
                         ppg_peaks=None, r_peaks=None, foot_indices=None):
    default = {
        'ptt_r_to_ppg_foot_ms': 0.0, 'ptt_r_to_ppg_peak_ms': 0.0,
        'ptt_beat_to_beat_std_ms': 0.0, 'ptt_beat_to_beat_cv': 0.0,
    }
    try:
        if ecg_signal is None or len(ecg_signal) < 100:
            return default

        ecg_c = ecg_clean if ecg_clean is not None else nk.ecg_clean(
            ecg_signal, sampling_rate=fs, method='neurokit')
        r_peaks_arr = r_peaks if r_peaks is not None else np.array(
            nk.ecg_peaks(ecg_c, sampling_rate=fs, method='neurokit')[1]['ECG_R_Peaks'],
            dtype=int)

        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(
            ppg_signal, sampling_rate=fs)
        ppg_peaks_arr = ppg_peaks if ppg_peaks is not None else extract_peak_indices(
            nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])

        if len(r_peaks_arr) < 2 or len(ppg_peaks_arr) < 2:
            return default
        if len(r_peaks_arr) < 3:
            return default

        ptt_foots, ptt_peaks = [], []
        for i, p_peak in enumerate(ppg_peaks_arr):
            prev_r = r_peaks_arr[r_peaks_arr < p_peak]
            if len(prev_r) == 0:
                continue
            r_idx = prev_r[-1]

            if foot_indices is not None and i < len(foot_indices):
                foot_idx = foot_indices[i]
                if not (r_idx < foot_idx < p_peak):
                    search_start = max(0, p_peak - int(0.4 * fs))
                    beat_window = ppg_c[search_start:p_peak + 1]
                    if len(beat_window) == 0:
                        continue
                    foot_rel = np.argmin(beat_window)
                    foot_idx = search_start + foot_rel
            else:
                search_start = max(0, p_peak - int(0.4 * fs))
                beat_window = ppg_c[search_start:p_peak + 1]
                if len(beat_window) == 0:
                    continue
                foot_rel = np.argmin(beat_window)
                foot_idx = search_start + foot_rel

            ptt_foot = (foot_idx - r_idx) / fs * 1000.0
            ptt_peak = (p_peak - r_idx) / fs * 1000.0

            if 60 < ptt_foot < 500 and 80 < ptt_peak < 700:
                ptt_foots.append(ptt_foot)
                ptt_peaks.append(ptt_peak)

        result = default.copy()
        if ptt_foots:
            result['ptt_r_to_ppg_foot_ms'] = safe_mean(ptt_foots)
        if ptt_peaks:
            result['ptt_r_to_ppg_peak_ms'] = safe_mean(ptt_peaks)
        if len(ptt_foots) > 1:
            result['ptt_beat_to_beat_std_ms'] = safe_std(ptt_foots)
            if result['ptt_r_to_ppg_foot_ms'] > 0:
                result['ptt_beat_to_beat_cv'] = (
                    result['ptt_beat_to_beat_std_ms'] /
                    result['ptt_r_to_ppg_foot_ms'])
        return result
    except Exception:
        return default


def extract_enhanced_derivative_features(ppg_signal, fs, ppg_clean=None, peaks=None):
    default = {
        'max_velocity': 0.0, 'min_velocity': 0.0, 'velocity_skewness': 0.0,
        'max_acceleration': 0.0, 'min_acceleration': 0.0,
        'upslope_mean': 0.0, 'downslope_mean': 0.0,
        'upslope_downslope_ratio': 0.0, 'normalized_max_rising_slope': 0.0,
    }
    try:
        sig = ppg_clean if ppg_clean is not None else ppg_signal
        if len(sig) < 10:
            return default

        vel = np.diff(sig)
        acc = np.diff(sig, 2) if len(sig) > 20 else np.array([])
        max_vel = float(np.max(vel)) if len(vel) > 0 else 0.0
        min_vel = float(np.min(vel)) if len(vel) > 0 else 0.0

        upslopes, downslopes, norm_rising_slopes = [], [], []
        if peaks is not None and len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                search_start = max(0, peaks[i] - int(0.3 * fs))
                foot_idx_global = search_start + np.argmin(sig[search_start:peaks[i]])
                search_end = min(len(sig), peaks[i + 1] + int(0.1 * fs))
                next_foot_idx_global = peaks[i] + np.argmin(sig[peaks[i]:search_end])
                next_foot_idx_global = min(next_foot_idx_global, len(sig) - 1)
                if next_foot_idx_global <= foot_idx_global:
                    continue
                beat = sig[foot_idx_global:next_foot_idx_global + 1]
                if len(beat) < 5:
                    continue
                peak_idx = np.argmax(beat)
                if peak_idx <= 0:
                    continue
                rise = beat[:peak_idx + 1]
                fall = beat[peak_idx:]
                if len(rise) > 1:
                    rise_vel = np.diff(rise)
                    upslopes.append(np.mean(rise_vel))
                    norm_rising_slopes.append(np.max(rise_vel) / (np.ptp(beat) + 1e-10))
                if len(fall) > 1:
                    downslopes.append(np.mean(np.diff(fall)))

        result = default.copy()
        result['max_velocity'] = max_vel
        result['min_velocity'] = min_vel
        result['velocity_skewness'] = float(skew(vel, bias=False)) if len(vel) > 2 else 0.0
        if len(acc) > 0:
            result['max_acceleration'] = float(np.max(acc))
            result['min_acceleration'] = float(np.min(acc))
        if upslopes: result['upslope_mean'] = safe_mean(upslopes)
        if downslopes: result['downslope_mean'] = safe_mean(downslopes)
        if upslopes and downslopes and abs(result['downslope_mean']) > 1e-10:
            result['upslope_downslope_ratio'] = result['upslope_mean'] / result['downslope_mean']
        if norm_rising_slopes: result['normalized_max_rising_slope'] = safe_mean(norm_rising_slopes)
        return result
    except Exception:
        return default


def extract_simplified_hrv(ppg_signal, fs, ppg_clean=None, peaks_df=None):
    default = {
        'hrv_mean_nn_ms': 0.0, 'hrv_sdnn_ms': 0.0, 'hrv_rmssd_ms': 0.0,
        'hrv_cvnn': 0.0, 'hrv_pnn50': 0.0,
        'hrv_lf_power': 0.0, 'hrv_hf_power': 0.0, 'hrv_lf_hf_ratio': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks_df is None:
            peaks_df, info = nk.ppg_peaks(ppg_c, sampling_rate=fs)
        peaks = extract_peak_indices(peaks_df)
        ibi_ms = np.diff(peaks) / fs * 1000
        if len(ibi_ms) < 3:
            return default

        time_domain = nk.hrv_time(peaks_df, sampling_rate=fs, show=False)
        freq_domain = nk.hrv_frequency(peaks_df, sampling_rate=fs, show=False)

        def safe_hrv(df, key):
            try:
                v = df[key]
                if hasattr(v, 'iloc'):
                    v = v.iloc[0]
                elif hasattr(v, '__len__') and not isinstance(v, str):
                    v = v[0] if len(v) > 0 else 0.0
                v = float(v)
                return 0.0 if np.isnan(v) else v
            except Exception:
                return 0.0

        rr = np.diff(peaks) / fs
        lf_manual = hf_manual = lfhf_manual = 0.0
        try:
            if len(rr) >= 4:
                t = np.cumsum(rr)
                duration = t[-1] - t[0]
                # Frequency-domain HRV requires >= 60 s of signal
                if duration >= 60.0:
                    fs_uniform = 4.0
                    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_uniform)
                    rr_uniform = np.interp(t_uniform, t, rr)
                    nperseg = min(len(rr_uniform), 256)
                    freqs_rr, psd_rr = welch(rr_uniform, fs=fs_uniform,
                                             nperseg=nperseg,
                                             nfft=max(256, nperseg * 2))
                    lf_mask = (freqs_rr >= 0.04) & (freqs_rr < 0.15)
                    hf_mask = (freqs_rr >= 0.15) & (freqs_rr < 0.40)
                    lf_manual = float(np.trapz(psd_rr[lf_mask], freqs_rr[lf_mask]))
                    hf_manual = float(np.trapz(psd_rr[hf_mask], freqs_rr[hf_mask]))
                    lfhf_manual = float(lf_manual / hf_manual) if hf_manual > 0 else 0.0
        except Exception:
            pass

        lf_nk = safe_hrv(freq_domain, 'HRV_LF')
        hf_nk = safe_hrv(freq_domain, 'HRV_HF')
        lfhf_nk = safe_hrv(freq_domain, 'HRV_LFHF')

        return {
            'hrv_mean_nn_ms': safe_hrv(time_domain, 'HRV_MeanNN'),
            'hrv_sdnn_ms': safe_hrv(time_domain, 'HRV_SDNN'),
            'hrv_rmssd_ms': safe_hrv(time_domain, 'HRV_RMSSD'),
            'hrv_cvnn': safe_hrv(time_domain, 'HRV_CVNN'),
            'hrv_pnn50': safe_hrv(time_domain, 'HRV_pNN50'),
            'hrv_lf_power': lf_nk if lf_nk > 0 else lf_manual,
            'hrv_hf_power': hf_nk if hf_nk > 0 else hf_manual,
            'hrv_lf_hf_ratio': lfhf_nk if lfhf_nk > 0 else lfhf_manual,
        }
    except Exception:
        return default

def extract_augmentation_index(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'aix_percent': 0.0, 'aix_amplitude_based': 0.0,
        'augmentation_pressure': 0.0, 'pulse_pressure_ppg': 0.0,
        'inflection_point_ratio': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        aix_values, aix_amp_values, aug_pressures, pulse_pressures, inflection_ratios = [], [], [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 20:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            foot_amp = fid['foot_amp']
            sys_amp = fid['sys_peak_amp']
            pp = sys_amp - foot_amp
            pulse_pressures.append(pp)
            if fid['dia_peak_idx'] is not None and fid['dia_peak_amp'] is not None:
                aix_amp_values.append((fid['dia_peak_amp'] - foot_amp) / pp * 100.0)
            if sys_idx < len(beat) - 5:
                descent_early = beat[sys_idx:sys_idx + int(0.6 * (len(beat) - sys_idx))]
                d2 = np.diff(descent_early, 2)
                if len(d2) > 3:
                    sign_changes = np.where((d2[:-1] < 0) & (d2[1:] >= 0))[0]
                    if len(sign_changes) > 0:
                        inf_idx_rel = sign_changes[0] + 1
                        inf_idx = sys_idx + inf_idx_rel
                        inf_amp = float(beat[inf_idx])
                        aug_p = sys_amp - inf_amp
                        aug_pressures.append(aug_p)
                        aix_values.append(aug_p / pp * 100.0)
                        inflection_ratios.append(inf_idx / len(beat) * 100.0)
        result = default.copy()
        if aix_values: result['aix_percent'] = safe_mean(aix_values)
        if aix_amp_values: result['aix_amplitude_based'] = safe_mean(aix_amp_values)
        if aug_pressures: result['augmentation_pressure'] = safe_mean(aug_pressures)
        if pulse_pressures: result['pulse_pressure_ppg'] = safe_mean(pulse_pressures)
        if inflection_ratios: result['inflection_point_ratio'] = safe_mean(inflection_ratios)
        return result
    except Exception:
        return default


def extract_stiffness_index(ppg_signal, fs, height_cm=170, ppg_clean=None, peaks=None, beats=None):
    default = {
        'stiffness_index': 0.0, 'lasi': 0.0,
        'reflection_index': 0.0, 'crest_time_ratio': 0.0,
        'diastolic_decay_time_ms': 0.0, 'diastolic_decay_constant': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        si_values, lasi_values, ri_values, crest_ratios, decay_times, decay_constants = [], [], [], [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 20:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            foot_amp = fid['foot_amp']
            sys_amp = fid['sys_peak_amp']
            if fid['dia_peak_idx'] is not None:
                delta_t = (fid['dia_peak_idx'] - foot_idx) / fs
                if delta_t > 0.05:
                    si_values.append(height_cm / 100.0 / delta_t)
                if sys_amp > foot_amp and fid['dia_peak_amp'] > foot_amp:
                    ri_values.append((fid['dia_peak_amp'] - foot_amp) / (sys_amp - foot_amp))
            crest_time = (sys_idx - foot_idx) / fs
            total_time = len(beat) / fs
            if total_time > 0:
                crest_ratios.append(crest_time / total_time)
            if sys_idx < len(beat) - 5:
                diastolic = beat[sys_idx:]
                t = np.arange(len(diastolic)) / fs
                y = np.maximum(diastolic - foot_amp, 1e-10)
                log_y = np.log(y)
                if len(t) > 3:
                    A = np.vstack([t, np.ones(len(t))]).T
                    slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]
                    if slope < 0 and abs(slope) > 1e-6:
                        lambda_decay = -slope
                        decay_constants.append(lambda_decay)
                        lasi_values.append(1.0 / lambda_decay * 1000.0)
                        decay_times.append(1.0 / lambda_decay * 1000.0)

        result = default.copy()
        if si_values: result['stiffness_index'] = safe_mean(si_values)
        if lasi_values: result['lasi'] = safe_mean(lasi_values)
        if ri_values: result['reflection_index'] = safe_mean(ri_values)
        if crest_ratios: result['crest_time_ratio'] = safe_mean(crest_ratios)
        if decay_times: result['diastolic_decay_time_ms'] = safe_mean(decay_times)
        if decay_constants: result['diastolic_decay_constant'] = safe_mean(decay_constants)
        return result
    except Exception:
        return default


def extract_pwv_surrogate(ptt_foot_ms, height_cm=170):
    default = {'pwv_surrogate': 0.0}
    try:
        if ptt_foot_ms is None or ptt_foot_ms <= 0:
            return default
        ptt_s = ptt_foot_ms / 1000.0
        distance_m = 0.8 * height_cm / 100.0
        pwv = distance_m / ptt_s
        return {'pwv_surrogate': float(pwv)}
    except Exception:
        return default


def extract_vascular_compliance(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {'compliance_index': 0.0, 'elasticity_index': 0.0, 'arterial_stiffness_proxy': 0.0}
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        ci_values, sys_areas, dia_areas = [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 10:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_amp = fid['foot_amp']
            sys_amp = fid['sys_peak_amp']
            pp = sys_amp - foot_amp
            dc_baseline = abs(foot_amp) if abs(foot_amp) > 1e-10 else abs(np.min(beat))
            if dc_baseline > 1e-10:
                ci_values.append(pp / dc_baseline)
            beat_rel = beat - foot_amp
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            notch_idx = fid['notch_idx']
            if notch_idx is None:
                notch_idx = int(0.65 * len(beat))
            notch_idx = max(sys_idx, min(notch_idx, len(beat) - 1))
            sys_area = trapezoid_compat(beat_rel[foot_idx:notch_idx + 1])
            dia_area = trapezoid_compat(beat_rel[notch_idx:])
            if dia_area > 0:
                sys_areas.append(sys_area)
                dia_areas.append(dia_area)

        result = default.copy()
        if ci_values:
            result['compliance_index'] = safe_mean(ci_values)
        if sys_areas and dia_areas:
            mean_sys = np.mean(sys_areas)
            mean_dia = np.mean(dia_areas)
            result['elasticity_index'] = mean_sys / mean_dia
            result['arterial_stiffness_proxy'] = 1.0 / (result['elasticity_index'] + 0.1)
        return result
    except Exception:
        return default


def extract_tdppg_features(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'tdppg_a': 0.0, 'tdppg_b': 0.0, 'tdppg_c': 0.0,
        'tdppg_d': 0.0, 'tdppg_e': 0.0, 'tdppg_f': 0.0,
        'tdppg_bd_index': 0.0, 'tdppg_cf_index': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        valid_beats = [b for b in get_core_beats(beats) if len(b) >= int(0.4 * fs)]
        if len(valid_beats) < 2:
            return default

        lengths = [len(b) for b in valid_beats]
        med_len = int(np.median(lengths))
        aligned = []
        for b in valid_beats:
            if len(b) == med_len:
                aligned.append(b)
            elif len(b) > med_len:
                aligned.append(b[:med_len])
            else:
                aligned.append(np.concatenate([b, np.full(med_len - len(b), b[-1])]))
        avg_beat = np.mean(aligned, axis=0)
        tdppg = np.diff(avg_beat, 3)
        if len(tdppg) < 30:
            return default

        b, a = butter(2, 10.0 / (fs / 2), btype='low')
        td_filt = filtfilt(b, a, tdppg)

        pos_peaks, _ = find_peaks(td_filt, distance=max(3, int(0.02 * fs)),
                                  prominence=0.02 * np.std(td_filt))
        neg_peaks, _ = find_peaks(-td_filt, distance=max(3, int(0.02 * fs)),
                                  prominence=0.02 * np.std(td_filt))

        extrema = []
        for idx in pos_peaks:
            extrema.append((idx, float(td_filt[idx]), 'pos'))
        for idx in neg_peaks:
            extrema.append((idx, float(td_filt[idx]), 'neg'))
        extrema.sort(key=lambda x: x[0])

        a_val = b_val = c_val = d_val = e_val = f_val = 0.0
        pos_count = 0
        neg_count = 0
        for idx, val, ptype in extrema:
            if ptype == 'pos':
                pos_count += 1
                if pos_count == 1: a_val = val
                elif pos_count == 2: c_val = val
                elif pos_count == 3: e_val = val
            else:
                neg_count += 1
                if neg_count == 1: b_val = val
                elif neg_count == 2: d_val = val
                elif neg_count == 3: f_val = val

        a_val = max(abs(a_val), 1e-10)
        return {
            'tdppg_a': float(a_val), 'tdppg_b': float(b_val), 'tdppg_c': float(c_val),
            'tdppg_d': float(d_val), 'tdppg_e': float(e_val), 'tdppg_f': float(f_val),
            'tdppg_bd_index': float((b_val + d_val) / a_val),
            'tdppg_cf_index': float((c_val + abs(f_val)) / a_val),
        }
    except Exception:
        return default


def extract_systolic_shoulder_features(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'shoulder_present_ratio': 0.0, 'shoulder_height_ratio': 0.0,
        'shoulder_time_ratio': 0.0, 'peak_bifurcation_index': 0.0,
        'systolic_roundness': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        shoulder_ratios, shoulder_heights, shoulder_times, bifurcation_scores, roundness_scores = [], [], [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 20:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            foot_amp = fid['foot_amp']
            sys_amp = fid['sys_peak_amp']
            if sys_idx <= foot_idx + 3 or sys_idx >= len(beat) - 3:
                continue

            window = 3
            peak_region = beat[max(0, sys_idx - window):min(len(beat), sys_idx + window + 1)]
            if len(peak_region) >= 3:
                roundness = np.std(peak_region) / (sys_amp - foot_amp + 1e-10)
                roundness_scores.append(roundness)

            upslope = beat[foot_idx:sys_idx + 1]
            if len(upslope) < 10:
                continue
            local_peaks, _ = find_peaks(upslope, distance=max(3, int(0.03 * fs)),
                                        prominence=0.05 * (sys_amp - foot_amp))
            local_peaks = local_peaks[local_peaks < sys_idx - foot_idx - 2]
            if len(local_peaks) > 0:
                shoulder_idx_rel = local_peaks[-1]
                shoulder_amp = upslope[shoulder_idx_rel]
                shoulder_ratios.append(1.0)
                shoulder_heights.append((shoulder_amp - foot_amp) / (sys_amp - foot_amp))
                shoulder_times.append(shoulder_idx_rel / (sys_idx - foot_idx))

            descent_early = beat[sys_idx:min(len(beat), sys_idx + int(0.15 * fs))]
            if len(descent_early) > 5:
                sec_peaks, _ = find_peaks(descent_early, distance=3, prominence=0.03 * (sys_amp - foot_amp))
                if len(sec_peaks) > 0:
                    shoulder_ratios.append(1.0)
                    sec_amp = descent_early[sec_peaks[0]]
                    shoulder_heights.append((sec_amp - foot_amp) / (sys_amp - foot_amp))

            all_peaks, _ = find_peaks(beat, distance=3, prominence=0.02 * (sys_amp - foot_amp))
            significant = [p for p in all_peaks if beat[p] > foot_amp + 0.5 * (sys_amp - foot_amp)]
            if len(significant) >= 2:
                sorted_peaks = sorted(significant, key=lambda p: beat[p], reverse=True)
                if len(sorted_peaks) >= 2:
                    bifurcation_scores.append(beat[sorted_peaks[1]] / beat[sorted_peaks[0]])

        result = default.copy()
        if shoulder_ratios: result['shoulder_present_ratio'] = np.mean(shoulder_ratios)
        if shoulder_heights: result['shoulder_height_ratio'] = safe_mean(shoulder_heights)
        if shoulder_times: result['shoulder_time_ratio'] = safe_mean(shoulder_times)
        if bifurcation_scores: result['peak_bifurcation_index'] = safe_mean(bifurcation_scores)
        if roundness_scores: result['systolic_roundness'] = safe_mean(roundness_scores)
        return result
    except Exception:
        return default


def extract_notch_characterization(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'notch_width_ms': 0.0, 'notch_sharpness': 0.0,
        'notch_foot_ratio': 0.0, 'post_notch_recovery_slope': 0.0,
        'notch_to_end_ratio': 0.0, 'notch_symmetry_index': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 3 or len(beats) < 2:
            return default

        widths, sharpnesses, foot_ratios, recovery_slopes, end_ratios, symmetries = [], [], [], [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 20:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            notch_idx = fid['notch_idx']
            foot_amp = fid['foot_amp']
            sys_amp = fid['sys_peak_amp']
            notch_amp = fid['notch_amp']
            if notch_idx is None or sys_idx >= len(beat) - 3:
                continue

            search_win = max(3, int(0.05 * fs))
            left_start = max(sys_idx, notch_idx - search_win)
            left_region = beat[left_start:notch_idx + 1]
            right_end = min(len(beat), notch_idx + search_win + 1)
            right_region = beat[notch_idx:right_end]

            left_max = np.max(left_region) if len(left_region) > 0 else notch_amp
            right_max = np.max(right_region) if len(right_region) > 0 else notch_amp
            shoulder_level = min(left_max, right_max)
            depth = shoulder_level - notch_amp
            if depth > 0:
                half_level = notch_amp + 0.5 * depth
                left_cross = notch_idx
                for i in range(notch_idx - 1, left_start - 1, -1):
                    if beat[i] >= half_level:
                        left_cross = i
                        break
                right_cross = notch_idx
                for i in range(notch_idx + 1, right_end):
                    if beat[i] >= half_level:
                        right_cross = i
                        break
                width = (right_cross - left_cross) / fs * 1000.0
                widths.append(width)

            if notch_idx > 1 and notch_idx < len(beat) - 2:
                d2 = beat[notch_idx - 1] - 2 * beat[notch_idx] + beat[notch_idx + 1]
                sharpnesses.append(d2)
            pp = sys_amp - foot_amp
            if pp > 0:
                foot_ratios.append((notch_amp - foot_amp) / pp)

            if fid['dia_peak_idx'] is not None:
                post_notch = beat[notch_idx:fid['dia_peak_idx'] + 1]
                if len(post_notch) > 1:
                    t = np.arange(len(post_notch)) / fs
                    slope = np.polyfit(t, post_notch, 1)[0]
                    recovery_slopes.append(slope)

            end_ratios.append((notch_idx - foot_idx) / (len(beat) - foot_idx))

            if left_max > notch_amp and right_max > notch_amp:
                left_drop = left_max - notch_amp
                right_drop = right_max - notch_amp
                if (left_drop + right_drop) > 0:
                    symmetries.append(abs(left_drop - right_drop) / (left_drop + right_drop))

        result = default.copy()
        if widths: result['notch_width_ms'] = safe_mean(widths)
        if sharpnesses: result['notch_sharpness'] = safe_mean(sharpnesses)
        if foot_ratios: result['notch_foot_ratio'] = safe_mean(foot_ratios)
        if recovery_slopes: result['post_notch_recovery_slope'] = safe_mean(recovery_slopes)
        if end_ratios: result['notch_to_end_ratio'] = safe_mean(end_ratios)
        if symmetries: result['notch_symmetry_index'] = safe_mean(symmetries)
        return result
    except Exception:
        return default

def extract_spectral_stiffness_features(ppg_signal, fs, pulse_rate, ppg_clean=None):
    default = {
        'spec_hf_power_ratio': 0.0, 'spec_very_hf_ratio': 0.0,
        'spec_slope_1_4_hz': 0.0, 'spec_entropy_highband': 0.0,
        'spec_centroid_above_fund': 0.0, 'spec_peak_skewness': 0.0,
    }
    try:
        sig = ppg_clean if ppg_clean is not None else ppg_signal
        ppg_detrended = sig - np.mean(sig)
        nperseg = min(512, len(ppg_signal))
        freqs, psd = welch(ppg_detrended, fs=fs, nperseg=nperseg, nfft=max(1024, nperseg * 2))
        if len(psd) == 0:
            return default

        total_power = np.sum(psd)
        fundamental = pulse_rate / 60.0

        hf_mask = freqs > 2.0
        vhf_mask = (freqs > 4.0) & (freqs <= 8.0)
        hf_power = np.sum(psd[hf_mask])
        vhf_power = np.sum(psd[vhf_mask])

        slope = 0.0
        slope_mask = (freqs >= 1.0) & (freqs <= 4.0) & (psd > 0)
        if np.sum(slope_mask) > 5:
            log_f = np.log10(freqs[slope_mask])
            log_p = np.log10(psd[slope_mask])
            slope = np.polyfit(log_f, log_p, 1)[0]

        hf_psd = psd[hf_mask]
        if np.sum(hf_psd) > 0:
            hf_norm = hf_psd / np.sum(hf_psd)
            hf_nz = hf_norm > 0
            hf_entropy = -np.sum(hf_norm[hf_nz] * np.log2(hf_norm[hf_nz]))
        else:
            hf_entropy = 0.0

        above_fund = freqs > fundamental + 0.5
        if np.sum(psd[above_fund]) > 0:
            centroid_above = np.sum(freqs[above_fund] * psd[above_fund]) / np.sum(psd[above_fund])
        else:
            centroid_above = 0.0

        around_fund = (freqs >= fundamental - 0.5) & (freqs <= fundamental + 2.0)
        if np.sum(around_fund) > 3:
            peak_skew = skew(psd[around_fund], bias=False)
        else:
            peak_skew = 0.0

        return {
            'spec_hf_power_ratio': float(hf_power / total_power) if total_power > 0 else 0.0,
            'spec_very_hf_ratio': float(vhf_power / total_power) if total_power > 0 else 0.0,
            'spec_slope_1_4_hz': float(slope),
            'spec_entropy_highband': float(hf_entropy),
            'spec_centroid_above_fund': float(centroid_above),
            'spec_peak_skewness': float(peak_skew),
        }
    except Exception:
        return default


def extract_ppg_complexity(ppg_signal, fs):
    default = {
        'ppg_sample_entropy': 0.0, 'ppg_dfa_alpha1': 0.0,
        'ppg_dfa_alpha2': 0.0, 'ppg_lle': 0.0, 'ppg_hurst': 0.0,
    }
    try:
        sig = ppg_signal.copy()
        if len(sig) > 5000:
            step = len(sig) // 2500
            sig = sig[::step]
        sig = sig - np.mean(sig)
        sig = sig / (np.std(sig) + 1e-10)

        # --- Sample Entropy (bottleneck fix) ---
        try:
            import antropy
            se = float(antropy.sample_entropy(sig, order=2))
        except Exception:
            # Fallback to custom slow implementation
            def sample_entropy(signal, m=2, r=0.2):
                N = len(signal)
                if N < m + 2:
                    return 0.0
                r = r * np.std(signal)
                def _phi(m_val):
                    x = np.array([signal[i:i + m_val] for i in range(N - m_val + 1)])
                    C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1) - 1
                    return np.sum(C) / (N - m_val + 1)
                try:
                    return -np.log(_phi(m + 1) / (_phi(m) + 1e-10))
                except Exception:
                    return 0.0
            se = sample_entropy(sig)

        def dfa(signal, scales=None):
            if scales is None:
                scales = np.unique(np.logspace(0.5, 2.5, 20).astype(int))
                scales = scales[(scales > 4) & (scales < len(signal) // 4)]
            y = np.cumsum(signal - np.mean(signal))
            fluctuations = []
            for scale in scales:
                n_segments = len(y) // scale
                if n_segments < 2:
                    continue
                y_trunc = y[:n_segments * scale]
                y_reshaped = y_trunc.reshape(n_segments, scale)
                x = np.arange(scale)
                f = 0.0
                for seg in y_reshaped:
                    slope, intercept = np.polyfit(x, seg, 1)
                    trend = slope * x + intercept
                    f += np.mean((seg - trend) ** 2)
                fluctuations.append(np.sqrt(f / n_segments))
            if len(fluctuations) < 4:
                return 0.0, 0.0, scales[:len(fluctuations)]
            log_s = np.log10(scales[:len(fluctuations)])
            log_f = np.log10(fluctuations)
            mid = len(log_s) // 2
            if mid > 2:
                alpha1 = np.polyfit(log_s[:mid], log_f[:mid], 1)[0]
                alpha2 = np.polyfit(log_s[mid:], log_f[mid:], 1)[0]
            else:
                alpha1 = alpha2 = np.polyfit(log_s, log_f, 1)[0]
            return alpha1, alpha2, scales[:len(fluctuations)]

        a1, a2, _ = dfa(sig)
        hurst = (a1 + a2) / 2.0

        def lle_surrogate(signal, tau=1, emb_dim=5):
            N = len(signal)
            if N < emb_dim * 10:
                return 0.0
            m = N - (emb_dim - 1) * tau
            if m < 10:
                return 0.0
            Y = np.array([signal[i:i + emb_dim * tau:tau] for i in range(m)])
            divergences = []
            for i in range(0, min(m, 200), 2):
                dists = np.linalg.norm(Y - Y[i], axis=1)
                dists[i] = np.inf
                j = np.argmin(dists)
                if dists[j] < 1e-10:
                    continue
                max_k = min(20, m - max(i, j) - 1)
                if max_k < 5:
                    continue
                for k in range(1, max_k):
                    d_new = np.linalg.norm(Y[i + k] - Y[j + k])
                    if d_new > 0:
                        divergences.append(np.log(d_new / dists[j]) / k)
                        break
            return np.mean(divergences) if divergences else 0.0

        lle = lle_surrogate(sig)

        return {
            'ppg_sample_entropy': float(se), 'ppg_dfa_alpha1': float(a1),
            'ppg_dfa_alpha2': float(a2), 'ppg_lle': float(lle), 'ppg_hurst': float(hurst),
        }
    except Exception:
        return default


def extract_ppg_variability(ppg_signal, fs, ppg_clean=None, peaks=None, beats=None):
    default = {
        'ppg_amp_sdnn': 0.0, 'ppg_amp_cv': 0.0, 'ppg_amp_rmssd': 0.0,
        'ppg_area_cv': 0.0, 'ppg_shape_consistency': 0.0, 'ppg_width_cv': 0.0,
    }
    try:
        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(ppg_signal, sampling_rate=fs)
        if peaks is None:
            peaks = extract_peak_indices(nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])
        if beats is None:
            beats, _, _ = segment_beats(ppg_c, peaks, fs)
        if len(peaks) < 5 or len(beats) < 4:
            return default

        amplitudes, areas, widths, shapes = [], [], [], []
        for beat in get_core_beats(beats):
            if len(beat) < 10:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            amp = fid['sys_peak_amp'] - fid['foot_amp']
            amplitudes.append(amp)
            areas.append(trapezoid_compat(beat - fid['foot_amp']))
            widths.append(len(beat) / fs * 1000.0)
            beat_norm = (beat - np.min(beat)) / (np.ptp(beat) + 1e-10)
            shapes.append(beat_norm)

        if len(amplitudes) < 3:
            return default

        amp_sdnn = np.std(amplitudes, ddof=1)
        amp_mean = np.mean(amplitudes)
        amp_cv = amp_sdnn / (amp_mean + 1e-10)
        amp_diffs = np.diff(amplitudes)
        amp_rmssd = np.sqrt(np.mean(amp_diffs ** 2))
        area_cv = np.std(areas, ddof=1) / (np.mean(areas) + 1e-10)
        width_cv = np.std(widths, ddof=1) / (np.mean(widths) + 1e-10)

        shape_corrs = []
        min_shape_len = min(len(s) for s in shapes)
        shapes_aligned = [s[:min_shape_len] for s in shapes]
        for i in range(len(shapes_aligned) - 1):
            if len(shapes_aligned[i]) > 5:
                c = np.corrcoef(shapes_aligned[i], shapes_aligned[i+1])[0, 1]
                if not np.isnan(c):
                    shape_corrs.append(c)
        shape_consistency = np.mean(shape_corrs) if shape_corrs else 0.0

        return {
            'ppg_amp_sdnn': float(amp_sdnn), 'ppg_amp_cv': float(amp_cv),
            'ppg_amp_rmssd': float(amp_rmssd), 'ppg_area_cv': float(area_cv),
            'ppg_shape_consistency': float(shape_consistency), 'ppg_width_cv': float(width_cv),
        }
    except Exception:
        return default


def extract_cardiac_intervals(ppg_signal, ecg_signal, fs, ppg_clean=None,
                              ecg_clean=None, ppg_peaks=None, beats=None,
                              foot_indices=None):
    default = {
        'pep_ms': 0.0, 'lvet_ms': 0.0, 'pep_lvet_ratio': 0.0,
        'pep_variability': 0.0, 'lvet_variability': 0.0,
    }
    try:
        if ecg_signal is None or len(ecg_signal) < 100:
            return default

        ecg_c = ecg_clean if ecg_clean is not None else nk.ecg_clean(
            ecg_signal, sampling_rate=fs, method='neurokit')
        r_peaks_arr = np.array(
            nk.ecg_peaks(ecg_c, sampling_rate=fs, method='neurokit')[1]['ECG_R_Peaks'],
            dtype=int)

        ppg_c = ppg_clean if ppg_clean is not None else nk.ppg_clean(
            ppg_signal, sampling_rate=fs)
        ppg_peaks_arr = ppg_peaks if ppg_peaks is not None else extract_peak_indices(
            nk.ppg_peaks(ppg_c, sampling_rate=fs)[0])

        if len(r_peaks_arr) < 2 or len(ppg_peaks_arr) < 2:
            return default
        if len(r_peaks_arr) < 3:
            return default

        if beats is None or foot_indices is None:
            beats, foot_indices, _ = segment_beats(ppg_c, ppg_peaks_arr, fs)

        core_beats = get_core_beats(beats)
        core_foot_indices = foot_indices[1:-1] if len(beats) > 2 else foot_indices

        expected_r = len(ppg_peaks_arr)
        ecg_reliable = len(r_peaks_arr) >= max(3, int(0.5 * expected_r))

        pep_values, lvet_values = [], []

        if ecg_reliable:
            for foot_idx, beat in zip(core_foot_indices, core_beats):
                if len(beat) < 10:
                    continue
                prev_r = r_peaks_arr[r_peaks_arr < foot_idx]
                if len(prev_r) == 0:
                    continue
                r_idx = prev_r[-1]
                pep = (foot_idx - r_idx) / fs * 1000.0
                if not (60 <= pep <= 350):
                    continue
                fid = detect_ppg_fiducials(beat, fs)
                if not is_valid_fiducials(fid):
                    continue
                pep_values.append(pep)
                if fid['notch_idx'] is not None:
                    lvet = fid['notch_idx'] / fs * 1000.0
                    if 200 < lvet <= 550:
                        lvet_values.append(lvet)

        if len(pep_values) < 3:
            pep_values.clear()
            lvet_values.clear()
            ft_notch = []
            for beat in core_beats:
                if len(beat) < 10:
                    continue
                fid = detect_ppg_fiducials(beat, fs)
                if not is_valid_fiducials(fid):
                    continue
                if fid['notch_idx'] is not None:
                    t = fid['notch_idx'] / fs * 1000.0
                    if 200 < t < 500:
                        ft_notch.append(t)

            ptt_dict = extract_ptt_features(
                ppg_signal, ecg_signal, fs,
                ppg_clean=ppg_clean, ecg_clean=ecg_clean,
                ppg_peaks=ppg_peaks_arr, r_peaks=r_peaks_arr,
                foot_indices=foot_indices)
            ptt_foot = ptt_dict.get('ptt_r_to_ppg_foot_ms', 0)

            if ft_notch:
                mean_ft_notch = safe_mean(ft_notch)
                if 200 < mean_ft_notch <= 550:
                    lvet_values.append(mean_ft_notch)

            if ptt_foot > 0:
                pep_est = ptt_foot * 0.40
                if 60 <= pep_est <= 350:
                    pep_values.append(pep_est)
            elif lvet_values:
                pep_est = lvet_values[0] * 0.25
                if 50 <= pep_est <= 150:
                    pep_values.append(pep_est)

        result = default.copy()
        if pep_values:
            result['pep_ms'] = safe_mean(pep_values)
            result['pep_variability'] = safe_std(pep_values)
        if lvet_values:
            result['lvet_ms'] = safe_mean(lvet_values)
            result['lvet_variability'] = safe_std(lvet_values)
        if pep_values and lvet_values and result['lvet_ms'] > 0:
            result['pep_lvet_ratio'] = result['pep_ms'] / result['lvet_ms']
        return result
    except Exception:
        return default

# ============================================================
# 6. UNIQUE PIPELINE-V5 EXTRACTORS (not in all_beats base)
# ============================================================
PERIOD_EXTREMA_MIN_PROMINENCE = 0.02
# ---------------------------------------------------------------------------
def _period_p_check(p):
    p = float(p)
    if not np.isfinite(p) or p == 0.0:
        raise ValueError("period_p must be finite and non-zero.")
    return p


def _paper_phi1_period(amps, durs, m_math, p):
    A = np.asarray(amps, dtype=float).reshape(-1)
    D = np.asarray(durs, dtype=float).reshape(-1)
    p = _period_p_check(p)
    if len(A) != len(D):
        raise ValueError("amps and durs must have the same length.")
    if m_math < 2 or m_math > len(A):
        raise ValueError("Phi1 requires 2 <= m_math <= number of steps.")

    m_py = int(m_math) - 1
    delta_T = float(np.sum(D[:m_py + 1]))
    denom = delta_T * m_py  # (m_math - 1) == m_py
    if denom == 0.0:
        raise ZeroDivisionError("Phi1 denominator is zero.")

    total = 0.0
    for n_py in range(0, m_py + 1):
        inner = 0.0
        for j_math in range(1, int(m_math) + 1):
            j_py = j_math - 1
            base = (float(D[j_py]) * p) / denom
            inner += base ** j_math
        total += float(A[n_py]) * inner
    return float(total)


def _paper_phi2_period(amps, durs, k1_math, m2_math, p):
    A = np.asarray(amps, dtype=float).reshape(-1)
    D = np.asarray(durs, dtype=float).reshape(-1)
    p = _period_p_check(p)
    if len(A) != len(D):
        raise ValueError("amps and durs must have the same length.")
    if not (1 <= k1_math < m2_math <= len(A)):
        raise ValueError("Require 1 <= k1_math < m2_math <= number of steps.")

    k1_py = int(k1_math) - 1
    m2_py = int(m2_math) - 1
    delta_T = float(np.sum(D[k1_py:m2_py + 1]))
    denom = delta_T * m2_py  # (m2_math - 1) == m2_py
    if denom == 0.0:
        raise ZeroDivisionError("Phi2 denominator is zero.")

    n_terms = int(m2_math) - int(k1_math) + 1
    total = 0.0
    for n_py in range(k1_py, m2_py + 1):
        inner = 0.0
        for j_math in range(1, n_terms + 1):
            j_idx_py = (j_math - 1) + k1_py
            dt = float(D[j_idx_py])
            base = (dt * p) / denom
            inner += base ** j_math
        total += float(A[n_py]) * inner
    return float(total)


def _paper_phi3_period(amps, durs, k2_math, m3_math, p):
    A = np.asarray(amps, dtype=float).reshape(-1)
    D = np.asarray(durs, dtype=float).reshape(-1)
    p = _period_p_check(p)
    if len(A) != len(D):
        raise ValueError("amps and durs must have the same length.")
    if not (1 <= k2_math < m3_math <= len(A)):
        raise ValueError("Require 1 <= k2_math < m3_math <= number of steps.")

    k2_py = int(k2_math) - 1
    m3_py = int(m3_math) - 1
    delta_T = float(np.sum(D[k2_py:m3_py + 1]))
    denom = delta_T * m3_py  # (m3_math - 1) == m3_py
    if denom == 0.0:
        raise ZeroDivisionError("Phi3 denominator is zero.")

    n_terms = int(m3_math) - int(k2_math) + 1
    total = 0.0
    for n_py in range(k2_py, m3_py + 1):
        inner = 0.0
        for j_math in range(1, n_terms + 1):
            j_idx_py = (j_math - 1) + k2_py
            dt = float(D[j_idx_py])
            base = (dt * p) / denom
            inner += base ** j_math
        total += float(A[n_py]) * inner
    return float(total)


def _paper_psi_period_midpoint(amps, durs, m_math, n_math, p):
    """Formula Psi, evaluated at the midpoint of every decline step.

    Midpoint aggregation is an explicit numerical convention for the discrete
    implementation; it is not claimed to be uniquely specified by the paper.
    """
    A = np.asarray(amps, dtype=float).reshape(-1)
    D = np.asarray(durs, dtype=float).reshape(-1)
    p = _period_p_check(p)
    if len(A) != len(D):
        raise ValueError("amps and durs must have the same length.")
    if not (1 <= m_math < n_math <= len(A)):
        raise ValueError("Require 1 <= m_math < n_math <= number of steps.")

    m_py = int(m_math) - 1
    tau = np.cumsum(D)
    tau_peak = float(tau[m_py])   # cumulative time at the peak
    values = []
    A_m = float(A[m_py])

    for k_math in range(int(m_math) + 1, int(n_math) + 1):
        k_py = k_math - 1
        start_t = 0.0 if k_math == 1 else float(tau[k_py - 1])
        end_t = float(tau[k_py])
        # Time from peak (t=0 at peak), not from beginning of pulse
        t_mid = 0.5 * (start_t + end_t) - tau_peak
        delta_tau_k = float(D[k_py])
        denom = (int(n_math) - int(m_math)) * delta_tau_k
        if denom == 0.0:
            continue
        exponent = (-t_mid * p) / denom
        amplitude_diff = float(A[k_py - 1] - A[k_py])
        linear_term = (int(n_math) - int(m_math)) / p
        psi = A_m * np.exp(exponent) * amplitude_diff * linear_term
        if np.isfinite(psi):
            values.append(float(psi))
    return np.asarray(values, dtype=float)

PERIOD_BASE_FEATURE_ORDER = BEAT_FEATURE_ORDER + [
    'f_mean', 'f_std', 'f_max',
    'phi_value', 'phi_case', 'phi_valid',
    'psi_mean', 'psi_std', 'psi_max',
    'n_local_maxima', 'n_local_minima',
    'pulse_duration_ms',
]


def normalize_pulse_period(pulse, n_points=1000):
    """Map one pulse period to x=0..1 s and y=0..1 relative units."""
    p = np.asarray(pulse, dtype=float).flatten()
    if len(p) < 10:
        return None
    p = p[np.isfinite(p)]
    if len(p) < 10:
        return None

    pmin = float(np.min(p))
    pmax = float(np.max(p))
    rng = pmax - pmin
    if not np.isfinite(rng) or rng <= 1e-12:
        return None

    y = (p - pmin) / rng
    if n_points and int(n_points) > 1:
        x_old = np.linspace(0.0, 1.0, len(y))
        x_new = np.linspace(0.0, 1.0, int(n_points))
        y = np.interp(x_new, x_old, y)
    return np.asarray(y, dtype=float)


def convert_normalized_pulse_to_step_representation(pulse_norm):
    """Convert a normalized pulse to a duration-weighted constant-amplitude step representation.

    The waveform is first quantized to 4 decimals (the project convention).
    Each step amplitude is the value at the beginning of a constant run, and
    each duration is the time from that run start to the next run start (or to
    the normalized endpoint t=1 for the final run). Thus:
        len(step_amps) == len(step_durations)
        sum(step_durations) == 1
        step_durations > 0
    """
    y = np.asarray(pulse_norm, dtype=float).flatten()
    if len(y) < 10:
        return None, None, None
    if not np.all(np.isfinite(y)):
        return None, None, None

    y = np.clip(y, 0.0, 1.0)
    y = np.round(y, decimals=4)
    n = len(y)
    x = np.linspace(0.0, 1.0, n)

    # Run starts: 0 plus every index whose amplitude differs from the prior
    # quantized sample. This preserves the first run instead of discarding it.
    change = np.flatnonzero(y[1:] != y[:-1]) + 1
    starts = np.r_[0, change].astype(int)

    # A final singleton at x=1 has zero duration under this endpoint convention;
    # it is not a time interval and is therefore omitted from the step sequence.
    valid = starts < (n - 1)
    starts = starts[valid]
    if len(starts) == 0:
        return None, None, None

    next_starts = np.r_[starts[1:], n - 1]
    step_amps = y[starts].astype(float)
    step_times = x[starts].astype(float)
    step_durations = (x[next_starts] - x[starts]).astype(float)

    # The last interval extends to the normalized endpoint t=1.
    step_durations[-1] = 1.0 - step_times[-1]

    # Guard against numerical endpoint artifacts and enforce the exact
    # partition of [0,1].
    keep = step_durations > 0.0
    step_amps = step_amps[keep]
    step_times = step_times[keep]
    step_durations = step_durations[keep]
    if len(step_amps) == 0:
        return None, None, None

    total = float(np.sum(step_durations))
    if not np.isfinite(total) or total <= 0.0:
        return None, None, None
    step_durations[-1] += 1.0 - total

    # Final numerical sanity checks. Any correction smaller than machine
    # precision is harmless; a genuinely negative duration is not accepted.
    if np.any(~np.isfinite(step_durations)) or np.any(step_durations <= 0.0):
        return None, None, None

    return step_amps, step_times, step_durations


def _robust_median_rr(peaks, fs):
    """Robust RR estimate in samples from an ordered peak sequence."""
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return float(max(1, round(0.8 * fs)))
    diffs = np.diff(peaks).astype(float)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return float(max(1, round(0.8 * fs)))
    med = float(np.median(diffs))
    return med if med > 0 else float(max(1, round(0.8 * fs)))


def _candidate_peak_prominence(signal, candidate_indices):
    """Return prominence values for candidate peak indices."""
    candidates = np.asarray(candidate_indices, dtype=int)
    if len(candidates) == 0:
        return np.array([], dtype=float)
    try:
        from scipy.signal import peak_prominences
        return np.asarray(
            peak_prominences(np.asarray(signal, dtype=float), candidates)[0],
            dtype=float,
        )
    except Exception:
        return np.full(len(candidates), np.nan, dtype=float)


def _find_missing_peaks_in_gap(ppg_clean, left_peak, right_peak, expected_rr, fs,
                               min_candidate_prominence_ratio=0.10,
                               min_side_amplitude_ratio=0.05,
                               min_strong_side_ratio=0.10,
                               min_side_balance=0.25):


    left_peak = int(left_peak)
    right_peak = int(right_peak)
    expected_rr = float(expected_rr)
    if right_peak <= left_peak + 1 or not np.isfinite(expected_rr) or expected_rr <= 1:
        return None

    gap = right_peak - left_peak
    # A genuine long RR is allowed to remain untouched unless a candidate can
    # split the interval into two plausible beats.
    lo = int(round(left_peak + 0.30 * expected_rr))
    hi = int(round(right_peak - 0.30 * expected_rr))
    lo = max(lo, left_peak + 1)
    hi = min(hi, right_peak - 1)
    if hi <= lo:
        return None

    seg = np.asarray(ppg_clean, dtype=float)
    finite = seg[np.isfinite(seg)]
    if finite.size < 10:
        return None
    robust_range = float(np.percentile(finite, 95) - np.percentile(finite, 5))
    if not np.isfinite(robust_range) or robust_range <= 1e-12:
        robust_range = float(np.ptp(finite))
    if not np.isfinite(robust_range) or robust_range <= 1e-12:
        return None

    prom = float(min_candidate_prominence_ratio) * robust_range
    distance = max(3, int(round(0.30 * expected_rr)))

    try:
        candidates, props = find_peaks(
            seg[lo:hi + 1],
            distance=distance,
            prominence=prom,
        )
    except Exception:
        return None

    if len(candidates) == 0:
        return None
    candidates = candidates.astype(int) + lo
    prominences = np.asarray(
        props.get('prominences', np.full(len(candidates), np.nan)),
        dtype=float,
    )

    # Two-sided morphology is evaluated in a local neighborhood around the
    # candidate. The window scales with expected RR but is capped so that a
    # distant large pulse does not define the candidate's local baseline.
    local_radius = int(round(0.10 * expected_rr))
    local_radius = max(40, min(local_radius, 150))

    valid = []
    for c, pr in zip(candidates, prominences):
        left_piece = float(c - left_peak)
        right_piece = float(right_peak - c)
        if not (0.50 * expected_rr <= left_piece <= 1.35 * expected_rr):
            continue
        if not (0.50 * expected_rr <= right_piece <= 1.35 * expected_rr):
            continue

        # Candidate must have a genuine local rise/fall on both sides.
        l0 = max(left_peak, int(c) - local_radius)
        r1 = min(right_peak, int(c) + local_radius)
        if c <= l0 or r1 <= c:
            continue
        left_local = seg[l0:c + 1]
        right_local = seg[c:r1 + 1]
        if not (np.any(np.isfinite(left_local)) and np.any(np.isfinite(right_local))):
            continue
        candidate_amp = float(seg[c])
        left_drop = candidate_amp - float(np.nanmin(left_local))
        right_drop = candidate_amp - float(np.nanmin(right_local))
        if not (np.isfinite(left_drop) and np.isfinite(right_drop)):
            continue
        if left_drop <= 0.0 or right_drop <= 0.0:
            continue

        strong_ratio = max(left_drop, right_drop) / robust_range
        weak_ratio = min(left_drop, right_drop) / robust_range
        balance = min(left_drop, right_drop) / max(left_drop, right_drop)

        if strong_ratio < float(min_strong_side_ratio):
            continue
        if weak_ratio < float(min_side_amplitude_ratio):
            continue
        if balance < float(min_side_balance):
            continue

        temporal_error = abs(left_piece - expected_rr) + abs(right_piece - expected_rr)
        valid.append((
            temporal_error,
            -float(pr) if np.isfinite(pr) else 0.0,
            -float(balance),
            int(c),
        ))

    if not valid:
        return None
    valid.sort(key=lambda z: (z[0], z[1], z[2], z[3]))
    return int(valid[0][3])

def refine_period_peak_sequence(ppg_clean, peaks, fs,
                                 long_gap_ratio=1.60,
                                 max_recovery_passes=2):
    """Repair a peak sequence for the period layer only.

    The original NeuroKit peak sequence is retained elsewhere in the pipeline.
    This function is intentionally conservative: it only inserts a peak into a
    long gap when the signal contains a candidate that creates two plausible
    RR intervals around the segment median RR.
    """
    peaks = np.asarray(peaks, dtype=int)
    peaks = np.unique(peaks[np.isfinite(peaks)]).astype(int) if len(peaks) else np.array([], dtype=int)
    if len(peaks) < 3:
        return peaks, np.zeros(len(peaks), dtype=bool), {
            'input_peak_count': int(len(peaks)),
            'refined_peak_count': int(len(peaks)),
            'inserted_peak_count': 0,
        }

    rr = _robust_median_rr(peaks, fs)
    refined = [int(p) for p in peaks]
    inserted_flags = [False] * len(refined)

    for _ in range(max(1, int(max_recovery_passes))):
        changed = False
        out = [refined[0]]
        out_flags = [inserted_flags[0]]
        local_rr = _robust_median_rr(np.asarray(refined, dtype=int), fs)
        for a, b, fb in zip(refined[:-1], refined[1:], inserted_flags[1:]):
            gap = b - a
            if gap >= float(long_gap_ratio) * local_rr:
                candidate = _find_missing_peaks_in_gap(
                    ppg_clean, a, b, local_rr, fs
                )
                if candidate is not None and a < candidate < b:
                    out.append(candidate)
                    out_flags.append(True)
                    changed = True
            out.append(b)
            out_flags.append(fb)
        refined, inserted_flags = out, out_flags
        if not changed:
            break

    refined_arr = np.asarray(refined, dtype=int)
    flag_arr = np.asarray(inserted_flags, dtype=bool)
    return refined_arr, flag_arr, {
        'input_peak_count': int(len(peaks)),
        'refined_peak_count': int(len(refined_arr)),
        'inserted_peak_count': int(np.sum(flag_arr)),
        'median_rr_samples': float(_robust_median_rr(refined_arr, fs)),
    }


def _find_refined_foot(ppg_clean, current_peak, previous_peak, next_peak, expected_rr, fs):
    """RR-adaptive foot estimate for one systolic peak."""
    p = int(current_peak)
    rr = float(expected_rr)
    if p <= 0:
        return None, 'invalid_peak'

    if previous_peak is None:
        # First-peak fallback: search a broad pre-peak fraction of expected RR.
        lo = max(0, int(round(p - 0.60 * rr)))
        hi = max(lo, int(round(p - 0.08 * rr)))
        if hi <= lo:
            return None, 'invalid_first_window'
        local = np.asarray(ppg_clean[lo:hi + 1], dtype=float)
        if not np.any(np.isfinite(local)):
            return None, 'nonfinite_first_window'
        foot = lo + int(np.nanargmin(local))
        edge = int(foot == lo)
        return int(foot), ('first_peak_edge_fallback' if edge else 'first_peak_adaptive')

    prev = int(previous_peak)
    gap = float(max(p - prev, 1))
    # Foot should occur after the previous systolic peak and sufficiently before
    # the current systolic peak. The search window adapts to the local RR.
    lo = max(prev + int(round(0.10 * gap)), int(round(p - 0.70 * rr)))
    hi = min(p - max(1, int(round(0.08 * gap))), p - 1)
    if hi <= lo:
        lo = prev + 1
        hi = p - 1
    if hi <= lo:
        return None, 'invalid_adjacent_window'

    local = np.asarray(ppg_clean[lo:hi + 1], dtype=float)
    if not np.any(np.isfinite(local)):
        return None, 'nonfinite_adjacent_window'

    foot = lo + int(np.nanargmin(local))
    if foot == lo or foot == hi:
        # Only use the full adjacent-peak interval as a fallback when the
        # adaptive window lands at an edge.
        flo = prev
        fhi = p
        full = np.asarray(ppg_clean[flo:fhi + 1], dtype=float)
        if np.any(np.isfinite(full)):
            foot = flo + int(np.nanargmin(full))
            return int(foot), 'adjacent_interval_fallback'
        return int(foot), 'adaptive_edge'

    return int(foot), 'adaptive'


def segment_pulse_periods_from_peaks(ppg_clean, peaks, fs,
                                    long_gap_ratio=1.60,
                                    max_recovery_passes=2,
                                    reject_mask=None):  # [join_mask_v2]
    """Build complete foot_i -> foot_(i+1) pulse periods with V9 repair."""
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 3:
        return [], []

    refined_peaks, inserted_flags, peak_stats = refine_period_peak_sequence(
        ppg_clean, peaks, fs,
        long_gap_ratio=long_gap_ratio,
        max_recovery_passes=max_recovery_passes,
    )
    expected_rr = float(peak_stats.get('median_rr_samples', _robust_median_rr(refined_peaks, fs)))

    feet = []
    foot_methods = []
    for i, p in enumerate(refined_peaks):
        prev = int(refined_peaks[i - 1]) if i > 0 else None
        nxt = int(refined_peaks[i + 1]) if i + 1 < len(refined_peaks) else None
        local_rr = expected_rr
        if prev is not None:
            local_rr = float(p - prev)
        elif nxt is not None:
            local_rr = float(nxt - p)
        local_rr = max(local_rr, 1.0)
        foot, method = _find_refined_foot(
            ppg_clean, int(p), prev, nxt, local_rr, fs
        )
        feet.append(foot)
        foot_methods.append(method)

    periods, meta = [], []

    # [join_mask_v2] reconstruct join positions from contiguous True runs
    _join_centers = []
    if reject_mask is not None:
        _in_run = False
        _run_start = 0
        for _i, _v in enumerate(reject_mask):
            if _v and not _in_run:
                _in_run = True
                _run_start = _i
            elif not _v and _in_run:
                _in_run = False
                _join_centers.append((_run_start + _i - 1) // 2)
        if _in_run:
            _join_centers.append((_run_start + len(reject_mask) - 1) // 2)
    min_len = max(20, int(round(0.30 * fs)))

    for i in range(len(refined_peaks) - 1):
        f0, f1 = feet[i], feet[i + 1]
        peak_i = int(refined_peaks[i])
        if f0 is None or f1 is None or f1 <= f0:
            continue
        if not (f0 <= peak_i <= f1):
            continue

        period = np.asarray(ppg_clean[f0:f1 + 1], dtype=float)
        if len(period) < min_len:
            continue

        # [join_mask_v2] skip only periods that STRADDLE a join
        if reject_mask is not None:
            _straddles = False
            for _j in _join_centers:
                if f0 < _j < f1:
                    _straddles = True
                    break
            if _straddles:
                continue

        periods.append(period)
        meta.append({
            'start_sample': int(f0),
            'end_sample': int(f1),
            'peak_sample': peak_i,
            'duration_ms': float((f1 - f0) / fs * 1000.0),
            # V10-native metadata plus V9-compatible aliases.
            'peak_inserted_v10': int(bool(inserted_flags[i])),
            'input_peak_count_v9': int(peak_stats.get('input_peak_count', len(peaks))),
            'refined_peak_count_v10': int(peak_stats.get('refined_peak_count', len(refined_peaks))),
            'inserted_peak_count_v10': int(peak_stats.get('inserted_peak_count', 0)),
            'peak_inserted_v9': int(bool(inserted_flags[i])),
            'input_peak_count_v8': int(peak_stats.get('input_peak_count', len(peaks))),
            'refined_peak_count_v9': int(peak_stats.get('refined_peak_count', len(refined_peaks))),
            'inserted_peak_count_v9': int(peak_stats.get('inserted_peak_count', 0)),
            'start_foot_method_v9': str(foot_methods[i]),
            'end_foot_method_v9': str(foot_methods[i + 1]),
        })
    return periods, meta


def detect_period_extrema(step_amps, min_prominence=0.02):
    """Prominence-aware extrema for the V9 period morphology layer."""
    if step_amps is None:
        empty = {'locations': np.array([], dtype=int), 'amplitudes': np.array([], dtype=float), 'prominences': np.array([], dtype=float)}
        return empty, empty.copy()

    y = np.asarray(step_amps, dtype=float)
    n = len(y)
    if n < 3 or not np.all(np.isfinite(y)):
        empty = {'locations': np.array([], dtype=int), 'amplitudes': np.array([], dtype=float), 'prominences': np.array([], dtype=float)}
        return empty, empty.copy()

    try:
        peak_locs, peak_props = find_peaks(y, prominence=float(min_prominence))
        valley_locs, valley_props = find_peaks(-y, prominence=float(min_prominence))
    except Exception:
        empty = {'locations': np.array([], dtype=int), 'amplitudes': np.array([], dtype=float), 'prominences': np.array([], dtype=float)}
        return empty, empty.copy()

    peak_locs = np.asarray(peak_locs, dtype=int)
    valley_locs = np.asarray(valley_locs, dtype=int)
    peak_prom = np.asarray(peak_props.get('prominences', np.full(len(peak_locs), np.nan)), dtype=float)
    valley_prom = np.asarray(valley_props.get('prominences', np.full(len(valley_locs), np.nan)), dtype=float)

    peaks = {
        'locations': peak_locs,
        'amplitudes': y[peak_locs] if len(peak_locs) else np.array([], dtype=float),
        'prominences': peak_prom,
    }
    valleys = {
        'locations': valley_locs,
        'amplitudes': y[valley_locs] if len(valley_locs) else np.array([], dtype=float),
        'prominences': valley_prom,
    }
    return peaks, valleys


def calculate_decay_function_period_v9(step_amps, step_times, step_durations, peak_loc, valley_loc, p):
    """Compatibility wrapper for the period Psi calculation.

    Uses the frozen formula PSI and the explicit midpoint-per-decline-step
    aggregation convention. step_times is retained in the signature for
    backward compatibility but is not used by the paper-faithful calculation.
    """
    peak_loc = int(peak_loc)
    valley_loc = int(valley_loc)
    if peak_loc < 0 or valley_loc >= len(step_amps) or peak_loc >= valley_loc:
        return 0.0
    values = _paper_psi_period_midpoint(
        step_amps, step_durations,
        m_math=peak_loc + 1,
        n_math=valley_loc + 1,
        p=p,
    )
    return float(np.mean(values)) if len(values) else 0.0

def calculate_period_f_phi_psi(pulse_norm, period_p=PERIOD_P_DEFAULT, min_extrema_prominence=0.02):
    """F/Phi/Psi for ONE normalized pulse period (V9 period layer)."""
    result = {
        'f_mean': np.nan, 'f_std': np.nan, 'f_max': np.nan,
        'phi_value': np.nan, 'phi_case': 0, 'phi_valid': 0,
        'psi_mean': np.nan, 'psi_std': np.nan, 'psi_max': np.nan,
        'n_local_maxima': 0, 'n_local_minima': 0,
    }

    step_amps, step_times, step_durations = convert_normalized_pulse_to_step_representation(
        pulse_norm
    )
    if step_amps is None:
        return result

    peaks_step, valleys_step = detect_period_extrema(
        step_amps, min_prominence=min_extrema_prominence
    )
    peak_locs = np.asarray(peaks_step['locations'], dtype=int)
    valley_locs = np.asarray(valleys_step['locations'], dtype=int)

    result['n_local_maxima'] = int(len(peak_locs))
    result['n_local_minima'] = int(len(valley_locs))

    p = float(period_p)
    _period_p_check(p)

    # F around EVERY significant maximum.
    f_values = []
    for loc in peak_locs:
        if 0 < int(loc) < len(step_amps):
            delta_a = float(step_amps[int(loc)] - step_amps[int(loc) - 1])
            delta_tau = float(step_durations[int(loc)])
            val = abs(delta_a / delta_tau) if delta_tau > 0.0 else np.nan
        else:
            val = np.nan
        if np.isfinite(val):
            f_values.append(float(val))
    if f_values:
        result['f_mean'] = float(np.mean(f_values))
        result['f_std'] = float(np.std(f_values))
        result['f_max'] = float(np.max(f_values))

    # Phi CASE selection is based on TOTAL significant maxima in this period.
    n_max = len(peak_locs)
    if n_max == 1:
        phi = _paper_phi1_period(
            step_amps, step_durations, int(peak_locs[0]) + 1, p
        )
        result['phi_case'] = 1
    elif n_max == 2:
        between = valley_locs[
            (valley_locs > peak_locs[0]) &
            (valley_locs < peak_locs[1])
        ]
        if len(between):
            phi = calculate_rise_phi2(
                step_amps, step_durations, int(between[0]),
                int(peak_locs[1]), p
            )
        else:
            phi = np.nan
        result['phi_case'] = 2
    elif n_max == 3:
        between = valley_locs[
            (valley_locs > peak_locs[1]) &
            (valley_locs < peak_locs[2])
        ]
        if len(between):
            phi = calculate_rise_phi3(
                step_amps, step_durations, int(between[0]),
                int(peak_locs[2]), p
            )
        else:
            phi = np.nan
        result['phi_case'] = 3
    else:
        phi = np.nan
        result['phi_case'] = 0

    if np.isfinite(phi):
        result['phi_value'] = float(phi)
        result['phi_valid'] = 1

    # Psi: every significant maximum -> following internal valley; if no
    # internal valley follows, the period-ending foot is used as n.  Eq. (5) is
    # evaluated at the midpoint of each falling step and then aggregated across
    # the decline steps exactly as the frozen reference convention specifies.
    psi_values = []
    endpoint_loc = len(step_amps) - 1
    for loc in peak_locs:
        nxt = valley_locs[valley_locs > loc]
        target = int(nxt[0]) if len(nxt) else int(endpoint_loc)
        if target <= int(loc):
            continue
        if not (np.isfinite(step_amps[int(loc)]) and np.isfinite(step_amps[target])):
            continue
        if float(step_amps[target]) >= float(step_amps[int(loc)]):
            continue
        psi_mean_value = calculate_decay_function_period_v9(
            step_amps, step_times, step_durations,
            int(loc), target, p
        )
        # The compatibility wrapper returns the mean over decline steps. For
        # period-level aggregation we retain one identical per-maximum value.
        psi_vec = np.asarray([psi_mean_value], dtype=float)
        psi_values.extend([float(v) for v in psi_vec if np.isfinite(v)])
    if psi_values:
        result['psi_mean'] = float(np.mean(psi_values))
        result['psi_std'] = float(np.std(psi_values))
        result['psi_max'] = float(np.max(psi_values))

    return result

def extract_corrected_period_features(periods, period_meta, age, fs, height_cm, period_p=PERIOD_P_DEFAULT):
    """Return one row + one numeric vector for every normalized foot-to-foot pulse period.

    Beat-level fiducial features may be missing for an otherwise usable period;
    those values are represented as NaN and marked by beat_fiducials_valid=0.
    """
    rows = []

    for idx, (period, meta) in enumerate(zip(periods, period_meta), start=1):
        norm = normalize_pulse_period(period, n_points=1000)
        if norm is None:
            continue

        bfeat = extract_beat_features_single(period, fs, height_cm)
        beat_fiducials_valid = int(bfeat is not None)
        if bfeat is None:
            # The period-level F/Phi/Psi representation is independent of the
            # auxiliary beat-fiducial feature set. Preserve the period and make
            # unavailable beat features explicitly missing rather than silently
            # dropping the period.
            bfeat = {k: np.nan for k in BEAT_FEATURE_ORDER}

        fp = calculate_period_f_phi_psi(
            norm, period_p=period_p, min_extrema_prominence=PERIOD_EXTREMA_MIN_PROMINENCE
        )
        row = {
            'period_index': int(idx),
            'period_start_sample': int(meta['start_sample']),
            'period_end_sample': int(meta['end_sample']),
            'period_peak_sample': int(meta['peak_sample']),
            **{k: float(bfeat[k]) for k in BEAT_FEATURE_ORDER},
            **fp,
            'beat_fiducials_valid': int(beat_fiducials_valid),
            'pulse_duration_ms': float(meta['duration_ms']),
            'period_p': float(period_p),
            'period_formula_reference': PERIOD_FORMULA_REFERENCE,
            'period_psi_evaluation_rule': PERIOD_PSI_EVALUATION_RULE,
            'peak_inserted_v9': int(meta.get('peak_inserted_v9', 0)),
            'input_peak_count_v8': int(meta.get('input_peak_count_v8', 0)),
            'refined_peak_count_v9': int(meta.get('refined_peak_count_v9', 0)),
            'inserted_peak_count_v9': int(meta.get('inserted_peak_count_v9', 0)),
            'start_foot_method_v9': str(meta.get('start_foot_method_v9', '')),
            'end_foot_method_v9': str(meta.get('end_foot_method_v9', '')),
        }
        rows.append(row)

    if not rows:
        return [], np.empty(
            (0, len(PERIOD_BASE_FEATURE_ORDER)), dtype=np.float32
        )

    matrix = np.asarray(
        [[row[k] for k in PERIOD_BASE_FEATURE_ORDER] for row in rows],
        dtype=np.float32
    )
    return rows, matrix


def summarize_period_features(period_rows):
    """
    Segment-level statistics across pulse periods:
    mean, std, median, min, max, CV for numerical period features.
    """
    out = {
        'period_count_valid': int(len(period_rows)),
        'period_phi1_fraction': 0.0,
        'period_phi2_fraction': 0.0,
        'period_phi3_fraction': 0.0,
        'period_phi_unsupported_fraction': 0.0,
        'period_phi_valid_fraction': 0.0,
    }
    if not period_rows:
        return out

    cases = np.asarray([r['phi_case'] for r in period_rows], dtype=int)
    out['period_phi1_fraction'] = float(np.mean(cases == 1))
    out['period_phi2_fraction'] = float(np.mean(cases == 2))
    out['period_phi3_fraction'] = float(np.mean(cases == 3))
    out['period_phi_unsupported_fraction'] = float(np.mean(cases == 0))
    out['period_phi_valid_fraction'] = float(
        np.mean([r['phi_valid'] == 1 for r in period_rows])
    )

    numeric_names = [
        k for k in PERIOD_BASE_FEATURE_ORDER
        if k not in {'phi_case', 'phi_valid'}
    ]

    for name in numeric_names:
        vals = np.asarray(
            [r[name] for r in period_rows if np.isfinite(float(r[name]))],
            dtype=float
        )
        if not len(vals):
            stats = {
                'mean': np.nan, 'std': np.nan, 'median': np.nan,
                'min': np.nan, 'max': np.nan, 'cv': np.nan
            }
        else:
            mean = float(np.mean(vals))
            stats = {
                'mean': mean,
                'std': float(np.std(vals)),
                'median': float(np.median(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'cv': float(np.std(vals) / abs(mean))
                if abs(mean) > 1e-12 else np.nan
            }
        for stat_name, value in stats.items():
            out[f'period_{stat_name}__{name}'] = value

    return out


def extract_step_cycle_features(beats, age, pulse_rate, fs):
    """Run step-representation on every beat and aggregate cycle statistics."""
    p = calculate_age_coefficient(age)
    all_cycles = []
    step_peak_count = 0
    step_valley_count = 0
    all_step_durations = []

    for beat in beats:
        step_amps, step_times, step_durations = convert_to_step_representation(beat, pulse_rate, fs)
        peaks_step, valleys_step = detect_step_extrema(step_amps)
        cycles = process_cardiac_cycles_per_beat(step_amps, step_times, step_durations, peaks_step, valleys_step, p)
        all_cycles.extend(cycles)
        step_peak_count += len(peaks_step['locations'])
        step_valley_count += len(valleys_step['locations'])
        all_step_durations.extend(step_durations.tolist())

    # --- Function features ---
    func = {
        'step_F_mean': 0.0, 'step_F_std': 0.0, 'step_F_cv': 0.0,
        'step_Phi_mean': 0.0, 'step_Phi_std': 0.0, 'step_Phi_cv': 0.0,
        'step_Psi_mean': 0.0, 'step_Psi_std': 0.0, 'step_Psi_cv': 0.0,
        'step_F_Psi_ratio': 0.0, 'step_Phi_Psi_ratio': 0.0, 'step_F_Phi_ratio': 0.0,
        'step_Psi_age_adjusted': 0.0, 'step_vascular_compliance_index': 0.0,
        'step_function_balance': 0.0,
    }
    if all_cycles:
        F_vals = np.array([c['F_value'] for c in all_cycles])
        Phi_vals = np.array([c['Phi_value'] for c in all_cycles])
        Psi_vals = np.array([c['Psi_value'] for c in all_cycles])
        func['step_F_mean'] = float(np.mean(F_vals))
        func['step_F_std'] = safe_std(F_vals)
        func['step_F_cv'] = func['step_F_std'] / func['step_F_mean'] if func['step_F_mean'] != 0 else 0.0
        func['step_Phi_mean'] = float(np.mean(Phi_vals))
        func['step_Phi_std'] = safe_std(Phi_vals)
        func['step_Phi_cv'] = func['step_Phi_std'] / func['step_Phi_mean'] if func['step_Phi_mean'] != 0 else 0.0
        func['step_Psi_mean'] = float(np.mean(Psi_vals))
        func['step_Psi_std'] = safe_std(Psi_vals)
        func['step_Psi_cv'] = func['step_Psi_std'] / func['step_Psi_mean'] if func['step_Psi_mean'] != 0 else 0.0
        func['step_F_Psi_ratio'] = func['step_F_mean'] / func['step_Psi_mean'] if func['step_Psi_mean'] != 0 else 0.0
        func['step_Phi_Psi_ratio'] = func['step_Phi_mean'] / func['step_Psi_mean'] if func['step_Psi_mean'] != 0 else 0.0
        func['step_F_Phi_ratio'] = func['step_F_mean'] / func['step_Phi_mean'] if func['step_Phi_mean'] != 0 else 0.0
        func['step_Psi_age_adjusted'] = func['step_Psi_mean'] * p
        func['step_vascular_compliance_index'] = func['step_Psi_mean'] / p if p != 0 else 0.0
        func['step_function_balance'] = (func['step_F_mean'] + func['step_Phi_mean']) / func['step_Psi_mean'] if func['step_Psi_mean'] != 0 else 0.0

    # --- Variability features ---
    var_feat = {
        'step_F_short_term_var': 0.0, 'step_Phi_short_term_var': 0.0, 'step_Psi_short_term_var': 0.0,
        'step_F_total_var': 0.0, 'step_Phi_total_var': 0.0, 'step_Psi_total_var': 0.0,
        'step_function_stability': 0.0,
    }
    if len(all_cycles) >= 2:
        F_vals = np.array([c['F_value'] for c in all_cycles])
        Phi_vals = np.array([c['Phi_value'] for c in all_cycles])
        Psi_vals = np.array([c['Psi_value'] for c in all_cycles])
        F_diff = np.diff(F_vals)
        Phi_diff = np.diff(Phi_vals)
        Psi_diff = np.diff(Psi_vals)
        var_feat['step_F_short_term_var'] = safe_std(F_diff) / np.mean(F_vals) if np.mean(F_vals) != 0 else 0.0
        var_feat['step_Phi_short_term_var'] = safe_std(Phi_diff) / np.mean(Phi_vals) if np.mean(Phi_vals) != 0 else 0.0
        var_feat['step_Psi_short_term_var'] = safe_std(Psi_diff) / np.mean(Psi_vals) if np.mean(Psi_vals) != 0 else 0.0
        var_feat['step_F_total_var'] = safe_std(F_vals) / np.mean(F_vals) if np.mean(F_vals) != 0 else 0.0
        var_feat['step_Phi_total_var'] = safe_std(Phi_vals) / np.mean(Phi_vals) if np.mean(Phi_vals) != 0 else 0.0
        var_feat['step_Psi_total_var'] = safe_std(Psi_vals) / np.mean(Psi_vals) if np.mean(Psi_vals) != 0 else 0.0
        avg_cv = (var_feat['step_F_total_var'] + var_feat['step_Phi_total_var'] + var_feat['step_Psi_total_var']) / 3.0
        var_feat['step_function_stability'] = 1.0 - avg_cv

    # --- Beat shape consistency (step-based) ---
    beat_shape = {
        'step_shape_consistency': 0.0, 'step_amplitude_modulation': 0.0,
        'step_duration_variability': 0.0, 'step_F_consistency': 0.0,
        'step_Phi_consistency': 0.0, 'step_Psi_consistency': 0.0,
        'step_augmentation_index_proxy': 0.0,
    }
    if len(all_cycles) >= 3:
        F_vals = np.array([c['F_value'] for c in all_cycles])
        Phi_vals = np.array([c['Phi_value'] for c in all_cycles])
        Psi_vals = np.array([c['Psi_value'] for c in all_cycles])
        beat_shape['step_F_consistency'] = 1.0 - (safe_std(F_vals) / np.mean(F_vals)) if np.mean(F_vals) != 0 else 0.0
        beat_shape['step_Phi_consistency'] = 1.0 - (safe_std(Phi_vals) / np.mean(Phi_vals)) if np.mean(Phi_vals) != 0 else 0.0
        beat_shape['step_Psi_consistency'] = 1.0 - (safe_std(Psi_vals) / np.mean(Psi_vals)) if np.mean(Psi_vals) != 0 else 0.0
        beat_shape['step_shape_consistency'] = (beat_shape['step_F_consistency'] + beat_shape['step_Phi_consistency'] + beat_shape['step_Psi_consistency']) / 3.0

    # --- Step morphology ---
    step_morph = {
        'step_peak_count': step_peak_count,
        'step_valley_count': step_valley_count,
        'mean_step_duration_ms': float(np.mean(all_step_durations)) if all_step_durations else 0.0,
        'step_duration_std': safe_std(all_step_durations),
    }

    return {**func, **var_feat, **beat_shape, **step_morph}


def extract_signal_quality_metrics(ppg_signal, fs):
    qual = {'snr_db': 0.0, 'baseline_wander_std': 0.0, 'clipping_fraction': 0.0}
    try:
        freqs, psd = welch(ppg_signal, fs=fs, nperseg=min(256, len(ppg_signal)))
        signal_band = (freqs >= 0.5) & (freqs <= 4.0)
        noise_band = ~signal_band
        signal_power = np.sum(psd[signal_band])
        noise_power = np.sum(psd[noise_band])
        if noise_power > 0:
            qual['snr_db'] = float(10 * np.log10(signal_power / noise_power))
        b, a = butter(2, 0.5 / (fs / 2), btype='low')
        baseline = filtfilt(b, a, ppg_signal)
        qual['baseline_wander_std'] = float(np.std(baseline))
        amp_range = np.max(ppg_signal) - np.min(ppg_signal)
        if amp_range > 0:
            upper_limit = np.max(ppg_signal) - 0.01 * amp_range
            lower_limit = np.min(ppg_signal) + 0.01 * amp_range
            clipping = np.sum((ppg_signal >= upper_limit) | (ppg_signal <= lower_limit))
            qual['clipping_fraction'] = float(clipping / len(ppg_signal))
    except Exception:
        pass
    return qual


def extract_time_frequency_detail(ppg_signal, fs):
    tf_feat = {'inst_freq_mean': 0.0, 'inst_freq_std': 0.0, 'inst_freq_range': 0.0}
    try:
        analytic = hilbert(ppg_signal)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
        valid = inst_freq[(inst_freq > 0.5) & (inst_freq < 4.0)]
        if len(valid) > 10:
            tf_feat['inst_freq_mean'] = float(np.mean(valid))
            tf_feat['inst_freq_std'] = safe_std(valid)
            tf_feat['inst_freq_range'] = float(np.max(valid) - np.min(valid))
    except Exception:
        pass
    return tf_feat


def extract_beat_morphology_dynamics(ppg_signal, fs):
    dyn = {'beat_corr_mean': 0.0, 'beat_corr_std': 0.0, 'beat_l2_distance_mean': 0.0}
    try:
        ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=fs)
        peaks = extract_peak_indices(nk.ppg_peaks(ppg_clean, sampling_rate=fs)[0])
        if len(peaks) < 4:
            return dyn
        pulse_len = int(0.7 * fs)
        pulses = []
        for p in peaks:
            start = max(0, p - int(0.2 * fs))
            end = min(len(ppg_clean), p + int(0.5 * fs))
            seg = ppg_clean[start:end]
            if len(seg) >= pulse_len:
                pulses.append(seg[:pulse_len])
        if len(pulses) < 3:
            return dyn
        corrs = []
        for i in range(len(pulses) - 1):
            if len(pulses[i]) == len(pulses[i + 1]):
                c = np.corrcoef(pulses[i], pulses[i + 1])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        if corrs:
            dyn['beat_corr_mean'] = float(np.mean(corrs))
            dyn['beat_corr_std'] = safe_std(corrs)
        dists = []
        for i in range(len(pulses) - 1):
            min_len = min(len(pulses[i]), len(pulses[i+1]))
            if min_len > 0:
                d = np.linalg.norm(pulses[i][:min_len] - pulses[i+1][:min_len])
                dists.append(d)
        if dists:
            dyn['beat_l2_distance_mean'] = float(np.mean(dists))
    except Exception:
        pass
    return dyn


def extract_pulse_decomposition(ppg_signal, fs):
    pda = {
        'forward_wave_amplitude': 0.0, 'forward_wave_time': 0.0,
        'reflected_wave_amplitude': 0.0, 'reflected_wave_time': 0.0,
        'dicrotic_notch_amplitude_pda': 0.0, 'dicrotic_notch_time_pda': 0.0,
        'reflection_index_pda': 0.0
    }
    try:
        ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=fs)
        peaks = extract_peak_indices(nk.ppg_peaks(ppg_clean, sampling_rate=fs)[0])
        if len(peaks) < 3:
            return pda
        pulse_len = int(0.8 * fs / (estimate_pulse_rate(ppg_signal, fs) / 60))
        if pulse_len < 20:
            pulse_len = int(0.6 * fs)
        pulses = []
        for p in peaks:
            start = p - int(0.15 * fs)
            end = p + pulse_len
            if start >= 0 and end < len(ppg_clean):
                pulses.append(ppg_clean[start:end])
        if len(pulses) < 2:
            return pda
        min_len = min([len(p) for p in pulses])
        avg_pulse = np.mean([p[:min_len] for p in pulses], axis=0)
        peaks_avg, _ = find_peaks(avg_pulse, distance=fs // 10)
        valleys_avg, _ = find_peaks(-avg_pulse, distance=fs // 10)
        if len(peaks_avg) == 0 or len(valleys_avg) == 0:
            return pda
        sys_peak = peaks_avg[0]
        valleys_after = valleys_avg[valleys_avg > sys_peak]
        if len(valleys_after) == 0:
            return pda
        notch = valleys_after[0]
        peaks_after = peaks_avg[peaks_avg > notch]
        dia_peak = peaks_after[0] if len(peaks_after) > 0 else None
        pda['forward_wave_amplitude'] = float(avg_pulse[sys_peak])
        pda['forward_wave_time'] = float(sys_peak / fs)
        pda['dicrotic_notch_amplitude_pda'] = float(avg_pulse[notch])
        pda['dicrotic_notch_time_pda'] = float(notch / fs)
        if dia_peak is not None:
            pda['reflected_wave_amplitude'] = float(avg_pulse[dia_peak])
            pda['reflected_wave_time'] = float(dia_peak / fs)
            if pda['forward_wave_amplitude'] != 0:
                pda['reflection_index_pda'] = pda['reflected_wave_amplitude'] / pda['forward_wave_amplitude']
    except Exception:
        pass
    return pda


def extract_htn_specific_features(ppg_signal, fs, beats, timing, aix, stiff, notch_deep, pulse_rate):
    htn = {
        'htn_stiffness_index_proxy': 0.0, 'htn_perfusion_index': 0.0,
        'htn_diastolic_decay_tau': 0.0, 'htn_normalized_pulse_volume': 0.0,
        'htn_notch_to_peak_ratio': 0.0, 'htn_systolic_rise_time_pct': 0.0,
        'htn_pulse_pressure_variability': 0.0, 'htn_vascular_compliance_score': 0.0,
        'htn_augmentation_index_validated': 0.0, 'htn_reflection_index_validated': 0.0,
    }
    try:
        ac = ppg_signal - np.mean(ppg_signal)
        htn['htn_perfusion_index'] = float(np.std(ac) / (np.mean(ppg_signal) + 1e-10))

        ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=fs)
        peaks = extract_peak_indices(nk.ppg_peaks(ppg_clean, sampling_rate=fs)[0])
        if len(peaks) >= 3:
            beat_amps = []
            for p in peaks:
                search = max(0, p - int(0.1 * fs))
                foot = search + np.argmin(ppg_clean[search:p])
                beat_amps.append(ppg_clean[p] - ppg_clean[foot])
            htn['htn_pulse_pressure_variability'] = float(safe_std(beat_amps) / (np.mean(beat_amps) + 1e-10))

        mean_rr = 60.0 / pulse_rate if pulse_rate > 0 else 1.0
        if timing['systolic_time_ms'] > 0 and mean_rr > 0:
            htn['htn_systolic_rise_time_pct'] = float(timing['systolic_time_ms'] / 1000.0 / mean_rr)

        htn['htn_augmentation_index_validated'] = float(aix.get('aix_percent', 0.0))
        htn['htn_reflection_index_validated'] = float(stiff.get('reflection_index', 0.0))

        if timing.get('delta_t_ms', 0) > 0:
            htn['htn_stiffness_index_proxy'] = 1.0 / (timing['delta_t_ms'] / 1000.0)

        if beats:
            first_beat = beats[0]
            p = np.argmax(first_beat)
            search = max(0, p - int(0.2 * fs))
            foot_idx = search + np.argmin(first_beat[search:p])
            foot_amp = first_beat[foot_idx]
            peak_amp = first_beat[p]
            amp_range = np.max(ppg_signal) - np.min(ppg_signal)
            htn['htn_normalized_pulse_volume'] = float((peak_amp - foot_amp) / (amp_range + 1e-10))
            end = min(len(first_beat), p + int(0.6 * fs))
            diast_seg = first_beat[p:end]
            if len(diast_seg) > 5:
                t = np.arange(len(diast_seg)) / fs
                y = diast_seg - np.min(diast_seg) + 1e-10
                log_y = np.log(y)
                A = np.vstack([t, np.ones_like(t)]).T
                m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]
                if m < 0:
                    htn['htn_diastolic_decay_tau'] = float(-1.0 / m)

        sys_areas, dia_areas = [], []
        for beat in beats:
            if len(beat) < 10:
                continue
            fid = detect_ppg_fiducials(beat, fs)
            if not is_valid_fiducials(fid):
                continue
            foot_idx = fid['foot_idx']
            sys_idx = fid['sys_peak_idx']
            notch_idx = fid['notch_idx']
            if notch_idx is None:
                notch_idx = int(0.65 * len(beat))
            notch_idx = max(sys_idx + 1, min(notch_idx, len(beat) - 2))
            beat_rel = beat - fid['foot_amp']
            sys_areas.append(trapezoid_compat(beat_rel[foot_idx:notch_idx + 1]))
            dia_areas.append(trapezoid_compat(beat_rel[notch_idx:]))
        if sys_areas and dia_areas and np.mean(sys_areas) > 0:
            htn['htn_vascular_compliance_score'] = float(np.mean(dia_areas) / np.mean(sys_areas))

        dnd = notch_deep.get('notch_depth_ratio', 0.0)
        if 0 <= dnd <= 1:
            htn['htn_notch_to_peak_ratio'] = float(1.0 - dnd)
    except Exception:
        pass
    return htn


# --- Nonlinear helpers ---
def compute_robust_approx_entropy(signal):
    N = len(signal)
    if N < 50:
        return 0.0
    m = 2
    r = 0.2 * np.std(signal)
    max_len = min(200, N)
    sig = signal[:max_len]
    N = len(sig)
    def phi(m_val):
        patterns = np.array([sig[i:i + m_val] for i in range(N - m_val + 1)])
        C = np.zeros(len(patterns))
        for i, pat in enumerate(patterns):
            dist = np.max(np.abs(patterns - pat), axis=1)
            C[i] = np.sum(dist <= r) / (N - m_val + 1)
        valid = C[C > 0]
        if len(valid) == 0:
            return 0.0
        return np.mean(np.log(valid))
    return max(0.0, phi(m) - phi(m + 1))


def compute_robust_recurrence_rate(signal):
    N = len(signal)
    if N < 20:
        return 0.0
    sig = signal[:min(N, 50)]
    sig_norm = (sig - np.mean(sig)) / np.std(sig) if np.std(sig) > 0 else sig
    threshold = 0.2
    M = len(sig_norm)
    rec = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if abs(sig_norm[i] - sig_norm[j]) < threshold:
                rec[i, j] = 1
    rate = (np.sum(rec) - M) / (M**2 - M) if M > 1 else 0.0
    return max(0.0, rate)


def compute_robust_multiscale_entropy(signal):
    N = len(signal)
    if N < 100:
        return 0.0
    entropies = []
    for scale in range(1, 4):
        coarse_len = N // scale
        if coarse_len < 20:
            continue
        coarse = np.array([np.mean(signal[i * scale:(i + 1) * scale]) for i in range(coarse_len)])
        if len(coarse) >= 50:
            ae = compute_robust_approx_entropy(coarse)
            if ae > 0:
                entropies.append(ae)
    if entropies:
        return float(np.mean(entropies))
    return 0.0


def extract_nonlinear_unique(ppg_signal):
    nln = {'approx_entropy': 0.0, 'recurrence_rate': 0.0, 'multiscale_entropy': 0.0}
    if len(ppg_signal) < 50:
        return nln
    try:
        ppg = ppg_signal.flatten()
        if len(ppg) >= 50:
            nln['approx_entropy'] = float(compute_robust_approx_entropy(ppg))
        if len(ppg) >= 30:
            nln['recurrence_rate'] = float(compute_robust_recurrence_rate(ppg))
        if len(ppg) >= 100:
            nln['multiscale_entropy'] = float(compute_robust_multiscale_entropy(ppg))
    except Exception:
        pass
    return nln


def extract_wavelet_energy(ppg_signal, wavelet='db4', level=4):
    features = {}
    try:
        coeffs = pywt.wavedec(ppg_signal, wavelet, level=level)
        total_energy = np.sum([np.sum(c**2) for c in coeffs]) + 1e-10
        for i, c in enumerate(coeffs):
            features[f'wavelet_energy_{i}'] = float(np.sum(c**2) / total_energy)
    except Exception:
        for i in range(level + 1):
            features[f'wavelet_energy_{i}'] = 0.0
    return features


def extract_hilbert_envelope_features(ppg_signal):
    try:
        analytic = hilbert(ppg_signal - np.mean(ppg_signal))
        envelope = np.abs(analytic)
        return {
            'hilbert_mean': float(np.mean(envelope)),
            'hilbert_std': safe_std(envelope),
            'hilbert_range': float(np.max(envelope) - np.min(envelope))
        }
    except Exception:
        return {'hilbert_mean': 0.0, 'hilbert_std': 0.0, 'hilbert_range': 0.0}


def extract_time_domain_extras(ppg_signal):
    try:
        ac = ppg_signal - np.mean(ppg_signal)
        rms = np.sqrt(np.mean(ac**2))
        crest = np.max(np.abs(ac)) / (rms + 1e-10)
        zcr = np.sum(np.diff(np.sign(ac)) != 0) / len(ac)
        return {'rms': float(rms), 'crest_factor': float(crest), 'zero_crossing_rate': float(zcr)}
    except Exception:
        return {'rms': 0.0, 'crest_factor': 0.0, 'zero_crossing_rate': 0.0}

# ============================================================
# 7. UNIFIED SEGMENT PROCESSOR
# ============================================================
def compute_signal_quality_score(ppg_clean, peaks_arr, fs):
    try:
        if len(peaks_arr) < 3:
            return 0.0
        unique_ratio = len(np.unique(ppg_clean)) / len(ppg_clean)
        if unique_ratio < 0.01:
            return 0.0
        beat_lengths = np.diff(peaks_arr) / fs
        if len(beat_lengths) < 2:
            return 0.3
        cv_length = np.std(beat_lengths) / (np.mean(beat_lengths) + 1e-10)
        score = max(0.0, 1.0 - cv_length)
        if unique_ratio < 0.05:
            score *= 0.5
        return float(score)
    except Exception:
        return 0.0


def process_single_segment_unified(seg_dict, mat_filepath, fs=125, period_p=PERIOD_P_DEFAULT):
    # ---------- Load & validate ----------
    subject_id = to_scalar(safe_get_field(seg_dict, 'SubjectID', 'Unknown'))
    age = to_scalar(safe_get_field(seg_dict, 'Age', 30))
    sbp = to_scalar(safe_get_field(seg_dict, 'SegSBP', 120))
    dbp = to_scalar(safe_get_field(seg_dict, 'SegDBP', 80))

    if not (40 <= sbp <= 250 and 30 <= dbp <= 150):
        return None
    map_val = dbp + (sbp - dbp) / 3
    if map_val < 50:
        return None

    if isinstance(age, str):
        try:
            age = float(age)
        except Exception:
            age = 30

    ppg_raw = safe_get_field(seg_dict, 'PPG_F', None)
    if ppg_raw is None:
        return None
    if not isinstance(ppg_raw, np.ndarray):
        ppg_raw = np.array(ppg_raw)
    if ppg_raw.ndim > 1:
        ppg_raw = ppg_raw.flatten()
    if len(ppg_raw) < 100:
        return None

    ecg_raw = safe_get_field(seg_dict, 'ECG_F', None)
    if ecg_raw is not None:
        if not isinstance(ecg_raw, np.ndarray):
            ecg_raw = np.array(ecg_raw)
        if ecg_raw.ndim > 1:
            ecg_raw = ecg_raw.flatten()
        if len(ecg_raw) != len(ppg_raw):
            ecg_raw = None

    # ---------- Canonical sampling rate (paper §3.3, design choice 8) ----------
    # All inputs are resampled to the canonical 125 Hz entry point so that
    # features are identical regardless of the caller-supplied fs. This is a
    # no-op for the production pipeline (all adapters pass fs=125).
    CANONICAL_FS = 125  # [canonical_v1]
    if fs != CANONICAL_FS:
        from math import gcd as _gcd
        from scipy.signal import resample_poly as _resample_poly
        _g = _gcd(int(fs), int(CANONICAL_FS))
        _up = CANONICAL_FS // _g
        _down = int(fs) // _g
        ppg_raw = _resample_poly(ppg_raw, _up, _down)
        if ecg_raw is not None:
            ecg_raw = _resample_poly(ecg_raw, _up, _down)
        fs = CANONICAL_FS

    # ---------- Pre-compute shared objects ONCE ----------
    ppg_clean = None
    peaks_df = None
    peaks_arr = np.array([])
    beats = None
    foot_indices = None
    ecg_clean = None
    r_peaks_arr = None

    try:
        ppg_clean = nk.ppg_clean(ppg_raw, sampling_rate=fs)
        peaks_df, info = nk.ppg_peaks(ppg_clean, sampling_rate=fs)
        peaks_arr = extract_peak_indices(peaks_df)
        # Extract join boundary mask from segment dict if present
        reject_mask = seg_dict.get("join_boundary_mask", None)
        if reject_mask is not None and len(reject_mask) != len(ppg_clean):
            reject_mask = None   # size mismatch — ignore
        beats, foot_indices, _ = segment_beats(ppg_clean, peaks_arr, fs,
                                                reject_mask=reject_mask)
        # beats, foot_indices, _ = segment_beats(ppg_clean, peaks_arr, fs)
    except Exception:
        pass

    if len(peaks_arr) < 3 or beats is None or len(beats) < 2:
        return None

    pulse_rate = estimate_pulse_rate(ppg_raw, fs)

    if ecg_raw is not None:
        try:
            ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=fs, method='neurokit')
            r_peaks_arr = np.array(
                nk.ecg_peaks(ecg_clean, sampling_rate=fs, method='neurokit')[1]['ECG_R_Peaks'],
                dtype=int)
        except Exception:
            ecg_clean = None
            r_peaks_arr = None

    height_cm = to_scalar(safe_get_field(seg_dict, 'Height', 170))
    if height_cm <= 50 or height_cm > 250:
        height_cm = 170

    quality_score = compute_signal_quality_score(ppg_clean, peaks_arr, fs)

    # ---------- Corrected pulse-period F/Phi/Psi analysis ----------
    # [join_mask_v2] forward join-boundary mask to period extraction
    _rmask_for_periods = None
    try:
        _rmask_for_periods = seg_dict.get("join_boundary_mask", None)
        if (_rmask_for_periods is not None
                and len(_rmask_for_periods) != len(ppg_clean)):
            _rmask_for_periods = None
    except Exception:
        _rmask_for_periods = None
    period_pulses, period_meta = segment_pulse_periods_from_peaks(
        ppg_clean, peaks_arr, fs, reject_mask=_rmask_for_periods
    )
    period_rows, period_matrix = extract_corrected_period_features(
        period_pulses, period_meta, age, fs, height_cm, period_p=period_p
    )
    period_summary = summarize_period_features(period_rows)

    step_features = {
        k: 0.0 for k in [
            'step_F_mean', 'step_F_std', 'step_F_cv',
            'step_Phi_mean', 'step_Phi_std', 'step_Phi_cv',
            'step_Psi_mean', 'step_Psi_std', 'step_Psi_cv',
            'step_F_Psi_ratio', 'step_Phi_Psi_ratio', 'step_F_Phi_ratio',
            'step_Psi_age_adjusted', 'step_vascular_compliance_index',
            'step_function_balance',
            'step_F_short_term_var', 'step_Phi_short_term_var',
            'step_Psi_short_term_var', 'step_F_total_var',
            'step_Phi_total_var', 'step_Psi_total_var',
            'step_function_stability',
            'step_peak_count', 'step_valley_count',
            'mean_pulse_duration_ms', 'step_duration_std',
            'step_shape_consistency', 'step_F_consistency',
            'step_Phi_consistency', 'step_Psi_consistency'
        ]
    }

    if period_rows:
        Fp = np.asarray([r['f_mean'] for r in period_rows], dtype=float)
        Phip = np.asarray([r['phi_value'] for r in period_rows], dtype=float)
        Psip = np.asarray([r['psi_mean'] for r in period_rows], dtype=float)

        def valid(x):
            return x[np.isfinite(x)]

        Fv, Phiv, Psiv = valid(Fp), valid(Phip), valid(Psip)

        if len(Fv):
            step_features['step_F_mean'] = float(np.mean(Fv))
            step_features['step_F_std'] = float(np.std(Fv))
            step_features['step_F_cv'] = (
                step_features['step_F_std'] / abs(step_features['step_F_mean'])
                if abs(step_features['step_F_mean']) > 1e-12 else 0.0
            )
        if len(Phiv):
            step_features['step_Phi_mean'] = float(np.mean(Phiv))
            step_features['step_Phi_std'] = float(np.std(Phiv))
            step_features['step_Phi_cv'] = (
                step_features['step_Phi_std'] / abs(step_features['step_Phi_mean'])
                if abs(step_features['step_Phi_mean']) > 1e-12 else 0.0
            )
        if len(Psiv):
            step_features['step_Psi_mean'] = float(np.mean(Psiv))
            step_features['step_Psi_std'] = float(np.std(Psiv))
            step_features['step_Psi_cv'] = (
                step_features['step_Psi_std'] / abs(step_features['step_Psi_mean'])
                if abs(step_features['step_Psi_mean']) > 1e-12 else 0.0
            )

        age_p = calculate_age_coefficient(age)
        if age_p != 0:
            step_features['step_Psi_age_adjusted'] = (
                step_features['step_Psi_mean'] * age_p
            )
            step_features['step_vascular_compliance_index'] = (
                step_features['step_Psi_mean'] / age_p
            )

        if abs(step_features['step_Psi_mean']) > 1e-12:
            step_features['step_F_Psi_ratio'] = (
                step_features['step_F_mean'] / step_features['step_Psi_mean']
            )
            step_features['step_Phi_Psi_ratio'] = (
                step_features['step_Phi_mean'] / step_features['step_Psi_mean']
            )
            step_features['step_function_balance'] = (
                step_features['step_F_mean'] + step_features['step_Phi_mean']
            ) / step_features['step_Psi_mean']
        if abs(step_features['step_Phi_mean']) > 1e-12:
            step_features['step_F_Phi_ratio'] = (
                step_features['step_F_mean'] / step_features['step_Phi_mean']
            )

        def cv(x):
            x = x[np.isfinite(x)]
            if not len(x):
                return 0.0
            m = float(np.mean(x))
            return float(np.std(x) / abs(m)) if abs(m) > 1e-12 else 0.0

        step_features['step_F_total_var'] = cv(Fp)
        step_features['step_Phi_total_var'] = cv(Phip)
        step_features['step_Psi_total_var'] = cv(Psip)

        step_features['step_F_short_term_var'] = cv(np.diff(Fv)) if len(Fv) >= 2 else 0.0
        step_features['step_Phi_short_term_var'] = cv(np.diff(Phiv)) if len(Phiv) >= 2 else 0.0
        step_features['step_Psi_short_term_var'] = cv(np.diff(Psiv)) if len(Psiv) >= 2 else 0.0

        step_features['step_function_stability'] = float(
            1.0 - np.mean([
                step_features['step_F_total_var'],
                step_features['step_Phi_total_var'],
                step_features['step_Psi_total_var']
            ])
        )

        step_features['step_peak_count'] = int(
            np.sum([r['n_local_maxima'] for r in period_rows])
        )
        step_features['step_valley_count'] = int(
            np.sum([r['n_local_minima'] for r in period_rows])
        )

        durations_sec = np.asarray(
            [r['pulse_duration_ms'] for r in period_rows], dtype=float
        ) / 1000.0
        step_features['mean_step_duration_ms'] = float(np.mean(durations_sec))
        step_features['step_duration_std'] = float(np.std(durations_sec))

        step_features['step_F_consistency'] = (
            1.0 - step_features['step_F_cv']
        )
        step_features['step_Phi_consistency'] = (
            1.0 - step_features['step_Phi_cv']
        )
        step_features['step_Psi_consistency'] = (
            1.0 - step_features['step_Psi_cv']
        )
        step_features['step_shape_consistency'] = float(
            np.mean([
                step_features['step_F_consistency'],
                step_features['step_Phi_consistency'],
                step_features['step_Psi_consistency']
            ])
        )

    # ---------- Per-beat sequence matrix (all_beats base) ----------
    beat_features_list = []
    core_beats = get_core_beats(beats)
    for beat in core_beats:
        if len(beat) < 20:
            continue
        bfeat = extract_beat_features_single(beat, fs, height_cm)
        if bfeat is not None:
            beat_features_list.append(bfeat)

    # Segment validity for the corrected period representation is based on
    # complete valid pulse periods, not on the legacy core-beat count.
    if len(period_rows) < 3:
        return None

    beat_matrix = np.asarray(
        [[b[f] for f in BEAT_FEATURE_ORDER] for b in beat_features_list],
        dtype=np.float32
    ) if beat_features_list else np.empty((0, len(BEAT_FEATURE_ORDER)), dtype=np.float32)

    # ---------- Segment-level base extractors (all_beats) ----------
    morphology = extract_enhanced_morphology(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    timing = extract_enhanced_timing_features(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    sdppg = extract_sdppg_features(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    spectral = extract_enhanced_spectral_features(ppg_raw, fs, pulse_rate, ppg_clean=ppg_clean)
    derivative = extract_enhanced_derivative_features(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr)
    hrv = extract_simplified_hrv(ppg_raw, fs, ppg_clean=ppg_clean, peaks_df=peaks_df)
    ptt = extract_ptt_features(ppg_raw, ecg_raw, fs, ppg_clean=ppg_clean, ecg_clean=ecg_clean,
                               ppg_peaks=peaks_arr, r_peaks=r_peaks_arr, foot_indices=foot_indices)
    aix = extract_augmentation_index(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    stiff = extract_stiffness_index(ppg_raw, fs, height_cm=height_cm, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    pwv = extract_pwv_surrogate(ptt['ptt_r_to_ppg_foot_ms'], height_cm=height_cm)
    compliance = extract_vascular_compliance(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    tdppg = extract_tdppg_features(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    shoulder = extract_systolic_shoulder_features(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    notch_deep = extract_notch_characterization(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    spec_stiff = extract_spectral_stiffness_features(ppg_raw, fs, pulse_rate, ppg_clean=ppg_clean)
    complexity = extract_ppg_complexity(ppg_raw, fs)
    ppg_var = extract_ppg_variability(ppg_raw, fs, ppg_clean=ppg_clean, peaks=peaks_arr, beats=beats)
    cardiac_int = extract_cardiac_intervals(ppg_raw, ecg_raw, fs, ppg_clean=ppg_clean, ecg_clean=ecg_clean,
                                             ppg_peaks=peaks_arr, beats=beats, foot_indices=foot_indices)

    # ---------- Unique pipeline-v5 extractors ----------
    signal_qual = extract_signal_quality_metrics(ppg_raw, fs)
    tf_detail = extract_time_frequency_detail(ppg_raw, fs)
    beat_dyn = extract_beat_morphology_dynamics(ppg_raw, fs)
    pulse_decomp = extract_pulse_decomposition(ppg_raw, fs)
    htn = extract_htn_specific_features(ppg_raw, fs, beats, timing, aix, stiff, notch_deep, pulse_rate)
    nln_unique = extract_nonlinear_unique(ppg_raw)
    wavelet = extract_wavelet_energy(ppg_raw)
    hilbert = extract_hilbert_envelope_features(ppg_raw)
    time_extras = extract_time_domain_extras(ppg_raw)

    # ---------- Assemble segment context vector ----------
    bmi = to_scalar(safe_get_field(seg_dict, 'BMI', 25))
    height = to_scalar(safe_get_field(seg_dict, 'Height', 170))
    weight = to_scalar(safe_get_field(seg_dict, 'Weight', 70))
    gender_raw = seg_dict.get('Gender', 'M') if isinstance(seg_dict, dict) else 'M'
    if isinstance(gender_raw, np.ndarray):
        gender_raw = gender_raw.item() if gender_raw.size == 1 else str(gender_raw.flatten()[0])
    gender_binary = decode_gender(gender_raw)
    patient_id = safe_extract_patient_id(mat_filepath, subject_id)
    fname = os.path.basename(mat_filepath)

    seg_ctx = {
        'n_beats': float(len(beat_features_list)),
        'demo_age': float(age), 'demo_bmi': float(bmi), 'demo_height': float(height),
        'demo_weight': float(weight), 'demo_gender': int(gender_binary),
        'pulse_rate': float(pulse_rate), 'signal_quality': float(quality_score),

        'morph_skewness': morphology['skewness_ppg'],
        'morph_kurtosis': morphology['kurtosis_ppg'],
        'morph_perfusion_index': morphology['perfusion_index'],

        'timing_systolic_ms': timing['systolic_time_ms'],
        'timing_diastolic_ms': timing['diastolic_time_ms'],
        'timing_ts_td_ratio': timing['ts_td_ratio'],
        'timing_delta_t_ms': timing['delta_t_ms'],
        'timing_deceleration_ms': timing['deceleration_time_ms'],
        'timing_notch_depth_ratio': timing['notch_depth_ratio'],
        'timing_notch_position_pct': timing['notch_position_pct'],
        'timing_reflection_index': timing['reflection_index_p2_p1'],

        'sdppg_a': sdppg['sdppg_a'], 'sdppg_b': sdppg['sdppg_b'],
        'sdppg_c': sdppg['sdppg_c'], 'sdppg_d': sdppg['sdppg_d'],
        'sdppg_e': sdppg['sdppg_e'], 'sdppg_aging_index': sdppg['sdppg_aging_index'],

        'spec_fundamental_freq': spectral['spectral_fundamental_freq'],
        'spec_centroid': spectral['spectral_centroid'],
        'spec_bandwidth': spectral['spectral_bandwidth'],
        'spec_entropy': spectral['spectral_entropy'],
        'spec_harmonic_h2_h1': spectral['harmonic_ratio_h2_h1'],
        'spec_harmonic_h3_h1': spectral['harmonic_ratio_h3_h1'],
        'spec_harmonic_h4_h1': spectral['harmonic_ratio_h4_h1'],

        'deriv_max_velocity': derivative['max_velocity'],
        'deriv_min_velocity': derivative['min_velocity'],
        'deriv_velocity_skewness': derivative['velocity_skewness'],
        'deriv_max_acceleration': derivative['max_acceleration'],
        'deriv_min_acceleration': derivative['min_acceleration'],
        'deriv_upslope_mean': derivative['upslope_mean'],
        'deriv_downslope_mean': derivative['downslope_mean'],
        'deriv_upslope_downslope_ratio': derivative['upslope_downslope_ratio'],
        'deriv_norm_max_rising_slope': derivative['normalized_max_rising_slope'],

        'hrv_mean_nn': hrv['hrv_mean_nn_ms'], 'hrv_sdnn': hrv['hrv_sdnn_ms'],
        'hrv_rmssd': hrv['hrv_rmssd_ms'], 'hrv_cvnn': hrv['hrv_cvnn'],
        'hrv_pnn50': hrv['hrv_pnn50'], 'hrv_lf_power': hrv['hrv_lf_power'],
        'hrv_hf_power': hrv['hrv_hf_power'], 'hrv_lf_hf_ratio': hrv['hrv_lf_hf_ratio'],

        'ptt_r_to_foot_ms': ptt['ptt_r_to_ppg_foot_ms'],
        'ptt_r_to_peak_ms': ptt['ptt_r_to_ppg_peak_ms'],
        'ptt_btb_std_ms': ptt['ptt_beat_to_beat_std_ms'],
        'ptt_btb_cv': ptt['ptt_beat_to_beat_cv'],

        'stiffness_index': stiff['stiffness_index'], 'lasi_ms': stiff['lasi'],
        'reflection_index_v2': stiff['reflection_index'],
        'crest_time_ratio': stiff['crest_time_ratio'],
        'diastolic_decay_time_ms': stiff['diastolic_decay_time_ms'],
        'diastolic_decay_constant': stiff['diastolic_decay_constant'],

        'aix_percent': aix['aix_percent'], 'aix_amplitude_based': aix['aix_amplitude_based'],
        'augmentation_pressure': aix['augmentation_pressure'],
        'pulse_pressure_ppg': aix['pulse_pressure_ppg'],
        'inflection_point_ratio': aix['inflection_point_ratio'],

        'pwv_surrogate': pwv['pwv_surrogate'],

        'compliance_index': compliance['compliance_index'],
        'elasticity_index': compliance['elasticity_index'],
        'arterial_stiffness_proxy': compliance['arterial_stiffness_proxy'],

        'tdppg_a': tdppg['tdppg_a'], 'tdppg_b': tdppg['tdppg_b'],
        'tdppg_c': tdppg['tdppg_c'], 'tdppg_d': tdppg['tdppg_d'],
        'tdppg_e': tdppg['tdppg_e'], 'tdppg_f': tdppg['tdppg_f'],
        'tdppg_bd_index': tdppg['tdppg_bd_index'], 'tdppg_cf_index': tdppg['tdppg_cf_index'],

        'shoulder_present_ratio': shoulder['shoulder_present_ratio'],
        'shoulder_height_ratio': shoulder['shoulder_height_ratio'],
        'shoulder_time_ratio': shoulder['shoulder_time_ratio'],
        'peak_bifurcation_index': shoulder['peak_bifurcation_index'],
        'systolic_roundness': shoulder['systolic_roundness'],

        'notch_width_ms': notch_deep['notch_width_ms'],
        'notch_sharpness': notch_deep['notch_sharpness'],
        'notch_foot_ratio': notch_deep['notch_foot_ratio'],
        'post_notch_recovery_slope': notch_deep['post_notch_recovery_slope'],
        'notch_to_end_ratio': notch_deep['notch_to_end_ratio'],
        'notch_symmetry_index': notch_deep['notch_symmetry_index'],

        'spec_hf_power_ratio': spec_stiff['spec_hf_power_ratio'],
        'spec_very_hf_ratio': spec_stiff['spec_very_hf_ratio'],
        'spec_slope_1_4_hz': spec_stiff['spec_slope_1_4_hz'],
        'spec_entropy_highband': spec_stiff['spec_entropy_highband'],
        'spec_centroid_above_fund': spec_stiff['spec_centroid_above_fund'],
        'spec_peak_skewness': spec_stiff['spec_peak_skewness'],

        'ppg_sample_entropy': complexity['ppg_sample_entropy'],
        'ppg_lle': complexity['ppg_lle'],

        'ppg_amp_sdnn': ppg_var['ppg_amp_sdnn'], 'ppg_amp_cv': ppg_var['ppg_amp_cv'],
        'ppg_amp_rmssd': ppg_var['ppg_amp_rmssd'], 'ppg_area_cv': ppg_var['ppg_area_cv'],
        'ppg_shape_consistency': ppg_var['ppg_shape_consistency'],
        'ppg_width_cv': ppg_var['ppg_width_cv'],

        'pep_ms': cardiac_int['pep_ms'], 'lvet_ms': cardiac_int['lvet_ms'],
        'pep_lvet_ratio': cardiac_int['pep_lvet_ratio'],
        'pep_variability': cardiac_int['pep_variability'],
        'lvet_variability': cardiac_int['lvet_variability'],
    }

    # Merge unique pipeline-v5 features
    seg_ctx.update(step_features)
    seg_ctx.update(signal_qual)
    seg_ctx.update(tf_detail)
    seg_ctx.update(beat_dyn)
    seg_ctx.update(pulse_decomp)
    seg_ctx.update(htn)
    seg_ctx.update(nln_unique)
    seg_ctx.update(wavelet)
    seg_ctx.update(hilbert)
    seg_ctx.update(time_extras)

    seg_vector = np.array([seg_ctx[k] for k in SEGMENT_FEATURE_ORDER], dtype=np.float32)

    # ---------- Flattened row for CSV ----------
    flat_row = {
        'subject_id': str(patient_id),
        'age': float(age),
        'sbp': float(sbp),
        'dbp': float(dbp),
        'pulse_rate': float(pulse_rate),
        'signal_quality': float(quality_score),
        'n_beats': int(len(beat_features_list)),
        'n_periods': int(len(period_rows)),
        'source_file': str(fname),
    }
    flat_row.update(seg_ctx)
    flat_row.update(period_summary)

    # ---------- Sequence structure for HDF5 ----------
    seq_struct = {
        'beat_features': beat_matrix,
        'period_features': period_matrix,
        'period_feature_names': PERIOD_BASE_FEATURE_ORDER,
        'period_summary': period_summary,
        'period_rows': period_rows,
        'segment_features': seg_vector,
        'labels': {'sbp': float(sbp), 'dbp': float(dbp)},
        'metadata': {
            'subject_id': str(patient_id),
            'age': float(age),
            'dataset_source_file': str(fname),
            'n_beats': int(len(beat_features_list)),
            'n_periods': int(len(period_rows)),
            'seq_length': int(len(period_rows)),
        }
    }

    return {'flat': flat_row, 'seq': seq_struct}

# ============================================================
# 8. PARALLEL I/O
# ============================================================
def flatten_features(feat_list):
    return [f['flat'] for f in feat_list if f is not None]


def load_processed_files(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return set(os.path.basename(line.strip()) for line in f if line.strip())
    return set()
def save_batch_csv(batch, output_csv, log_file, fnames_batch, global_seg_idx_start=0):
    if not batch:
        return
    df = pd.DataFrame(batch)
    # --- ADD THIS LINE ---
    df['segment_id'] = [f"segment_{global_seg_idx_start + i:08d}" for i in range(len(batch))]
    # ---------------------
    header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    df.to_csv(output_csv, mode='a', header=header, index=False)
    with open(log_file, 'a') as f:
        for fname in fnames_batch:
            f.write(fname + '\n')
    logging.info(f"Saved {len(batch)} CSV rows from {len(fnames_batch)} files")


def save_batch_period_csv(batch, period_output_csv, global_seg_idx_start):
    """Append one row per pulse period to a separate CSV."""
    rows = []
    for i, seg in enumerate(batch):
        segment_id = f"segment_{global_seg_idx_start + i:08d}"
        meta = seg.get('metadata', {})
        for row in seg.get('period_rows', []):
            rows.append({
                'segment_id': segment_id,
                'subject_id': meta.get('subject_id', ''),
                'source_file': meta.get('dataset_source_file', ''),
                **row,
            })
    if rows:
        df = pd.DataFrame(rows)
        header = not os.path.exists(period_output_csv) or os.path.getsize(period_output_csv) == 0
        df.to_csv(period_output_csv, mode='a', header=header, index=False)


def save_batch_sequences(batch, h5_path, log_file, fnames_batch,
                         global_seg_idx_start, period_output_csv=None):
    if not batch:
        return global_seg_idx_start

    with h5py.File(h5_path, 'a') as f:
        for i, seg in enumerate(batch):
            grp_name = f'segment_{global_seg_idx_start + i:08d}'
            if grp_name in f:
                del f[grp_name]
            grp = f.create_group(grp_name)

            grp.create_dataset(
                'beat_features', data=seg['beat_features'],
                compression='gzip', chunks=True
            )
            grp.create_dataset(
                'period_features', data=seg['period_features'],
                compression='gzip', chunks=True
            )
            grp.create_dataset(
                'segment_features', data=seg['segment_features'],
                compression='gzip'
            )
            grp.create_dataset(
                'period_feature_names',
                data=np.asarray(seg.get('period_feature_names', []), dtype='S128')
            )
            grp.create_dataset(
                'labels',
                data=np.array(
                    [seg['labels']['sbp'], seg['labels']['dbp']],
                    dtype=np.float32
                )
            )

            for k, v in seg.get('metadata', {}).items():
                if isinstance(v, (str, bytes)):
                    grp.attrs[k] = str(v)
                elif isinstance(v, (int, np.integer)):
                    grp.attrs[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    grp.attrs[k] = float(v)
                else:
                    grp.attrs[k] = str(v)

            for k, v in seg.get('period_summary', {}).items():
                if isinstance(v, (str, bytes)):
                    grp.attrs[k] = str(v)
                elif isinstance(v, (int, np.integer)):
                    grp.attrs[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    grp.attrs[k] = float(v)
                else:
                    grp.attrs[k] = str(v)

    if period_output_csv:
        save_batch_period_csv(batch, period_output_csv, global_seg_idx_start)

    with open(log_file, 'a') as lf:
        for fname in fnames_batch:
            lf.write(fname + '\n')

    logging.info(
        f"Saved {len(batch)} segments to HDF5 "
        f"(indices {global_seg_idx_start}-{global_seg_idx_start + len(batch) - 1})"
    )
    return global_seg_idx_start + len(batch)

def process_mat_file_unified(mat_filepath, max_segments=None, fs=125, period_p=PERIOD_P_DEFAULT):
    file_features = []
    try:
        segments = load_v73_mat_to_dict_list(mat_filepath)
        if not segments:
            return file_features
        if max_segments and max_segments < len(segments):
            segments = segments[:max_segments]
        for idx, seg_dict in enumerate(segments):
            try:
                feat = process_single_segment_unified(seg_dict, mat_filepath, fs=fs, period_p=period_p)
                if feat is not None:
                    file_features.append(feat)
            except Exception as e:
                logging.error(f"Error segment {idx} in {mat_filepath}: {e}")
    except Exception as e:
        logging.error(f"Error loading {mat_filepath}: {e}")
    return file_features


def process_all_files_unified(data_root, output_csv, output_h5, log_file, log_file_seq,
                              limit_files=None, max_segments_per_file=None, fs=125,
                              period_p=PERIOD_P_DEFAULT, max_workers=MAX_WORKERS):
    logging.info(f"Starting FINAL v1 feature extraction (paper-reference period F/Phi/Psi; p={period_p})...")
    period_output_csv = os.path.splitext(output_csv)[0] + "_period_features.csv"
    if not os.path.isdir(data_root):
        logging.error(f"Folder not found: {data_root}")
        return

    mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]
    if not mat_files:
        logging.error(f"No .mat files in {data_root}")
        return
    logging.info(f"Found {len(mat_files)} .mat files")
    if limit_files:
        mat_files = mat_files[:limit_files]

    processed_csv = load_processed_files(log_file)
    processed_seq = load_processed_files(log_file_seq)
    remaining_files = [f for f in mat_files if f not in processed_csv or f not in processed_seq]

    # Initialize HDF5 if fresh
    if not os.path.exists(output_h5):
        with h5py.File(output_h5, 'w') as f:
            f.attrs['description'] = 'Unified PPG beat sequences for DL models'
            f.attrs['beat_feature_names'] = BEAT_FEATURE_ORDER
            f.attrs['segment_feature_names'] = SEGMENT_FEATURE_ORDER
            f.attrs['period_feature_names'] = PERIOD_BASE_FEATURE_ORDER
            f.attrs['description'] = 'Unified PPG sequences with final paper-reference pulse-period F/Phi/Psi analysis'
            f.attrs['period_p'] = float(period_p)
            f.attrs['period_formula_reference'] = PERIOD_FORMULA_REFERENCE
            f.attrs['period_psi_evaluation_rule'] = PERIOD_PSI_EVALUATION_RULE

    # Resume: count existing segments
    global_seg_idx = 0
    if os.path.exists(output_h5):
        with h5py.File(output_h5, 'r') as f:
            global_seg_idx = len([k for k in f.keys() if k.startswith('segment_')])

    batch_csv = []
    batch_seq = []
    fnames_batch = []
    total_processed_files = len(processed_csv)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fname in remaining_files:
            filepath = os.path.join(data_root, fname)
            futures[executor.submit(process_mat_file_unified, filepath, max_segments_per_file, fs, period_p)] = fname

        for future in as_completed(futures):
            fname = futures[future]
            try:
                feats = future.result()
                if feats:
                    batch_csv.extend(flatten_features(feats))
                    batch_seq.extend([f['seq'] for f in feats])
                    fnames_batch.append(fname)
                    total_processed_files += 1
                    logging.info(f"Completed {fname} ({total_processed_files}/{len(mat_files)})")
                    if len(batch_csv) >= SAVE_INTERVAL:
                        # save_batch_csv(batch_csv, output_csv, log_file, fnames_batch)
                        save_batch_csv(batch_csv, output_csv, log_file, fnames_batch, global_seg_idx)
                        global_seg_idx = save_batch_sequences(
                            batch_seq, output_h5, log_file_seq, fnames_batch, global_seg_idx,
                            period_output_csv=period_output_csv)
                        batch_csv = []
                        batch_seq = []
                        fnames_batch = []
                else:
                    with open(log_file, 'a') as f:
                        f.write(fname + '\n')
                    with open(log_file_seq, 'a') as f:
                        f.write(fname + '\n')
                    total_processed_files += 1
            except Exception as e:
                logging.error(f"Failed {fname}: {e}\n{traceback.format_exc()}")

    if batch_csv:
        # save_batch_csv(batch_csv, output_csv, log_file, fnames_batch)
        save_batch_csv(batch_csv, output_csv, log_file, fnames_batch, global_seg_idx)
        global_seg_idx = save_batch_sequences(
            batch_seq, output_h5, log_file_seq, fnames_batch, global_seg_idx,
            period_output_csv=period_output_csv)

    logging.info(f"Done. Files: {total_processed_files}. Segments in HDF5: {global_seg_idx}. "
                 f"CSV: {output_csv}")


# ============================================================
# 9. MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Final unified PPG extractor candidate v2 with literature-faithful period F/Phi/Psi formulas."
    )
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--output-csv', default=OUTPUT_CSV)
    parser.add_argument('--output-h5', default=OUTPUT_H5)
    parser.add_argument('--log-file', default=LOG_FILE)
    parser.add_argument('--log-file-seq', default=LOG_FILE_SEQ)
    parser.add_argument('--fs', type=float, default=125.0)
    parser.add_argument('--period-p', type=float, default=PERIOD_P_DEFAULT)
    parser.add_argument('--limit-files', type=int, default=None)
    parser.add_argument('--max-segments-per-file', type=int, default=None)
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_all_files_unified(
        data_root=args.data_root,
        output_csv=args.output_csv,
        output_h5=args.output_h5,
        log_file=args.log_file,
        log_file_seq=args.log_file_seq,
        limit_files=args.limit_files,
        max_segments_per_file=args.max_segments_per_file,
        fs=args.fs,
        period_p=args.period_p,
        max_workers=args.max_workers,
    )


# [join_mask_v2] synthetic test
def test_join_mask_filtering(fs=125, verbose=True):
    """Verify periods that overlap the reject_mask are dropped."""
    import numpy as _np
    t = _np.arange(int(10 * fs)) / fs
    sig = _np.sin(2 * _np.pi * 1.0 * t).astype(float)
    peaks = _np.arange(int(1 * fs), int(9 * fs), fs)
    mask = _np.zeros(len(sig), dtype=bool)
    mask[480:520] = True

    periods, meta = segment_pulse_periods_from_peaks(
        sig, peaks, fs, reject_mask=mask)
    for m in meta:
        s, e = int(m['start_sample']), int(m['end_sample'])
        assert not mask[s:e + 1].any(), (
            f"period [{s}, {e}] overlaps mask; filtering failed")
    _, meta_unmasked = segment_pulse_periods_from_peaks(sig, peaks, fs)
    assert len(meta) <= len(meta_unmasked)
    if verbose:
        print("[join_mask_v2] test passed:",
              len(meta), "retained of", len(meta_unmasked), "unmasked")
    return True

