"""Prétraitement séparant faits observés, artefacts, interpolation et blinks EyeLink.

Mesures physiologiques de blinks = EBLINK uniquement; le padding n'est jamais
compté dans leurs durées. Les lacunes réelles ne sont jamais masquées par le filtre.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from config import Config


def _robust_outlier_mask(values: pd.Series, threshold: float) -> np.ndarray:
    a = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    valid = a[np.isfinite(a)]
    if len(valid) < 10:
        return np.zeros(len(a), dtype=bool)
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros(len(a), dtype=bool)
    return np.isfinite(a) & (np.abs(a - med) > threshold * 1.4826 * mad)


def _contiguous_intervals(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def _mask_from_intervals(t: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Masque O(n log m) via bornes; supppose timestamps croissants."""
    if not len(starts):
        return np.zeros(len(t), dtype=bool)
    diff = np.zeros(len(t) + 1, dtype=np.int32)
    a = np.searchsorted(t, starts, side='left')
    b = np.searchsorted(t, ends, side='right')
    np.add.at(diff, a, 1)
    np.add.at(diff, b, -1)
    return np.cumsum(diff[:-1]) > 0


def _expand_contiguous_segments(mask: np.ndarray, timestamps_ms: np.ndarray,
                                before_ms: int, after_ms: int) -> np.ndarray:
    starts, ends = _contiguous_intervals(mask)
    if not len(starts):
        return mask.copy()
    return _mask_from_intervals(timestamps_ms, timestamps_ms[starts] - before_ms,
                                timestamps_ms[ends-1] + after_ms)


def _mask_speed_near_missing_segments(speed_mask: np.ndarray, missing_mask: np.ndarray,
                                      timestamps_ms: np.ndarray, vicinity_ms: int = 150) -> np.ndarray:
    """Masquage vectorisé sans boucle quadratique sur les samples manquants."""
    return speed_mask & _expand_contiguous_segments(missing_mask, timestamps_ms,
                                                     vicinity_ms, vicinity_ms)


def _interpolate_short_gaps(signal: pd.Series, timestamps_ms: np.ndarray,
                            max_gap_ms: float, segment_ids=None) -> tuple[pd.Series, np.ndarray]:
    result = signal.to_numpy(dtype=float, copy=True)
    interpolated_mask = np.zeros(len(result), dtype=bool)
    dt = float(np.median(np.diff(timestamps_ms))) if len(result) > 1 else 1.0
    starts, ends = _contiguous_intervals(~np.isfinite(result))
    for a, b in zip(starts, ends):
        if a == 0 or b == len(result):
            continue
        # Trou après padding ; aucune reconstruction d'une interruption.
        if timestamps_ms[b] - timestamps_ms[a-1] - dt > max_gap_ms + 1e-9:
            continue
        if np.any(np.diff(timestamps_ms[a-1:b+1]) > 1.5 * dt):
            continue
        if segment_ids is not None and np.any(np.diff(segment_ids[a-1:b+1]) != 0):
            continue
        if not np.isfinite(result[a-1]) or not np.isfinite(result[b]):
            continue
        result[a:b] = np.interp(timestamps_ms[a:b],
                               [timestamps_ms[a-1], timestamps_ms[b]],
                               [result[a-1], result[b]])
        interpolated_mask[a:b] = True
    return pd.Series(result, index=signal.index), interpolated_mask


def _filter_valid_segments(signal: pd.Series, t_ms: np.ndarray, cfg: Config,
                           segment_ids=None) -> pd.Series:
    if not cfg.lowpass_enabled:
        return signal
    a = signal.to_numpy(dtype=float, copy=True)
    fs = 1000.0 / np.nanmedian(np.diff(t_ms)) if len(t_ms) > 2 else cfg.sampling_rate_hz
    if not np.isfinite(fs) or fs <= 2 * cfg.lowpass_cutoff_hz:
        return signal
    sos = butter(cfg.lowpass_order, cfg.lowpass_cutoff_hz,
                 btype='low', fs=fs, output='sos')
    valid = np.isfinite(a)
    # Séparer aussi les interruptions d'horloge : ne pas lisser au travers.
    gaps = np.flatnonzero(np.diff(t_ms) > 1.5 * (1000.0 / fs)) + 1
    if segment_ids is not None:
        gaps = np.union1d(gaps, np.flatnonzero(np.diff(segment_ids) != 0) + 1)
    starts, ends = _contiguous_intervals(valid)
    for start, end in zip(starts, ends):
        edges = np.r_[start, gaps[(gaps > start) & (gaps < end)], end]
        for left, right in zip(edges[:-1], edges[1:]):
            if right - left >= 40:
                try:
                    a[left:right] = sosfiltfilt(sos, a[left:right])
                except ValueError:
                    pass
    return pd.Series(a, index=signal.index)


def _eyelink_blink_intervals(events: pd.DataFrame) -> pd.DataFrame:
    columns = ['blink_start_ms', 'blink_end_ms', 'blink_duration_ms']
    if events.empty or 'event_type' not in events:
        return pd.DataFrame(columns=columns)
    e = events.loc[events['event_type'].eq('EBLINK')].copy()
    if e.empty:
        return pd.DataFrame(columns=columns)
    e = e.dropna(subset=['start_ms', 'end_ms'])
    e = e.loc[e['end_ms'] > e['start_ms']]
    result = e[['start_ms','end_ms','duration_ms']].copy()
    result.columns = columns
    duration = pd.to_numeric(result['blink_duration_ms'], errors='coerce')
    result['blink_duration_ms'] = duration.where(np.isfinite(duration) & duration.gt(0),
                                               result['blink_end_ms'] - result['blink_start_ms'])
    return result.reset_index(drop=True)


def preprocess_pupil(samples: pd.DataFrame, events: pd.DataFrame,
                     cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = samples.reset_index(drop=True).copy()
    blink_intervals = _eyelink_blink_intervals(events)
    if df.empty:
        return df, blink_intervals
    t = df['timestamp_ms'].to_numpy(dtype=float)
    if not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError('Samples must have finite, strictly increasing timestamps')
    segment_ids = df.get('segment_id', pd.Series(0, index=df.index)).to_numpy()
    pupil = pd.to_numeric(df['pupil_size'], errors='coerce').to_numpy(dtype=float)
    acquisition_missing = ~np.isfinite(pupil) | (pupil <= cfg.min_plausible_pupil)
    event_blinks = _mask_from_intervals(t, blink_intervals['blink_start_ms'].to_numpy(dtype=float),
                                        blink_intervals['blink_end_ms'].to_numpy(dtype=float))
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_speed = np.abs(np.diff(pupil) / (np.diff(t) / 1000.0))
    good_pairs = (~acquisition_missing[:-1] & ~acquisition_missing[1:]
                  & (np.diff(t) <= 1.5 * 1000.0 / cfg.sampling_rate_hz)
                  & (np.diff(segment_ids) == 0))
    pair_speed[~good_pairs] = np.nan
    speed = np.fmax(np.r_[np.nan, pair_speed], np.r_[pair_speed, np.nan])
    finite_speed = speed[np.isfinite(speed)]
    speed_artifact = np.zeros(len(t), dtype=bool)
    if len(finite_speed) >= 10:
        med = np.median(finite_speed)
        mad = np.median(np.abs(finite_speed - med))
        speed_artifact = speed > med + max(1e-9, cfg.pupil_speed_mad_threshold * 1.4826 * mad)
    near_missing = _mask_speed_near_missing_segments(speed_artifact, acquisition_missing, t)
    # MASQUE DE NETTOYAGE: ne pas l'assimiler aux clignements physiologiques.
    rejected_speed = speed_artifact if cfg.pupil_reject_speed_outliers else near_missing
    artifact = _expand_contiguous_segments(acquisition_missing | event_blinks | rejected_speed,
                                           t, cfg.blink_padding_before_ms, cfg.blink_padding_after_ms)
    clean_input = pupil.copy()
    clean_input[artifact] = np.nan
    outliers = _robust_outlier_mask(pd.Series(clean_input), cfg.pupil_outlier_mad_threshold)
    clean_input[outliers] = np.nan
    artifact |= outliers
    interp, interpolated = _interpolate_short_gaps(pd.Series(clean_input), t,
                                                   cfg.interpolation_max_gap_ms, segment_ids)
    df['pupil_clean'] = _filter_valid_segments(interp, t, cfg, segment_ids)
    df['pupil_missing_acquisition'] = acquisition_missing
    df['pupil_artifact_mask'] = artifact
    df['pupil_observed_valid'] = ~artifact
    df['pupil_interpolated'] = interpolated
    df['pupil_missing_qc'] = artifact
    df['pupil_missing_raw'] = artifact  # compatibilité métriques anciennes
    df['blink_mask'] = event_blinks  # durée EBLINK sans padding
    df['event_blink_mask'] = event_blinks
    df['speed_artifact_mask'] = speed_artifact
    df['speed_near_missing_mask'] = near_missing
    df['residual_outlier_mask'] = outliers
    gaze_x = pd.to_numeric(df['gaze_x'], errors='coerce').to_numpy(dtype=float)
    gaze_y = pd.to_numeric(df['gaze_y'], errors='coerce').to_numpy(dtype=float)
    gaze_valid = (np.isfinite(gaze_x) & np.isfinite(gaze_y)
                  & (gaze_x >= 0) & (gaze_x < cfg.gaze_screen_width_px)
                  & (gaze_y >= 0) & (gaze_y < cfg.gaze_screen_height_px)
                  & ~event_blinks & ~acquisition_missing)
    df['gaze_valid'] = gaze_valid
    df['gaze_x_clean'] = np.where(gaze_valid, gaze_x, np.nan)
    df['gaze_y_clean'] = np.where(gaze_valid, gaze_y, np.nan)
    return df, blink_intervals
