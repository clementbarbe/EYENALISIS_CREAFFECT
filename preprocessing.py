"""Prétraitement pupillométrique : blinks, interpolation, filtrage, outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import Config


def _blink_mask_from_events(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Construit un masque de blink depuis EBLINK ou SBLINK/EBLINK disponibles."""
    mask = np.zeros(len(samples), dtype=bool)
    if events.empty:
        return mask

    blink_events = events.loc[events["event_type"] == "EBLINK"]
    for _, event in blink_events.iterrows():
        start = event.get("start_ms", np.nan)
        end = event.get("end_ms", np.nan)
        if pd.notna(start) and pd.notna(end):
            start -= cfg.blink_padding_before_ms
            end += cfg.blink_padding_after_ms
            mask |= samples["timestamp_ms"].between(start, end).to_numpy()

    return mask


def _mad_outliers(values: pd.Series, threshold: float) -> np.ndarray:
    """Détection robuste des outliers par MAD."""
    valid = values.dropna()
    if valid.empty:
        return np.zeros(len(values), dtype=bool)

    median = valid.median()
    mad = np.median(np.abs(valid - median))
    if mad == 0:
        return np.zeros(len(values), dtype=bool)

    robust_z = 0.6745 * (values - median) / mad
    return (np.abs(robust_z) > threshold).fillna(False).to_numpy()


def _interpolate_short_gaps(
    signal: pd.Series,
    max_gap_samples: int,
) -> pd.Series:
    """Interpole linéairement seulement les segments NaN courts."""
    out = signal.copy()
    is_na = out.isna().to_numpy()

    start = 0
    while start < len(out):
        if not is_na[start]:
            start += 1
            continue

        end = start
        while end < len(out) and is_na[end]:
            end += 1

        if (end - start) <= max_gap_samples and start > 0 and end < len(out):
            out.iloc[start:end] = np.linspace(
                out.iloc[start - 1],
                out.iloc[end],
                end - start + 2,
            )[1:-1]
        start = end

    return out


def _filter_contiguous_segments(signal: pd.Series, cfg: Config) -> pd.Series:
    """Applique un Butterworth séparément sur chaque segment non-NaN."""
    if not cfg.lowpass_enabled:
        return signal

    out = signal.copy()
    nyquist = cfg.sampling_rate_hz / 2.0
    b, a = butter(
        cfg.lowpass_order,
        cfg.lowpass_cutoff_hz / nyquist,
        btype="low",
    )

    valid = out.notna().to_numpy()
    start = 0

    while start < len(out):
        if not valid[start]:
            start += 1
            continue

        end = start
        while end < len(out) and valid[end]:
            end += 1

        segment = out.iloc[start:end].to_numpy(dtype=float)
        min_length = 3 * max(len(a), len(b))
        if len(segment) >= min_length:
            out.iloc[start:end] = filtfilt(b, a, segment)

        start = end

    return out


def preprocess_pupil(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Produit un signal nettoyé.

    pupil_missing_raw conserve les données invalides avant interpolation,
    afin de calculer correctement le taux de données manquantes.
    """
    df = samples.copy().sort_values("timestamp_ms").reset_index(drop=True)
    if df.empty:
        return df

    pupil = pd.to_numeric(df["pupil_size"], errors="coerce")
    invalid = pupil.isna() | (pupil <= cfg.min_plausible_pupil)

    blink_mask = _blink_mask_from_events(df, events, cfg)
    pupil.loc[invalid | blink_mask] = np.nan

    # Outliers bruts avant interpolation.
    outliers = _mad_outliers(pupil, cfg.pupil_outlier_mad_threshold)
    pupil.loc[outliers] = np.nan

    df["pupil_missing_raw"] = pupil.isna()
    df["blink_or_invalid"] = invalid | blink_mask | outliers

    max_gap_samples = int(cfg.interpolation_max_gap_ms * cfg.sampling_rate_hz / 1000)
    interpolated = _interpolate_short_gaps(pupil, max_gap_samples)
    df["pupil_clean"] = _filter_contiguous_segments(interpolated, cfg)

    return df
