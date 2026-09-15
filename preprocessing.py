"""Prétraitement pupillométrique."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import Config


def _mad_outlier_mask(signal: pd.Series, threshold: float) -> np.ndarray:
    """Détecte les valeurs aberrantes avec la MAD."""
    valid = signal.dropna()

    if valid.empty:
        return np.zeros(len(signal), dtype=bool)

    median = valid.median()
    mad = np.median(np.abs(valid - median))

    if mad <= 0 or not np.isfinite(mad):
        return np.zeros(len(signal), dtype=bool)

    robust_z = 0.6745 * (signal - median) / mad
    return (robust_z.abs() > threshold).fillna(False).to_numpy()


def _blink_mask_from_events(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Crée un masque de blink à partir des EBLINK."""
    mask = np.zeros(len(samples), dtype=bool)

    if events.empty:
        return mask

    blink_events = events.loc[events["event_type"] == "EBLINK"]

    for _, event in blink_events.iterrows():
        start_ms = event.get("start_ms", np.nan)
        end_ms = event.get("end_ms", np.nan)

        if pd.isna(start_ms) or pd.isna(end_ms):
            continue

        padded_start = start_ms - cfg.blink_padding_before_ms
        padded_end = end_ms + cfg.blink_padding_after_ms

        mask |= samples["timestamp_ms"].between(
            padded_start,
            padded_end,
            inclusive="both",
        ).to_numpy()

    return mask


def _interpolate_short_gaps(
    signal: pd.Series,
    max_gap_samples: int,
) -> pd.Series:
    """Interpole les trous NaN dont la durée est inférieure au seuil."""
    output = signal.copy()
    missing = output.isna().to_numpy()

    start = 0

    while start < len(output):
        if not missing[start]:
            start += 1
            continue

        end = start
        while end < len(output) and missing[end]:
            end += 1

        gap_length = end - start

        if (
            gap_length <= max_gap_samples
            and start > 0
            and end < len(output)
            and pd.notna(output.iloc[start - 1])
            and pd.notna(output.iloc[end])
        ):
            output.iloc[start:end] = np.linspace(
                output.iloc[start - 1],
                output.iloc[end],
                gap_length + 2,
            )[1:-1]

        start = end

    return output


def _filter_segments(signal: pd.Series, cfg: Config) -> pd.Series:
    """Applique le filtre passe-bas séparément à chaque segment valide."""
    if not cfg.lowpass_enabled:
        return signal

    output = signal.copy()
    valid = output.notna().to_numpy()

    nyquist = cfg.sampling_rate_hz / 2.0
    normalized_cutoff = cfg.lowpass_cutoff_hz / nyquist

    b, a = butter(
        cfg.lowpass_order,
        normalized_cutoff,
        btype="low",
    )

    min_segment_length = 3 * max(len(a), len(b))
    start = 0

    while start < len(output):
        if not valid[start]:
            start += 1
            continue

        end = start
        while end < len(output) and valid[end]:
            end += 1

        segment = output.iloc[start:end].to_numpy(dtype=float)

        if len(segment) >= min_segment_length:
            try:
                output.iloc[start:end] = filtfilt(b, a, segment)
            except ValueError:
                pass

        start = end

    return output


def preprocess_pupil(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Nettoie le signal pupillaire.

    Colonnes produites
    ------------------
    pupil_missing_raw:
        Manquant / invalide avant interpolation. Utilisé pour QC.
    pupil_clean:
        Signal interpolé et filtré.
    blink_or_invalid:
        Indique les échantillons masqués avant interpolation.
    """
    df = samples.copy().sort_values("timestamp_ms").reset_index(drop=True)

    if df.empty:
        return df

    pupil = pd.to_numeric(df["pupil_size"], errors="coerce")

    invalid_mask = pupil.isna() | (pupil <= cfg.min_plausible_pupil)
    blink_mask = _blink_mask_from_events(df, events, cfg)

    pupil.loc[invalid_mask | blink_mask] = np.nan

    outlier_mask = _mad_outlier_mask(
        pupil,
        cfg.pupil_outlier_mad_threshold,
    )

    pupil.loc[outlier_mask] = np.nan

    df["pupil_missing_raw"] = pupil.isna()
    df["blink_or_invalid"] = invalid_mask | blink_mask | outlier_mask

    max_gap_samples = int(
        cfg.interpolation_max_gap_ms
        * cfg.sampling_rate_hz
        / 1000.0
    )

    pupil_interpolated = _interpolate_short_gaps(
        pupil,
        max_gap_samples=max_gap_samples,
    )

    df["pupil_clean"] = _filter_segments(
        pupil_interpolated,
        cfg,
    )

    return df