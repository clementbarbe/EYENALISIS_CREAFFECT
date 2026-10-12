"""Prétraitement pupillométrique inspiré de Kret & Sjak-Shie (2019).

Important
---------
La vitesse pupillaire est utilisée comme information complémentaire autour
des trous de signal, et non comme une règle globale supprimant tout échantillon
dont la dérivée est élevée. Cela évite de transformer des fluctuations normales
à 1000 Hz en dizaines de pourcents de données manquantes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import Config


def _robust_outlier_mask(
    values: pd.Series,
    threshold: float,
) -> np.ndarray:
    """Détecte les outliers robustes par MAD."""
    valid = values.dropna()

    if valid.empty:
        return np.zeros(len(values), dtype=bool)

    median = valid.median()
    mad = np.median(np.abs(valid - median))

    if not np.isfinite(mad) or mad <= 0:
        return np.zeros(len(values), dtype=bool)

    robust_z = 0.6745 * (values - median) / mad

    return (
        robust_z.abs() > threshold
    ).fillna(False).to_numpy()


def _pupil_speed_artifact_mask(
    pupil: pd.Series,
    timestamps_ms: pd.Series,
    cfg: Config,
) -> np.ndarray:
    """
    Détecte des vitesses pupillaires anormalement fortes.

    Cette fonction ne doit pas être utilisée seule comme masque de blink global.
    Elle sert uniquement à identifier des artefacts proches des trous du signal.
    """
    dt_s = timestamps_ms.diff() / 1000.0
    dp = pupil.diff()

    speed = (dp / dt_s).abs()
    speed = speed.replace([np.inf, -np.inf], np.nan)

    return _robust_outlier_mask(
        speed,
        cfg.pupil_speed_mad_threshold,
    )


def _event_blink_mask(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Construit un masque à partir des événements EBLINK, s'ils existent."""
    mask = np.zeros(len(samples), dtype=bool)

    if events.empty or "event_type" not in events.columns:
        return mask

    blink_events = events.loc[
        events["event_type"] == "EBLINK"
    ]

    for _, event in blink_events.iterrows():
        start_ms = event.get("start_ms", np.nan)
        end_ms = event.get("end_ms", np.nan)

        if pd.isna(start_ms) or pd.isna(end_ms):
            continue

        mask |= samples["timestamp_ms"].between(
            start_ms,
            end_ms,
            inclusive="both",
        ).to_numpy()

    return mask


def _expand_contiguous_segments(
    mask: np.ndarray,
    timestamps_ms: np.ndarray,
    before_ms: int,
    after_ms: int,
) -> np.ndarray:
    """
    Étend chaque segment CONTIGU d'artefact.

    Contrairement à l'ancien code, on n'étend pas séparément chaque échantillon.
    Cela évite d'ajouter +/-100 ms des milliers de fois.
    """
    expanded = mask.copy()

    if not mask.any():
        return expanded

    start = 0

    while start < len(mask):
        if not mask[start]:
            start += 1
            continue

        end = start

        while end < len(mask) and mask[end]:
            end += 1

        segment_start_ms = timestamps_ms[start]
        segment_end_ms = timestamps_ms[end - 1]

        expanded |= (
            (timestamps_ms >= segment_start_ms - before_ms)
            & (timestamps_ms <= segment_end_ms + after_ms)
        )

        start = end

    return expanded


def _mask_speed_near_missing_segments(
    speed_mask: np.ndarray,
    missing_mask: np.ndarray,
    timestamps_ms: np.ndarray,
    vicinity_ms: int = 150,
) -> np.ndarray:
    """
    Ne conserve les artefacts de vitesse que proches d'un vrai trou de signal.

    Un changement rapide de pupille peut être physiologique ou être du bruit.
    Il n'est traité comme artefact de blink que s'il est proche d'une période
    réellement invalide/manquante.
    """
    if not speed_mask.any() or not missing_mask.any():
        return np.zeros(len(speed_mask), dtype=bool)

    nearby_missing = np.zeros(len(missing_mask), dtype=bool)

    missing_indices = np.flatnonzero(missing_mask)

    for index in missing_indices:
        timestamp = timestamps_ms[index]

        nearby_missing |= (
            (timestamps_ms >= timestamp - vicinity_ms)
            & (timestamps_ms <= timestamp + vicinity_ms)
        )

    return speed_mask & nearby_missing


def _interpolate_short_gaps(
    signal: pd.Series,
    max_gap_samples: int,
) -> pd.Series:
    """Interpole linéairement les trous courts uniquement."""
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


def _filter_valid_segments(
    signal: pd.Series,
    cfg: Config,
) -> pd.Series:
    """Applique un filtre passe-bas aux segments continus valides."""
    if not cfg.lowpass_enabled:
        return signal

    output = signal.copy()
    valid_mask = output.notna().to_numpy()

    nyquist = cfg.sampling_rate_hz / 2.0
    normalized_cutoff = cfg.lowpass_cutoff_hz / nyquist

    b, a = butter(
        cfg.lowpass_order,
        normalized_cutoff,
        btype="low",
    )

    minimum_length = 3 * max(len(a), len(b))

    start = 0

    while start < len(output):
        if not valid_mask[start]:
            start += 1
            continue

        end = start

        while end < len(output) and valid_mask[end]:
            end += 1

        segment = output.iloc[start:end].to_numpy(dtype=float)

        if len(segment) >= minimum_length:
            try:
                output.iloc[start:end] = filtfilt(b, a, segment)
            except ValueError:
                pass

        start = end

    return output


def _make_blink_intervals(
    blink_mask: np.ndarray,
    timestamps_ms: np.ndarray,
) -> pd.DataFrame:
    """Transforme le masque de blink en tableau d'intervalles."""
    rows: list[dict] = []

    start = 0

    while start < len(blink_mask):
        if not blink_mask[start]:
            start += 1
            continue

        end = start

        while end < len(blink_mask) and blink_mask[end]:
            end += 1

        start_ms = timestamps_ms[start]
        end_ms = timestamps_ms[end - 1]

        rows.append(
            {
                "blink_start_ms": start_ms,
                "blink_end_ms": end_ms,
                "blink_duration_ms": end_ms - start_ms,
            }
        )

        start = end

    return pd.DataFrame(
        rows,
        columns=[
            "blink_start_ms",
            "blink_end_ms",
            "blink_duration_ms",
        ],
    )


def preprocess_pupil(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nettoie le signal pupillaire.

    Returns
    -------
    preprocessed:
        Contient :
        - pupil_missing_acquisition : trous réellement présents dans le fichier ;
        - pupil_missing_qc : trous + fenêtres de blink ;
        - pupil_clean : signal interpolé/filtré ;
        - blink_mask : masque utilisé pour taux de blink.

    blink_intervals:
        Tableau des épisodes de blink identifiés.
    """
    df = samples.copy().sort_values(
        "timestamp_ms"
    ).reset_index(drop=True)

    empty_blinks = pd.DataFrame(
        columns=[
            "blink_start_ms",
            "blink_end_ms",
            "blink_duration_ms",
        ]
    )

    if df.empty:
        return df, empty_blinks

    pupil_raw = pd.to_numeric(
        df["pupil_size"],
        errors="coerce",
    )

    timestamps = df["timestamp_ms"].to_numpy(dtype=float)

    # 1. Données réellement manquantes/inexploitables dans le fichier EyeLink.
    acquisition_missing_mask = (
        pupil_raw.isna()
        | (pupil_raw <= cfg.min_plausible_pupil)
    ).to_numpy()

    # 2. EBLINK éventuels : complément.
    event_blink_mask = _event_blink_mask(
        df,
        events,
        cfg,
    )

    # 3. Détection de vitesse élevée, mais elle n'est utilisée qu'autour
    # de vrais trous de tracking.
    speed_mask = _pupil_speed_artifact_mask(
        pupil_raw,
        df["timestamp_ms"],
        cfg,
    )

    speed_near_missing_mask = _mask_speed_near_missing_segments(
        speed_mask=speed_mask,
        missing_mask=acquisition_missing_mask,
        timestamps_ms=timestamps,
        vicinity_ms=150,
    )

    # 4. Blinks initiaux : vrais trous + tags EyeLink + vitesse près d'un trou.
    initial_blink_mask = (
        acquisition_missing_mask
        | event_blink_mask
        | speed_near_missing_mask
    )

    # 5. Padding seulement une fois par segment continu.
    blink_mask = _expand_contiguous_segments(
        mask=initial_blink_mask,
        timestamps_ms=timestamps,
        before_ms=cfg.blink_padding_before_ms,
        after_ms=cfg.blink_padding_after_ms,
    )

    pupil_for_processing = pupil_raw.copy()
    pupil_for_processing.loc[blink_mask] = np.nan

    # 6. Outliers restants : ils sont retirés du signal, mais on les stocke
    # séparément. Ils ne gonflent pas automatiquement le taux QC run.
    residual_outlier_mask = _robust_outlier_mask(
        pupil_for_processing,
        cfg.pupil_outlier_mad_threshold,
    )

    pupil_for_processing.loc[residual_outlier_mask] = np.nan

    # QC acquisition : ce qui était réellement manquant dans le fichier.
    df["pupil_missing_acquisition"] = acquisition_missing_mask

    # QC fenêtre : acquisition + blink padding.
    # Les outliers résiduels restent dans une colonne séparée.
    df["pupil_missing_qc"] = blink_mask

    # Compatibilité avec les autres modules actuels :
    # le pipeline utilise cette colonne pour exclusion fenêtre.
    df["pupil_missing_raw"] = df["pupil_missing_qc"]

    df["blink_mask"] = blink_mask
    df["event_blink_mask"] = event_blink_mask
    df["speed_artifact_mask"] = speed_mask
    df["speed_near_missing_mask"] = speed_near_missing_mask
    df["residual_outlier_mask"] = residual_outlier_mask

    blink_intervals = _make_blink_intervals(
        blink_mask,
        timestamps,
    )

    max_gap_samples = int(
        cfg.interpolation_max_gap_ms
        * cfg.sampling_rate_hz
        / 1000.0
    )

    interpolated = _interpolate_short_gaps(
        pupil_for_processing,
        max_gap_samples=max_gap_samples,
    )

    df["pupil_clean"] = _filter_valid_segments(
        interpolated,
        cfg,
    )

    return df, blink_intervals