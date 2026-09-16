"""Registre extensible de métriques pupillométriques."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config import Config


MetricFunction = Callable[[pd.DataFrame, float], float]

METRIC_REGISTRY: dict[str, MetricFunction] = {}


def register_metric(name: str) -> Callable:
    """Décorateur pour enregistrer une nouvelle métrique."""
    def decorator(function: MetricFunction) -> MetricFunction:
        METRIC_REGISTRY[name] = function
        return function
    return decorator


@register_metric("pupil_mean")
def pupil_mean(window: pd.DataFrame, baseline: float) -> float:
    """Moyenne de taille pupillaire."""
    return float(window["pupil_clean"].mean())


@register_metric("pupil_sd")
def pupil_sd(window: pd.DataFrame, baseline: float) -> float:
    """Ecart-type pupillaire dans la fenêtre."""
    return float(window["pupil_clean"].std(ddof=1))


@register_metric("pupil_baseline_corrected")
def pupil_baseline_corrected(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Moyenne pupillaire corrigée par la baseline."""
    if not np.isfinite(baseline):
        return np.nan

    return float(window["pupil_clean"].mean() - baseline)


@register_metric("pupil_slope_per_s")
def pupil_slope(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Pente de dilatation pupillaire par seconde."""
    valid = window.dropna(subset=["pupil_clean"])

    if len(valid) < 3:
        return np.nan

    time_s = (
        valid["timestamp_ms"]
        - valid["timestamp_ms"].iloc[0]
    ) / 1000.0

    return float(
        linregress(
            time_s,
            valid["pupil_clean"],
        ).slope
    )


@register_metric("pupil_auc")
def pupil_auc(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Aire sous la courbe pupillaire."""
    valid = window.dropna(subset=["pupil_clean"])

    if len(valid) < 2:
        return np.nan

    time_s = valid["timestamp_ms"].to_numpy() / 1000.0
    signal = valid["pupil_clean"].to_numpy()

    return float(np.trapz(signal, time_s))


@register_metric("pupil_peak")
def pupil_peak(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Pic de dilatation pupillaire."""
    return float(window["pupil_clean"].max())


@register_metric("pupil_peak_latency_s")
def pupil_peak_latency(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Latence du pic de dilatation."""
    valid = window.dropna(subset=["pupil_clean"])

    if valid.empty:
        return np.nan

    peak_index = valid["pupil_clean"].idxmax()

    return float(
        (
            valid.loc[peak_index, "timestamp_ms"]
            - valid["timestamp_ms"].iloc[0]
        ) / 1000.0
    )


# Métriques de QC / oculomotrices secondaires.
@register_metric("missing_fraction")
def missing_fraction(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Proportion d'échantillons manquants dans la fenêtre."""
    return float(window["pupil_missing_raw"].mean())


@register_metric("gaze_dispersion_x")
def gaze_dispersion_x(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Dispersion horizontale du regard."""
    return float(window["gaze_x"].std(ddof=1))


@register_metric("gaze_dispersion_y")
def gaze_dispersion_y(
    window: pd.DataFrame,
    baseline: float,
) -> float:
    """Dispersion verticale du regard."""
    return float(window["gaze_y"].std(ddof=1))


def extract_metrics(
    windows: pd.DataFrame,
    preprocessed: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Calcule toutes les métriques pour chaque fenêtre.

    Une baseline manquante n'invalide pas une fenêtre entière :
    seules les métriques nécessitant une baseline, notamment
    pupil_baseline_corrected, deviennent NaN.
    """
    rows: list[dict] = []

    for _, window in windows.iterrows():
        segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                window["start_eyelink_ms"],
                window["end_eyelink_ms"],
                inclusive="both",
            )
        ].copy()

        baseline_segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                window["baseline_start_eyelink_ms"],
                window["baseline_end_eyelink_ms"],
                inclusive="both",
            )
        ].copy()

        missing_window = (
            float(segment["pupil_missing_raw"].mean())
            if not segment.empty
            else 1.0
        )

        baseline_missing_fraction = (
            float(baseline_segment["pupil_missing_raw"].mean())
            if not baseline_segment.empty
            else 1.0
        )

        if (
            baseline_segment.empty
            or baseline_missing_fraction > cfg.max_baseline_missing_fraction
        ):
            baseline_value = np.nan
        else:
            baseline_value = float(
                baseline_segment["pupil_clean"].mean()
            )

        trial_invalid = (
            bool(window["timeout"])
            or bool(window["is_resumed"])
        )

        duration_invalid = not bool(window["duration_valid"])
        no_samples = segment.empty

        too_missing = (
            missing_window > cfg.max_moment_missing_fraction
        )

        moment_valid = not (
            trial_invalid
            or duration_invalid
            or no_samples
            or too_missing
        )

        reason = ""

        if trial_invalid:
            reason = "timeout_or_resumed"
        elif duration_invalid:
            reason = "window_too_short"
        elif no_samples:
            reason = "no_samples_in_window"
        elif too_missing:
            reason = "moment_missing_fraction_above_threshold"

        for metric_name, function in METRIC_REGISTRY.items():
            value = (
                function(segment, baseline_value)
                if moment_valid
                else np.nan
            )

            rows.append(
                {
                    "subject_id": window["subject_id"],
                    "run_id": window["run_id"],
                    "trial_n": window["trial_n"],
                    "trial_n_global": window.get(
                        "trial_n_global",
                        np.nan,
                    ),
                    "induction_block_id": window["induction_block_id"],
                    "moment": window["moment"],
                    "window_x_s": window["window_x_s"],
                    "metric": metric_name,
                    "value": value,
                    "moment_valid": moment_valid,
                    "moment_exclusion_reason": reason,
                    "missing_fraction_window": missing_window,
                    "baseline_value": baseline_value,
                    "baseline_missing_fraction": baseline_missing_fraction,
                    "baseline_source": window["baseline_source"],
                    "actual_duration_s": window["actual_duration_s"],
                    "group_5cat": window["group_5cat"],
                    "group_valence": window["group_valence"],
                    "group_arousal": window["group_arousal"],
                    "emo_grid_valence": window.get(
                        "emo_grid_valence",
                        np.nan,
                    ),
                    "emo_grid_arousal": window.get(
                        "emo_grid_arousal",
                        np.nan,
                    ),
                    "LTask_rating": window.get(
                        "LTask_rating",
                        np.nan,
                    ),
                }
            )

    return pd.DataFrame(rows)