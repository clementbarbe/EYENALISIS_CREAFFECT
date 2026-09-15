"""Registre extensible de métriques pupillométriques."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import linregress


MetricFunction = Callable[[pd.DataFrame, float], float]

METRIC_REGISTRY: dict[str, MetricFunction] = {}


def register_metric(name: str) -> Callable:
    """Décorateur permettant d'ajouter une métrique au registre."""
    def decorator(function: MetricFunction) -> MetricFunction:
        METRIC_REGISTRY[name] = function
        return function
    return decorator


@register_metric("pupil_mean")
def pupil_mean(window: pd.DataFrame, baseline: float) -> float:
    """Moyenne de pupille."""
    return float(window["pupil_clean"].mean())


@register_metric("pupil_sd")
def pupil_sd(window: pd.DataFrame, baseline: float) -> float:
    """Ecart-type de pupille."""
    return float(window["pupil_clean"].std(ddof=1))


@register_metric("pupil_baseline_corrected")
def pupil_baseline_corrected(window: pd.DataFrame, baseline: float) -> float:
    """Moyenne corrigée par baseline."""
    if not np.isfinite(baseline):
        return np.nan

    return float(window["pupil_clean"].mean() - baseline)


@register_metric("pupil_slope_per_s")
def pupil_slope(window: pd.DataFrame, baseline: float) -> float:
    """Pente de dilatation pupillaire par seconde."""
    valid = window.dropna(subset=["pupil_clean"])

    if len(valid) < 3:
        return np.nan

    time_s = (
        valid["timestamp_ms"] - valid["timestamp_ms"].iloc[0]
    ) / 1000.0

    return float(linregress(time_s, valid["pupil_clean"]).slope)


@register_metric("pupil_auc")
def pupil_auc(window: pd.DataFrame, baseline: float) -> float:
    """Aire sous la courbe pupillaire."""
    valid = window.dropna(subset=["pupil_clean"])

    if len(valid) < 2:
        return np.nan

    x_s = valid["timestamp_ms"].to_numpy() / 1000.0
    y = valid["pupil_clean"].to_numpy()

    return float(np.trapz(y, x_s))


@register_metric("pupil_peak")
def pupil_peak(window: pd.DataFrame, baseline: float) -> float:
    """Pic pupillaire."""
    return float(window["pupil_clean"].max())


@register_metric("pupil_peak_latency_s")
def pupil_peak_latency(window: pd.DataFrame, baseline: float) -> float:
    """Latence du pic pupillaire."""
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


@register_metric("missing_fraction")
def missing_fraction(window: pd.DataFrame, baseline: float) -> float:
    """Proportion de données manquantes brutes."""
    return float(window["pupil_missing_raw"].mean())


@register_metric("gaze_dispersion_x")
def gaze_dispersion_x(window: pd.DataFrame, baseline: float) -> float:
    """Dispersion horizontale du regard."""
    return float(window["gaze_x"].std(ddof=1))


@register_metric("gaze_dispersion_y")
def gaze_dispersion_y(window: pd.DataFrame, baseline: float) -> float:
    """Dispersion verticale du regard."""
    return float(window["gaze_y"].std(ddof=1))


def extract_metrics(
    windows: pd.DataFrame,
    preprocessed: pd.DataFrame,
    max_missing_fraction: float,
) -> pd.DataFrame:
    """Calcule les métriques pour chaque fenêtre essai x moment."""
    output_rows: list[dict] = []

    for _, row in windows.iterrows():
        segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                row["start_eyelink_ms"],
                row["end_eyelink_ms"],
                inclusive="both",
            )
        ].copy()

        baseline_segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                row["baseline_start_eyelink_ms"],
                row["baseline_end_eyelink_ms"],
                inclusive="left",
            )
        ].copy()

        if segment.empty:
            missing_fraction_window = 1.0
        else:
            missing_fraction_window = float(
                segment["pupil_missing_raw"].mean()
            )

        baseline_value = float(baseline_segment["pupil_clean"].mean())

        trial_invalid = bool(row["timeout"]) or bool(row["is_resumed"])
        duration_invalid = not bool(row["duration_valid"])
        no_samples = segment.empty
        too_missing = missing_fraction_window > max_missing_fraction

        moment_valid = not (
            trial_invalid
            or duration_invalid
            or no_samples
            or too_missing
        )

        exclusion_reason = ""

        if trial_invalid:
            exclusion_reason = "timeout_or_resumed"
        elif duration_invalid:
            exclusion_reason = "window_too_short"
        elif no_samples:
            exclusion_reason = "no_samples_in_window"
        elif too_missing:
            exclusion_reason = "moment_missing_fraction_above_threshold"

        for metric_name, function in METRIC_REGISTRY.items():
            value = function(segment, baseline_value) if moment_valid else np.nan

            output_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "run_id": row["run_id"],
                    "trial_n": row["trial_n"],
                    "trial_n_global": row.get("trial_n_global", np.nan),
                    "induction_block_id": row["induction_block_id"],
                    "moment": row["moment"],
                    "metric": metric_name,
                    "value": value,
                    "moment_valid": moment_valid,
                    "moment_exclusion_reason": exclusion_reason,
                    "missing_fraction_window": missing_fraction_window,
                    "baseline_value": baseline_value,
                    "actual_duration_s": row["actual_duration_s"],
                    "group_5cat": row["group_5cat"],
                    "group_valence": row["group_valence"],
                    "group_arousal": row["group_arousal"],
                    "emo_grid_valence": row.get("emo_grid_valence", np.nan),
                    "emo_grid_arousal": row.get("emo_grid_arousal", np.nan),
                    "LTask_rating": row.get("LTask_rating", np.nan),
                }
            )

    return pd.DataFrame(output_rows)