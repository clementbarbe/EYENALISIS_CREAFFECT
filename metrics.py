"""Calcul des métriques pupillaires et oculomotrices."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config import Config


MetricFunction = Callable[
    [pd.DataFrame, float, pd.DataFrame, pd.DataFrame, Config],
    float,
]

METRIC_REGISTRY: dict[str, MetricFunction] = {}


def register_metric(name: str) -> Callable:
    """Décorateur pour enregistrer une métrique."""
    def decorator(function: MetricFunction) -> MetricFunction:
        METRIC_REGISTRY[name] = function
        return function
    return decorator


def _window_duration_s(window: pd.DataFrame) -> float:
    """Durée réelle d'une fenêtre en secondes."""
    if window.empty:
        return np.nan

    return float(
        (
            window["timestamp_ms"].iloc[-1]
            - window["timestamp_ms"].iloc[0]
        ) / 1000.0
    )


def _events_overlapping_window(
    events: pd.DataFrame,
    start_ms: float,
    end_ms: float,
    event_type: str,
) -> pd.DataFrame:
    """Retourne les évènements chevauchant une fenêtre."""
    selected = events.loc[
        events["event_type"] == event_type
    ].copy()

    if selected.empty:
        return selected

    selected = selected.loc[
        selected["start_ms"].notna()
        & selected["end_ms"].notna()
    ]

    return selected.loc[
        (selected["end_ms"] >= start_ms)
        & (selected["start_ms"] <= end_ms)
    ]


@register_metric("pupil_mean")
def pupil_mean(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Taille pupillaire moyenne."""
    return float(window["pupil_clean"].mean())


@register_metric("pupil_sd")
def pupil_sd(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Ecart-type pupillaire."""
    return float(window["pupil_clean"].std(ddof=1))


@register_metric("pupil_baseline_corrected")
def pupil_baseline_corrected(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Taille moyenne corrigée par baseline."""
    if not np.isfinite(baseline):
        return np.nan

    return float(window["pupil_clean"].mean() - baseline)


@register_metric("pupil_slope_per_s")
def pupil_slope(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Pente pupillaire linéaire par seconde."""
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
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Aire sous la courbe pupillaire."""
    valid = window.dropna(subset=["pupil_clean"])

    if len(valid) < 2:
        return np.nan

    x_s = valid["timestamp_ms"].to_numpy() / 1000.0
    y = valid["pupil_clean"].to_numpy()

    return float(np.trapz(y, x_s))


@register_metric("pupil_peak")
def pupil_peak(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Pic pupillaire."""
    return float(window["pupil_clean"].max())


@register_metric("pupil_peak_latency_s")
def pupil_peak_latency(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
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


@register_metric("blink_rate_per_s")
def blink_rate(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Taux de blink par seconde."""
    duration_s = _window_duration_s(window)

    if not np.isfinite(duration_s) or duration_s <= 0:
        return np.nan

    start_ms = window["timestamp_ms"].iloc[0]
    end_ms = window["timestamp_ms"].iloc[-1]

    overlapping = blinks.loc[
        (blinks["blink_end_ms"] >= start_ms)
        & (blinks["blink_start_ms"] <= end_ms)
    ]

    return float(len(overlapping) / duration_s)


@register_metric("blink_mean_duration_ms")
def blink_mean_duration(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Durée moyenne des blinks dans la fenêtre."""
    start_ms = window["timestamp_ms"].iloc[0]
    end_ms = window["timestamp_ms"].iloc[-1]

    overlapping = blinks.loc[
        (blinks["blink_end_ms"] >= start_ms)
        & (blinks["blink_start_ms"] <= end_ms)
    ]

    if overlapping.empty:
        return 0.0

    return float(overlapping["blink_duration_ms"].mean())


@register_metric("microsaccade_rate_per_s")
def microsaccade_rate(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Taux de microsaccades (<1 degré)."""
    duration_s = _window_duration_s(window)

    if not np.isfinite(duration_s) or duration_s <= 0:
        return np.nan

    start_ms = window["timestamp_ms"].iloc[0]
    end_ms = window["timestamp_ms"].iloc[-1]

    saccades = _events_overlapping_window(
        events,
        start_ms,
        end_ms,
        "ESACC",
    )

    microsaccades = saccades.loc[
        saccades["amplitude_deg"]
        < cfg.microsaccade_max_amplitude_deg
    ]

    return float(len(microsaccades) / duration_s)


@register_metric("saccade_rate_per_s")
def saccade_rate(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Fréquence de saccades hors microsaccades."""
    duration_s = _window_duration_s(window)

    if not np.isfinite(duration_s) or duration_s <= 0:
        return np.nan

    start_ms = window["timestamp_ms"].iloc[0]
    end_ms = window["timestamp_ms"].iloc[-1]

    saccades = _events_overlapping_window(
        events,
        start_ms,
        end_ms,
        "ESACC",
    )

    large_saccades = saccades.loc[
        saccades["amplitude_deg"]
        >= cfg.microsaccade_max_amplitude_deg
    ]

    return float(len(large_saccades) / duration_s)


@register_metric("saccade_mean_amplitude_deg")
def saccade_mean_amplitude(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Amplitude moyenne des saccades hors microsaccades."""
    start_ms = window["timestamp_ms"].iloc[0]
    end_ms = window["timestamp_ms"].iloc[-1]

    saccades = _events_overlapping_window(
        events,
        start_ms,
        end_ms,
        "ESACC",
    )

    large_saccades = saccades.loc[
        saccades["amplitude_deg"]
        >= cfg.microsaccade_max_amplitude_deg
    ]

    return float(large_saccades["amplitude_deg"].mean())


@register_metric("gaze_dispersion_x")
def gaze_dispersion_x(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Dispersion horizontale du regard."""
    return float(window["gaze_x"].std(ddof=1))


@register_metric("gaze_dispersion_y")
def gaze_dispersion_y(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Dispersion verticale du regard."""
    return float(window["gaze_y"].std(ddof=1))


@register_metric("gaze_aversion_fraction")
def gaze_aversion_fraction(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Proportion de temps passé hors ROI du cue."""
    if (
        cfg.cue_roi_center_x is None
        or cfg.cue_roi_center_y is None
    ):
        return np.nan

    gaze = window.dropna(subset=["gaze_x", "gaze_y"])

    if gaze.empty:
        return np.nan

    inside_roi = (
        gaze["gaze_x"].between(
            cfg.cue_roi_center_x - cfg.cue_roi_half_width_px,
            cfg.cue_roi_center_x + cfg.cue_roi_half_width_px,
        )
        & gaze["gaze_y"].between(
            cfg.cue_roi_center_y - cfg.cue_roi_half_height_px,
            cfg.cue_roi_center_y + cfg.cue_roi_half_height_px,
        )
    )

    return float(1.0 - inside_roi.mean())


@register_metric("missing_fraction")
def missing_fraction(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Proportion réellement manquante dans le signal brut."""
    missing_column = (
        "pupil_missing_acquisition"
        if "pupil_missing_acquisition" in window.columns
        else "pupil_missing_raw"
    )

    return float(window[missing_column].mean())


def extract_metrics(
    windows: pd.DataFrame,
    preprocessed: pd.DataFrame,
    blink_intervals: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Calcule les métriques de toutes les fenêtres."""
    rows: list[dict] = []

    missing_column = (
        "pupil_missing_acquisition"
        if "pupil_missing_acquisition" in preprocessed.columns
        else "pupil_missing_raw"
    )

    for _, window_meta in windows.iterrows():
        segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                window_meta["start_eyelink_ms"],
                window_meta["end_eyelink_ms"],
                inclusive="both",
            )
        ].copy()

        baseline_segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                window_meta["baseline_start_eyelink_ms"],
                window_meta["baseline_end_eyelink_ms"],
                inclusive="both",
            )
        ].copy()

        missing_window = (
            float(segment[missing_column].mean())
            if not segment.empty
            else 1.0
        )

        baseline_missing = (
            float(baseline_segment[missing_column].mean())
            if not baseline_segment.empty
            else 1.0
        )

        baseline_value = np.nan

        if (
            not baseline_segment.empty
            and baseline_missing <= cfg.max_baseline_missing_fraction
        ):
            baseline_value = float(
                baseline_segment["pupil_clean"].mean()
            )

        trial_invalid = (
            bool(window_meta["timeout"])
            or bool(window_meta["is_resumed"])
        )

        moment_valid = not (
            trial_invalid
            or not bool(window_meta["duration_valid"])
            or segment.empty
            or missing_window > cfg.max_moment_missing_fraction
        )

        if trial_invalid:
            exclusion_reason = "timeout_or_resumed"
        elif not bool(window_meta["duration_valid"]):
            exclusion_reason = "window_too_short"
        elif segment.empty:
            exclusion_reason = "no_samples_in_window"
        elif missing_window > cfg.max_moment_missing_fraction:
            exclusion_reason = "too_much_acquisition_missing"
        else:
            exclusion_reason = ""

        for metric_name, metric_function in METRIC_REGISTRY.items():
            value = (
                metric_function(
                    segment,
                    baseline_value,
                    blink_intervals,
                    events,
                    cfg,
                )
                if moment_valid
                else np.nan
            )

            rows.append(
                {
                    "subject_id": window_meta["subject_id"],
                    "run_id": window_meta["run_id"],
                    "trial_n": window_meta["trial_n"],
                    "induction_block_id": window_meta[
                        "induction_block_id"
                    ],
                    "moment": window_meta["moment"],
                    "window_x_s": window_meta["window_x_s"],
                    "metric": metric_name,
                    "value": value,
                    "moment_valid": moment_valid,
                    "moment_exclusion_reason": exclusion_reason,
                    "missing_fraction_window": missing_window,
                    "baseline_value": baseline_value,
                    "baseline_missing_fraction": baseline_missing,
                    "actual_duration_s": window_meta[
                        "actual_duration_s"
                    ],
                    "group_5cat": window_meta["group_5cat"],
                    "group_valence": window_meta["group_valence"],
                    "group_arousal": window_meta["group_arousal"],
                    "emo_grid_valence": window_meta.get(
                        "emo_grid_valence",
                        np.nan,
                    ),
                    "emo_grid_arousal": window_meta.get(
                        "emo_grid_arousal",
                        np.nan,
                    ),
                    "LTask_rating": window_meta.get(
                        "LTask_rating",
                        np.nan,
                    ),
                }
            )

    return pd.DataFrame(rows)


def extract_h4_gaze_samples(
    windows: pd.DataFrame,
    preprocessed: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Extrait un sous-échantillon de gaze pendant la recherche d'idée.

    Cette table sert aux heatmaps H4 par catégorie.
    """
    rows: list[pd.DataFrame] = []

    h4_windows = windows.loc[
        windows["moment"] == "H4_full_search_gaze"
    ].copy()

    for _, window_meta in h4_windows.iterrows():
        if (
            bool(window_meta["timeout"])
            or bool(window_meta["is_resumed"])
            or not bool(window_meta["duration_valid"])
        ):
            continue

        gaze = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                window_meta["start_eyelink_ms"],
                window_meta["end_eyelink_ms"],
                inclusive="both",
            )
        ][["timestamp_ms", "gaze_x", "gaze_y"]].dropna()

        if gaze.empty:
            continue

        if len(gaze) > cfg.gaze_plot_max_samples_per_trial:
            step = int(
                np.ceil(
                    len(gaze) / cfg.gaze_plot_max_samples_per_trial
                )
            )
            gaze = gaze.iloc[::step].copy()

        gaze["subject_id"] = window_meta["subject_id"]
        gaze["run_id"] = window_meta["run_id"]
        gaze["trial_n"] = window_meta["trial_n"]
        gaze["group_5cat"] = window_meta["group_5cat"]
        gaze["group_valence"] = window_meta["group_valence"]
        gaze["group_arousal"] = window_meta["group_arousal"]

        rows.append(gaze)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp_ms",
                "gaze_x",
                "gaze_y",
                "subject_id",
                "run_id",
                "trial_n",
                "group_5cat",
                "group_valence",
                "group_arousal",
            ]
        )

    return pd.concat(rows, ignore_index=True)

@register_metric("all_saccade_rate_per_s")
def all_saccade_rate(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """
    Fréquence de toutes les saccades ESACC, sans seuil d'amplitude.

    Cette métrique est utile pour vérifier que les résultats sur les
    saccades ne dépendent pas uniquement du seuil choisi pour distinguer
    microsaccades et saccades.
    """
    duration_s = _window_duration_s(window)

    if not np.isfinite(duration_s) or duration_s <= 0:
        return np.nan

    start_ms = window["timestamp_ms"].iloc[0]
    end_ms = window["timestamp_ms"].iloc[-1]

    saccades = _events_overlapping_window(
        events=events,
        start_ms=start_ms,
        end_ms=end_ms,
        event_type="ESACC",
    )

    return float(len(saccades) / duration_s)