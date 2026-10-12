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
def pupil_auc(window: pd.DataFrame, baseline: float, blinks: pd.DataFrame,
              events: pd.DataFrame, cfg: Config) -> float:
    """Aire baseline-corrigée sur sous-segments observés/filtrés contigus.

    La valeur n'est PAS extrapolée aux lacunes: comparer avec couverture et durée.
    """
    if not np.isfinite(baseline) or window.empty:
        return np.nan
    t = window['timestamp_ms'].to_numpy(dtype=float)
    y = window['pupil_clean'].to_numpy(dtype=float) - baseline
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    dt_expected = 1000.0 / cfg.sampling_rate_hz
    good_pairs = valid[:-1] & valid[1:] & (np.diff(t) <= dt_expected * 1.5)
    return float(np.sum((y[:-1][good_pairs] + y[1:][good_pairs]) *
                        np.diff(t)[good_pairs] / 2000.0))


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
    return float(window["gaze_x_clean"].std(ddof=1))


@register_metric("gaze_dispersion_y")
def gaze_dispersion_y(
    window: pd.DataFrame,
    baseline: float,
    blinks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Config,
) -> float:
    """Dispersion verticale du regard."""
    return float(window["gaze_y_clean"].std(ddof=1))


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

    gaze = window.dropna(subset=["gaze_x_clean", "gaze_y_clean"])

    if gaze.empty:
        return np.nan

    inside_roi = (
        gaze["gaze_x_clean"].between(
            cfg.cue_roi_center_x - cfg.cue_roi_half_width_px,
            cfg.cue_roi_center_x + cfg.cue_roi_half_width_px,
        )
        & gaze["gaze_y_clean"].between(
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


def _qc_segment(segment: pd.DataFrame, start_ms: float, end_ms: float,
                cfg: Config) -> dict:
    """QC indépendant de la condition expérimentale; compter les trous d'horloge."""
    duration_ms = max(0., float(end_ms) - float(start_ms))
    if segment.empty or duration_ms <= 0:
        return dict(observed_valid_fraction=0., artifact_fraction=1.,
                    interpolated_fraction=0., acquisition_missing_fraction=1.,
                    gaze_valid_fraction=0., longest_gap_ms=duration_ms,
                    sampling_coverage_fraction=0., n_samples=0)
    t = segment['timestamp_ms'].to_numpy(dtype=float)
    artifact = segment['pupil_artifact_mask'].to_numpy(dtype=bool)
    observed = segment['pupil_observed_valid'].to_numpy(dtype=bool)
    interpol = segment['pupil_interpolated'].to_numpy(dtype=bool)
    gaze = segment['gaze_valid'].to_numpy(dtype=bool)
    # Fraction du temps couverte (tenant compte des samples manquants dans l'ASC)
    expected = 1.0 + duration_ms * cfg.sampling_rate_hz / 1000.0
    coverage = min(1., len(segment) / expected)
    # Longueur des interruptions: bords + blocs d'artefacts + trous entre samples.
    missing_t = t[artifact]
    breaks = np.diff(t)
    largest_timestamp_gap = float(max(np.max(breaks) - 1000. / cfg.sampling_rate_hz, 0)) if len(breaks) else duration_ms
    largest_artifact_gap = 0.
    starts = np.flatnonzero(np.diff(np.r_[0, artifact.astype(np.int8), 0]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[0, artifact.astype(np.int8), 0]) == -1)
    if len(starts):
        largest_artifact_gap = float(np.max(t[ends-1] - t[starts] + 1000./cfg.sampling_rate_hz))
    edge_gap = max(float(t[0] - start_ms), float(end_ms - t[-1]), 0.)
    longest = max(largest_timestamp_gap, largest_artifact_gap, edge_gap)
    return dict(observed_valid_fraction=float(observed.mean()*coverage),
                artifact_fraction=float(1-observed.mean()*coverage),
                interpolated_fraction=float(interpol.mean()*coverage),
                acquisition_missing_fraction=float(segment['pupil_missing_acquisition'].mean()*coverage + 1-coverage),
                gaze_valid_fraction=float(gaze.mean()*coverage),
                longest_gap_ms=longest,
                sampling_coverage_fraction=float(coverage), n_samples=len(segment))


def extract_metrics(windows: pd.DataFrame, preprocessed: pd.DataFrame,
                    blink_intervals: pd.DataFrame, events: pd.DataFrame,
                    cfg: Config) -> pd.DataFrame:
    """QC par essai/fenêtre/métrique. Les métriques non pupillaires restent accessibles."""
    rows = []
    if windows.empty or preprocessed.empty:
        return pd.DataFrame()
    # Découpage par searchsorted: O(log n), sans refaire un masque de 400k points
    timestamps = preprocessed['timestamp_ms'].to_numpy(dtype=float)
    def segment(left, right):
        if pd.isna(left) or pd.isna(right) or right <= left:
            return preprocessed.iloc[0:0]
        a, b = np.searchsorted(timestamps, [left, right], side='left')
        return preprocessed.iloc[a:b]
    pupil_metrics = {k for k in METRIC_REGISTRY if k.startswith('pupil_')}
    gaze_metrics = {k for k in METRIC_REGISTRY if k.startswith('gaze_')}
    blink_metrics = {k for k in METRIC_REGISTRY if k.startswith('blink_')}
    # saccades sont une famille distincte, sensible aux problèmes de calibration.
    for _, meta in windows.iterrows():
        start, end = float(meta['start_eyelink_ms']), float(meta['end_eyelink_ms'])
        win = segment(start, end)
        b_start, b_end = meta['baseline_start_eyelink_ms'], meta['baseline_end_eyelink_ms']
        base = segment(b_start, b_end)
        q = _qc_segment(win, start, end, cfg)
        bq = _qc_segment(base, b_start, b_end, cfg)
        baseline_valid = (bq['observed_valid_fraction'] >= cfg.min_observed_fraction_baseline
                          and bq['acquisition_missing_fraction'] <= cfg.max_baseline_missing_fraction
                          and bq['n_samples'] > 0)
        baseline = float(base.loc[base['pupil_observed_valid'], 'pupil_clean'].median()) if baseline_valid else np.nan
        trial_invalid = bool(meta['timeout']) or bool(meta['is_resumed'])
        common_reasons = []
        if trial_invalid: common_reasons.append('timeout_or_resumed')
        if not bool(meta['duration_valid']): common_reasons.append('window_too_short')
        if q['n_samples'] == 0: common_reasons.append('no_samples')
        if q['sampling_coverage_fraction'] < 1-cfg.max_window_missing_sampling_fraction:
            common_reasons.append('missing_sampling_coverage')
        common_ok = not common_reasons
        pupil_reasons = common_reasons.copy()
        if q['observed_valid_fraction'] < cfg.min_observed_fraction_window:
            pupil_reasons.append('low_observed_pupil_fraction')
        if q['interpolated_fraction'] > cfg.max_interpolated_fraction_window:
            pupil_reasons.append('excessive_pupil_interpolation')
        if q['longest_gap_ms'] > cfg.max_longest_gap_window_ms:
            pupil_reasons.append('long_artifact_gap')
        pupil_ok = not pupil_reasons
        gaze_ok = common_ok and q['gaze_valid_fraction'] >= cfg.min_gaze_valid_fraction
        # Uniquement les métriques soustractives et AUC dépendent de la baseline.
        baseline_metrics = {'pupil_baseline_corrected', 'pupil_auc'}
        for name, metric in METRIC_REGISTRY.items():
            reason = list(pupil_reasons) if name in pupil_metrics else list(common_reasons)
            if name in gaze_metrics and not gaze_ok:
                reason.append('low_valid_gaze_fraction')
            if name in baseline_metrics and not baseline_valid:
                reason.append('invalid_baseline')
            if name == 'gaze_aversion_fraction' and (cfg.cue_roi_center_x is None or cfg.cue_roi_center_y is None):
                reason.append('roi_not_configured')
            if name == 'microsaccade_rate_per_s':
                reason.append('exploratory_small_saccades_not_validated')
            # Blinks et saccades calculables même avec une pupille partiellement invalide.
            ok = not reason
            value = metric(win, baseline, blink_intervals, events, cfg) if ok else np.nan
            rows.append({
                'subject_id':meta['subject_id'], 'run_id':meta['run_id'],
                'trial_n':meta['trial_n'], 'induction_block_id':meta['induction_block_id'],
                'moment':meta['moment'], 'window_x_s':meta['window_x_s'],
                'metric':name, 'value':value, 'moment_valid':bool(ok),
                'moment_exclusion_reason':';'.join(dict.fromkeys(reason)),
                'pupil_window_valid':bool(pupil_ok), 'gaze_window_valid':bool(gaze_ok),
                'baseline_valid':bool(baseline_valid),
                'missing_fraction_window':q['acquisition_missing_fraction'],
                'baseline_value':baseline,
                'baseline_missing_fraction':bq['acquisition_missing_fraction'],
                'baseline_observed_valid_fraction':bq['observed_valid_fraction'],
                'actual_duration_s':meta['actual_duration_s'],
                'observed_valid_fraction':q['observed_valid_fraction'],
                'artifact_fraction':q['artifact_fraction'],
                'interpolated_fraction':q['interpolated_fraction'],
                'gaze_valid_fraction':q['gaze_valid_fraction'],
                'sampling_coverage_fraction':q['sampling_coverage_fraction'],
                'longest_gap_ms':q['longest_gap_ms'],
                'stim_id':meta.get('stim_id', np.nan),
                'stim_image':meta.get('stim_image', np.nan),
                'stim_sound':meta.get('stim_sound', np.nan),
                'trial_n_global':meta.get('trial_n_global', np.nan),
                'cue_text':meta.get('cues', np.nan),
                'group_5cat':meta['group_5cat'], 'group_valence':meta['group_valence'],
                'group_arousal':meta['group_arousal'],
                'emo_grid_valence':meta.get('emo_grid_valence', np.nan),
                'emo_grid_arousal':meta.get('emo_grid_arousal', np.nan),
                'LTask_rating':meta.get('LTask_rating', np.nan),
                'fgat_search_duration_s': (float(meta['fgat_input_time'])-float(meta['fgat_cue_onset'])
                     if pd.notna(meta.get('fgat_input_time')) and pd.notna(meta.get('fgat_cue_onset')) else np.nan),
                'LTask_response_duration_s': (float(meta['LTask_validation'])-float(meta['LTask_onset'])
                     if pd.notna(meta.get('LTask_validation')) and pd.notna(meta.get('LTask_onset')) else np.nan),
            })
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
        ][["timestamp_ms", "gaze_x_clean", "gaze_y_clean", "gaze_valid"]].dropna()

        gaze = gaze.loc[gaze["gaze_valid"]].copy()
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

        gaze = gaze.rename(columns={"gaze_x_clean": "gaze_x", "gaze_y_clean": "gaze_y"})
        gaze["sample_weight"] = 1.0 / max(len(gaze), 1)
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