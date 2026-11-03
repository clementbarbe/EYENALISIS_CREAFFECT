"""Métriques déclarées avec unités et prérequis ; extraction commune aux projets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config import Config
from quality import evaluate_window, qc_segment as _qc_segment


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    family: str
    unit: str
    baseline: bool = False
    event_type: str = ""
    roi: bool = False
    validated: bool = True
    description: str = ""


METRIC_REGISTRY: dict[str, Callable] = {}
METRIC_DEFINITIONS: dict[str, MetricDefinition] = {}
LEGACY_METRICS = (
    "pupil_mean", "pupil_sd", "pupil_baseline_corrected", "pupil_slope_per_s",
    "pupil_auc", "pupil_peak", "pupil_peak_latency_s", "blink_rate_per_s",
    "blink_mean_duration_ms", "microsaccade_rate_per_s", "saccade_rate_per_s",
    "saccade_mean_amplitude_deg", "gaze_dispersion_x", "gaze_dispersion_y",
    "gaze_aversion_fraction", "missing_fraction", "all_saccade_rate_per_s",
)


def register_metric(name, family, unit, **requirements):
    def decorate(function):
        if name in METRIC_REGISTRY:
            raise ValueError(f"Duplicate metric: {name}")
        METRIC_REGISTRY[name] = function
        METRIC_DEFINITIONS[name] = MetricDefinition(
            name, family, unit, description=function.__doc__ or "", **requirements)
        return function
    return decorate


def metric_catalog():
    return pd.DataFrame([asdict(spec) for spec in METRIC_DEFINITIONS.values()])


def _bounds(window):
    if "window_start_ms" in window.attrs:
        return window.attrs["window_start_ms"], window.attrs["window_end_ms"]
    if window.empty:
        return np.nan, np.nan
    time = window.timestamp_ms.to_numpy(dtype=float)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    return time[0], time[-1] + dt


def _window_duration_s(window):
    start, stop = _bounds(window)
    return float((stop - start) / 1000.0)


def _exposure(window, family):
    return window.attrs.get(f"{family}_exposure_s", _window_duration_s(window))


def _onsets(window, events, kind):
    start, stop = _bounds(window)
    if events.empty:
        return events
    return events.loc[events.event_type.eq(kind) & events.start_ms.ge(start)
                      & events.start_ms.lt(stop) & events.end_ms.ge(events.start_ms)]


def _events_overlapping_window(events, start_ms, end_ms, event_type):
    if events.empty:
        return events
    return events.loc[events.event_type.eq(event_type) & events.end_ms.gt(start_ms)
                      & events.start_ms.lt(end_ms) & events.end_ms.gt(events.start_ms)]


def _blink_onsets(window, blinks):
    start, stop = _bounds(window)
    return blinks.loc[blinks.blink_start_ms.ge(start) & blinks.blink_start_ms.lt(stop)]


def _integral_parts(window, cfg):
    time = window.timestamp_ms.to_numpy(dtype=float)
    values = window.pupil_clean.to_numpy(dtype=float)
    dt = np.diff(time) / 1000.0
    pairs = (np.isfinite(values[:-1]) & np.isfinite(values[1:])
             & (dt > 0) & (dt <= 1.5 / cfg.sampling_rate_hz))
    if "segment_id" in window:
        pairs &= np.diff(window.segment_id.to_numpy()) == 0
    return values, dt, pairs


@register_metric("pupil_mean", "pupil", "pupil_AU")
def pupil_mean(w, baseline, blinks, events, cfg):
    """Moyenne des samples pupillaires nettoyés, interpolation comprise."""
    return float(w.pupil_clean.mean())


@register_metric("pupil_median", "pupil", "pupil_AU")
def pupil_median(w, baseline, blinks, events, cfg):
    """Médiane des samples pupillaires nettoyés."""
    return float(w.pupil_clean.median())


@register_metric("pupil_sd", "pupil", "pupil_AU")
def pupil_sd(w, baseline, blinks, events, cfg):
    """Écart-type descriptif de la pupille (ddof=1)."""
    return float(w.pupil_clean.std(ddof=1))


@register_metric("pupil_baseline_corrected", "pupil", "pupil_AU", baseline=True)
def pupil_baseline_corrected(w, baseline, blinks, events, cfg):
    """Moyenne pupillaire moins médiane de la baseline observée."""
    return float(w.pupil_clean.mean() - baseline)


@register_metric("pupil_slope_per_s", "pupil", "pupil_AU/s")
def pupil_slope(w, baseline, blinks, events, cfg):
    """Pente linéaire descriptive sur les timestamps disponibles."""
    valid = w.dropna(subset=["pupil_clean"])
    if len(valid) < 3:
        return np.nan
    return float(linregress(valid.timestamp_ms / 1000.0, valid.pupil_clean).slope)


@register_metric("pupil_auc", "pupil", "pupil_AU*s", baseline=True)
def pupil_auc(window, baseline, blinks, events, cfg):
    """AUC signée soustractive, sans franchir les trous/segments d'acquisition."""
    if not np.isfinite(baseline) or window.empty:
        return np.nan
    y, dt, pairs = _integral_parts(window, cfg)
    if not pairs.any():
        return np.nan
    return float(np.sum(((y[:-1] + y[1:]) / 2 - baseline)[pairs] * dt[pairs]))


@register_metric("pupil_integrated_duration_s", "pupil", "s")
def pupil_integrated_duration(w, baseline, blinks, events, cfg):
    """Durée effectivement intégrée dans l'AUC, trous exclus."""
    _, dt, pairs = _integral_parts(w, cfg)
    return float(dt[pairs].sum())


@register_metric("pupil_auc_per_observed_s", "pupil", "pupil_AU", baseline=True)
def pupil_auc_mean(w, baseline, blinks, events, cfg):
    """AUC divisée par la durée intégrée ; ne reconstruit pas le temps absent."""
    duration = pupil_integrated_duration(w, baseline, blinks, events, cfg)
    return pupil_auc(w, baseline, blinks, events, cfg) / duration if duration > 0 else np.nan


@register_metric("pupil_peak", "pupil", "pupil_AU")
def pupil_peak(w, baseline, blinks, events, cfg):
    """Maximum absolu de la pupille nettoyée."""
    return float(w.pupil_clean.max())


@register_metric("pupil_peak_delta", "pupil", "pupil_AU", baseline=True)
def pupil_peak_delta(w, baseline, blinks, events, cfg):
    """Maximum pupillaire moins baseline."""
    return float(w.pupil_clean.max() - baseline)


@register_metric("pupil_peak_latency_s", "pupil", "s")
def pupil_peak_latency(window, baseline, blinks, events, cfg):
    """Latence du maximum depuis l'ancre déclarée (début nominal par défaut)."""
    valid = window.dropna(subset=["pupil_clean"])
    if valid.empty:
        return np.nan
    anchor = window.attrs.get("anchor_eyelink_ms", _bounds(window)[0])
    return float((valid.loc[valid.pupil_clean.idxmax(), "timestamp_ms"] - anchor) / 1000.0)


@register_metric("blink_count", "blink", "count", event_type="EBLINK")
def blink_count(w, baseline, blinks, events, cfg):
    """Nombre d'onsets de blinks dans [début, fin)."""
    return float(len(_blink_onsets(w, blinks)))


@register_metric("blink_rate_per_s", "blink", "1/s_tracking", event_type="EBLINK")
def blink_rate(window, baseline, blinks, events, cfg):
    """Onsets de blinks / seconde de tracking, fermetures EBLINK comprises."""
    duration = _exposure(window, "blink")
    return blink_count(window, baseline, blinks, events, cfg) / duration if duration > 0 else np.nan


@register_metric("blink_mean_duration_ms", "blink", "ms", event_type="EBLINK")
def blink_mean_duration(w, baseline, blinks, events, cfg):
    """Durée complète moyenne des blinks dont l'onset appartient à la fenêtre."""
    return float(_blink_onsets(w, blinks).blink_duration_ms.mean())


def _union_duration(starts, ends):
    total, last = 0.0, -np.inf
    for left, right in sorted(zip(starts, ends)):
        total += max(0.0, right - max(left, last))
        last = max(last, right)
    return total


@register_metric("blink_closed_fraction", "blink", "fraction_window", event_type="EBLINK")
def blink_closed_fraction(w, baseline, blinks, events, cfg):
    """Temps EBLINK intersectant la fenêtre / durée nominale, sans padding."""
    start, stop = _bounds(w)
    duration = _union_duration(np.maximum(blinks.blink_start_ms, start),
                               np.minimum(blinks.blink_end_ms, stop))
    return duration / (stop - start) if stop > start else np.nan


@register_metric("microsaccade_rate_per_s", "saccade", "1/s_valid_gaze",
                 event_type="ESACC", validated=False)
def microsaccade_rate(w, baseline, blinks, events, cfg):
    """Petites ESACC ; métrique désactivée car ce détecteur n'est pas validé."""
    return np.nan


@register_metric("all_saccade_count", "saccade", "count", event_type="ESACC")
def all_saccade_count(w, baseline, blinks, events, cfg):
    """Nombre d'onsets de toutes les ESACC."""
    return float(len(_onsets(w, events, "ESACC")))


@register_metric("all_saccade_rate_per_s", "saccade", "1/s_valid_gaze", event_type="ESACC")
def all_saccade_rate(window, baseline, blinks, events, cfg):
    """Onsets ESACC / seconde de gaze valide, sans seuil d'amplitude."""
    duration = _exposure(window, "gaze")
    return all_saccade_count(window, baseline, blinks, events, cfg) / duration if duration > 0 else np.nan


@register_metric("saccade_rate_per_s", "saccade", "1/s_valid_gaze", event_type="ESACC")
def saccade_rate(w, baseline, blinks, events, cfg):
    """Onsets ESACC au-dessus du seuil d'amplitude / seconde de gaze valide."""
    selected = _onsets(w, events, "ESACC")
    duration = _exposure(w, "gaze")
    return float(selected.amplitude_deg.ge(cfg.microsaccade_max_amplitude_deg).sum() / duration) if duration > 0 else np.nan


@register_metric("saccade_mean_amplitude_deg", "saccade", "deg", event_type="ESACC")
def saccade_mean_amplitude(w, baseline, blinks, events, cfg):
    """Amplitude moyenne des ESACC au-dessus du seuil, par onset."""
    selected = _onsets(w, events, "ESACC")
    return float(selected.loc[selected.amplitude_deg.ge(cfg.microsaccade_max_amplitude_deg), "amplitude_deg"].mean())


@register_metric("saccade_mean_duration_ms", "saccade", "ms", event_type="ESACC")
def saccade_mean_duration(w, baseline, blinks, events, cfg):
    """Durée moyenne complète des ESACC dont l'onset appartient à la fenêtre."""
    return float(_onsets(w, events, "ESACC").duration_ms.mean())


@register_metric("saccade_mean_peak_velocity_deg_s", "saccade", "deg/s", event_type="ESACC")
def saccade_peak_velocity(w, baseline, blinks, events, cfg):
    """Moyenne des pics de vitesse ESACC, selon le détecteur EyeLink."""
    return float(_onsets(w, events, "ESACC").peak_velocity.mean())


@register_metric("fixation_count", "fixation", "count", event_type="EFIX")
def fixation_count(w, baseline, blinks, events, cfg):
    """Nombre d'onsets de fixations EFIX."""
    return float(len(_onsets(w, events, "EFIX")))


@register_metric("fixation_mean_duration_ms", "fixation", "ms", event_type="EFIX")
def fixation_mean_duration(w, baseline, blinks, events, cfg):
    """Durée complète moyenne des EFIX dont l'onset appartient à la fenêtre."""
    return float(_onsets(w, events, "EFIX").duration_ms.mean())


@register_metric("fixation_time_fraction", "fixation", "fraction_window", event_type="EFIX")
def fixation_time_fraction(w, baseline, blinks, events, cfg):
    """Durée d'intersection des EFIX / durée nominale de fenêtre."""
    start, stop = _bounds(w)
    selected = _events_overlapping_window(events, start, stop, "EFIX")
    duration = _union_duration(np.maximum(selected.start_ms, start), np.minimum(selected.end_ms, stop))
    return duration / (stop - start) if stop > start else np.nan


@register_metric("gaze_dispersion_x", "gaze", "px")
def gaze_dispersion_x(w, baseline, blinks, events, cfg):
    """SD horizontale descriptive du gaze ; pas une précision instrumentale."""
    return float(w.gaze_x_clean.std(ddof=1))


@register_metric("gaze_dispersion_y", "gaze", "px")
def gaze_dispersion_y(w, baseline, blinks, events, cfg):
    """SD verticale descriptive du gaze."""
    return float(w.gaze_y_clean.std(ddof=1))


def _inside_roi(w, cfg):
    return (w.gaze_valid & w.gaze_x_clean.between(cfg.cue_roi_center_x - cfg.cue_roi_half_width_px,
                                               cfg.cue_roi_center_x + cfg.cue_roi_half_width_px)
            & w.gaze_y_clean.between(cfg.cue_roi_center_y - cfg.cue_roi_half_height_px,
                                    cfg.cue_roi_center_y + cfg.cue_roi_half_height_px))


@register_metric("gaze_aversion_fraction", "gaze", "fraction_valid_gaze", roi=True)
def gaze_aversion_fraction(w, baseline, blinks, events, cfg):
    """Fraction des samples gaze valides hors de la ROI rectangulaire."""
    if cfg.cue_roi_center_x is None or cfg.cue_roi_center_y is None or not w.gaze_valid.any():
        return np.nan
    return float(1 - _inside_roi(w, cfg).sum() / w.gaze_valid.sum())


@register_metric("gaze_roi_dwell_s", "gaze", "s", roi=True)
def gaze_roi_dwell(w, baseline, blinks, events, cfg):
    """Temps de gaze valide dans la ROI, non extrapolé aux samples absents."""
    return float(_inside_roi(w, cfg).sum() / cfg.sampling_rate_hz)


@register_metric("missing_fraction", "qc", "fraction_expected_samples")
def missing_fraction(w, baseline, blinks, events, cfg):
    """Fraction d'acquisition manquante, trous d'horloge compris."""
    return float(w.attrs.get("acquisition_missing_fraction", w.pupil_missing_acquisition.mean()))


def _stream_available(events, kind, blinks):
    declared = events.attrs.get("eyelink_metadata", {}).get("event_streams", [])
    return (kind in declared or (not events.empty and events.event_type.eq(kind).any())
            or (kind == "EBLINK" and not blinks.empty))


def extract_metrics(windows, preprocessed, blink_intervals, events, cfg: Config):
    """Une ligne par fenêtre × métrique, inclusions et prérequis explicites."""
    rows = []
    for _, meta in windows.iterrows():
        roi_fields = {k: meta[k] for k in ("cue_roi_center_x", "cue_roi_center_y",
                      "cue_roi_half_width_px", "cue_roi_half_height_px") if k in meta and pd.notna(meta[k])}
        local_cfg = replace(cfg, **roi_fields)
        evaluated = evaluate_window(meta, preprocessed, local_cfg)
        win, q, bq = evaluated["window"], evaluated["qc"], evaluated["baseline_qc"]
        baseline_valid = not evaluated["baseline_reasons"]
        names = meta.get("metrics", LEGACY_METRICS)
        for name in names:
            spec = METRIC_DEFINITIONS[name]
            reasons = list(evaluated["pupil_reasons"] if spec.family == "pupil" else
                           evaluated["gaze_reasons"] if spec.family in {"gaze", "saccade", "fixation"}
                           else evaluated["common_reasons"])
            if spec.family == "blink" and q["blink_tracking_fraction"] < cfg.min_gaze_valid_fraction:
                reasons.append("low_blink_tracking_exposure")
            if spec.baseline and not baseline_valid:
                reasons.append("invalid_baseline")
            if spec.event_type and not _stream_available(events, spec.event_type, blink_intervals):
                reasons.append("event_stream_unavailable")
            if spec.roi and (local_cfg.cue_roi_center_x is None or local_cfg.cue_roi_center_y is None):
                reasons.append("roi_not_configured")
            if not spec.validated:
                reasons.append("exploratory_small_saccades_not_validated")
            value = np.nan
            if not reasons:
                value = METRIC_REGISTRY[name](win, evaluated["baseline_value"], blink_intervals, events, local_cfg)
                if not np.isfinite(value):
                    reasons.append("undefined_no_events_or_samples")
            # Conserver les métadonnées de projet ; éviter les longues traces curseur FGAT.
            context = ({k: v for k, v in meta.items() if k != "metrics"}
                       if "project_id" in meta else {k: meta.get(k, np.nan) for k in (
                           "subject_id", "run_id", "trial_n", "induction_block_id", "moment",
                           "window_x_s", "actual_duration_s", "stim_id", "stim_image", "stim_sound",
                           "trial_n_global", "group_5cat", "group_valence", "group_arousal",
                           "emo_grid_valence", "emo_grid_arousal", "LTask_rating",
                           "start_eyelink_ms", "end_eyelink_ms", "baseline_start_eyelink_ms",
                           "baseline_end_eyelink_ms", "baseline_source", "requested_duration_s")})
            context["cue_text"] = meta.get("cue_text", meta.get("cues", np.nan))
            for label, left, right in (("fgat_search_duration_s", "fgat_cue_onset", "fgat_input_time"),
                                        ("LTask_response_duration_s", "LTask_onset", "LTask_validation")):
                if left in meta and right in meta:
                    context[label] = float(meta[right] - meta[left])
            rows.append({**context, "metric": name, "value": value, "unit": spec.unit,
                         "metric_family": spec.family, "moment_valid": not reasons,
                         "moment_exclusion_reason": ";".join(dict.fromkeys(reasons)),
                         "pupil_window_valid": not evaluated["pupil_reasons"],
                         "gaze_window_valid": not evaluated["gaze_reasons"],
                         "baseline_valid": baseline_valid, "baseline_value": evaluated["baseline_value"],
                         "baseline_exclusion_reason": ";".join(evaluated["baseline_reasons"]),
                         "baseline_missing_fraction": bq["acquisition_missing_fraction"],
                         "baseline_observed_valid_fraction": bq["observed_valid_fraction"],
                         "missing_fraction_window": q["acquisition_missing_fraction"], **q})
    return pd.DataFrame(rows)


def extract_h4_gaze_samples(windows, preprocessed, cfg):
    """Samples H4 avec le même QC de fenêtre gaze que les métriques."""
    rows = []
    for _, meta in windows.loc[windows.moment.eq("H4_full_search_gaze")].iterrows():
        assessed = evaluate_window(meta, preprocessed, cfg)
        if assessed["gaze_reasons"]:
            continue
        gaze = assessed["window"].loc[lambda d: d.gaze_valid,
                                       ["timestamp_ms", "gaze_x_clean", "gaze_y_clean"]].copy()
        step = max(1, int(np.ceil(len(gaze) / cfg.gaze_plot_max_samples_per_trial)))
        gaze = gaze.iloc[::step].rename(columns={"gaze_x_clean": "gaze_x", "gaze_y_clean": "gaze_y"})
        for key in ("subject_id", "run_id", "trial_n", "group_5cat", "group_valence", "group_arousal"):
            gaze[key] = meta[key]
        gaze["sample_weight"] = 1.0 / max(1, len(gaze))
        rows.append(gaze)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
