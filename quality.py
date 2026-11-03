"""Découpage [début, fin), couverture temporelle et QC partagés."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def segment(samples, start_ms, end_ms):
    if not np.isfinite(start_ms) or not np.isfinite(end_ms) or end_ms <= start_ms:
        return samples.iloc[0:0].copy()
    time = samples.timestamp_ms.to_numpy(dtype=float)
    a, b = np.searchsorted(time, [start_ms, end_ms], side="left")
    return samples.iloc[a:b].copy()


def qc_segment(data, start_ms, end_ms, cfg: Config):
    duration = float(end_ms - start_ms)
    if not np.isfinite(duration) or duration <= 0:
        duration = 0.0
    expected = duration * cfg.sampling_rate_hz / 1000.0
    empty = dict(observed_valid_fraction=0.0, artifact_fraction=1.0,
                 interpolated_fraction=0.0, acquisition_missing_fraction=1.0,
                 gaze_valid_fraction=0.0, blink_tracking_fraction=0.0,
                 longest_gap_ms=duration, sampling_coverage_fraction=0.0,
                 n_samples=0, expected_samples=expected, gaze_exposure_s=0.0,
                 blink_exposure_s=0.0, pupil_integrated_duration_s=0.0)
    if data.empty or expected <= 0:
        return empty
    dt = 1000.0 / cfg.sampling_rate_hz
    time = data.timestamp_ms.to_numpy(dtype=float)
    observed = data.pupil_observed_valid.to_numpy(dtype=bool)
    artifact = ~observed
    pupil = data.pupil_clean.to_numpy(dtype=float)
    gaze = data.gaze_valid.to_numpy(dtype=bool)
    blink = data.event_blink_mask.to_numpy(dtype=bool)
    missing = data.pupil_missing_acquisition.to_numpy(dtype=bool)
    coverage = min(1.0, len(data) / expected)
    starts = np.flatnonzero(np.diff(np.r_[0, artifact.astype(int), 0]) == 1)
    stops = np.flatnonzero(np.diff(np.r_[0, artifact.astype(int), 0]) == -1)
    longest_artifact = max((time[b-1] - time[a] + dt for a, b in zip(starts, stops)), default=0.0)
    intervals = np.diff(time)
    good_pairs = np.isfinite(pupil[:-1]) & np.isfinite(pupil[1:]) & (intervals <= 1.5 * dt)
    if "segment_id" in data:
        good_pairs &= np.diff(data.segment_id.to_numpy()) == 0
    gaze_fraction = float(gaze.mean() * coverage)
    blink_fraction = float((~missing | blink).mean() * coverage)
    return dict(
        observed_valid_fraction=float(observed.mean() * coverage),
        artifact_fraction=float(1 - observed.mean() * coverage),
        interpolated_fraction=float(data.pupil_interpolated.mean() * coverage),
        acquisition_missing_fraction=float(missing.mean() * coverage + 1 - coverage),
        gaze_valid_fraction=gaze_fraction, blink_tracking_fraction=blink_fraction,
        longest_gap_ms=float(max(longest_artifact, max(intervals - dt, default=0.0),
                                 time[0] - start_ms, end_ms - time[-1] - dt, 0.0)),
        sampling_coverage_fraction=coverage, n_samples=len(data), expected_samples=expected,
        gaze_exposure_s=gaze_fraction * duration / 1000.0,
        blink_exposure_s=blink_fraction * duration / 1000.0,
        pupil_integrated_duration_s=float(intervals[good_pairs].sum() / 1000.0),
    )


def _flag(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def evaluate_window(meta, samples, cfg: Config):
    """Une définition de validité pour métriques et courbes temporelles."""
    start, end = float(meta["start_eyelink_ms"]), float(meta["end_eyelink_ms"])
    left = float(meta.get("baseline_start_eyelink_ms", np.nan))
    right = float(meta.get("baseline_end_eyelink_ms", np.nan))
    win, base = segment(samples, start, end), segment(samples, left, right)
    q, bq = qc_segment(win, start, end, cfg), qc_segment(base, left, right, cfg)
    declared_baseline = meta.get("baseline_definition_reason", "")
    baseline_reasons = [declared_baseline] if isinstance(declared_baseline, str) and declared_baseline else []
    if not np.isfinite(left) or not np.isfinite(right):
        baseline_reasons.append("baseline_not_available")
    elif (right - left) / 1000.0 + 1e-9 < cfg.minimum_baseline_duration_s:
        baseline_reasons.append("baseline_too_short")
    if bq["observed_valid_fraction"] < cfg.min_observed_fraction_baseline:
        baseline_reasons.append("low_observed_baseline_fraction")
    if bq["acquisition_missing_fraction"] > cfg.max_baseline_missing_fraction:
        baseline_reasons.append("missing_baseline_acquisition")
    if bq["sampling_coverage_fraction"] < 1 - cfg.max_window_missing_sampling_fraction:
        baseline_reasons.append("missing_baseline_sampling_coverage")
    baseline_value = (float(base.loc[base.pupil_observed_valid, "pupil_clean"].median())
                      if not baseline_reasons else np.nan)
    if not baseline_reasons and not np.isfinite(baseline_value):
        baseline_reasons.append("nonfinite_baseline")
    common = []
    declared_reason = meta.get("window_exclusion_reason", "")
    if isinstance(declared_reason, str) and declared_reason:
        common.extend(declared_reason.split(";"))
    if _flag(meta.get("timeout", False)) or _flag(meta.get("is_resumed", False)):
        common.append("timeout_or_resumed")
    if not _flag(meta.get("duration_valid", end - start >= cfg.minimum_usable_window_s * 1000)):
        common.append("window_too_short")
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        common.append("invalid_window_bounds")
    if not q["n_samples"]:
        common.append("no_samples")
    if q["sampling_coverage_fraction"] < 1 - cfg.max_window_missing_sampling_fraction:
        common.append("missing_sampling_coverage")
    pupil_reasons = common.copy()
    if q["observed_valid_fraction"] < cfg.min_observed_fraction_window:
        pupil_reasons.append("low_observed_pupil_fraction")
    if q["interpolated_fraction"] > cfg.max_interpolated_fraction_window:
        pupil_reasons.append("excessive_pupil_interpolation")
    if q["longest_gap_ms"] > cfg.max_longest_gap_window_ms:
        pupil_reasons.append("long_artifact_gap")
    gaze_reasons = common.copy()
    if q["gaze_valid_fraction"] < cfg.min_gaze_valid_fraction:
        gaze_reasons.append("low_valid_gaze_fraction")
    win.attrs.update(window_start_ms=start, window_end_ms=end,
                     anchor_eyelink_ms=float(meta.get("anchor_eyelink_ms", start)),
                     sampling_rate_hz=cfg.sampling_rate_hz, **q)
    return dict(window=win, baseline_segment=base, qc=q, baseline_qc=bq,
                baseline_value=baseline_value, baseline_reasons=baseline_reasons,
                common_reasons=common, pupil_reasons=pupil_reasons,
                gaze_reasons=gaze_reasons)
