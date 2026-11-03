"""Événements longs normalisés et fenêtres déclaratives, indépendants de FGAT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class EventResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ClockMap:
    offset_ms: float
    slope: float = 1.0
    n_anchors: int = 0
    max_residual_ms: float = 0.0
    quality: str = "declared_common_clock"

    def to_ms(self, seconds):
        return self.offset_ms + 1000.0 * self.slope * seconds


def read_table(path):
    path = Path(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                       dtype=str, keep_default_na=False)


def fit_clock(recording, synchronization, manifest_dir):
    mode = synchronization["mode"]
    if mode == "eyelink_clock":
        return ClockMap(0.0)
    if mode == "offset":
        try:
            offset = float(recording["offset_ms"])
        except (ValueError, KeyError) as exc:
            raise ValueError("offset mode requires offset_ms in each manifest row") from exc
        if not np.isfinite(offset):
            raise ValueError("Nonfinite clock offset")
        return ClockMap(offset, quality="offset_declared_drift_unverified")
    if not recording.get("sync_path"):
        raise ValueError("paired_anchors mode requires sync_path in the manifest")
    anchors = read_table(Path(manifest_dir) / recording["sync_path"])
    if not {"reference_s", "eyelink_ms"}.issubset(anchors):
        raise ValueError("Anchor table requires reference_s and eyelink_ms")
    x = pd.to_numeric(anchors.reference_s, errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(anchors.eyelink_ms, errors="raise").to_numpy(dtype=float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Empty or nonfinite clock anchors")
    if len(x) > 1 and (not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)):
        raise ValueError("Paired anchors must increase strictly, without duplicates")
    if len(x) == 1:
        return ClockMap(float(y[0] - 1000 * x[0]), n_anchors=1, quality="single_anchor_drift_unverified")
    dx = (x - x.mean()) * 1000.0
    slope = float(np.dot(dx, y - y.mean()) / np.dot(dx, dx))
    offset = float(y.mean() - 1000.0 * slope * x.mean())
    residual = float(np.abs(y - (offset + slope * x * 1000)).max())
    if slope <= 0 or residual > synchronization.get("max_residual_ms", 5.0):
        raise ValueError(f"Clock anchor residual too large: {residual:.6g} ms")
    drift = abs(slope - 1) * 1e6
    quality = "drift_warning" if drift > synchronization.get("drift_warning_ppm", 150.0) else "paired_anchors"
    return ClockMap(offset, slope, len(x), residual, quality)


def read_task_events(path, event_spec, recording):
    events = read_table(path)
    columns = event_spec.get("columns", {})
    if any(source not in events for source in columns.values()):
        raise ValueError("An events.columns source column is missing")
    events = events.rename(columns={v: k for k, v in columns.items()})
    if events.columns.duplicated().any():
        raise ValueError("Event column mapping creates duplicate names")
    required = {"trial_id", "event_type", "onset"}
    if not required.issubset(events) or events.empty:
        raise ValueError("Nonempty event table requires trial_id, event_type, onset")
    for key in ("participant_id", "session_id", "run_id"):
        if key in events and not events[key].eq(str(recording[key])).all():
            raise ValueError(f"Event table {key} differs from manifest")
    if events.trial_id.eq("").any() or events.event_type.eq("").any():
        raise ValueError("Empty trial_id or event_type")
    if "event_id" not in events:
        events["event_id"] = [f"event-{i:06}" for i in range(len(events))]
    if events.event_id.eq("").any() or events.event_id.duplicated().any():
        raise ValueError("Event ids must be unique and nonempty within a recording")
    if "duration" not in events:
        events["duration"] = "0"
    for key in ("onset", "duration"):
        events[key] = pd.to_numeric(events[key], errors="raise").astype(float)
        if not np.isfinite(events[key]).all():
            raise ValueError(f"Nonfinite event {key}")
    if events.duration.lt(0).any():
        raise ValueError("Event durations must be nonnegative")
    # mapping = nom canonique -> nom dans les données.
    events.event_type = events.event_type.replace({v: k for k, v in event_spec.get("mapping", {}).items()})
    for key in ("timeout", "is_resumed"):
        if key in events:
            lower = events[key].str.strip().str.lower()
            if not lower.isin({"", "0", "1", "false", "true", "no", "yes"}).all():
                raise ValueError(f"Unrecognized boolean in {key}")
            events[key] = lower.map({"0": False, "1": True, "false": False,
                                     "true": True, "no": False, "yes": True, "": np.nan})
    return events.sort_values("onset", kind="stable").reset_index(drop=True)


def _choose(events, selector, anchor=None):
    subset = events.loc[events.event_type.eq(selector["event"])]
    occurrence = selector.get("occurrence", "only")
    if occurrence == "last_before_anchor":
        if anchor is None:
            raise EventResolutionError("last_before_anchor_requires_anchor")
        subset = subset.loc[(subset.onset + subset.duration).le(anchor + 1e-10)]
        occurrence = "last"
    if subset.empty:
        raise EventResolutionError(f"missing_event:{selector['event']}")
    if occurrence == "only" and len(subset) != 1:
        raise EventResolutionError(f"ambiguous_event:{selector['event']}")
    index = -1 if occurrence == "last" else occurrence - 1 if type(occurrence) is int else 0
    if index >= len(subset):
        raise EventResolutionError(f"missing_event_occurrence:{selector['event']}")
    return subset.iloc[index]


def _edge(event, selector):
    return float(event.onset + (event.duration if selector.get("edge", "onset") == "offset" else 0)
                 + selector.get("offset_s", 0))


def _resolve(selector, events, anchor, anchor_event, anchor_spec):
    if "anchor_offset_s" in selector:
        return anchor + selector["anchor_offset_s"]
    for operator, function in (("min", min), ("max", max)):
        if operator in selector:
            return function(_resolve(child, events, anchor, anchor_event, anchor_spec)
                            for child in selector[operator])
    # Une référence à l'événement d'ancrage vise l'occurrence sélectionnée.
    event = (anchor_event if selector["event"] == anchor_spec["event"] and "occurrence" not in selector
             else _choose(events, selector, anchor))
    return _edge(event, selector)


def _variants(window):
    if "sliding" not in window:
        yield "main", window["start"], window["end"]
        return
    sliding = window["sliding"]
    left, right = sliding["range_s"]
    step = sliding["step_s"]
    for width in sliding["widths_s"]:
        count = int(np.floor((right - left - width) / step + 1e-9)) + 1
        for i in range(count):
            start = left + i * step
            yield f"w{width:g}_i{i}", {"anchor_offset_s": start}, {"anchor_offset_s": start + width}


def _baseline(events, spec, anchor, clock):
    if spec is None:
        return dict(baseline_start_eyelink_ms=np.nan, baseline_end_eyelink_ms=np.nan,
                    baseline_definition_reason="baseline_not_configured", baseline_source="none")
    try:
        selector = {"event": spec["event"], "occurrence": spec.get("occurrence", "last_before_anchor"),
                    "edge": spec.get("end_edge", "offset"), "offset_s": spec.get("offset_s", 0)}
        event = _choose(events, selector, anchor)
        end = _edge(event, selector)
        start = end - spec["duration_s"]
        reason = ""
        if end > anchor + 1e-10:
            reason = "baseline_after_anchor"
        if start < event.onset - 1e-10 or end > event.onset + event.duration + 1e-10:
            if spec.get("clipping_policy", "reject") == "reject":
                reason = "baseline_outside_event"
            else:
                start, end = max(start, event.onset), min(end, event.onset + event.duration)
        return dict(baseline_start_eyelink_ms=clock.to_ms(start), baseline_end_eyelink_ms=clock.to_ms(end),
                    baseline_start_behavior_s=start, baseline_end_behavior_s=end,
                    baseline_requested_duration_s=spec["duration_s"], baseline_source=event.event_id,
                    baseline_definition_reason=reason)
    except EventResolutionError as exc:
        return dict(baseline_start_eyelink_ms=np.nan, baseline_end_eyelink_ms=np.nan,
                    baseline_definition_reason=str(exc), baseline_source="missing")


def build_project_windows(events, project, recording, clock, cfg):
    rows = []
    structural = {"event_id", "event_type", "onset", "duration", "participant_id", "session_id", "run_id"}
    reserved = {"subject_id", "trial_n", "moment", "project_id", "window_id", "window_spec_id",
                "metrics", "metric", "value", "unit", "moment_valid", "window_exclusion_reason"}
    if reserved.intersection(events.columns):
        raise ValueError(f"Reserved event metadata columns: {sorted(reserved.intersection(events.columns))}")
    for trial_id, trial in events.groupby("trial_id", sort=False):
        context = {}
        for key in set(events) - structural - {"trial_id"}:
            values = trial[key]
            values = values.loc[values.ne("") & values.notna()].drop_duplicates()
            if len(values) > 1:
                raise ValueError(f"Trial {trial_id} has inconsistent metadata: {key}")
            context[key] = values.iloc[0] if len(values) else np.nan
        for window in project["windows"]:
            anchor_spec = window["anchor"]
            error = ""
            try:
                anchors = (list(trial.loc[trial.event_type.eq(anchor_spec["event"])].iterrows())
                           if anchor_spec.get("occurrence") == "each"
                           else [(None, _choose(trial, anchor_spec))])
                if not anchors:
                    raise EventResolutionError(f"missing_event:{anchor_spec['event']}")
            except EventResolutionError as exc:
                anchors, error = [(None, None)], str(exc)
            for _, anchor_event in anchors:
                anchor = _edge(anchor_event, anchor_spec) if anchor_event is not None else np.nan
                base = _baseline(trial, project.get("baselines", {}).get(window.get("baseline")), anchor, clock)
                for variant, start_spec, end_spec in _variants(window):
                    reason, start, end = error, np.nan, np.nan
                    requested_start, requested_end = np.nan, np.nan
                    truncated = False
                    try:
                        if reason:
                            raise EventResolutionError(reason)
                        start = requested_start = _resolve(start_spec, trial, anchor, anchor_event, anchor_spec)
                        end = requested_end = _resolve(end_spec, trial, anchor, anchor_event, anchor_spec)
                        if "lower_bound" in window:
                            start = max(start, _resolve(window["lower_bound"], trial, anchor, anchor_event, anchor_spec))
                        if "upper_bound" in window:
                            end = min(end, _resolve(window["upper_bound"], trial, anchor, anchor_event, anchor_spec))
                        truncated = abs(start - requested_start) > 1e-10 or abs(end - requested_end) > 1e-10
                        if truncated and window.get("clipping_policy", "reject") == "reject":
                            reason = "window_requires_truncation"
                        if end <= start:
                            reason = "nonpositive_window_duration"
                    except EventResolutionError as exc:
                        reason = str(exc)
                    duration = max(0.0, end - start) if np.isfinite(end - start) else 0.0
                    win_id = window["id"] + (f"__{variant}" if variant != "main" else "")
                    roi = project.get("rois", {}).get(window.get("roi"), {})
                    roi_fields = {"cue_roi_center_x": roi.get("center_x"), "cue_roi_center_y": roi.get("center_y"),
                                  "cue_roi_half_width_px": roi.get("half_width_px"),
                                  "cue_roi_half_height_px": roi.get("half_height_px")}
                    rows.append({**context, **recording, **base, **roi_fields,
                                 "project_id": project["project_id"], "subject_id": recording["participant_id"],
                                 "trial_id": trial_id, "trial_n": trial_id,
                                 "induction_block_id": context.get("block_id", trial_id),
                                 "window_spec_id": window["id"], "window_id": win_id, "moment": win_id,
                                 "analysis_variant_id": variant, "analysis_role": window.get("role", "exploratory"),
                                 "anchor_event_id": anchor_event.event_id if anchor_event is not None else "",
                                 "anchor_eyelink_ms": clock.to_ms(anchor), "metrics": window["metrics"],
                                 "clock_slope": clock.slope,
                                 "start_behavior_s": start, "end_behavior_s": end,
                                 "start_eyelink_ms": clock.to_ms(start), "end_eyelink_ms": clock.to_ms(end),
                                 "requested_start_behavior_s": requested_start, "requested_end_behavior_s": requested_end,
                                 "requested_duration_s": requested_end - requested_start,
                                 "actual_duration_s": duration, "truncated": truncated,
                                 "actual_duration_eyelink_s": duration * clock.slope,
                                 "duration_valid": duration + 1e-9 >= window.get("min_duration_s", cfg.minimum_usable_window_s),
                                 "window_exclusion_reason": reason, "window_x_s": duration,
                                 "timeout": context.get("timeout", False), "is_resumed": context.get("is_resumed", False)})
    return pd.DataFrame(rows)
