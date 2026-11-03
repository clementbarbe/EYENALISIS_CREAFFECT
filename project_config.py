"""Configuration versionnée du premier mode générique (JSON ou YAML)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import math

from config import DEFAULT_CONFIG
from metrics import METRIC_REGISTRY, METRIC_DEFINITIONS

PARAMETERS = {
    "eye", "sampling_rate_hz", "blink_padding_before_ms", "blink_padding_after_ms",
    "min_plausible_pupil", "pupil_speed_mad_threshold", "pupil_outlier_mad_threshold",
    "pupil_reject_speed_outliers", "interpolation_max_gap_ms", "lowpass_enabled",
    "lowpass_cutoff_hz", "lowpass_order", "min_observed_fraction_window",
    "max_interpolated_fraction_window", "min_observed_fraction_baseline",
    "max_baseline_missing_fraction", "max_longest_gap_window_ms",
    "max_window_missing_sampling_fraction", "min_gaze_valid_fraction",
    "gaze_screen_width_px", "gaze_screen_height_px", "minimum_usable_window_s",
    "minimum_baseline_duration_s", "microsaccade_max_amplitude_deg",
    "min_rows_for_model", "min_subjects_for_model", "alpha", "fdr_method",
}


def keys(mapping, allowed, context):
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {context} fields: {sorted(unknown)}")


def number(value, context, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"Invalid {context}: {value}")
    return float(value)


def _selector(spec, context, *, anchor=False, depth=0):
    if depth > 8:
        raise ValueError(f"{context}: selectors nested too deeply")
    keys(spec, {"event", "edge", "offset_s", "occurrence", "anchor_offset_s", "min", "max"}, context)
    modes = [key for key in ("event", "anchor_offset_s", "min", "max") if key in spec]
    if len(modes) != 1:
        raise ValueError(f"{context} requires exactly one of event/anchor_offset_s/min/max")
    if anchor and modes != ["event"]:
        raise ValueError(f"{context} must identify an event")
    if modes[0] in {"min", "max"}:
        if set(spec) != {modes[0]} or not isinstance(spec[modes[0]], list) or not spec[modes[0]]:
            raise ValueError(f"Invalid {context} min/max")
        for child in spec[modes[0]]:
            _selector(child, context, depth=depth + 1)
    elif modes[0] == "anchor_offset_s":
        if set(spec) != {"anchor_offset_s"}:
            raise ValueError(f"Unexpected fields in {context}")
        number(spec["anchor_offset_s"], context)
    else:
        if not isinstance(spec["event"], str) or not spec["event"]:
            raise ValueError(f"Invalid event in {context}")
        if spec.get("edge", "onset") not in {"onset", "offset"}:
            raise ValueError(f"Invalid edge in {context}")
        occurrence = spec.get("occurrence", "only")
        choices = {"first", "last", "only"} | ({"each"} if anchor else {"last_before_anchor"})
        if not (isinstance(occurrence, str) and occurrence in choices
                or type(occurrence) is int and occurrence >= 1):
            raise ValueError(f"Invalid occurrence in {context}")
        number(spec.get("offset_s", 0), context)


def validate_project(spec):
    keys(spec, {"version", "project_id", "description", "recordings_manifest", "parameters",
                "events", "synchronization", "baselines", "windows", "rois",
                "timecourses", "hypotheses"}, "project")
    if type(spec.get("version")) is not int or spec["version"] != 1:
        raise ValueError("Project version must be 1; the design-only proposal is not executable")
    for field in ("project_id", "recordings_manifest"):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise ValueError(f"Missing {field}")
    parameters = spec.get("parameters", {})
    keys(parameters, PARAMETERS, "parameters")
    for key, value in parameters.items():
        default = getattr(DEFAULT_CONFIG, key)
        if isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
        else:
            number(value, key, nonnegative=True)
            if isinstance(default, int) and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            if ("fraction" in key or key == "alpha") and not 0 <= value <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")
    cfg = replace(DEFAULT_CONFIG, **parameters)
    if cfg.eye not in {"L", "R"}:
        raise ValueError("eye must be L or R")
    for field in ("sampling_rate_hz", "gaze_screen_width_px", "gaze_screen_height_px",
                  "lowpass_cutoff_hz", "lowpass_order", "min_subjects_for_model", "min_rows_for_model"):
        number(getattr(cfg, field), field, positive=True)
    if cfg.min_subjects_for_model < 2:
        raise ValueError("Clustered models require at least 2 subjects")
    if cfg.fdr_method not in {"fdr_bh", "fdr_by", "holm", "bonferroni"}:
        raise ValueError("Unsupported multiplicity correction")
    events = spec.get("events", {})
    keys(events, {"columns", "mapping"}, "events")
    keys(events.get("columns", {}), {"trial_id", "event_id", "event_type", "onset", "duration"}, "event columns")
    for mapping in (events.get("columns", {}), events.get("mapping", {})):
        if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                for k, v in mapping.items()):
            raise ValueError("Event mappings must contain strings")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Event mappings must be one-to-one")
    sync = spec.get("synchronization", {})
    keys(sync, {"mode", "max_residual_ms", "drift_warning_ppm"}, "synchronization")
    if sync.get("mode") not in {"eyelink_clock", "offset", "paired_anchors"}:
        raise ValueError("Declare synchronization.mode: eyelink_clock, offset or paired_anchors")
    number(sync.get("max_residual_ms", 5.0), "max_residual_ms", nonnegative=True)
    number(sync.get("drift_warning_ppm", 150.0), "drift_warning_ppm", nonnegative=True)
    baselines = spec.get("baselines", {})
    if not isinstance(baselines, dict):
        raise ValueError("baselines must be a mapping")
    for name, base in baselines.items():
        keys(base, {"event", "duration_s", "occurrence", "end_edge", "offset_s", "clipping_policy"}, f"baseline {name}")
        _selector({"event": base.get("event"), "occurrence": base.get("occurrence", "last_before_anchor"),
                   "edge": base.get("end_edge", "offset"), "offset_s": base.get("offset_s", 0)}, f"baseline {name}")
        number(base.get("duration_s"), f"baseline {name} duration", positive=True)
        if base.get("clipping_policy", "reject") not in {"reject", "truncate"}:
            raise ValueError(f"Invalid baseline clipping policy: {name}")
    rois = spec.get("rois", {})
    if not isinstance(rois, dict):
        raise ValueError("rois must be a mapping")
    for name, roi in rois.items():
        keys(roi, {"center_x", "center_y", "half_width_px", "half_height_px"}, f"ROI {name}")
        for field in ("center_x", "center_y", "half_width_px", "half_height_px"):
            number(roi.get(field), f"ROI {name} {field}", positive=field.startswith("half"))
    windows = spec.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("At least one window is required")
    ids = []
    for window in windows:
        keys(window, {"id", "anchor", "start", "end", "sliding", "lower_bound", "upper_bound",
                      "clipping_policy", "min_duration_s", "baseline", "metrics", "roi", "role"}, "window")
        if not isinstance(window.get("id"), str) or not window["id"]:
            raise ValueError("Window id must be a nonempty string")
        ids.append(window["id"])
        _selector(window.get("anchor"), f"{window['id']} anchor", anchor=True)
        if window.get("role", "exploratory") not in {"primary", "exploratory", "sensitivity"}:
            raise ValueError("Invalid window role")
        if window.get("clipping_policy", "reject") not in {"reject", "truncate"}:
            raise ValueError("Invalid window clipping policy")
        number(window.get("min_duration_s", cfg.minimum_usable_window_s), "min_duration_s", nonnegative=True)
        if "sliding" in window:
            if "start" in window or "end" in window:
                raise ValueError("sliding replaces window start/end")
            sliding = window["sliding"]
            keys(sliding, {"range_s", "widths_s", "step_s"}, "sliding")
            span, widths = sliding.get("range_s"), sliding.get("widths_s")
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("sliding.range_s needs two bounds")
            for value in span:
                number(value, "sliding bound")
            if span[1] <= span[0] or not isinstance(widths, list) or not widths:
                raise ValueError("Invalid sliding range/widths")
            for width in widths:
                number(width, "sliding width", positive=True)
                if width > span[1] - span[0]:
                    raise ValueError("Sliding width exceeds range")
            number(sliding.get("step_s"), "sliding step", positive=True)
            count = sum(math.floor((span[1] - span[0] - w) / sliding["step_s"] + 1e-9) + 1 for w in widths)
            if count > 10000 or len(widths) != len(set(widths)):
                raise ValueError("Too many sliding windows or duplicate widths")
        else:
            for field in ("start", "end"):
                _selector(window.get(field), f"{window['id']} {field}")
        for field in ("lower_bound", "upper_bound"):
            if field in window:
                _selector(window[field], field)
        metrics = window.get("metrics")
        if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) for m in metrics):
            raise ValueError("Every window requires a list of metrics")
        if len(metrics) != len(set(metrics)) or set(metrics) - METRIC_REGISTRY.keys():
            raise ValueError(f"Unknown or duplicate metrics in {window['id']}: {metrics}")
        if window.get("baseline") is not None and window["baseline"] not in baselines:
            raise ValueError("Unknown baseline")
        if window.get("roi") is not None and window["roi"] not in rois:
            raise ValueError("Unknown ROI")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate window ids")
    tc = spec.get("timecourses", {})
    keys(tc, {"windows", "bin_width_s"}, "timecourses")
    if not isinstance(tc.get("windows", []), list) or set(tc.get("windows", [])) - set(ids):
        raise ValueError("Unknown timecourse windows")
    number(tc.get("bin_width_s", 0.05), "bin_width_s", positive=True)
    hypotheses = spec.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        raise ValueError("hypotheses must be a list")
    seen = set()
    for hyp in hypotheses:
        keys(hyp, {"id", "description", "window", "metric", "predictor", "contrast",
                   "numeric_covariates", "categorical_covariates", "family"}, "hypothesis")
        if not isinstance(hyp.get("id"), str) or not hyp["id"] or hyp["id"] in seen:
            raise ValueError("Hypothesis ids must be unique nonempty strings")
        seen.add(hyp["id"])
        win = next((w for w in windows if w["id"] == hyp.get("window")), None)
        if win is None or hyp.get("metric") not in win["metrics"]:
            raise ValueError("Hypothesis must select a metric computed in its window")
        unit = METRIC_DEFINITIONS[hyp["metric"]].unit
        if unit == "count" or unit.startswith(("fraction", "1/s")):
            raise ValueError("Project v1 inference supports continuous outcomes; count/proportion/rate models are not implemented")
        if "sliding" in win or win["anchor"].get("occurrence") == "each":
            raise ValueError("Inferential models for sliding/repeated-anchor windows are not implemented")
        if not isinstance(hyp.get("predictor"), str) or not hyp["predictor"]:
            raise ValueError("Hypothesis requires a categorical predictor")
        contrast = hyp.get("contrast")
        if not isinstance(contrast, list) or len(contrast) != 2 or contrast[0] == contrast[1] or any(not isinstance(c, str) for c in contrast):
            raise ValueError("contrast must be [level_positive, level_reference]")
        if not isinstance(hyp.get("family", "default"), str):
            raise ValueError("Hypothesis family must be a string")
        for field in ("numeric_covariates", "categorical_covariates"):
            values = hyp.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
                raise ValueError(f"Invalid {field}")
    return cfg


def load_project(path):
    path = Path(path).resolve()
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        spec = json.loads(content)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML needs PyYAML; install requirements or use a .json project") from exc
        spec = yaml.safe_load(content)
    cfg = validate_project(spec)
    return spec, cfg
