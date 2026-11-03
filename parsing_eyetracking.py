"""Lecture EyeLink ASC : sélection d'œil, métadonnées et segments d'acquisition."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "event_type", "timestamp_ms", "start_ms", "end_ms", "duration_ms", "eye",
    "button_code", "button_state", "x_start", "y_start", "x_end", "y_end",
    "x_mean", "y_mean", "pupil_mean", "amplitude_deg", "peak_velocity", "raw_line",
]
SAMPLE_COLUMNS = ["timestamp_ms", "gaze_x", "gaze_y", "pupil_size", "segment_id"]
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_END_FIELDS = {
    "EBLINK": ["start_ms", "end_ms", "duration_ms"],
    "EFIX": ["start_ms", "end_ms", "duration_ms", "x_mean", "y_mean", "pupil_mean"],
    "ESACC": ["start_ms", "end_ms", "duration_ms", "x_start", "y_start",
              "x_end", "y_end", "amplitude_deg", "peak_velocity"],
}


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_asc(asc_path: Path, eye: str = "R", *, strict: bool = False):
    """Retourne samples et événements de l'œil choisi, métadonnées dans attrs.

    ``strict`` impose SAMPLES/GAZE, RATE et PUPIL. Hors mode strict, les anciens
    exports monoculaires sans en-têtes restent lisibles. Les changements de
    format et les horloges réinitialisées/dupliquées sont refusés.
    """
    asc_path = Path(asc_path)
    eye = eye.upper()
    if eye not in {"L", "R"}:
        raise ValueError("eye must be L or R")
    samples, events = [], []
    metadata = {"selected_eye": eye, "recorded_eyes": [], "sampling_rate_hz": None,
                "coordinate_system": None, "pupil_measure": None,
                "tracker_filter": None, "gaze_coords": None, "event_streams": [], "segments": 0}
    sample_eyes = []
    segment = -1
    has_start = False
    recording = True
    sample_header_seen = False
    digest = hashlib.sha256()

    def set_consistent(key, value):
        old = metadata[key]
        if old is not None and old != value:
            raise ValueError(f"ASC changes {key}: {old!r} -> {value!r}")
        metadata[key] = value

    with asc_path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            line = raw.decode("latin-1").strip()
            if not line or line.startswith("**"):
                continue
            tokens = line.split()
            kind = tokens[0]
            if kind == "START":
                has_start, recording = True, True
                segment += 1
                sample_eyes = [e for e, word in (("L", "LEFT"), ("R", "RIGHT")) if word in tokens]
                continue
            if kind == "END":
                recording = False
                continue
            if kind == "PUPIL" and len(tokens) >= 2:
                if tokens[1] not in {"AREA", "DIAMETER"}:
                    raise ValueError(f"Unknown pupil measure: {tokens[1]}")
                set_consistent("pupil_measure", tokens[1].lower())
                continue
            if kind in {"SAMPLES", "EVENTS"}:
                if len(tokens) < 2 or tokens[1] != "GAZE":
                    raise ValueError("Only ASC GAZE screen coordinates are supported")
                set_consistent("coordinate_system", "gaze")
                eyes = [e for e, word in (("L", "LEFT"), ("R", "RIGHT")) if word in tokens]
                if eyes:
                    if metadata["recorded_eyes"] and metadata["recorded_eyes"] != eyes:
                        raise ValueError("ASC changes recorded eyes")
                    metadata["recorded_eyes"] = eyes
                    if eye not in eyes:
                        raise ValueError(f"Requested eye {eye} absent; available eyes: {eyes}")
                    if kind == "SAMPLES":
                        sample_header_seen = True
                        sample_eyes = eyes
                if "RATE" in tokens:
                    set_consistent("sampling_rate_hz", float(tokens[tokens.index("RATE") + 1]))
                if "FILTER" in tokens:
                    set_consistent("tracker_filter", tokens[tokens.index("FILTER") + 1])
                if kind == "EVENTS":
                    metadata["event_streams"] = sorted(_END_FIELDS)
                continue
            if _NUMBER.fullmatch(kind):
                if has_start and not recording:
                    continue
                if sample_eyes and eye not in sample_eyes:
                    raise ValueError(f"Requested eye {eye} absent from recording segment")
                offset = 1 + 3 * (sample_eyes.index(eye) if sample_eyes else 0)
                if len(tokens) < offset + 3:
                    raise ValueError("Incomplete ASC sample row")
                samples.append((_to_float(kind), *map(_to_float, tokens[offset:offset + 3]),
                                max(0, segment)))
                continue
            event = {column: np.nan for column in EVENT_COLUMNS}
            event.update(event_type=kind, raw_line=line)
            if kind in {"BUTTON", "INPUT", "MSG"} and len(tokens) >= 2:
                event["timestamp_ms"] = _to_float(tokens[1])
                if kind == "MSG" and len(tokens) >= 7 and tokens[2] == "GAZE_COORDS":
                    set_consistent("gaze_coords", list(map(float, tokens[3:7])))
                if kind == "BUTTON" and len(tokens) >= 4:
                    event.update(button_code=_to_float(tokens[2]), button_state=_to_float(tokens[3]))
            elif kind in {"SFIX", "SSACC", "SBLINK"} and len(tokens) >= 3:
                if tokens[1] != eye:
                    continue
                event.update(eye=eye, timestamp_ms=_to_float(tokens[2]))
            elif kind in _END_FIELDS and len(tokens) >= 2:
                if tokens[1] != eye:
                    continue
                fields = _END_FIELDS[kind]
                if len(tokens) < len(fields) + 2:
                    raise ValueError(f"Incomplete {kind} event")
                event.update(eye=eye)
                event.update(zip(fields, map(_to_float, tokens[2:2 + len(fields)])))
                if (not np.isfinite([event[k] for k in ("start_ms", "end_ms", "duration_ms")]).all()
                        or event["end_ms"] < event["start_ms"] or event["duration_ms"] <= 0):
                    raise ValueError(f"Invalid {kind} timing")
                if kind not in metadata["event_streams"]:
                    metadata["event_streams"].append(kind)
            else:
                continue
            if kind not in _END_FIELDS and not np.isfinite(event["timestamp_ms"]):
                raise ValueError(f"Invalid {kind} timestamp")
            events.append(event)

    if strict and (not sample_header_seen or not sample_eyes or any(metadata[k] is None for k in
                   ("sampling_rate_hz", "coordinate_system", "pupil_measure"))):
        raise ValueError("ASC requires SAMPLES GAZE eye/RATE and PUPIL headers in project mode")
    samples = pd.DataFrame(samples, columns=SAMPLE_COLUMNS)
    if len(samples) and (not np.isfinite(samples.timestamp_ms).all()
                         or not np.all(np.diff(samples.timestamp_ms) > 0)):
        raise ValueError("ASC timestamps must increase strictly across recording segments")
    metadata["segments"] = int(samples.segment_id.nunique())
    metadata["sha256"] = digest.hexdigest()
    metadata["event_streams"] = sorted(set(metadata["event_streams"]))
    events = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if not events.empty:
        events = events.sort_values(["timestamp_ms", "start_ms"], na_position="last").reset_index(drop=True)
    samples.attrs["eyelink_metadata"] = metadata.copy()
    events.attrs["eyelink_metadata"] = metadata.copy()
    return samples, events
