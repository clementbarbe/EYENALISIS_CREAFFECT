"""Parseur défensif EyeLink ASCII (.asc)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_type",
    "timestamp_ms",
    "start_ms",
    "end_ms",
    "duration_ms",
    "eye",
    "button_code",
    "button_state",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "x_mean",
    "y_mean",
    "pupil_mean",
    "amplitude_deg",
    "peak_velocity",
    "raw_line",
]


def _to_float(value: str) -> float:
    """Convertit une valeur texte en float ou NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _new_event(event_type: str, raw_line: str) -> dict[str, Any]:
    """Initialise un évènement EyeLink."""
    event = {column: np.nan for column in EVENT_COLUMNS}
    event["event_type"] = event_type
    event["raw_line"] = raw_line
    return event


def parse_asc(asc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse un fichier ASC EyeLink.

    Returns
    -------
    samples:
        timestamp_ms, gaze_x, gaze_y, pupil_size.
    events:
        Tableau d'évènements EyeLink.
    """
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    with asc_path.open(
        "r",
        encoding="latin-1",
        errors="replace",
    ) as file:

        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("**"):
                continue

            tokens = line.split()
            first_token = tokens[0]

            # Echantillon brut :
            # <timestamp> <gaze_x> <gaze_y> <pupil_size> ...
            if first_token.lstrip("-").isdigit():
                if len(tokens) >= 4:
                    samples.append(
                        {
                            "timestamp_ms": _to_float(tokens[0]),
                            "gaze_x": _to_float(tokens[1]),
                            "gaze_y": _to_float(tokens[2]),
                            "pupil_size": _to_float(tokens[3]),
                        }
                    )
                continue

            if first_token == "BUTTON" and len(tokens) >= 4:
                event = _new_event(first_token, line)
                event["timestamp_ms"] = _to_float(tokens[1])
                event["button_code"] = _to_float(tokens[2])
                event["button_state"] = _to_float(tokens[3])
                events.append(event)
                continue

            if first_token == "MSG" and len(tokens) >= 2:
                event = _new_event(first_token, line)
                event["timestamp_ms"] = _to_float(tokens[1])
                events.append(event)
                continue

            if first_token == "SFIX" and len(tokens) >= 3:
                event = _new_event(first_token, line)
                event["eye"] = tokens[1]
                event["timestamp_ms"] = _to_float(tokens[2])
                events.append(event)
                continue

            # EFIX R start end duration mean_x mean_y mean_pupil
            if first_token == "EFIX" and len(tokens) >= 8:
                event = _new_event(first_token, line)
                event["eye"] = tokens[1]
                event["start_ms"] = _to_float(tokens[2])
                event["end_ms"] = _to_float(tokens[3])
                event["duration_ms"] = _to_float(tokens[4])
                event["x_mean"] = _to_float(tokens[5])
                event["y_mean"] = _to_float(tokens[6])
                event["pupil_mean"] = _to_float(tokens[7])
                events.append(event)
                continue

            if first_token == "SSACC" and len(tokens) >= 3:
                event = _new_event(first_token, line)
                event["eye"] = tokens[1]
                event["timestamp_ms"] = _to_float(tokens[2])
                events.append(event)
                continue

            # ESACC R start end duration xstart ystart xend yend amp peakvel
            if first_token == "ESACC" and len(tokens) >= 11:
                event = _new_event(first_token, line)
                event["eye"] = tokens[1]
                event["start_ms"] = _to_float(tokens[2])
                event["end_ms"] = _to_float(tokens[3])
                event["duration_ms"] = _to_float(tokens[4])
                event["x_start"] = _to_float(tokens[5])
                event["y_start"] = _to_float(tokens[6])
                event["x_end"] = _to_float(tokens[7])
                event["y_end"] = _to_float(tokens[8])
                event["amplitude_deg"] = _to_float(tokens[9])
                event["peak_velocity"] = _to_float(tokens[10])
                events.append(event)
                continue

            # EBLINK R start end duration
            if first_token == "EBLINK" and len(tokens) >= 5:
                event = _new_event(first_token, line)
                event["eye"] = tokens[1]
                event["start_ms"] = _to_float(tokens[2])
                event["end_ms"] = _to_float(tokens[3])
                event["duration_ms"] = _to_float(tokens[4])
                events.append(event)
                continue

            # SBLINK R timestamp
            if first_token == "SBLINK" and len(tokens) >= 3:
                event = _new_event(first_token, line)
                event["eye"] = tokens[1]
                event["timestamp_ms"] = _to_float(tokens[2])
                events.append(event)
                continue

    samples_df = pd.DataFrame(samples)

    if samples_df.empty:
        samples_df = pd.DataFrame(
            columns=[
                "timestamp_ms",
                "gaze_x",
                "gaze_y",
                "pupil_size",
            ]
        )
    else:
        samples_df = samples_df.sort_values(
            "timestamp_ms"
        ).reset_index(drop=True)

    events_df = pd.DataFrame(events)

    if events_df.empty:
        events_df = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        events_df = events_df.sort_values(
            ["timestamp_ms", "start_ms"],
            na_position="last",
        ).reset_index(drop=True)

    return samples_df, events_df