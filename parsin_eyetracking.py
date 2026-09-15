"""Parseur défensif de fichiers EyeLink ASCII (.asc)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_type", "timestamp_ms", "start_ms", "end_ms", "duration_ms",
    "eye", "button_code", "button_state", "x_start", "y_start",
    "x_end", "y_end", "x_mean", "y_mean", "pupil_mean",
    "amplitude_deg", "peak_velocity", "raw_line",
]


def _to_float(token: str) -> float:
    """Convertit un token en float, sinon NaN."""
    try:
        return float(token)
    except (ValueError, TypeError):
        return np.nan


def parse_asc(asc_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse un .asc EyeLink.

    Returns
    -------
    samples:
        timestamp_ms, gaze_x, gaze_y, pupil_size
    events:
        Évènements EyeLink structurés.
    """
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    with asc_path.open("r", encoding="latin-1", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue

            tokens = line.split()
            first = tokens[0]

            # Lignes d'échantillons : premier token entier.
            if first.lstrip("-").isdigit():
                if len(tokens) >= 4:
                    samples.append({
                        "timestamp_ms": int(tokens[0]),
                        "gaze_x": _to_float(tokens[1]),
                        "gaze_y": _to_float(tokens[2]),
                        "pupil_size": _to_float(tokens[3]),
                    })
                continue

            base = {col: np.nan for col in EVENT_COLUMNS}
            base["event_type"] = first
            base["raw_line"] = line

            if first == "BUTTON" and len(tokens) >= 4:
                base.update({
                    "timestamp_ms": _to_float(tokens[1]),
                    "button_code": _to_float(tokens[2]),
                    "button_state": _to_float(tokens[3]),
                })
                events.append(base)

            elif first == "EFIX" and len(tokens) >= 8:
                base.update({
                    "eye": tokens[1],
                    "start_ms": _to_float(tokens[2]),
                    "end_ms": _to_float(tokens[3]),
                    "duration_ms": _to_float(tokens[4]),
                    "x_mean": _to_float(tokens[5]),
                    "y_mean": _to_float(tokens[6]),
                    "pupil_mean": _to_float(tokens[7]),
                })
                events.append(base)

            elif first == "SFIX" and len(tokens) >= 3:
                base.update({"eye": tokens[1], "timestamp_ms": _to_float(tokens[2])})
                events.append(base)

            elif first == "ESACC" and len(tokens) >= 11:
                base.update({
                    "eye": tokens[1],
                    "start_ms": _to_float(tokens[2]),
                    "end_ms": _to_float(tokens[3]),
                    "duration_ms": _to_float(tokens[4]),
                    "x_start": _to_float(tokens[5]),
                    "y_start": _to_float(tokens[6]),
                    "x_end": _to_float(tokens[7]),
                    "y_end": _to_float(tokens[8]),
                    "amplitude_deg": _to_float(tokens[9]),
                    "peak_velocity": _to_float(tokens[10]),
                })
                events.append(base)

            elif first == "SSACC" and len(tokens) >= 3:
                base.update({"eye": tokens[1], "timestamp_ms": _to_float(tokens[2])})
                events.append(base)

            elif first in {"SBLINK", "EBLINK"} and len(tokens) >= 3:
                base["eye"] = tokens[1]
                if first == "EBLINK" and len(tokens) >= 5:
                    base.update({
                        "start_ms": _to_float(tokens[2]),
                        "end_ms": _to_float(tokens[3]),
                        "duration_ms": _to_float(tokens[4]),
                    })
                else:
                    base["timestamp_ms"] = _to_float(tokens[2])
                events.append(base)

            elif first == "MSG" and len(tokens) >= 2:
                base["timestamp_ms"] = _to_float(tokens[1])
                events.append(base)

    sample_df = pd.DataFrame(samples)
    if sample_df.empty:
        sample_df = pd.DataFrame(columns=["timestamp_ms", "gaze_x", "gaze_y", "pupil_size"])

    event_df = pd.DataFrame(events)
    if event_df.empty:
        event_df = pd.DataFrame(columns=EVENT_COLUMNS)

    return sample_df.sort_values("timestamp_ms"), event_df
