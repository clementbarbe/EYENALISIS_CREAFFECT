import numpy as np
import pandas as pd


def parse_asc(filepath):
    samples = {
        "time": [],
        "x": [],
        "y": [],
        "pupil": [],
    }

    messages = []
    saccades = []
    fixations = []
    blinks = []
    buttons = []

    recording = False

    def parse_float(value):
        return np.nan if value == "." else float(value)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:

        for line in file:
            parts = line.split()

            if not parts:
                continue

            token = parts[0]

            if token == "START":
                recording = True
                continue

            if token == "END":
                recording = False
                continue

            if token == "MSG" and len(parts) >= 3:
                try:
                    messages.append({
                        "time": int(parts[1]),
                        "text": " ".join(parts[2:]),
                    })
                except ValueError:
                    pass
                continue

            if token == "BUTTON" and len(parts) >= 4:
                try:
                    buttons.append({
                        "time": int(parts[1]),
                        "button": int(parts[2]),
                        "state": int(parts[3]),
                    })
                except ValueError:
                    pass
                continue

            if token == "INPUT" and len(parts) >= 3:
                try:
                    buttons.append({
                        "time": int(parts[1]),
                        "button": int(parts[2]),
                        "state": 1,
                    })
                except ValueError:
                    pass
                continue

            if token == "ESACC" and len(parts) >= 10:
                try:
                    saccades.append({
                        "start": int(parts[2]),
                        "end": int(parts[3]),
                        "duration": int(parts[4]),
                        "sx": parse_float(parts[5]),
                        "sy": parse_float(parts[6]),
                        "ex": parse_float(parts[7]),
                        "ey": parse_float(parts[8]),
                        "amplitude": parse_float(parts[9]),
                        "peak_vel": parse_float(parts[10])
                        if len(parts) > 10 else np.nan,
                    })
                except (ValueError, IndexError):
                    pass
                continue

            if token == "EFIX" and len(parts) >= 7:
                try:
                    fixations.append({
                        "start": int(parts[2]),
                        "end": int(parts[3]),
                        "duration": int(parts[4]),
                        "x": parse_float(parts[5]),
                        "y": parse_float(parts[6]),
                        "pupil": parse_float(parts[7])
                        if len(parts) > 7 else np.nan,
                    })
                except (ValueError, IndexError):
                    pass
                continue

            if token == "EBLINK" and len(parts) >= 5:
                try:
                    blinks.append({
                        "start": int(parts[2]),
                        "end": int(parts[3]),
                        "duration": int(parts[4]),
                    })
                except (ValueError, IndexError):
                    pass
                continue

            if recording:
                try:
                    samples["time"].append(int(parts[0]))
                    samples["x"].append(parse_float(parts[1]))
                    samples["y"].append(parse_float(parts[2]))
                    samples["pupil"].append(parse_float(parts[3]))
                except (ValueError, IndexError):
                    pass

    samples = {
        key: np.asarray(value, dtype=float)
        for key, value in samples.items()
    }

    return {
        "samples": samples,
        "messages": pd.DataFrame(messages),
        "saccades": pd.DataFrame(saccades),
        "fixations": pd.DataFrame(fixations),
        "blinks": pd.DataFrame(blinks),
        "buttons": pd.DataFrame(
            buttons,
            columns=["time", "button", "state"],
        ),
    }