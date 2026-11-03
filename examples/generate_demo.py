"""Crée un petit projet EyeLink entièrement synthétique, sans données privées."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo(root, subjects=3, sampling_hz=500):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Demo directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    project = json.loads(Path(__file__).with_name("project_example.json").read_text())
    (root / "project.json").write_text(json.dumps(project, indent=2, ensure_ascii=False), encoding="utf-8")
    records = []
    for subject in range(subjects):
        participant = f"volunteer-{subject + 1:02}"
        folder = root / participant
        folder.mkdir()
        rng = np.random.default_rng(21 + subject)
        offset = 100000.0 + 25000 * subject
        slope = 1 + subject * 20e-6
        total_s = 24.0
        time = offset + np.arange(int(total_s * sampling_hz)) * 1000.0 / sampling_hz
        reference = (time - offset) / (1000 * slope)
        pupil = 2400 + subject * 50 + 12 * np.sin(reference * 2 * np.pi / 9) + rng.normal(0, 1, len(time))
        x = 960 + 80 * np.sin(reference * 1.7)
        y = 540 + 25 * np.cos(reference * 1.1)
        events, ocular = [], []
        for trial in range(6):
            origin = 4.0 * trial
            stim = origin + 0.8
            rt = 1.0 + 0.07 * trial + rng.uniform(0, 0.12)
            response = stim + rt
            rating = response + 0.3
            condition = "A" if trial % 2 == 0 else "B"
            timeout = trial == 5
            shared = dict(trial_id=f"trial-{trial + 1}", condition=condition,
                          response_time_s=rt, item_id=f"item-{trial}", timeout=timeout)
            specifications = [("FIX", origin + 0.1, 0.7), ("STIM", stim, 0.15),
                              ("RESP", response, 0), ("RATE", rating, 0)]
            if trial != 4:
                specifications.append(("MOVE", rating + 0.8, 0))
            for kind, onset, duration in specifications:
                events.append({**shared, "event_id": f"{trial}-{kind}", "event_type": kind,
                               "onset": onset, "duration": duration})
            effect = 70 + 25 * (condition == "B")
            pupil += effect * np.exp(-0.5 * ((reference - stim - 0.85) / 0.35) ** 2)
            to_ms = lambda seconds: offset + 1000 * slope * seconds
            ocular.append(f"EFIX R {to_ms(origin + .1):.3f} {to_ms(stim):.3f} 700 960 540 2400")
            ocular.append(f"ESACC R {to_ms(stim + .18):.3f} {to_ms(stim + .21):.3f} 30 960 540 990 540 1.5 180")
            ocular.append(f"EFIX R {to_ms(stim + .21):.3f} {to_ms(response):.3f} {(rt - .21)*1000:.3f} 990 540 2430")
            if trial == 1:
                blink_start, blink_stop = stim + .4, stim + .49
                missing = (reference >= blink_start) & (reference < blink_stop)
                pupil[missing], x[missing], y[missing] = 0, np.nan, np.nan
                ocular.append(f"EBLINK R {to_ms(blink_start):.3f} {to_ms(blink_stop):.3f} 90")
        pd.DataFrame(events).to_csv(folder / "events.tsv", sep="\t", index=False)
        pd.DataFrame({"reference_s": [0, 12, 24], "eyelink_ms": offset + 1000 * slope * np.array([0, 12, 24])}).to_csv(
            folder / "anchors.tsv", sep="\t", index=False)
        with (folder / "recording.asc").open("w", encoding="latin-1") as stream:
            stream.write(f"MSG {offset:.0f} GAZE_COORDS 0 0 1919 1079\nSTART {offset:.0f} RIGHT SAMPLES EVENTS\n"
                         f"PUPIL DIAMETER\nSAMPLES GAZE RIGHT RATE {sampling_hz} TRACKING CR FILTER 2\n"
                         f"EVENTS GAZE RIGHT RATE {sampling_hz} TRACKING CR FILTER 2\n")
            for i in range(len(time)):
                # Interruption d'acquisition de 40 ms sur un essai.
                if 12.9 <= reference[i] < 12.94:
                    continue
                gaze_x = f"{x[i]:.2f}" if np.isfinite(x[i]) else "."
                gaze_y = f"{y[i]:.2f}" if np.isfinite(y[i]) else "."
                stream.write(f"{time[i]:.3f} {gaze_x} {gaze_y} {pupil[i]:.3f}\n")
            stream.write("\n".join(ocular) + f"\nEND {time[-1]:.3f} SAMPLES EVENTS\n")
        records.append(dict(participant_id=participant, session_id="visit-a", run_id="task-one",
                            asc_path=f"{participant}/recording.asc", events_path=f"{participant}/events.tsv",
                            sync_path=f"{participant}/anchors.tsv"))
    pd.DataFrame(records).to_csv(root / "recordings.tsv", sep="\t", index=False)
    return root / "project.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--sampling-hz", type=int, choices=[500, 1000, 2000], default=500)
    args = parser.parse_args()
    if args.subjects < 1:
        parser.error("--subjects must be positive")
    print(generate_demo(args.output_dir, args.subjects, args.sampling_hz))
