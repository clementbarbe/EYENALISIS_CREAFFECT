"""Construction des fenêtres temporelles par essai et moment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from sync import SyncResult, behavioral_to_eyelink_ms


def _make_window(
    name: str,
    start_s: float,
    end_s: float,
    requested_s: float,
    cfg: Config,
) -> dict:
    """Construit une fenêtre, tronquée si nécessaire."""
    available_s = max(0.0, end_s - start_s)
    actual_end_s = min(end_s, start_s + requested_s)
    actual_duration_s = max(0.0, actual_end_s - start_s)

    valid_duration = actual_duration_s >= cfg.minimum_usable_window_s

    return {
        "moment": name,
        "start_behavior_s": start_s,
        "end_behavior_s": actual_end_s,
        "requested_duration_s": requested_s,
        "available_duration_s": available_s,
        "actual_duration_s": actual_duration_s,
        "duration_valid": valid_duration,
    }


def build_moment_windows(
    trials: pd.DataFrame,
    timing: pd.Series,
    sync: SyncResult,
    cfg: Config,
) -> pd.DataFrame:
    """
    Crée les fenêtres Moment 1, 2a, 2b, 3a et 3b.

    Moment 1 n'est produit que pour les essais portant réellement l'induction,
    afin d'éviter de tripler artificiellement la même période d'induction.
    """
    rows: list[dict] = []

    for _, trial in trials.iterrows():
        common = trial.to_dict()
        candidate_windows: list[dict] = []

        if bool(trial["has_induction"]) and pd.notna(trial["emo_stim_onset"]):
            candidate_windows.append(
                _make_window(
                    "M1_induction",
                    float(trial["emo_stim_onset"]),
                    float(trial["emo_stim_onset"]) + cfg.induction_duration_s,
                    cfg.induction_duration_s,
                    cfg,
                )
            )

        cue = trial.get("fgat_cue_onset", np.nan)
        response = trial.get("fgat_input_time", np.nan)
        if pd.notna(cue) and pd.notna(response):
            candidate_windows.append(
                _make_window(
                    "M2a_search_start",
                    float(cue),
                    float(response),
                    cfg.moment_window_s["M2a_search_start"],
                    cfg,
                )
            )
            start = max(float(cue), float(response) - cfg.moment_window_s["M2b_search_end"])
            candidate_windows.append(
                _make_window(
                    "M2b_search_end",
                    start,
                    float(response),
                    cfg.moment_window_s["M2b_search_end"],
                    cfg,
                )
            )

        rating_onset = trial.get("LTask_onset", np.nan)
        validation = trial.get("LTask_validation", np.nan)
        if pd.notna(rating_onset) and pd.notna(validation):
            candidate_windows.append(
                _make_window(
                    "M3a_rating_start",
                    float(rating_onset),
                    float(validation),
                    cfg.moment_window_s["M3a_rating_start"],
                    cfg,
                )
            )
            start = max(
                float(rating_onset),
                float(validation) - cfg.moment_window_s["M3b_rating_end"],
            )
            candidate_windows.append(
                _make_window(
                    "M3b_rating_end",
                    start,
                    float(validation),
                    cfg.moment_window_s["M3b_rating_end"],
                    cfg,
                )
            )

        for window in candidate_windows:
            onset_ms = behavioral_to_eyelink_ms(
                window["start_behavior_s"], sync, timing
            )
            end_ms = behavioral_to_eyelink_ms(
                window["end_behavior_s"], sync, timing
            )

            baseline_start_s = window["start_behavior_s"] - cfg.baseline_duration_s
            baseline_start_ms = behavioral_to_eyelink_ms(
                baseline_start_s, sync, timing
            )

            rows.append({
                **common,
                **window,
                "start_eyelink_ms": onset_ms,
                "end_eyelink_ms": end_ms,
                "baseline_start_eyelink_ms": baseline_start_ms,
                "baseline_end_eyelink_ms": onset_ms,
            })

    return pd.DataFrame(rows)
