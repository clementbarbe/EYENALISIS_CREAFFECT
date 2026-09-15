"""Construction des fenêtres temporelles expérimentales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from sync import SyncResult, behavioral_to_eyelink_ms


def _create_window(
    moment: str,
    start_s: float,
    available_end_s: float,
    requested_duration_s: float,
    cfg: Config,
) -> dict:
    """Construit une fenêtre tronquée si nécessaire."""
    end_s = min(
        available_end_s,
        start_s + requested_duration_s,
    )

    actual_duration_s = max(0.0, end_s - start_s)

    return {
        "moment": moment,
        "start_behavior_s": start_s,
        "end_behavior_s": end_s,
        "requested_duration_s": requested_duration_s,
        "available_duration_s": max(
            0.0,
            available_end_s - start_s,
        ),
        "actual_duration_s": actual_duration_s,
        "duration_valid": (
            actual_duration_s >= cfg.minimum_usable_window_s
        ),
    }


def build_moment_windows(
    trials: pd.DataFrame,
    timing: pd.Series,
    sync: SyncResult,
    cfg: Config,
) -> pd.DataFrame:
    """
    Crée M1, M2a, M2b, M3a et M3b.

    M1 est créé une fois par induction, seulement sur has_induction=True.
    """
    rows: list[dict] = []

    for _, trial in trials.iterrows():
        windows: list[dict] = []

        if (
            bool(trial["has_induction"])
            and pd.notna(trial["emo_stim_onset"])
        ):
            onset = float(trial["emo_stim_onset"])

            windows.append(
                _create_window(
                    "M1_induction",
                    onset,
                    onset + cfg.induction_duration_s,
                    cfg.induction_duration_s,
                    cfg,
                )
            )

        cue = trial.get("fgat_cue_onset", np.nan)
        response = trial.get("fgat_input_time", np.nan)

        if pd.notna(cue) and pd.notna(response):
            cue = float(cue)
            response = float(response)

            windows.append(
                _create_window(
                    "M2a_search_start",
                    cue,
                    response,
                    cfg.moment_window_s["M2a_search_start"],
                    cfg,
                )
            )

            start_end_window = max(
                cue,
                response - cfg.moment_window_s["M2b_search_end"],
            )

            windows.append(
                _create_window(
                    "M2b_search_end",
                    start_end_window,
                    response,
                    cfg.moment_window_s["M2b_search_end"],
                    cfg,
                )
            )

        rating_onset = trial.get("LTask_onset", np.nan)
        validation = trial.get("LTask_validation", np.nan)

        if pd.notna(rating_onset) and pd.notna(validation):
            rating_onset = float(rating_onset)
            validation = float(validation)

            windows.append(
                _create_window(
                    "M3a_rating_start",
                    rating_onset,
                    validation,
                    cfg.moment_window_s["M3a_rating_start"],
                    cfg,
                )
            )

            start_end_window = max(
                rating_onset,
                validation - cfg.moment_window_s["M3b_rating_end"],
            )

            windows.append(
                _create_window(
                    "M3b_rating_end",
                    start_end_window,
                    validation,
                    cfg.moment_window_s["M3b_rating_end"],
                    cfg,
                )
            )

        for window in windows:
            start_ms = behavioral_to_eyelink_ms(
                window["start_behavior_s"],
                timing,
                sync,
            )

            end_ms = behavioral_to_eyelink_ms(
                window["end_behavior_s"],
                timing,
                sync,
            )

            baseline_start_s = (
                window["start_behavior_s"]
                - cfg.baseline_duration_s
            )

            baseline_start_ms = behavioral_to_eyelink_ms(
                baseline_start_s,
                timing,
                sync,
            )

            rows.append(
                {
                    **trial.to_dict(),
                    **window,
                    "start_eyelink_ms": start_ms,
                    "end_eyelink_ms": end_ms,
                    "baseline_start_eyelink_ms": baseline_start_ms,
                    "baseline_end_eyelink_ms": start_ms,
                }
            )

    return pd.DataFrame(rows)