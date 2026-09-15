"""Création des fenêtres temporelles des moments expérimentaux."""

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
    """Construit une fenêtre potentiellement tronquée."""
    available_duration_s = max(0.0, available_end_s - start_s)

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
        "available_duration_s": available_duration_s,
        "actual_duration_s": actual_duration_s,
        "duration_valid": actual_duration_s >= cfg.minimum_usable_window_s,
    }


def build_moment_windows(
    trials: pd.DataFrame,
    timing: pd.Series,
    sync: SyncResult,
    cfg: Config,
) -> pd.DataFrame:
    """
    Construit les fenêtres M1, M2a, M2b, M3a et M3b.

    M1 est calculé seulement pour l'essai portant réellement l'induction,
    afin de ne pas tripler artificiellement une même induction de 10 s.
    """
    rows: list[dict] = []

    for _, trial in trials.iterrows():
        candidate_windows: list[dict] = []

        # Moment 1 : induction, uniquement premier essai du bloc.
        if (
            bool(trial["has_induction"])
            and pd.notna(trial["emo_stim_onset"])
        ):
            onset = float(trial["emo_stim_onset"])

            candidate_windows.append(
                _create_window(
                    moment="M1_induction",
                    start_s=onset,
                    available_end_s=onset + cfg.induction_duration_s,
                    requested_duration_s=cfg.induction_duration_s,
                    cfg=cfg,
                )
            )

        # Moment 2a et 2b : recherche d'association.
        cue = trial.get("fgat_cue_onset", np.nan)
        response = trial.get("fgat_input_time", np.nan)

        if pd.notna(cue) and pd.notna(response):
            cue = float(cue)
            response = float(response)

            candidate_windows.append(
                _create_window(
                    moment="M2a_search_start",
                    start_s=cue,
                    available_end_s=response,
                    requested_duration_s=cfg.moment_window_s["M2a_search_start"],
                    cfg=cfg,
                )
            )

            end_window_start = max(
                cue,
                response - cfg.moment_window_s["M2b_search_end"],
            )

            candidate_windows.append(
                _create_window(
                    moment="M2b_search_end",
                    start_s=end_window_start,
                    available_end_s=response,
                    requested_duration_s=cfg.moment_window_s["M2b_search_end"],
                    cfg=cfg,
                )
            )

        # Moment 3a et 3b : évaluation.
        rating_onset = trial.get("LTask_onset", np.nan)
        validation = trial.get("LTask_validation", np.nan)

        if pd.notna(rating_onset) and pd.notna(validation):
            rating_onset = float(rating_onset)
            validation = float(validation)

            candidate_windows.append(
                _create_window(
                    moment="M3a_rating_start",
                    start_s=rating_onset,
                    available_end_s=validation,
                    requested_duration_s=cfg.moment_window_s["M3a_rating_start"],
                    cfg=cfg,
                )
            )

            end_window_start = max(
                rating_onset,
                validation - cfg.moment_window_s["M3b_rating_end"],
            )

            candidate_windows.append(
                _create_window(
                    moment="M3b_rating_end",
                    start_s=end_window_start,
                    available_end_s=validation,
                    requested_duration_s=cfg.moment_window_s["M3b_rating_end"],
                    cfg=cfg,
                )
            )

        for window in candidate_windows:
            start_eyelink_ms = behavioral_to_eyelink_ms(
                behavioral_time_s=window["start_behavior_s"],
                timing=timing,
                sync=sync,
            )

            end_eyelink_ms = behavioral_to_eyelink_ms(
                behavioral_time_s=window["end_behavior_s"],
                timing=timing,
                sync=sync,
            )

            baseline_start_behavior_s = (
                window["start_behavior_s"] - cfg.baseline_duration_s
            )

            baseline_start_eyelink_ms = behavioral_to_eyelink_ms(
                behavioral_time_s=baseline_start_behavior_s,
                timing=timing,
                sync=sync,
            )

            rows.append(
                {
                    **trial.to_dict(),
                    **window,
                    "start_eyelink_ms": start_eyelink_ms,
                    "end_eyelink_ms": end_eyelink_ms,
                    "baseline_start_eyelink_ms": baseline_start_eyelink_ms,
                    "baseline_end_eyelink_ms": start_eyelink_ms,
                }
            )

    return pd.DataFrame(rows)