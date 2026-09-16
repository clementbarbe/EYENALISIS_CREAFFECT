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
    window_x_s: float,
    cfg: Config,
) -> dict:
    """Construit une fenêtre, tronquée si la phase est trop courte."""
    end_s = min(
        available_end_s,
        start_s + requested_duration_s,
    )

    actual_duration_s = max(0.0, end_s - start_s)

    return {
        "moment": moment,
        "window_x_s": window_x_s,
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


def _get_baseline_bounds(
    trial: pd.Series,
    onset_column: str,
    duration_column: str,
    cfg: Config,
) -> tuple[float, float, str]:
    """
    Retourne les bornes de baseline depuis la croix précédente.

    Baseline = dernières 500 ms de la croix de fixation.
    """
    cross_onset = trial.get(onset_column, np.nan)
    cross_duration = trial.get(duration_column, np.nan)

    if pd.isna(cross_onset) or pd.isna(cross_duration):
        return np.nan, np.nan, "missing_cross_timestamp"

    cross_onset = float(cross_onset)
    cross_duration = float(cross_duration)

    cross_end = cross_onset + cross_duration

    baseline_start = max(
        cross_onset,
        cross_end - cfg.baseline_duration_s,
    )

    return baseline_start, cross_end, "preceding_fixation_cross"


def _append_window(
    rows: list[dict],
    trial: pd.Series,
    window: dict,
    baseline_start_s: float,
    baseline_end_s: float,
    baseline_source: str,
    timing: pd.Series,
    sync: SyncResult,
) -> None:
    """Convertit les bornes comportementales en timestamps EyeLink."""
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

    baseline_start_ms = behavioral_to_eyelink_ms(
        baseline_start_s,
        timing,
        sync,
    )

    baseline_end_ms = behavioral_to_eyelink_ms(
        baseline_end_s,
        timing,
        sync,
    )

    rows.append(
        {
            **trial.to_dict(),
            **window,
            "baseline_source": baseline_source,
            "baseline_start_behavior_s": baseline_start_s,
            "baseline_end_behavior_s": baseline_end_s,
            "start_eyelink_ms": start_ms,
            "end_eyelink_ms": end_ms,
            "baseline_start_eyelink_ms": baseline_start_ms,
            "baseline_end_eyelink_ms": baseline_end_ms,
        }
    )


def build_moment_windows(
    trials: pd.DataFrame,
    timing: pd.Series,
    sync: SyncResult,
    cfg: Config,
) -> pd.DataFrame:
    """
    Crée M1, M2a, M2b, M3a et M3b.

    M1 :
        Une seule fenêtre de 10 secondes par induction.

    M2/M3 :
        Une fenêtre par valeur de X entre 0.5 et 2.0 secondes.

    Baselines :
        M1    -> croix avant induction.
        M2a/b -> croix avant cue FGAT.
        M3a/b -> croix avant LTask.

    Notes
    -----
    window_x_s = 0.0 pour M1 car M1 est fixe et ne dépend pas de X.
    """
    rows: list[dict] = []

    for _, trial in trials.iterrows():

        # ----------------------------------------------------------
        # Moment 1 : induction émotionnelle.
        # Une seule fois par bloc.
        # ----------------------------------------------------------
        if (
            bool(trial["has_induction"])
            and pd.notna(trial["emo_stim_onset"])
        ):
            induction_onset = float(trial["emo_stim_onset"])

            baseline_start_s, baseline_end_s, baseline_source = (
                _get_baseline_bounds(
                    trial,
                    "t_onset_cross_induc",
                    "t_duration_cross_induc",
                    cfg,
                )
            )

            window = _create_window(
                moment="M1_induction",
                start_s=induction_onset,
                available_end_s=(
                    induction_onset + cfg.induction_duration_s
                ),
                requested_duration_s=cfg.induction_duration_s,
                window_x_s=0.0,
                cfg=cfg,
            )

            _append_window(
                rows,
                trial,
                window,
                baseline_start_s,
                baseline_end_s,
                baseline_source,
                timing,
                sync,
            )

        # ----------------------------------------------------------
        # Moments 2a et 2b : recherche créative.
        # ----------------------------------------------------------
        cue = trial.get("fgat_cue_onset", np.nan)
        response = trial.get("fgat_input_time", np.nan)

        if pd.notna(cue) and pd.notna(response):
            cue = float(cue)
            response = float(response)

            baseline_start_s, baseline_end_s, baseline_source = (
                _get_baseline_bounds(
                    trial,
                    "t_onset_cross_cue_fgat",
                    "t_duration_cross_cue_fgat",
                    cfg,
                )
            )

            for x_s in cfg.dynamic_x_values_s:

                # M2a : début de recherche.
                window_start = _create_window(
                    moment="M2a_search_start",
                    start_s=cue,
                    available_end_s=response,
                    requested_duration_s=x_s,
                    window_x_s=x_s,
                    cfg=cfg,
                )

                _append_window(
                    rows,
                    trial,
                    window_start,
                    baseline_start_s,
                    baseline_end_s,
                    baseline_source,
                    timing,
                    sync,
                )

                # M2b : fin de recherche.
                end_window_start = max(cue, response - x_s)

                window_end = _create_window(
                    moment="M2b_search_end",
                    start_s=end_window_start,
                    available_end_s=response,
                    requested_duration_s=x_s,
                    window_x_s=x_s,
                    cfg=cfg,
                )

                _append_window(
                    rows,
                    trial,
                    window_end,
                    baseline_start_s,
                    baseline_end_s,
                    baseline_source,
                    timing,
                    sync,
                )

        # ----------------------------------------------------------
        # Moments 3a et 3b : évaluation.
        # ----------------------------------------------------------
        ltask_onset = trial.get("LTask_onset", np.nan)
        validation = trial.get("LTask_validation", np.nan)

        if pd.notna(ltask_onset) and pd.notna(validation):
            ltask_onset = float(ltask_onset)
            validation = float(validation)

            baseline_start_s, baseline_end_s, baseline_source = (
                _get_baseline_bounds(
                    trial,
                    "t_onset_cross_LTask",
                    "t_duration_cross_LTask",
                    cfg,
                )
            )

            for x_s in cfg.dynamic_x_values_s:

                # M3a : début de l'évaluation.
                window_start = _create_window(
                    moment="M3a_rating_start",
                    start_s=ltask_onset,
                    available_end_s=validation,
                    requested_duration_s=x_s,
                    window_x_s=x_s,
                    cfg=cfg,
                )

                _append_window(
                    rows,
                    trial,
                    window_start,
                    baseline_start_s,
                    baseline_end_s,
                    baseline_source,
                    timing,
                    sync,
                )

                # M3b : fin de l'évaluation.
                end_window_start = max(
                    ltask_onset,
                    validation - x_s,
                )

                window_end = _create_window(
                    moment="M3b_rating_end",
                    start_s=end_window_start,
                    available_end_s=validation,
                    requested_duration_s=x_s,
                    window_x_s=x_s,
                    cfg=cfg,
                )

                _append_window(
                    rows,
                    trial,
                    window_end,
                    baseline_start_s,
                    baseline_end_s,
                    baseline_source,
                    timing,
                    sync,
                )

    return pd.DataFrame(rows)