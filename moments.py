"""Construction des fenêtres temporelles confirmatoires FGAT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from sync import SyncResult, behavioral_to_eyelink_ms


def _create_window(
    moment: str,
    start_s: float,
    end_s: float,
    requested_duration_s: float | None,
    cfg: Config,
) -> dict:
    """Construit une fenêtre avec métadonnées de durée."""
    actual_duration_s = max(0.0, end_s - start_s)

    return {
        "moment": moment,
        "window_x_s": (
            requested_duration_s
            if requested_duration_s is not None
            else 0.0
        ),
        "start_behavior_s": start_s,
        "end_behavior_s": end_s,
        "requested_duration_s": requested_duration_s,
        "actual_duration_s": actual_duration_s,
        "duration_valid": (
            actual_duration_s >= cfg.minimum_usable_window_s
        ),
    }


def _get_cross_baseline(
    trial: pd.Series,
    onset_column: str,
    duration_column: str,
    cfg: Config,
) -> tuple[float, float, str]:
    """Retourne les dernières ms de la croix précédente."""
    cross_onset = trial.get(onset_column, np.nan)
    cross_duration = trial.get(duration_column, np.nan)

    if pd.isna(cross_onset) or pd.isna(cross_duration):
        return np.nan, np.nan, "missing_cross_timestamp"

    cross_onset = float(cross_onset)
    cross_end = cross_onset + float(cross_duration)

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
    """Ajoute une fenêtre et ses timestamps EyeLink."""
    rows.append(
        {
            **trial.to_dict(),
            **window,
            "baseline_source": baseline_source,
            "baseline_start_behavior_s": baseline_start_s,
            "baseline_end_behavior_s": baseline_end_s,
            "start_eyelink_ms": behavioral_to_eyelink_ms(
                window["start_behavior_s"],
                timing,
                sync,
            ),
            "end_eyelink_ms": behavioral_to_eyelink_ms(
                window["end_behavior_s"],
                timing,
                sync,
            ),
            "baseline_start_eyelink_ms": behavioral_to_eyelink_ms(
                baseline_start_s,
                timing,
                sync,
            ),
            "baseline_end_eyelink_ms": behavioral_to_eyelink_ms(
                baseline_end_s,
                timing,
                sync,
            ),
        }
    )


def build_moment_windows(
    trials: pd.DataFrame,
    timing: pd.Series,
    sync: SyncResult,
    cfg: Config,
) -> pd.DataFrame:
    """
    Construit uniquement les fenêtres confirmatoires fixes.

    Fenêtres :
    - M1_early_photometric : induction 0–2 s.
    - H1_induction_late : induction 2–10 s.
    - H2_pre_idea : X secondes avant fgat_input_time.
    - H3_liking : LTask_onset -> LTask_first_movement.
    - H4_full_search_gaze : fgat_cue_onset -> fgat_input_time.
    """
    rows: list[dict] = []

    for _, trial in trials.iterrows():

        # ----------------------------------------------------------
        # H1 : induction émotionnelle.
        # ----------------------------------------------------------
        if (
            bool(trial["has_induction"])
            and pd.notna(trial["emo_stim_onset"])
        ):
            induction_onset = float(trial["emo_stim_onset"])

            baseline = _get_cross_baseline(
                trial,
                "t_onset_cross_induc",
                "t_duration_cross_induc",
                cfg,
            )

            early_window = _create_window(
                moment="M1_early_photometric",
                start_s=induction_onset + cfg.induction_early_start_s,
                end_s=induction_onset + cfg.induction_early_end_s,
                requested_duration_s=(
                    cfg.induction_early_end_s
                    - cfg.induction_early_start_s
                ),
                cfg=cfg,
            )

            late_window = _create_window(
                moment="H1_induction_late",
                start_s=induction_onset + cfg.induction_late_start_s,
                end_s=induction_onset + cfg.induction_late_end_s,
                requested_duration_s=(
                    cfg.induction_late_end_s
                    - cfg.induction_late_start_s
                ),
                cfg=cfg,
            )

            _append_window(
                rows,
                trial,
                early_window,
                *baseline,
                timing,
                sync,
            )

            _append_window(
                rows,
                trial,
                late_window,
                *baseline,
                timing,
                sync,
            )

        # ----------------------------------------------------------
        # H2 et H4 : recherche d'idée.
        # ----------------------------------------------------------
        cue = trial.get("fgat_cue_onset", np.nan)
        idea_time = trial.get("fgat_input_time", np.nan)

        if pd.notna(cue) and pd.notna(idea_time):
            cue = float(cue)
            idea_time = float(idea_time)

            baseline = _get_cross_baseline(
                trial,
                "t_onset_cross_cue_fgat",
                "t_duration_cross_cue_fgat",
                cfg,
            )

            # H2 : période fixe juste avant le clic "idée trouvée".
            h2_start = max(
                cue,
                idea_time - cfg.fixed_pre_idea_window_s,
            )

            pre_idea_window = _create_window(
                moment="H2_pre_idea",
                start_s=h2_start,
                end_s=idea_time,
                requested_duration_s=cfg.fixed_pre_idea_window_s,
                cfg=cfg,
            )

            # Sensibilité H2 sans 200 dernières ms de préparation / clic moteur.
            motor_end = max(cue, idea_time - cfg.pre_idea_motor_exclusion_s)
            no_motor = _create_window(
                moment="H2_pre_idea_without_motor",
                start_s=max(cue, motor_end - cfg.fixed_pre_idea_window_s),
                end_s=motor_end,
                requested_duration_s=cfg.fixed_pre_idea_window_s,
                cfg=cfg,
            )
            _append_window(rows, trial, no_motor, *baseline, timing, sync)

            # H4 : période entière cue -> clic idée, pour les heatmaps gaze.
            full_search_window = _create_window(
                moment="H4_full_search_gaze",
                start_s=cue,
                end_s=idea_time,
                requested_duration_s=None,
                cfg=cfg,
            )

            _append_window(
                rows,
                trial,
                pre_idea_window,
                *baseline,
                timing,
                sync,
            )

            _append_window(
                rows,
                trial,
                full_search_window,
                *baseline,
                timing,
                sync,
            )

        # ----------------------------------------------------------
        # H3 : liking de l'idée.
        # ----------------------------------------------------------
        ltask_onset = trial.get("LTask_onset", np.nan)
        first_movement = trial.get("LTask_first_movement", np.nan)
        validation = trial.get("LTask_validation", np.nan)

        liking_end = np.nan

        if cfg.liking_use_first_movement and pd.notna(first_movement):
            liking_end = first_movement

        elif pd.notna(validation):
            liking_end = validation

        if pd.notna(ltask_onset) and pd.notna(liking_end):
            ltask_onset = float(ltask_onset)
            liking_end = float(liking_end)

            baseline = _get_cross_baseline(
                trial,
                "t_onset_cross_LTask",
                "t_duration_cross_LTask",
                cfg,
            )

            liking_window = _create_window(
                moment="H3_liking",
                start_s=ltask_onset,
                end_s=liking_end,
                requested_duration_s=None,
                cfg=cfg,
            )

            _append_window(
                rows, trial, liking_window, *baseline, timing, sync,
            )
            # Sensibilité: évaluation initiale au plus 1 s, sans mouvement.
            if pd.notna(first_movement):
                fixed = _create_window(
                    moment="H3_liking_first_1s_pre_move",
                    start_s=ltask_onset,
                    end_s=min(ltask_onset + cfg.liking_fixed_window_s, float(first_movement)),
                    requested_duration_s=cfg.liking_fixed_window_s, cfg=cfg,
                )
                _append_window(rows, trial, fixed, *baseline, timing, sync)

    return pd.DataFrame(rows)