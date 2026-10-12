"""Configuration centralisée du pipeline FGAT eyetracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Paramètres modifiables du pipeline."""

    # --------------------------------------------------------------
    # Acquisition EyeLink
    # --------------------------------------------------------------
    sampling_rate_hz: float = 1000.0
    eye: str = "R"

    # --------------------------------------------------------------
    # Synchronisation
    # --------------------------------------------------------------
    sync_button_code: int = 1
    sync_button_state: int = 1
    max_sync_end_difference_s: float = 2.0
    exclude_bad_sync_end_check: bool = True

    # --------------------------------------------------------------
    # Prétraitement pupillaire
    # --------------------------------------------------------------
    blink_padding_before_ms: int = 100
    blink_padding_after_ms: int = 100

    min_plausible_pupil: float = 0.01
    min_unique_pupil_values: int = 20

    pupil_speed_mad_threshold: float = 6.0
    pupil_outlier_mad_threshold: float = 6.0

    interpolation_max_gap_ms: int = 200

    lowpass_enabled: bool = True
    lowpass_cutoff_hz: float = 6.0
    lowpass_order: int = 4

    # --------------------------------------------------------------
    # Exclusions volontairement permissives
    # --------------------------------------------------------------
    # Exclusion run seulement si >60% réellement absent dans l'ASC.
    # Ce taux utilise pupil_missing_acquisition, pas les blinks/padding.
    max_run_acquisition_missing_fraction: float = 0.45

    # Une fenêtre est exclue si >50% du signal est vraiment absent.
    max_moment_missing_fraction: float = 0.30

    # Baseline invalide si >50% manquant.
    max_baseline_missing_fraction: float = 0.25

    # Conservé pour compatibilité, mais le run n'est plus exclu sur ce critère.
    min_valid_trial_fraction_per_run: float = 0.0

    # Sujet exclu seulement s'il a moins de deux runs réellement exploitables.
    min_valid_runs_per_subject: int = 2

    # Paramètres QC exploratoires (à valider sur distributions aveugles aux conditions)
    max_interpolated_fraction_window: float = 0.20
    min_observed_fraction_window: float = 0.70
    min_observed_fraction_baseline: float = 0.75
    max_longest_gap_window_ms: float = 500.0
    max_window_missing_sampling_fraction: float = 0.05
    min_gaze_valid_fraction: float = 0.70
    max_sync_drift_ppm_warning: float = 150.0
    max_sync_pulse_interval_cv: float = 0.05
    gaze_screen_width_px: float = 1920.0
    gaze_screen_height_px: float = 1080.0
    # --------------------------------------------------------------
    # Fenêtres temporelles : analyses principales
    # --------------------------------------------------------------

    # Induction précoce : réflexe photomoteur/luminance.
    # Cette fenêtre est séparée et n'est pas interprétée comme arousal.
    induction_early_start_s: float = 0.0
    induction_early_end_s: float = 2.0

    # H1 principal : réponse émotionnelle soutenue.
    induction_late_start_s: float = 2.0
    induction_late_end_s: float = 10.0

    # H2 principal : fenêtre juste avant le clic "idée trouvée".
    fixed_pre_idea_window_s: float = 1.0
    pre_idea_motor_exclusion_s: float = 0.2
    liking_fixed_window_s: float = 1.0

    # H3 principal : première appréciation, avant le premier mouvement.
    liking_use_first_movement: bool = True

    # Une fenêtre tronquée est conservée si elle contient au moins 500 ms.
    minimum_usable_window_s: float = 0.5

    # Baseline soustractive : dernières 500 ms de la croix de fixation précédente.
    baseline_duration_s: float = 0.5

    # --------------------------------------------------------------
    # Analyses de sensibilité / robustesse
    #
    # Ces paramètres ne doivent pas être décrits comme confirmatoires
    # puisque les résultats principaux ont déjà été observés.
    # --------------------------------------------------------------

    # H1 : sensibilité au début de la fenêtre émotionnelle tardive.
    # La fin reste fixée à 10 s, durée de l'induction image+son.
    induction_late_start_values_s: tuple[float, ...] = (
        1.5,
        2.0,
        2.5,
        3.0,
    )

    # H2 : sensibilité à la durée précédant le clic idée.
    pre_idea_window_values_s: tuple[float, ...] = (
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    )

    # H3 : deux définitions conceptuellement distinctes.
    # first_movement = évaluation initiale avant mouvement curseur.
    # validation = évaluation complète jusqu'à validation finale.
    liking_endpoints: tuple[str, ...] = (
        "first_movement",
        "validation",
    )

    # Baseline : sensibilité optionnelle, non activée par défaut.
    # Garder une seule valeur pour l'analyse principale.
    baseline_duration_values_s: tuple[float, ...] = (
        0.5,
    )

    # --------------------------------------------------------------
    # Evènements oculomoteurs
    # --------------------------------------------------------------

    # Valeur principale utilisée pour séparer microsaccades / saccades.
    microsaccade_max_amplitude_deg: float = 1.0

    # Sensibilité du seuil microsaccade.
    microsaccade_threshold_values_deg: tuple[float, ...] = (
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    )

    # --------------------------------------------------------------
    # ROI gaze aversion
    # --------------------------------------------------------------
    # À renseigner si les coordonnées réelles du mot-indice sont connues.
    # Si None, la métrique gaze_aversion_fraction devient NaN.
    cue_roi_center_x: float | None = None
    cue_roi_center_y: float | None = None

    cue_roi_half_width_px: float = 200.0
    cue_roi_half_height_px: float = 100.0

    # Limite mémoire pour les données utilisées dans les heatmaps gaze.
    gaze_plot_max_samples_per_trial: int = 2000

    # --------------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------------
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"

    min_rows_for_model: int = 30
    min_subjects_for_model: int = 12
    min_levels_for_model: int = 2

    # --------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------
    dpi: int = 180


DEFAULT_CONFIG = Config()


def make_output_dirs(output_root: Path) -> dict[str, Path]:
    """Crée l'arborescence de sortie."""
    paths = {
        "root": output_root,
        "tables": output_root / "tables",
        "logs": output_root / "logs",
        "figures": output_root / "figures",
        "figures_exclusion": output_root / "figures" / "exclusion",
        "figures_results": output_root / "figures" / "results",
        "figures_gaze": output_root / "figures" / "gaze_heatmaps",
        "figures_distributions": output_root / "figures" / "distributions",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths