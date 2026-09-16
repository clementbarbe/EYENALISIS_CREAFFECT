"""Configuration centralisée du pipeline FGAT eyetracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Tous les paramètres modifiables du pipeline."""

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
    max_sync_end_difference_s: float = 10.0
    exclude_bad_sync_end_check: bool = False

    # --------------------------------------------------------------
    # Prétraitement pupillaire
    # --------------------------------------------------------------
    blink_padding_before_ms: int = 100
    blink_padding_after_ms: int = 100

    min_plausible_pupil: float = 0.01
    min_unique_pupil_values: int = 20

    pupil_outlier_mad_threshold: float = 6.0
    pupil_drop_mad_threshold: float = 8.0

    interpolation_max_gap_ms: int = 500

    lowpass_enabled: bool = True
    lowpass_cutoff_hz: float = 6.0
    lowpass_order: int = 4

    # --------------------------------------------------------------
    # Exclusions
    # --------------------------------------------------------------
    # Seuil confirmé dans le brief : >20% de données invalides => run exclu.
    max_run_missing_fraction: float = 0.35

    # Fenêtre invalide si plus de 25% d'échantillons manquants.
    max_moment_missing_fraction: float = 0.25

    # Baseline invalide si plus de 25% d'échantillons manquants.
    max_baseline_missing_fraction: float = 0.25

    # Exclusion du run si trop peu d'essais dynamiques valides.
    min_valid_trial_fraction_per_run: float = 0.50

    # Valeur de X utilisée uniquement pour la QC run.
    # Les analyses statistiques sont ensuite faites pour TOUS les X.
    run_qc_x_s: float = 1.0

    # Sujet exclu si moins de trois runs valides.
    min_valid_runs_per_subject: int = 3

    # --------------------------------------------------------------
    # Fenêtres temporelles
    # --------------------------------------------------------------
    induction_duration_s: float = 10.0

    # Exploration systématique :
    # 0.5, 0.6, 0.7, ..., 2.0 secondes.
    dynamic_x_values_s: tuple[float, ...] = tuple(
        round(0.5 + 0.1 * index, 1)
        for index in range(16)
    )

    # Une fenêtre tronquée plus courte que 0.5 s est rejetée.
    minimum_usable_window_s: float = 0.5

    # Baseline : dernières 500 ms de la croix de fixation précédente.
    baseline_duration_s: float = 0.5

    # --------------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------------
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"

    min_rows_for_model: int = 12
    min_subjects_for_model: int = 5
    min_levels_for_model: int = 2

    # Important : la FDR inclut toutes les valeurs X explorées.
    # Cela évite de sélectionner a posteriori le X avec la plus petite p-value.
    fdr_across_all_x_values: bool = True

    # --------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------
    dpi: int = 180

    # Produit aussi les figures descriptives si modèle non estimable.
    plot_non_estimable_hypotheses: bool = True


DEFAULT_CONFIG = Config()


def make_output_dirs(output_root: Path) -> dict[str, Path]:
    """Crée l'arborescence complète de sortie."""
    paths = {
        "root": output_root,
        "tables": output_root / "tables",
        "logs": output_root / "logs",
        "figures": output_root / "figures",
        "figures_exclusion": output_root / "figures" / "exclusion",
        "figures_heatmaps": output_root / "figures" / "heatmaps",
        "figures_hypotheses": output_root / "figures" / "hypotheses",
        "figures_results_tables": output_root / "figures" / "results_tables",
        "figures_x_stability": output_root / "figures" / "x_stability",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths