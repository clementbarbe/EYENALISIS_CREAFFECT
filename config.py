"""Configuration centralisée du pipeline FGAT eyetracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Tous les paramètres modifiables du pipeline."""

    # Acquisition EyeLink
    sampling_rate_hz: float = 1000.0
    eye: str = "R"

    # Synchronisation
    sync_button_code: int = 1
    sync_button_state: int = 1
    max_sync_end_difference_s: float = 10.0
    exclude_bad_sync_end_check: bool = False

    # Prétraitement pupillaire
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

    # Exclusion des runs
    # Valeur confirmée dans le brief : >20 % => run exclu.
    max_run_missing_fraction: float = 0.35

    # Exclusion essai x moment
    max_moment_missing_fraction: float = 0.25
    min_valid_trial_fraction_per_run: float = 0.50
    min_valid_runs_per_subject: int = 3

    # Fenêtres temporelles
    induction_duration_s: float = 10.0

    moment_window_s: dict[str, float] = field(default_factory=lambda: {
        "M2a_search_start": 1.0,
        "M2b_search_end": 1.0,
        "M3a_rating_start": 1.0,
        "M3b_rating_end": 1.0,
    })

    minimum_usable_window_s: float = 0.5

    # Approximation provisoire : 500 ms avant l'onset du moment.
    # À remplacer si les timestamps exacts des croix de fixation sont disponibles.
    baseline_duration_s: float = 1

    # Statistiques
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"

    min_rows_for_model: int = 12
    min_subjects_for_model: int = 5
    min_levels_for_model: int = 2

    # Figures
    dpi: int = 180
    plot_non_estimable_hypotheses: bool = True


DEFAULT_CONFIG = Config()


def make_output_dirs(output_root: Path) -> dict[str, Path]:
    """Crée l'arborescence de sortie."""
    paths = {
        "root": output_root,
        "tables": output_root / "tables",
        "logs": output_root / "logs",
        "figures": output_root / "figures",
        "figures_exclusion": output_root / "figures" / "exclusion",
        "figures_heatmaps": output_root / "figures" / "heatmaps",
        "figures_hypotheses": output_root / "figures" / "hypotheses",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths