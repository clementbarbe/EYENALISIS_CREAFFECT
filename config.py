"""Configuration centralisée du pipeline FGAT x eyetracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class Config:
    """Paramètres globaux, modifiables sans toucher aux modules."""

    # -----------------------------
    # Sorties
    # -----------------------------
    output_dir_name: str = "output"
    figures_dir_name: str = "figures"
    tables_dir_name: str = "tables"
    logs_dir_name: str = "logs"

    # Produire également des figures pour les hypothèses non estimables :
    # elles porteront l'annotation "modèle non estimable" ou la raison.
    plot_non_estimable_hypotheses: bool = True

    # -----------------------------
    # Acquisition
    # -----------------------------
    sampling_rate_hz: float = 1000.0
    eye: str = "R"

    # -----------------------------
    # Synchronisation
    # -----------------------------
    sync_button_code: int = 1
    sync_button_state: int = 1
    max_sync_end_difference_s: float = 10.0
    exclude_bad_sync_end_check: bool = False

    # -----------------------------
    # Prétraitement pupillaire
    # -----------------------------
    blink_padding_before_ms: int = 100
    blink_padding_after_ms: int = 100

    # Si SBLINK/EBLINK absents : détection défensive complémentaire
    pupil_drop_mad_threshold: float = 8.0
    pupil_outlier_mad_threshold: float = 6.0

    interpolation_max_gap_ms: int = 500

    lowpass_enabled: bool = True
    lowpass_cutoff_hz: float = 6.0
    lowpass_order: int = 4

    min_plausible_pupil: float = 0.01
    min_unique_pupil_values: int = 20

    # -----------------------------
    # Exclusions
    # -----------------------------
    # Valeur confirmée : un run est exclu au-delà de 20%.
    max_run_missing_fraction: float = 0.20
    max_moment_missing_fraction: float = 0.25
    min_valid_trial_fraction_per_run: float = 0.50
    min_valid_runs_per_subject: int = 3

    # -----------------------------
    # Fenêtres temporelles
    # -----------------------------
    dynamic_window_default_s: float = 2.0
    moment_window_s: Dict[str, float] = field(default_factory=lambda: {
        "M2a_search_start": 2.0,
        "M2b_search_end": 2.0,
        "M3a_rating_start": 2.0,
        "M3b_rating_end": 2.0,
    })

    induction_duration_s: float = 10.0
    minimum_usable_window_s: float = 0.5

    # Les CSV ne fournissent pas explicitement l'onset de chaque croix.
    # La baseline est donc une approximation : N dernières ms avant l'onset
    # du moment. Si des timestamps de croix deviennent disponibles, remplacer
    # cette stratégie dans moments.py.
    baseline_duration_s: float = 0.5

    # -----------------------------
    # Analyses statistiques
    # -----------------------------
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"
    min_rows_for_model: int = 12
    min_subjects_for_model: int = 3
    min_levels_for_model: int = 2

    # -----------------------------
    # Visualisation
    # -----------------------------
    dpi: int = 160
    temporal_curve_window_s: float = 2.0
    temporal_curve_bin_ms: int = 50


DEFAULT_CONFIG = Config()


def make_output_dirs(output_root: Path, cfg: Config) -> dict[str, Path]:
    """
    Crée et retourne une arborescence de sortie structurée.

    Parameters
    ----------
    output_root:
        Dossier parent où le dossier output sera créé.
    """
    out = output_root / cfg.output_dir_name

    paths = {
        "root": out,
        "figures": out / cfg.figures_dir_name,
        "tables": out / cfg.tables_dir_name,
        "logs": out / cfg.logs_dir_name,
        "figures_exclusion": out / cfg.figures_dir_name / "exclusion",
        "figures_heatmaps": out / cfg.figures_dir_name / "heatmaps",
        "figures_hypotheses": out / cfg.figures_dir_name / "hypotheses",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths
