"""Configuration centralisée du pipeline FGAT x eyetracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Paramètres modifiables du pipeline."""

    # ------------------------------------------------------------------
    # Acquisition EyeLink
    # ------------------------------------------------------------------
    sampling_rate_hz: float = 1000.0
    eye: str = "R"

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------
    sync_button_code: int = 1
    sync_button_state: int = 1

    # Contrôle secondaire de cohérence temporelle.
    max_sync_end_difference_s: float = 10.0

    # False recommandé : le contrôle de fin est informatif, mais il ne faut
    # pas exclure automatiquement avant validation du format réel des ASC.
    exclude_bad_sync_end_check: bool = False

    # ------------------------------------------------------------------
    # Prétraitement pupillaire
    # ------------------------------------------------------------------
    blink_padding_before_ms: int = 100
    blink_padding_after_ms: int = 100

    # Valeur minimale plausible pour la pupille.
    min_plausible_pupil: float = 0.01

    # Seuil MAD pour les valeurs aberrantes de pupille.
    pupil_outlier_mad_threshold: float = 6.0

    # Interpolation des trous courts uniquement.
    interpolation_max_gap_ms: int = 500

    # Filtre passe-bas Butterworth.
    lowpass_enabled: bool = True
    lowpass_cutoff_hz: float = 6.0
    lowpass_order: int = 4

    # Signal considéré comme anormalement constant sous ce seuil.
    min_unique_pupil_values: int = 20

    # ------------------------------------------------------------------
    # Exclusions
    # ------------------------------------------------------------------
    # Valeur confirmée dans le brief : >20 % => run exclu.
    max_run_missing_fraction: float = 0.20

    # Exclusion d'une fenêtre essai x moment.
    max_moment_missing_fraction: float = 0.25

    # Run exclu si moins de 50 % des essais possèdent au moins un moment valide.
    min_valid_trial_fraction_per_run: float = 0.50

    # Sujet exclu si moins de 3 runs expérimentaux valides.
    min_valid_runs_per_subject: int = 3

    # ------------------------------------------------------------------
    # Fenêtres temporelles
    # ------------------------------------------------------------------
    induction_duration_s: float = 10.0

    moment_window_s: dict[str, float] = field(default_factory=lambda: {
        "M2a_search_start": 2.0,
        "M2b_search_end": 2.0,
        "M3a_rating_start": 2.0,
        "M3b_rating_end": 2.0,
    })

    minimum_usable_window_s: float = 0.5

    # Baseline provisoire : 500 ms avant l'onset du moment.
    # À remplacer par la croix précédente si ses timestamps deviennent disponibles.
    baseline_duration_s: float = 0.5

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------
    alpha: float = 0.05
    fdr_method: str = "fdr_bh"

    min_rows_for_model: int = 12
    min_subjects_for_model: int = 5
    min_levels_for_model: int = 2

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    dpi: int = 160

    # Si True, produit aussi les graphiques descriptifs lorsque le modèle
    # est non estimable/non fiable.
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