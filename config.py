from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

TRIAL_DIR = RESULTS_DIR / "trial_level"
QC_DIR = RESULTS_DIR / "qc"
SUBJECT_DIR = RESULTS_DIR / "subject"
GROUP_DIR = RESULTS_DIR / "group"

for directory in [
    RESULTS_DIR,
    TRIAL_DIR,
    QC_DIR,
    SUBJECT_DIR,
    GROUP_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PARAMÈTRES EYE-TRACKING
# ─────────────────────────────────────────────────────────────

PARAMS = {
    "screen_width": 1920,
    "screen_height": 1080,

    "sampling_rate": 1000,

    "blink_margin_ms": 100,

    "min_pupil": 100,

    # Recommandé plutôt que 2.0 :
    "mad_threshold": 3.0,

    "pupil_lowpass_cutoff": 4.0,
    "pupil_lowpass_order": 2,

    "baseline_pre_ms": 500,

    "entropy_bins_x": 24,
    "entropy_bins_y": 14,
    "entropy_min_samples": 100,
    "entropy_smoothing_sigma": 0.0,
    "normalize_entropy": True,

    "emotion_categories": [
        "HVHA",
        "HVLA",
        "LVHA",
        "LVLA",
        "MVMA",
    ],

    "emotion_colors": {
        "HVHA": "#e74c3c",
        "HVLA": "#e67e22",
        "LVHA": "#3498db",
        "LVLA": "#2ecc71",
        "MVMA": "#95a5a6",
    },
}


ANALYSIS_WINDOWS = {
    "h1": {
        "main": 10.0,
        "exploratory": [3.0, 5.0, 8.0, 10.0],
    },

    "h2": {
        "max_ideation_duration_s": 60.0,
        "fixed_windows_s": [3.0, 5.0, 8.0, 10.0],
        "include_full_trial": True,
    },

    "h3": {
        "fixed_windows_s": [3.0, 5.0, 8.0],
        "include_to_movement": True,
        "include_to_validation": True,
    },

    "h4": {
        "fixed_windows_s": [3.0, 5.0, 8.0, 10.0],
        "include_full_ideation": True,
    },
}
# ─────────────────────────────────────────────────────────────
# CRITÈRES QC GROUPE
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# CRITÈRES QC POUR LES ANALYSES DE GROUPE
# ─────────────────────────────────────────────────────────────
#
# Ces critères sont appliqués au niveau participant × run.
#
# Mettre une valeur à None désactive le critère correspondant.
#
# Important :
# - Ne pas modifier les seuils après avoir regardé les résultats
#   statistiques pour "obtenir" un effet.
# - Les seuils ci-dessous sont un point de départ raisonnable.
# - Vérifier ensuite combien de runs/sujets sont retenus.
# ─────────────────────────────────────────────────────────────

GROUP_QC = {

    # Qualité du signal pupillaire APRÈS prétraitement.
    # Exclure le run si moins de 60% de samples pupillaires restent valides.
    "min_pupil_valid_pct": 82,

    # Perte brute pupillaire avant interpolation / nettoyage.
    "max_pupil_data_loss_pct": 35.0,

    # Perte brute du regard (x/y invalides ou hors écran).
    "max_gaze_data_loss_pct": 35.0,

    # Temps total couvert par des événements EBLINK.
    # Un run avec > 15% de temps de blink sera exclu.
    "max_blink_time_pct": 15,

    # Critère optionnel.
    # 60 blinks/min est volontairement assez permissif.
    # Mettre None si tu ne veux pas exclure sur ce critère.
    "max_blink_rate_per_min": 60.0,

    # Run trop court : peut indiquer un enregistrement interrompu.
    "min_run_duration_s": 60.0,

    # Nombre minimal de samples acquis.
    # À 1000 Hz, 60 000 = environ 60 secondes.
    "min_n_samples_total": 60000,

    # Nombre minimal de runs retenus chez un sujet pour le garder
    # dans les analyses groupe.
    #
    # 1 = un sujet peut contribuer même avec un seul run propre.
    # 2 ou 3 = plus strict.
    "min_valid_runs_per_participant": 1,

    # Critère pour décider si on applique le QC.
    "apply_qc": True,

    # Alpha statistique.
    "alpha": 0.05,
}


# ─────────────────────────────────────────────────────────────
# MÉTRIQUES PAR HYPOTHÈSE
# ─────────────────────────────────────────────────────────────

H1_METRICS = [
    "pup_mean",
    "pup_max",
    "pup_slope",
    "sac_freq",
    "gaze_disp",
    "gaze_entropy",
]

H2_METRICS = [
    "pup_mean",
    "pup_max",
    "pup_slope",
    "pup_auc_per_s",
    "gaze_entropy",
    "ideation_dur",
]

H3_METRICS = [
    "pup_mean_m",
    "pup_max_m",
    "pup_slope_m",
    "pup_mean_v",
    "pup_max_v",
    "pup_slope_v",
]

H4_METRICS = [
    "gaze_disp",
    "gaze_path_len_per_s",
    "gaze_entropy",
    "fix_rate",
    "sac_rate",
    "mean_fix_dur",
]