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


# ─────────────────────────────────────────────────────────────
# FENÊTRES D'ANALYSE
# ─────────────────────────────────────────────────────────────

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

GROUP_QC = {
    "max_pupil_data_loss_pct": 30.0,
    "max_gaze_data_loss_pct": 30.0,
    "min_pupil_valid_pct": 60.0,
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