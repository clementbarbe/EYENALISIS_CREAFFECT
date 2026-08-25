import numpy as np


def classify_data_loss(loss_pct):
    if not np.isfinite(loss_pct):
        return "inconnu"

    if loss_pct < 10:
        return "bon"

    if loss_pct < 20:
        return "acceptable"

    if loss_pct < 30:
        return "à surveiller"

    return "faible"


def classify_blink_rate(rate_per_min):
    if not np.isfinite(rate_per_min):
        return "inconnu"

    if rate_per_min < 5:
        return "faible"

    if rate_per_min <= 30:
        return "typique"

    if rate_per_min <= 50:
        return "élevé"

    return "très élevé"


def compute_blink_qc(blinks_df, time_raw):
    if len(time_raw) < 2:
        duration_ms = np.nan
    else:
        duration_ms = time_raw[-1] - time_raw[0]

    duration_min = duration_ms / 60000 if duration_ms > 0 else np.nan

    if blinks_df is None or len(blinks_df) == 0:
        return {
            "n_blinks": 0,
            "blink_rate_per_min": 0.0,
            "blink_duration_mean_ms": np.nan,
            "blink_duration_median_ms": np.nan,
            "blink_time_total_ms": 0.0,
            "blink_time_pct": 0.0,
        }

    blink_durations = blinks_df["duration"].dropna().values

    total_blink_time = np.sum(blink_durations)

    return {
        "n_blinks": int(len(blinks_df)),
        "blink_rate_per_min": (
            float(len(blinks_df) / duration_min)
            if np.isfinite(duration_min) and duration_min > 0
            else np.nan
        ),
        "blink_duration_mean_ms": (
            float(np.mean(blink_durations))
            if len(blink_durations)
            else np.nan
        ),
        "blink_duration_median_ms": (
            float(np.median(blink_durations))
            if len(blink_durations)
            else np.nan
        ),
        "blink_time_total_ms": float(total_blink_time),
        "blink_time_pct": (
            float(100 * total_blink_time / duration_ms)
            if np.isfinite(duration_ms) and duration_ms > 0
            else np.nan
        ),
    }


def compute_data_loss_qc(
    time_raw,
    x_raw,
    y_raw,
    pupil_raw,
    params,
):
    n_total = len(time_raw)

    pupil_nan = np.isnan(pupil_raw)
    pupil_zero_or_negative = pupil_raw <= 0
    pupil_too_small = pupil_raw < params["min_pupil"]

    pupil_missing = (
        pupil_nan
        |
        pupil_zero_or_negative
        |
        pupil_too_small
    )

    gaze_missing = (
        np.isnan(x_raw)
        |
        np.isnan(y_raw)
        |
        (x_raw < 0)
        |
        (x_raw > params["screen_width"])
        |
        (y_raw < 0)
        |
        (y_raw > params["screen_height"])
    )

    return {
        "n_samples_total": int(n_total),

        "n_pupil_missing_raw": int(np.sum(pupil_missing)),
        "pupil_data_loss_pct": (
            float(100 * np.mean(pupil_missing))
            if n_total else np.nan
        ),

        "n_gaze_missing_raw": int(np.sum(gaze_missing)),
        "gaze_data_loss_pct": (
            float(100 * np.mean(gaze_missing))
            if n_total else np.nan
        ),

        "n_pupil_zero_or_negative": int(
            np.sum(pupil_zero_or_negative)
        ),

        "n_pupil_nan": int(
            np.sum(pupil_nan)
        ),
    }


def compute_pupil_qc(
    pupil_raw,
    pupil_clean,
    sampling_rate,
):
    raw_valid = pupil_raw[
        np.isfinite(pupil_raw)
        &
        (pupil_raw > 0)
    ]

    clean_valid = pupil_clean[
        np.isfinite(pupil_clean)
    ]

    if len(clean_valid) >= 2:
        derivative = np.diff(clean_valid)

        median_derivative = np.median(derivative)
        mad_derivative = np.median(
            np.abs(derivative - median_derivative)
        )

        jump_threshold = (
            5 * 1.4826 * mad_derivative
            if mad_derivative > 0
            else np.nan
        )

        n_jumps = (
            int(np.sum(np.abs(derivative) > jump_threshold))
            if np.isfinite(jump_threshold)
            else 0
        )

    else:
        jump_threshold = np.nan
        n_jumps = 0

    return {
        "pupil_raw_mean": (
            float(np.mean(raw_valid))
            if len(raw_valid) else np.nan
        ),

        "pupil_raw_median": (
            float(np.median(raw_valid))
            if len(raw_valid) else np.nan
        ),

        "pupil_raw_min": (
            float(np.min(raw_valid))
            if len(raw_valid) else np.nan
        ),

        "pupil_raw_max": (
            float(np.max(raw_valid))
            if len(raw_valid) else np.nan
        ),

        "pupil_clean_mean": (
            float(np.mean(clean_valid))
            if len(clean_valid) else np.nan
        ),

        "pupil_clean_sd": (
            float(np.std(clean_valid))
            if len(clean_valid) else np.nan
        ),

        "n_pupil_jumps": n_jumps,
        "jump_threshold": jump_threshold,
        "sampling_rate": sampling_rate,
    }