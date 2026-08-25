import numpy as np
from scipy import signal


def remove_blinks(time, pupil, blinks_df, margin_ms):
    output = pupil.copy()

    if blinks_df is None or len(blinks_df) == 0:
        return output

    for _, blink in blinks_df.iterrows():
        mask = (
            (time >= blink["start"] - margin_ms)
            & (time <= blink["end"] + margin_ms)
        )
        output[mask] = np.nan

    return output


def interpolate_nans(data, max_gap_samples=500):
    output = data.copy()

    nan_mask = np.isnan(output)

    if not nan_mask.any() or nan_mask.all():
        return output

    transitions = np.diff(
        np.concatenate(([0], nan_mask.astype(int), [0]))
    )

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    valid_indices = np.where(~nan_mask)[0]

    if len(valid_indices) < 2:
        return output

    for start, end in zip(starts, ends):
        if (end - start) <= max_gap_samples:
            indices = np.arange(start, end)

            output[indices] = np.interp(
                indices,
                valid_indices,
                output[valid_indices],
            )

    return output


def lowpass_filter(data, fs, cutoff, order=2):
    output = data.copy()
    valid = ~np.isnan(output)

    if valid.sum() < 20:
        return output

    b, a = signal.butter(
        order,
        cutoff / (fs / 2),
        btype="low",
    )

    padlen = 3 * (max(len(a), len(b)) - 1)

    changes = np.diff(
        np.concatenate(([0], valid.astype(int), [0]))
    )

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        segment = output[start:end]

        if len(segment) > padlen:
            try:
                output[start:end] = signal.filtfilt(b, a, segment)
            except ValueError:
                pass

    return output


def reject_artifacts_mad(data, threshold=3.0):
    """
    À utiliser avec prudence : un seuil trop faible peut retirer
    des variations physiologiques réelles de pupille.
    """
    output = data.copy()
    valid = output[np.isfinite(output)]

    if len(valid) < 10:
        return output

    median = np.median(valid)
    mad = np.median(np.abs(valid - median))

    if mad == 0:
        return output

    robust_z = 0.6745 * np.abs(output - median) / mad
    output[robust_z > threshold] = np.nan

    return output


def preprocess_pupil(time, pupil, blinks_df, params):
    pupil_clean = pupil.copy()

    pupil_clean[pupil_clean <= 0] = np.nan
    pupil_clean[pupil_clean < params["min_pupil"]] = np.nan

    pupil_clean = remove_blinks(
        time,
        pupil_clean,
        blinks_df,
        params["blink_margin_ms"],
    )

    pupil_clean = reject_artifacts_mad(
        pupil_clean,
        threshold=params["mad_threshold"],
    )

    pupil_clean = interpolate_nans(
        pupil_clean,
        max_gap_samples=int(0.5 * params["sampling_rate"]),
    )

    pupil_clean = lowpass_filter(
        pupil_clean,
        fs=params["sampling_rate"],
        cutoff=params["pupil_lowpass_cutoff"],
        order=params["pupil_lowpass_order"],
    )

    return pupil_clean


def preprocess_gaze(x, y, time, blinks_df, params):
    x_clean = x.copy()
    y_clean = y.copy()

    invalid = (
        (x_clean < 0)
        | (x_clean > params["screen_width"])
        | (y_clean < 0)
        | (y_clean > params["screen_height"])
    )

    x_clean[invalid] = np.nan
    y_clean[invalid] = np.nan

    if blinks_df is not None and len(blinks_df):

        for _, blink in blinks_df.iterrows():
            mask = (
                (time >= blink["start"] - params["blink_margin_ms"])
                & (time <= blink["end"] + params["blink_margin_ms"])
            )

            x_clean[mask] = np.nan
            y_clean[mask] = np.nan

    return x_clean, y_clean