import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter


def baseline_correct_global(
    pupil,
    time_rel,
    event_time_rel,
    pre_ms=500,
):
    baseline_start = event_time_rel - pre_ms / 1000

    mask = (
        (time_rel >= baseline_start)
        & (time_rel <= event_time_rel)
    )

    values = pupil[mask]
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def pupil_metrics(pupil, time):
    valid = np.isfinite(pupil)

    output = {
        "n_valid": int(valid.sum()),
        "pct_valid": 100 * valid.sum() / max(len(pupil), 1),
    }

    if valid.sum() < 2:
        for key in [
            "mean", "std", "max", "min",
            "range", "slope", "auc",
            "auc_per_s", "peak_time",
        ]:
            output[key] = np.nan

        return output

    p = pupil[valid]
    t = time[valid]

    duration = t[-1] - t[0] if len(t) > 1 else np.nan

    auc = np.trapezoid(p, t) if hasattr(np, "trapezoid") else np.trapz(p, t)

    output["mean"] = float(np.mean(p))
    output["std"] = float(np.std(p))
    output["max"] = float(np.max(p))
    output["min"] = float(np.min(p))
    output["range"] = output["max"] - output["min"]

    output["auc"] = float(auc)
    output["auc_per_s"] = float(auc / duration) if duration > 0 else np.nan

    output["peak_time"] = float(t[np.argmax(p)] - t[0])

    output["slope"] = (
        float(stats.linregress(t, p).slope)
        if len(t) > 2 else np.nan
    )

    return output


def gaze_metrics(x, y, duration_s=None):
    valid = np.isfinite(x) & np.isfinite(y)

    output = {
        "n_valid": int(valid.sum()),
    }

    if valid.sum() < 2:
        for key in [
            "mean_x", "mean_y",
            "disp_x", "disp_y", "disp",
            "path_len", "path_len_per_s",
        ]:
            output[key] = np.nan

        return output

    x = x[valid]
    y = y[valid]

    output["mean_x"] = float(np.mean(x))
    output["mean_y"] = float(np.mean(y))

    output["disp_x"] = float(np.std(x))
    output["disp_y"] = float(np.std(y))

    output["disp"] = float(
        np.sqrt(output["disp_x"]**2 + output["disp_y"]**2)
    )

    path_length = np.sum(
        np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    )

    output["path_len"] = float(path_length)

    output["path_len_per_s"] = (
        float(path_length / duration_s)
        if duration_s is not None and duration_s > 0
        else np.nan
    )

    return output


def gaze_entropy(x, y, params):
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < params["entropy_min_samples"]:
        return np.nan

    nx = params["entropy_bins_x"]
    ny = params["entropy_bins_y"]

    histogram, _, _ = np.histogram2d(
        x,
        y,
        bins=[nx, ny],
        range=[
            [0, params["screen_width"]],
            [0, params["screen_height"]],
        ],
    )

    sigma = params.get("entropy_smoothing_sigma", 0.0)

    if sigma > 0:
        histogram = gaussian_filter(histogram, sigma=sigma)

    total = histogram.sum()

    if total <= 0:
        return np.nan

    probabilities = histogram.ravel() / total
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log(probabilities))

    if params.get("normalize_entropy", True):
        entropy /= np.log(nx * ny)

    return float(entropy)