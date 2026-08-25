import numpy as np

from scipy import stats
from scipy.ndimage import gaussian_filter


def baseline_correct_global(
    pupil,
    time_rel,
    event_time_rel,
    pre_ms=500,
):
    """
    Calcule la baseline pupillaire précédant l'événement.
    """

    baseline_start = event_time_rel - pre_ms / 1000.0

    mask = (
        (time_rel >= baseline_start)
        &
        (time_rel <= event_time_rel)
    )

    values = pupil[mask]
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def pupil_metrics(pupil, time):
    """
    Métriques pupillaires.

    Les métriques sont calculées sur les données déjà baseline-corrigées
    ou brutes selon le signal fourni à la fonction.
    """

    valid = np.isfinite(pupil)

    output = {
        "n_valid": int(valid.sum()),
        "pct_valid": (
            100 * valid.sum() / max(len(pupil), 1)
        ),
    }

    metric_names = [
        "mean",
        "std",
        "max",
        "min",
        "range",
        "slope",
        "auc",
        "auc_per_s",
        "peak_time",
    ]

    if valid.sum() < 2:
        for name in metric_names:
            output[name] = np.nan
        return output

    p = pupil[valid]
    t = time[valid]

    duration = (
        float(t[-1] - t[0])
        if len(t) > 1
        else np.nan
    )

    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(p, t)
    else:
        auc = np.trapz(p, t)

    output["mean"] = float(np.mean(p))
    output["std"] = float(np.std(p))

    output["max"] = float(np.max(p))
    output["min"] = float(np.min(p))
    output["range"] = output["max"] - output["min"]

    output["auc"] = float(auc)

    output["auc_per_s"] = (
        float(auc / duration)
        if np.isfinite(duration) and duration > 0
        else np.nan
    )

    output["peak_time"] = float(
        t[np.argmax(p)] - t[0]
    )

    output["slope"] = (
        float(stats.linregress(t, p).slope)
        if len(t) > 2
        else np.nan
    )

    return output


def gaze_metrics(
    x,
    y,
    duration_s=None,
):
    """
    Métriques spatiales de regard.

    path_len_per_s est préférable à path_len quand les essais
    n'ont pas tous la même durée.
    """

    valid = np.isfinite(x) & np.isfinite(y)

    output = {
        "n_valid": int(valid.sum()),
    }

    metric_names = [
        "mean_x",
        "mean_y",
        "disp_x",
        "disp_y",
        "disp",
        "path_len",
        "path_len_per_s",
    ]

    if valid.sum() < 2:
        for name in metric_names:
            output[name] = np.nan
        return output

    x_valid = x[valid]
    y_valid = y[valid]

    output["mean_x"] = float(np.mean(x_valid))
    output["mean_y"] = float(np.mean(y_valid))

    output["disp_x"] = float(np.std(x_valid))
    output["disp_y"] = float(np.std(y_valid))

    output["disp"] = float(
        np.sqrt(
            output["disp_x"] ** 2
            +
            output["disp_y"] ** 2
        )
    )

    path_length = np.sum(
        np.sqrt(
            np.diff(x_valid) ** 2
            +
            np.diff(y_valid) ** 2
        )
    )

    output["path_len"] = float(path_length)

    output["path_len_per_s"] = (
        float(path_length / duration_s)
        if duration_s is not None
        and np.isfinite(duration_s)
        and duration_s > 0
        else np.nan
    )

    return output


def gaze_entropy(
    x,
    y,
    params,
):
    """
    Entropie spatiale normalisée de Shannon.

    0 : regard très concentré.
    1 : regard très dispersé sur la grille spatiale.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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

    sigma = params.get(
        "entropy_smoothing_sigma",
        0.0,
    )

    if sigma > 0:
        histogram = gaussian_filter(
            histogram,
            sigma=sigma,
        )

    total = histogram.sum()

    if total <= 0:
        return np.nan

    probabilities = histogram.ravel() / total
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    if params.get("normalize_entropy", True):
        entropy = entropy / np.log(nx * ny)

    return float(entropy)