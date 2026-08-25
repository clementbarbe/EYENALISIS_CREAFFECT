import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from analysis.common import (
    get_subject_data,
    subject_output_dir,
)


H3_WINDOWS = {
    "movement": {
        "mean": "pup_mean_m",
        "max": "pup_max_m",
        "slope": "pup_slope_m",
        "auc": "pup_auc_m",
        "duration": "dur_to_move",
    },

    "validation": {
        "mean": "pup_mean_v",
        "max": "pup_max_v",
        "slope": "pup_slope_v",
        "auc": "pup_auc_v",
        "duration": "dur_to_valid",
    },
}


def analyse_subject_h3(
    participant_id,
    metric="pup_mean_v",
    correlation_method="spearman",
):
    """
    H3 — Évaluation de l'idée.

    Étudie la relation entre une métrique pupillaire et le rating.
    """

    df = get_subject_data("h3", participant_id)

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H3 : {metric}"
        )

    if "rating" not in df.columns:
        raise ValueError(
            "La colonne rating est absente du fichier H3."
        )

    output_dir = subject_output_dir(participant_id, "h3")

    corr_data = df[
        [metric, "rating"]
    ].dropna()

    result = {
        "participant": participant_id,
        "metric": metric,
        "method": correlation_method,
        "n_trials": len(corr_data),
        "r": np.nan,
        "p": np.nan,
    }

    if (
        len(corr_data) >= 4
        and corr_data[metric].nunique() >= 2
        and corr_data["rating"].nunique() >= 2
    ):
        if correlation_method == "spearman":
            r_value, p_value = stats.spearmanr(
                corr_data[metric],
                corr_data["rating"],
            )
        else:
            r_value, p_value = stats.pearsonr(
                corr_data[metric],
                corr_data["rating"],
            )

        result["r"] = r_value
        result["p"] = p_value

    print("\n" + "=" * 60)
    print("H3 — ÉVALUATION")
    print("=" * 60)
    print(f"Participant : {participant_id}")
    print(f"Métrique : {metric}")
    print(
        f"{correlation_method}: "
        f"r = {result['r']:.3f}, "
        f"p = {result['p']:.4f}, "
        f"n = {result['n_trials']}"
    )

    # Résumé par niveau de rating
    rating_summary = (
        corr_data.groupby("rating")[metric]
        .agg(["mean", "std", "median", "count"])
        .sort_index()
    )

    rating_summary.to_csv(
        output_dir / f"h3_{metric}_by_rating.csv"
    )

    # ─────────────────────────────────────────────────────────
    # FIGURE
    # ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    # Scatter plot
    axes[0].scatter(
        corr_data["rating"],
        corr_data[metric],
        alpha=0.75,
        color="darkred",
    )

    if len(corr_data) >= 2:
        slope, intercept, _, _, _ = stats.linregress(
            corr_data["rating"],
            corr_data[metric],
        )

        x_values = np.linspace(
            corr_data["rating"].min(),
            corr_data["rating"].max(),
            100,
        )

        axes[0].plot(
            x_values,
            intercept + slope * x_values,
            color="black",
            linewidth=2,
        )

    axes[0].set_title(
        f"{metric} × rating\n"
        f"{correlation_method} r={result['r']:.3f}, "
        f"p={result['p']:.4f}"
    )

    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel(metric)
    axes[0].grid(alpha=0.25)

    # Moyenne par rating
    axes[1].errorbar(
        rating_summary.index,
        rating_summary["mean"],
        yerr=rating_summary["std"],
        marker="o",
        linewidth=2,
        capsize=5,
        color="darkred",
    )

    axes[1].set_title(
        f"H3 — {participant_id}\n{metric} par niveau de rating"
    )
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel(metric)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h3_{metric}.png",
        dpi=150,
    )

    plt.show()

    pd.DataFrame([result]).to_csv(
        output_dir / f"h3_{metric}_rating_correlation.csv",
        index=False,
    )

    return {
        "rating_summary": rating_summary,
        "correlation": result,
    }