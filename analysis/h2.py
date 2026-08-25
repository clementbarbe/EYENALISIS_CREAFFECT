import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import PARAMS
from analysis.common import (
    get_subject_data,
    subject_output_dir,
)


def analyse_subject_h2(
    participant_id,
    metric="pup_mean",
    correlation_method="spearman",
):
    """
    H2 — Recherche / génération d'idée.

    Analyse :
    1. Descriptif de la métrique par catégorie émotionnelle.
    2. Corrélation entre la métrique et ideation_dur.
    """

    df = get_subject_data("h2", participant_id)

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H2 : {metric}"
        )

    if "ideation_dur" not in df.columns:
        raise ValueError(
            "La colonne ideation_dur est absente du fichier H2."
        )

    output_dir = subject_output_dir(participant_id, "h2")

    categories = PARAMS["emotion_categories"]

    # ─────────────────────────────────────────────────────────
    # RÉSUMÉ PAR CATÉGORIE
    # ─────────────────────────────────────────────────────────

    df_cat = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    summary = (
        df_cat.groupby("stim_category")[metric]
        .agg(["mean", "std", "median", "count"])
        .reindex(categories)
    )

    print("\n" + "=" * 60)
    print("H2 — GÉNÉRATION D'IDÉE")
    print("=" * 60)
    print(f"Participant : {participant_id}")
    print(f"Métrique : {metric}")
    print("\nRésumé par catégorie :")
    print(summary)

    summary.to_csv(
        output_dir / f"h2_{metric}_by_category.csv"
    )

    # ─────────────────────────────────────────────────────────
    # CORRÉLATION AVEC DURÉE D'IDÉATION
    # ─────────────────────────────────────────────────────────

    corr_data = df[
        [metric, "ideation_dur"]
    ].dropna()

    correlation_result = {
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
        and corr_data["ideation_dur"].nunique() >= 2
    ):
        if correlation_method == "spearman":
            r_value, p_value = stats.spearmanr(
                corr_data[metric],
                corr_data["ideation_dur"],
            )
        else:
            r_value, p_value = stats.pearsonr(
                corr_data[metric],
                corr_data["ideation_dur"],
            )

        correlation_result["r"] = r_value
        correlation_result["p"] = p_value

    print("\nRelation métrique × durée d'idéation :")
    print(
        f"{correlation_method}: "
        f"r = {correlation_result['r']:.3f}, "
        f"p = {correlation_result['p']:.4f}, "
        f"n = {correlation_result['n_trials']}"
    )

    # ─────────────────────────────────────────────────────────
    # FIGURE
    # ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    # A. Barplot par catégorie
    colors = [
        PARAMS["emotion_colors"].get(cat, "gray")
        for cat in categories
    ]

    axes[0].bar(
        categories,
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
        color=colors,
        alpha=0.75,
    )

    axes[0].set_title(
        f"H2 — {participant_id}\n{metric} par catégorie"
    )
    axes[0].set_xlabel("Catégorie émotionnelle")
    axes[0].set_ylabel(metric)
    axes[0].grid(axis="y", alpha=0.25)

    # B. Scatter métrique × durée
    axes[1].scatter(
        corr_data["ideation_dur"],
        corr_data[metric],
        alpha=0.75,
        color="steelblue",
    )

    if len(corr_data) >= 2:
        slope, intercept, _, _, _ = stats.linregress(
            corr_data["ideation_dur"],
            corr_data[metric],
        )

        x_values = np.linspace(
            corr_data["ideation_dur"].min(),
            corr_data["ideation_dur"].max(),
            100,
        )

        axes[1].plot(
            x_values,
            intercept + slope * x_values,
            color="black",
            linewidth=2,
        )

    axes[1].set_title(
        f"{metric} × durée d'idéation\n"
        f"{correlation_method} r={correlation_result['r']:.3f}, "
        f"p={correlation_result['p']:.4f}"
    )

    axes[1].set_xlabel("Durée d'idéation (s)")
    axes[1].set_ylabel(metric)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h2_{metric}.png",
        dpi=150,
    )

    plt.show()

    import pandas as pd

    pd.DataFrame([correlation_result]).to_csv(
        output_dir / f"h2_{metric}_duration_correlation.csv",
        index=False,
    )

    return {
        "summary": summary,
        "correlation": correlation_result,
    }