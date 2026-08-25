import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import PARAMS
from analysis.common import (
    get_subject_data,
    subject_output_dir,
)


def analyse_subject_h4(
    participant_id,
    metric="gaze_entropy",
    correlation_method="spearman",
):
    """
    H4 — Dynamique spatiale du regard pendant l'idéation.

    Exemples de métriques :
        gaze_disp
        gaze_path_len_per_s
        gaze_entropy
        fix_rate
        sac_rate
        mean_fix_dur
    """

    df = get_subject_data("h4", participant_id)

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H4 : {metric}"
        )

    if "ideation_dur" not in df.columns:
        raise ValueError(
            "La colonne ideation_dur est absente du fichier H4."
        )

    output_dir = subject_output_dir(participant_id, "h4")

    categories = PARAMS["emotion_categories"]

    # ─────────────────────────────────────────────────────────
    # PAR CATÉGORIE
    # ─────────────────────────────────────────────────────────

    df_cat = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    summary = (
        df_cat.groupby("stim_category")[metric]
        .agg(["mean", "std", "median", "count"])
        .reindex(categories)
    )

    summary.to_csv(
        output_dir / f"h4_{metric}_by_category.csv"
    )

    # ─────────────────────────────────────────────────────────
    # CORRÉLATION AVEC DURÉE
    # ─────────────────────────────────────────────────────────

    corr_data = df[
        [metric, "ideation_dur"]
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

        result["r"] = r_value
        result["p"] = p_value

    print("\n" + "=" * 60)
    print("H4 — DYNAMIQUE SPATIALE DU REGARD")
    print("=" * 60)
    print(f"Participant : {participant_id}")
    print(f"Métrique : {metric}")
    print("\nRésumé par catégorie :")
    print(summary)

    print("\nRelation métrique × durée d'idéation :")
    print(
        f"{correlation_method}: "
        f"r = {result['r']:.3f}, "
        f"p = {result['p']:.4f}, "
        f"n = {result['n_trials']}"
    )

    # ─────────────────────────────────────────────────────────
    # FIGURES
    # ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

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
        f"H4 — {participant_id}\n{metric} par catégorie"
    )

    axes[0].set_xlabel("Catégorie émotionnelle")
    axes[0].set_ylabel(metric)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(
        corr_data["ideation_dur"],
        corr_data[metric],
        alpha=0.75,
        color="seagreen",
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
        f"{correlation_method} r={result['r']:.3f}, "
        f"p={result['p']:.4f}"
    )

    axes[1].set_xlabel("Durée d'idéation (s)")
    axes[1].set_ylabel(metric)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h4_{metric}.png",
        dpi=150,
    )

    plt.show()

    pd.DataFrame([result]).to_csv(
        output_dir / f"h4_{metric}_duration_correlation.csv",
        index=False,
    )

    return {
        "summary": summary,
        "correlation": result,
    }