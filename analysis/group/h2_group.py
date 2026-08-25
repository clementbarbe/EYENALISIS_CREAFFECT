import pandas as pd
import matplotlib.pyplot as plt

from config import (
    PARAMS,
    GROUP_DIR,
    H2_METRICS,
)
from tools.statistics import (
    paired_friedman,
    pairwise_wilcoxon,
    within_subject_correlations,
    group_test_fisher_correlations,
)
from analysis.group.common import (
    load_group_data,
    load_qc,
    apply_group_qc,
    subject_condition_table,
)


def analyse_group_h2(
    metric="pup_mean",
    mode="duration",
    correlation_method="spearman",
):
    """
    mode :
        category : comparaison des catégories émotionnelles.
        duration : relation métrique × durée d'idéation.
    """

    if metric not in H2_METRICS:
        print(
            f"⚠ {metric} n'est pas dans H2_METRICS, "
            "mais l'analyse est tentée."
        )

    df = load_group_data("h2")
    qc = load_qc()

    df = apply_group_qc(df, qc)

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H2 : {metric}"
        )

    output_dir = GROUP_DIR / "h2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "category":
        return _analyse_h2_categories(
            df=df,
            metric=metric,
            output_dir=output_dir,
        )

    if mode == "duration":
        return _analyse_h2_duration(
            df=df,
            metric=metric,
            output_dir=output_dir,
            correlation_method=correlation_method,
        )

    raise ValueError(
        "mode doit être 'category' ou 'duration'."
    )


def _analyse_h2_categories(df, metric, output_dir):
    categories = PARAMS["emotion_categories"]

    df = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    subject_table = subject_condition_table(
        df,
        metric=metric,
    ).reindex(columns=categories)

    stat, pvalue, n_subjects = paired_friedman(
        subject_table,
        categories,
        value_col=metric,
    )

    posthoc = pairwise_wilcoxon(
        subject_table,
        categories,
    )

    print("\nH2 groupe — comparaison entre catégories")
    print(f"Métrique : {metric}")
    print(f"Friedman χ² = {stat:.4f}")
    print(f"p = {pvalue:.6f}")
    print(f"n sujets complets = {n_subjects}")

    subject_table.to_csv(
        output_dir / f"h2_{metric}_category_subject_table.csv"
    )

    posthoc.to_csv(
        output_dir / f"h2_{metric}_category_posthoc.csv",
        index=False,
    )

    pd.DataFrame([{
        "metric": metric,
        "friedman_stat": stat,
        "friedman_p": pvalue,
        "n_subjects": n_subjects,
    }]).to_csv(
        output_dir / f"h2_{metric}_category_friedman.csv",
        index=False,
    )

    _plot_group_categories(
        subject_table,
        categories,
        metric,
        output_dir / f"h2_{metric}_category.png",
        title=f"H2 groupe — {metric} par catégorie",
    )

    return {
        "subject_table": subject_table,
        "friedman_stat": stat,
        "friedman_p": pvalue,
        "posthoc": posthoc,
    }


def _analyse_h2_duration(
    df,
    metric,
    output_dir,
    correlation_method,
):
    correlations = within_subject_correlations(
        df=df,
        x_col=metric,
        y_col="ideation_dur",
        method=correlation_method,
    )

    group_result = group_test_fisher_correlations(
        correlations
    )

    print("\nH2 groupe — métrique × durée d'idéation")
    print(f"Métrique : {metric}")
    print(f"Méthode : {correlation_method}")
    print(
        f"r moyen = {group_result['mean_r']:.3f}, "
        f"t = {group_result['t']:.3f}, "
        f"p = {group_result['p']:.5f}, "
        f"n = {group_result['n']}"
    )

    correlations.to_csv(
        output_dir / f"h2_{metric}_duration_subject_correlations.csv",
        index=False,
    )

    pd.DataFrame([group_result]).to_csv(
        output_dir / f"h2_{metric}_duration_group_test.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0, color="black", linewidth=1)

    ax.bar(
        correlations["participant"],
        correlations["r"],
        color="steelblue",
        alpha=0.75,
    )

    ax.set_title(
        f"H2 — Corrélations intra-sujet\n"
        f"{metric} × durée d'idéation"
    )
    ax.set_xlabel("Participant")
    ax.set_ylabel("Corrélation intra-sujet r")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h2_{metric}_duration_subject_r.png",
        dpi=150,
    )

    plt.show()

    return {
        "subject_correlations": correlations,
        "group_result": group_result,
    }


def _plot_group_categories(
    subject_table,
    categories,
    metric,
    output_path,
    title,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in subject_table.iterrows():
        ax.plot(
            categories,
            row.reindex(categories),
            color="gray",
            alpha=0.4,
            marker="o",
        )

    means = subject_table[categories].mean()
    sems = subject_table[categories].sem()

    ax.errorbar(
        categories,
        means,
        yerr=sems,
        color="black",
        linewidth=3,
        marker="o",
        markersize=8,
        capsize=5,
        label="Moyenne ± SEM",
    )

    ax.set_title(title)
    ax.set_xlabel("Catégorie émotionnelle")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()