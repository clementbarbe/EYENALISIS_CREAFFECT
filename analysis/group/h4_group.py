import pandas as pd

from config import (
    H4_METRICS,
    PARAMS,
)

from tools.statistics import (
    group_test_fisher_correlations,
    paired_friedman,
    pairwise_wilcoxon,
    within_subject_correlations,
)

from analysis.group.common import (
    apply_group_qc,
    get_group_output_dir,
    load_group_data,
    load_qc,
    plot_condition_boxplot,
    plot_correlation_forest,
    plot_subject_faceted_scatter,
    print_condition_availability,
    subject_condition_table,
)


def analyse_group_h4(
    metric="gaze_entropy",
    mode="duration",
    correlation_method="spearman",
):
    """
    H4 — Dynamique spatiale du regard.

    Modes
    -----
    duration :
        Relation entre métrique de regard et durée d'idéation.

    category :
        Comparaison entre catégories émotionnelles.
    """

    if metric not in H4_METRICS:
        print(
            f"⚠ {metric} n'est pas dans H4_METRICS, "
            "mais l'analyse est tout de même lancée."
        )

    output_dir = get_group_output_dir("h4")

    df = load_group_data("h4")
    qc_df = load_qc()

    df = apply_group_qc(
        df,
        qc_df,
    )

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H4 : {metric}"
        )

    if mode == "duration":
        return _analyse_h4_duration(
            df=df,
            metric=metric,
            output_dir=output_dir,
            correlation_method=correlation_method,
        )

    if mode == "category":
        return _analyse_h4_category(
            df=df,
            metric=metric,
            output_dir=output_dir,
        )

    raise ValueError(
        "mode doit être 'duration' ou 'category'."
    )


def _analyse_h4_duration(
    df,
    metric,
    output_dir,
    correlation_method,
):
    """
    H4 : métrique oculomotrice × durée d'idéation.
    """

    if "ideation_dur" not in df.columns:
        raise ValueError(
            "Colonne ideation_dur absente de H4."
        )

    correlations = within_subject_correlations(
        df=df,
        x_col=metric,
        y_col="ideation_dur",
        method=correlation_method,
    )

    group_result = group_test_fisher_correlations(
        correlations
    )

    print("\n" + "=" * 60)
    print("H4 — GROUPE : REGARD × DURÉE D'IDÉATION")
    print("=" * 60)
    print(f"Métrique : {metric}")
    print(f"Méthode : {correlation_method}")
    print(f"Sujets avec corrélation : {group_result['n']}")
    print(f"r moyen : {group_result['mean_r']:.4f}")
    print(f"t : {group_result['t']:.4f}")
    print(f"p : {group_result['p']:.6f}")

    correlations.to_csv(
        output_dir
        / f"h4_{metric}_duration_subject_correlations.csv",
        index=False,
    )

    pd.DataFrame([group_result]).to_csv(
        output_dir
        / f"h4_{metric}_duration_group_test.csv",
        index=False,
    )

    # Scatterplots par participant.
    plot_subject_faceted_scatter(
        df=df,
        x_col="ideation_dur",
        y_col=metric,
        title=(
            "H4 — Dynamique du regard\n"
            f"{metric} × durée d'idéation"
        ),
        output_path=(
            output_dir
            / f"h4_{metric}_duration_facets.png"
        ),
        correlation_method=correlation_method,
        x_label="Durée d'idéation (s)",
        y_label=metric,
        x_jitter=False,
    )

    # Forest plot.
    plot_correlation_forest(
        correlations_df=correlations,
        group_result=group_result,
        title=(
            "H4 — Corrélations intra-sujets\n"
            f"{metric} × durée d'idéation"
        ),
        output_path=(
            output_dir
            / f"h4_{metric}_duration_forest.png"
        ),
    )

    return {
        "data": df,
        "subject_correlations": correlations,
        "group_result": group_result,
    }


def _analyse_h4_category(
    df,
    metric,
    output_dir,
):
    """
    H4 : comparaison de la dynamique du regard entre catégories.
    """

    categories = PARAMS["emotion_categories"]

    df = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    subject_table = subject_condition_table(
        df,
        metric=metric,
    ).reindex(columns=categories)

    print_condition_availability(
        subject_table,
        categories,
    )

    friedman_stat, friedman_p, n_subjects = (
        paired_friedman(
            subject_table=subject_table,
            categories=categories,
            value_col=metric,
        )
    )

    posthoc = pairwise_wilcoxon(
        subject_table=subject_table,
        categories=categories,
    )

    print("\n" + "=" * 60)
    print("H4 — GROUPE : COMPARAISON PAR CATÉGORIE")
    print("=" * 60)
    print(f"Métrique : {metric}")
    print(f"Sujets complets : {n_subjects}")
    print(f"Friedman χ² = {friedman_stat:.4f}")
    print(f"p = {friedman_p:.6f}")

    subject_table.to_csv(
        output_dir
        / f"h4_{metric}_category_subject_table.csv"
    )

    posthoc.to_csv(
        output_dir
        / f"h4_{metric}_category_posthoc.csv",
        index=False,
    )

    pd.DataFrame([{
        "metric": metric,
        "friedman_stat": friedman_stat,
        "friedman_p": friedman_p,
        "n_complete_subjects": n_subjects,
    }]).to_csv(
        output_dir
        / f"h4_{metric}_category_friedman.csv",
        index=False,
    )

    colors = [
        PARAMS["emotion_colors"].get(
            category,
            "#95a5a6",
        )
        for category in categories
    ]

    plot_condition_boxplot(
        subject_table=subject_table,
        categories=categories,
        metric=metric,
        title=(
            "H4 — Dynamique spatiale du regard\n"
            f"{metric} par catégorie émotionnelle"
        ),
        output_path=(
            output_dir
            / f"h4_{metric}_category_boxplot.png"
        ),
        colors=colors,
    )

    return {
        "data": df,
        "subject_table": subject_table,
        "friedman_stat": friedman_stat,
        "friedman_p": friedman_p,
        "posthoc": posthoc,
    }