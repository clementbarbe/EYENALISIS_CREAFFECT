import pandas as pd

from config import H3_METRICS

from tools.statistics import (
    group_test_fisher_correlations,
    within_subject_correlations,
)

from analysis.group.common import (
    apply_group_qc,
    get_group_output_dir,
    load_group_data,
    load_qc,
    plot_correlation_forest,
    plot_subject_faceted_scatter,
)


def analyse_group_h3(
    metric="pup_mean_v",
    correlation_method="spearman",
):
    """
    H3 — Évaluation de l'idée.

    Analyse principale :
    relation intra-sujet entre une métrique pupillaire et le rating.

    Figures :
    - scatterplot pupille × rating facetté par sujet ;
    - forest plot des corrélations intra-sujets.
    """

    if metric not in H3_METRICS:
        print(
            f"⚠ {metric} n'est pas dans H3_METRICS, "
            "mais l'analyse est tout de même lancée."
        )

    output_dir = get_group_output_dir("h3")

    df = load_group_data("h3")
    qc_df = load_qc()

    df = apply_group_qc(
        df,
        qc_df,
    )

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H3 : {metric}"
        )

    if "rating" not in df.columns:
        raise ValueError(
            "Colonne rating absente du fichier H3."
        )

    correlations = within_subject_correlations(
        df=df,
        x_col="rating",
        y_col=metric,
        method=correlation_method,
    )

    group_result = group_test_fisher_correlations(
        correlations
    )

    print("\n" + "=" * 60)
    print("H3 — GROUPE : PUPILLE × RATING")
    print("=" * 60)
    print(f"Métrique : {metric}")
    print(f"Méthode : {correlation_method}")
    print(f"Sujets avec corrélation : {group_result['n']}")
    print(f"r moyen : {group_result['mean_r']:.4f}")
    print(f"t : {group_result['t']:.4f}")
    print(f"p : {group_result['p']:.6f}")

    correlations.to_csv(
        output_dir
        / f"h3_{metric}_rating_subject_correlations.csv",
        index=False,
    )

    pd.DataFrame([group_result]).to_csv(
        output_dir
        / f"h3_{metric}_rating_group_test.csv",
        index=False,
    )

    # Scatter facetté par sujet.
    plot_subject_faceted_scatter(
        df=df,
        x_col="rating",
        y_col=metric,
        title=(
            "H3 — Évaluation de l'idée\n"
            f"{metric} × rating"
        ),
        output_path=(
            output_dir
            / f"h3_{metric}_rating_facets.png"
        ),
        correlation_method=correlation_method,
        x_label="Rating",
        y_label=metric,
        x_jitter=True,
    )

    # Forest plot.
    plot_correlation_forest(
        correlations_df=correlations,
        group_result=group_result,
        title=(
            "H3 — Corrélations intra-sujets\n"
            f"{metric} × rating"
        ),
        output_path=(
            output_dir
            / f"h3_{metric}_rating_forest.png"
        ),
    )

    return {
        "data": df,
        "subject_correlations": correlations,
        "group_result": group_result,
    }