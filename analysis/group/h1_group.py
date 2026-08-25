import pandas as pd

from config import (
    H1_METRICS,
    PARAMS,
)

from tools.statistics import (
    paired_friedman,
    pairwise_wilcoxon,
)

from analysis.group.common import (
    apply_group_qc,
    get_group_output_dir,
    load_group_data,
    load_qc,
    plot_condition_boxplot,
    print_condition_availability,
    subject_condition_table,
)


def analyse_group_h1(metric="pup_mean"):
    """
    H1 — Induction émotionnelle.

    Compare les catégories émotionnelles avec :
    - Friedman apparié ;
    - Wilcoxon post-hoc avec FDR ;
    - boxplot + points individuels au niveau sujet.
    """

    if metric not in H1_METRICS:
        print(
            f"⚠ {metric} n'est pas dans H1_METRICS, "
            "mais l'analyse est tout de même lancée."
        )

    output_dir = get_group_output_dir("h1")

    df = load_group_data("h1")
    qc_df = load_qc()

    df = apply_group_qc(
        df,
        qc_df,
    )

    categories = PARAMS["emotion_categories"]

    df = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H1 : {metric}"
        )

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
    print("H1 — ANALYSE DE GROUPE")
    print("=" * 60)
    print(f"Métrique : {metric}")
    print(f"Sujets complets dans Friedman : {n_subjects}")
    print(f"Friedman χ² = {friedman_stat:.4f}")
    print(f"p = {friedman_p:.6f}")

    # ─────────────────────────────────────────────────────────
    # Exports tableaux
    # ─────────────────────────────────────────────────────────

    subject_table.to_csv(
        output_dir / f"h1_{metric}_subject_table.csv"
    )

    posthoc.to_csv(
        output_dir / f"h1_{metric}_posthoc.csv",
        index=False,
    )

    pd.DataFrame([{
        "metric": metric,
        "friedman_stat": friedman_stat,
        "friedman_p": friedman_p,
        "n_complete_subjects": n_subjects,
    }]).to_csv(
        output_dir / f"h1_{metric}_friedman.csv",
        index=False,
    )

    # ─────────────────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────────────────

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
            "H1 — Induction émotionnelle\n"
            f"{metric} par catégorie"
        ),
        output_path=(
            output_dir
            / f"h1_{metric}_boxplot.png"
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