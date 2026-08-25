import pandas as pd
import matplotlib.pyplot as plt

from config import GROUP_DIR, H3_METRICS
from tools.statistics import (
    within_subject_correlations,
    group_test_fisher_correlations,
)


from analysis.group.common import (
    load_group_data,
    load_qc,
    apply_group_qc,
)


def analyse_group_h3(
    metric="pup_mean_v",
    correlation_method="spearman",
):
    """
    H3 — Relation entre métrique pupillaire et rating.

    Une corrélation est calculée séparément pour chaque participant.
    Les r sont ensuite agrégés avec la transformation de Fisher.
    """

    if metric not in H3_METRICS:
        print(
            f"⚠ {metric} n'est pas dans H3_METRICS, "
            "mais l'analyse est tentée."
        )

    df = load_group_data("h3")
    qc = load_qc()

    df = apply_group_qc(df, qc)

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente du fichier H3 : {metric}"
        )

    if "rating" not in df.columns:
        raise ValueError(
            "La colonne rating est absente du fichier H3."
        )

    output_dir = GROUP_DIR / "h3"
    output_dir.mkdir(parents=True, exist_ok=True)

    correlations = within_subject_correlations(
        df=df,
        x_col=metric,
        y_col="rating",
        method=correlation_method,
    )

    group_result = group_test_fisher_correlations(
        correlations
    )

    print("\n" + "=" * 60)
    print("H3 — ANALYSE DE GROUPE")
    print("=" * 60)
    print(f"Métrique : {metric}")
    print(f"Méthode : {correlation_method}")
    print(
        f"r moyen = {group_result['mean_r']:.3f}, "
        f"t = {group_result['t']:.3f}, "
        f"p = {group_result['p']:.5f}, "
        f"n = {group_result['n']}"
    )

    correlations.to_csv(
        output_dir / f"h3_{metric}_rating_subject_correlations.csv",
        index=False,
    )

    pd.DataFrame([group_result]).to_csv(
        output_dir / f"h3_{metric}_rating_group_test.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0, color="black", linewidth=1)

    ax.bar(
        correlations["participant"],
        correlations["r"],
        color="darkred",
        alpha=0.75,
    )

    ax.set_title(
        f"H3 — Corrélations intra-sujet\n"
        f"{metric} × rating"
    )

    ax.set_xlabel("Participant")
    ax.set_ylabel("r intra-sujet")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h3_{metric}_rating_subject_r.png",
        dpi=150,
    )

    plt.show()

    return {
        "subject_correlations": correlations,
        "group_result": group_result,
    }