import pandas as pd
import matplotlib.pyplot as plt

from config import (
    PARAMS,
    GROUP_DIR,
    H1_METRICS,
)
from tools.statistics import paired_friedman, pairwise_wilcoxon
from analysis.group.common import (
    load_group_data,
    load_qc,
    apply_group_qc,
    subject_condition_table,
)


def analyse_group_h1(metric="pup_mean"):
    if metric not in H1_METRICS:
        print(f"⚠ Métrique non listée dans H1_METRICS : {metric}")

    df = load_group_data("h1")
    qc = load_qc()

    df = apply_group_qc(df, qc)

    categories = PARAMS["emotion_categories"]

    df = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    if metric not in df.columns:
        raise ValueError(f"Métrique absente : {metric}")

    subject_table = subject_condition_table(
        df,
        metric=metric,
    )

    subject_table = subject_table.reindex(columns=categories)

    stat, pvalue, n = paired_friedman(
        subject_table=subject_table,
        categories=categories,
        value_col=metric,
    )

    posthoc = pairwise_wilcoxon(
        subject_table=subject_table,
        categories=categories,
    )

    print("\n" + "=" * 60)
    print("H1 — ANALYSE DE GROUPE")
    print("=" * 60)
    print(f"Métrique : {metric}")
    print(f"Sujets complets dans Friedman : {n}")
    print(f"Friedman χ² = {stat:.4f}")
    print(f"p = {pvalue:.6f}")

    output_dir = GROUP_DIR / "h1"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_table.to_csv(
        output_dir / f"h1_{metric}_subject_table.csv"
    )

    posthoc.to_csv(
        output_dir / f"h1_{metric}_posthoc.csv",
        index=False,
    )

    pd.DataFrame([{
        "metric": metric,
        "friedman_stat": stat,
        "friedman_p": pvalue,
        "n_subjects": n,
    }]).to_csv(
        output_dir / f"h1_{metric}_friedman.csv",
        index=False,
    )

    plot_h1_group(
        subject_table,
        metric,
        output_dir,
    )

    return {
        "subject_table": subject_table,
        "friedman_stat": stat,
        "friedman_p": pvalue,
        "posthoc": posthoc,
    }


def plot_h1_group(subject_table, metric, output_dir):
    categories = PARAMS["emotion_categories"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in subject_table.iterrows():
        ax.plot(
            categories,
            row.reindex(categories),
            color="gray",
            alpha=0.45,
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

    ax.set_title(f"H1 groupe — {metric}")
    ax.set_xlabel("Catégorie émotionnelle")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h1_{metric}_group.png",
        dpi=150,
    )

    plt.show()