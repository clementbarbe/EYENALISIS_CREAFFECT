import matplotlib.pyplot as plt
import pandas as pd

from config import PARAMS
from analysis.common import get_subject_data, subject_output_dir


def analyse_subject_h1(participant_id, metric="pup_mean"):
    df = get_subject_data("h1", participant_id)

    categories = PARAMS["emotion_categories"]

    df = df.loc[
        df["stim_category"].isin(categories)
    ].copy()

    if metric not in df.columns:
        raise ValueError(f"Métrique absente : {metric}")

    summary = (
        df.groupby("stim_category")[metric]
        .agg(["mean", "std", "count", "median"])
        .reindex(categories)
    )

    print("\nH1 — Induction émotionnelle")
    print(f"Participant : {participant_id}")
    print(f"Métrique : {metric}")
    print(summary)

    output_dir = subject_output_dir(participant_id, "h1")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        PARAMS["emotion_colors"].get(category, "gray")
        for category in categories
    ]

    ax.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
        color=colors,
        alpha=0.75,
    )

    ax.set_title(
        f"H1 — {participant_id} — {metric}"
    )
    ax.set_xlabel("Catégorie émotionnelle")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"h1_{metric}.png",
        dpi=150,
    )

    summary.to_csv(
        output_dir / f"h1_{metric}_summary.csv"
    )

    plt.show()

    return summary