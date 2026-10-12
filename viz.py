"""Visualisations confirmatoires FGAT eyetracking."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config


SCHEME_TO_GROUP_COLUMN = {
    "five_categories": "group_5cat",
    "valence": "group_valence",
    "arousal": "group_arousal",
}


def plot_exclusion_funnel(
    run_qc: pd.DataFrame,
    subject_qc: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Crée un graphique de synthèse des exclusions."""
    data = pd.DataFrame(
        {
            "Étape": [
                "Runs trouvés",
                "Runs valides",
                "Sujets trouvés",
                "Sujets retenus",
            ],
            "Nombre": [
                len(run_qc),
                int(run_qc["run_valid"].sum()),
                len(subject_qc),
                int(subject_qc["subject_valid"].sum()),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.barplot(
        data=data,
        x="Étape",
        y="Nombre",
        color="#4C78A8",
        ax=ax,
    )

    ax.set_title("Entonnoir d'exclusion")
    ax.set_xlabel("")
    ax.set_ylabel("Nombre")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()

    fig.savefig(
        output_dir / "exclusion_funnel.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_run_exclusion_reasons(
    run_exclusion_summary: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Crée un graphique des raisons d'exclusion."""
    if run_exclusion_summary.empty:
        return

    data = run_exclusion_summary.sort_values(
        "n_runs",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, max(4, len(data) * 0.6))
    )

    sns.barplot(
        data=data,
        x="n_runs",
        y="reason",
        color="#D95F02",
        ax=ax,
    )

    ax.set_title("Raisons d'exclusion des runs")
    ax.set_xlabel("Nombre de runs exclus")
    ax.set_ylabel("Raison")

    fig.tight_layout()

    fig.savefig(
        output_dir / "run_exclusion_reasons.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_results_table(
    stats_df: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """
    Produit un tableau PNG de résultats confirmatoires.

    Vert : significatif après FDR.
    Gris foncé : non significatif.
    Gris clair : modèle non estimable.
    """
    if stats_df.empty:
        return

    display_rows = []

    for _, row in stats_df.iterrows():
        group_p = row.get("p_group_fdr", np.nan)
        rating_p = row.get("p_rating_fdr", np.nan)

        if pd.notna(rating_p):
            test_name = (
                f"{row['hypothesis']} | {row['metric']} | rating"
            )
            p_value = rating_p
            effect = row.get("rating_effect_std", np.nan)
        else:
            test_name = (
                f"{row['hypothesis']} | {row['scheme']} | "
                f"{row['metric']} | groupe"
            )
            p_value = group_p
            effect = row.get("group_effect_std", np.nan)

        reliable = bool(row.get("model_reliable", False))

        if not reliable or pd.isna(p_value):
            color = 0
            annotation = "NA"
        elif float(p_value) < cfg.alpha:
            color = 2
            annotation = (
                f"p={float(p_value):.3f}\n"
                f"effet={float(effect):.2f}"
            )
        else:
            color = 1
            annotation = (
                f"p={float(p_value):.3f}\n"
                f"effet={float(effect):.2f}"
            )

        display_rows.append(
            {
                "test": test_name,
                "color": color,
                "annotation": annotation,
            }
        )

    display = pd.DataFrame(display_rows)

    color_matrix = display[["color"]].copy()
    color_matrix.index = display["test"]

    annotation_matrix = display[["annotation"]].copy()
    annotation_matrix.index = display["test"]

    cmap = ListedColormap(
        [
            "#D1D5DB",
            "#1F2937",
            "#16A34A",
        ]
    )

    fig_height = max(5, len(display) * 0.65)

    fig, ax = plt.subplots(figsize=(8, fig_height))

    sns.heatmap(
        color_matrix,
        annot=annotation_matrix,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 10},
    )

    for text in ax.texts:
        text.set_color("white")

    ax.set_title(
        "Résultats confirmatoires\n"
        "Vert : p FDR < .05 | "
        "Gris foncé : non significatif | "
        "Gris clair : non estimable",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()

    fig.savefig(
        output_dir / "confirmatory_results_table.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def _subject_condition_means(
    metrics_long: pd.DataFrame,
    moment: str,
    metric: str,
    group_column: str,
) -> pd.DataFrame:
    """Produit une moyenne par sujet et condition."""
    data = metrics_long.loc[
        metrics_long["moment_valid"]
        & (metrics_long["moment"] == moment)
        & (metrics_long["metric"] == metric)
        & metrics_long["value"].notna()
        & metrics_long[group_column].notna()
    ].copy()

    if data.empty:
        return data

    return (
        data.groupby(
            ["subject_id", group_column],
            as_index=False,
        )["value"]
        .mean()
        .rename(columns={"value": "subject_mean"})
    )


def plot_hypothesis_distributions(
    metrics_long: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Produit des distributions sujet × condition."""
    for _, result in stats_df.iterrows():
        if not bool(result.get("model_reliable", False)):
            continue

        scheme = str(result["scheme"])
        group_column = SCHEME_TO_GROUP_COLUMN[scheme]

        data = _subject_condition_means(
            metrics_long,
            moment=str(result["moment"]),
            metric=str(result["metric"]),
            group_column=group_column,
        )

        if data.empty:
            continue

        p_value = result.get("p_group_fdr", np.nan)

        fig, ax = plt.subplots(figsize=(8, 5.5))

        sns.violinplot(
            data=data,
            x=group_column,
            y="subject_mean",
            inner=None,
            cut=0,
            color="#B8D8E8",
            ax=ax,
        )

        sns.boxplot(
            data=data,
            x=group_column,
            y="subject_mean",
            width=0.25,
            color="white",
            showfliers=False,
            ax=ax,
        )

        sns.stripplot(
            data=data,
            x=group_column,
            y="subject_mean",
            color="black",
            alpha=0.6,
            size=4,
            jitter=0.18,
            ax=ax,
        )

        ax.set_title(
            f"{result['hypothesis']} | {result['moment']} | "
            f"{result['metric']}\n"
            f"p FDR groupe = {float(p_value):.4g}"
        )

        ax.set_xlabel(group_column)
        ax.set_ylabel("Moyenne par sujet")

        filename = (
            f"{result['hypothesis']}_"
            f"{scheme}_"
            f"{result['metric']}.png"
        )

        fig.tight_layout()

        fig.savefig(
            output_dir / filename,
            dpi=cfg.dpi,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_liking_scatter(
    metrics_long: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Produit un scatter pupil × LTask_rating pour H3."""
    h3_results = stats_df.loc[
        (stats_df["hypothesis"] == "H3_liking")
        & stats_df["model_reliable"]
    ]

    for _, result in h3_results.iterrows():
        data = metrics_long.loc[
            metrics_long["moment_valid"]
            & (metrics_long["moment"] == "H3_liking")
            & (metrics_long["metric"] == result["metric"])
            & metrics_long["value"].notna()
            & metrics_long["LTask_rating"].notna()
        ].copy()

        if data.empty:
            continue

        subject_data = (
            data.groupby(
                ["subject_id", "LTask_rating"],
                as_index=False,
            )["value"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.regplot(
            data=subject_data,
            x="LTask_rating",
            y="value",
            scatter_kws={"alpha": 0.45, "s": 25},
            line_kws={"color": "red"},
            ax=ax,
        )

        p_value = result.get("p_rating_fdr", np.nan)

        ax.set_title(
            f"H3 liking | {result['metric']}\n"
            f"p FDR rating = {float(p_value):.4g}"
        )

        ax.set_xlabel("LTask_rating")
        ax.set_ylabel(result["metric"])

        fig.tight_layout()

        fig.savefig(
            output_dir / f"H3_liking_rating_{result['metric']}.png",
            dpi=cfg.dpi,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_gaze_heatmaps(
    gaze_samples: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """
    Crée une heatmap gaze pour chaque catégorie émotionnelle.

    Les heatmaps H4 correspondent à :
    fgat_cue_onset -> fgat_input_time.
    """
    if gaze_samples.empty:
        return

    categories = [
        "LVLA",
        "LVHA",
        "MVMA",
        "HVLA",
        "HVHA",
    ]

    valid = gaze_samples.dropna(subset=["gaze_x", "gaze_y"]).copy()

    for category in categories:
        data = valid.loc[
            valid["group_5cat"] == category
        ].copy()

        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.kdeplot(
            data=data,
            x="gaze_x",
            y="gaze_y",
            fill=True,
            thresh=0.05,
            levels=50,
            cmap="mako",
            ax=ax,
        )

        if (
            cfg.cue_roi_center_x is not None
            and cfg.cue_roi_center_y is not None
        ):
            rectangle = plt.Rectangle(
                (
                    cfg.cue_roi_center_x
                    - cfg.cue_roi_half_width_px,
                    cfg.cue_roi_center_y
                    - cfg.cue_roi_half_height_px,
                ),
                2 * cfg.cue_roi_half_width_px,
                2 * cfg.cue_roi_half_height_px,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                label="ROI cue",
            )

            ax.add_patch(rectangle)
            ax.legend()

        ax.invert_yaxis()

        ax.set_title(
            f"H4 — Densité du regard pendant recherche d'idée\n"
            f"Catégorie : {category}"
        )

        ax.set_xlabel("Gaze X (pixels)")
        ax.set_ylabel("Gaze Y (pixels)")

        fig.tight_layout()

        fig.savefig(
            output_dir / f"gaze_heatmap_{category}.png",
            dpi=cfg.dpi,
            bbox_inches="tight",
        )

        plt.close(fig)