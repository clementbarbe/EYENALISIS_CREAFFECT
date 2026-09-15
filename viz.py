"""Visualisations du pipeline FGAT eyetracking."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

from config import Config


SCHEME_TO_GROUP_COLUMN = {
    "five_categories": "group_5cat",
    "valence": "group_valence",
    "arousal": "group_arousal",
}

MOMENTS_ORDER = [
    "M1_induction",
    "M2a_search_start",
    "M2b_search_end",
    "M3a_rating_start",
    "M3b_rating_end",
]

PRIMARY_METRICS_ORDER = [
    "pupil_baseline_corrected",
    "pupil_mean",
    "pupil_sd",
    "pupil_slope_per_s",
    "pupil_auc",
    "pupil_peak",
    "pupil_peak_latency_s",
]


def plot_exclusion_funnel(
    run_qc: pd.DataFrame,
    subject_qc: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Graphique global des exclusions."""
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
    """Graphique des raisons d'exclusion des runs."""
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


def make_subject_condition_summary(
    metrics_long: pd.DataFrame,
    scheme: str,
    moment: str,
    metric: str,
) -> tuple[pd.DataFrame, str]:
    """
    Calcule les moyennes descriptives sujet x condition.

    Essais -> moyenne de bloc -> moyenne sujet x condition.
    """
    group_column = SCHEME_TO_GROUP_COLUMN[scheme]

    data = metrics_long.loc[
        metrics_long["moment_valid"]
        & (metrics_long["moment"] == moment)
        & (metrics_long["metric"] == metric)
        & metrics_long["value"].notna()
        & metrics_long[group_column].notna()
    ].copy()

    if data.empty:
        return pd.DataFrame(), group_column

    block_data = (
        data.groupby(
            [
                "subject_id",
                "run_id",
                "induction_block_id",
                group_column,
            ],
            as_index=False,
        )["value"]
        .mean()
        .rename(columns={"value": "block_mean"})
    )

    subject_data = (
        block_data.groupby(
            [
                "subject_id",
                group_column,
            ],
            as_index=False,
        )["block_mean"]
        .mean()
        .rename(columns={"block_mean": "subject_mean"})
    )

    return subject_data, group_column


def plot_distribution(
    metrics_long: pd.DataFrame,
    stats_df: pd.DataFrame,
    scheme: str,
    moment: str,
    metric: str,
    hypotheses_root: Path,
    cfg: Config,
) -> None:
    """Figure descriptive sujet x condition pour une hypothèse."""
    data, group_column = make_subject_condition_summary(
        metrics_long,
        scheme,
        moment,
        metric,
    )

    if data.empty:
        return

    stat_row = stats_df.loc[
        (stats_df["scheme"] == scheme)
        & (stats_df["moment"] == moment)
        & (stats_df["metric"] == metric)
    ]

    model_reliable = False
    p_fdr = float("nan")
    effect = float("nan")
    error = ""

    if not stat_row.empty:
        result = stat_row.iloc[0]
        model_reliable = bool(
            result.get("model_reliable", False)
        )
        p_fdr = result.get("p_fdr", float("nan"))
        effect = result.get(
            "effect_size_std_max_contrast",
            float("nan"),
        )
        error = str(result.get("error", "") or "")

    if model_reliable and pd.notna(p_fdr):
        subtitle = (
            f"p FDR = {float(p_fdr):.4g} | "
            f"effet standardisé = {float(effect):.2f}"
        )
    elif error:
        subtitle = f"Descriptif seulement : {error}"
    else:
        subtitle = "Descriptif seulement : modèle non estimable"

    output_dir = hypotheses_root / scheme / moment
    output_dir.mkdir(parents=True, exist_ok=True)

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
        showfliers=False,
        color="white",
        ax=ax,
    )

    sns.stripplot(
        data=data,
        x=group_column,
        y="subject_mean",
        color="black",
        alpha=0.60,
        size=4,
        jitter=0.18,
        ax=ax,
    )

    ax.set_title(
        f"{scheme} | {moment} | {metric}\n{subtitle}"
    )

    ax.set_xlabel(group_column)
    ax.set_ylabel(f"{metric} — moyenne par sujet")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{metric}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_fdr_heatmap(
    stats_df: pd.DataFrame,
    scheme: str,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Heatmap classique des p-values FDR."""
    data = stats_df.loc[
        (stats_df["scheme"] == scheme)
        & stats_df["model_reliable"]
    ].copy()

    if data.empty:
        return

    pivot = data.pivot(
        index="metric",
        columns="moment",
        values="p_fdr",
    )

    pivot = pivot.reindex(
        index=PRIMARY_METRICS_ORDER,
        columns=MOMENTS_ORDER,
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis_r",
        vmin=0.0,
        vmax=0.10,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(f"P-values corrigées FDR — {scheme}")
    ax.set_xlabel("Moment")
    ax.set_ylabel("Métrique")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"fdr_heatmap_{scheme}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_results_table_png(
    stats_df: pd.DataFrame,
    scheme: str,
    output_dir: Path,
    cfg: Config,
) -> None:
    """
    Produit un tableau PNG final pour un schéma.

    Vert :
        p_FDR < 0.05.
    Gris foncé :
        p_FDR >= 0.05.
    Gris clair :
        modèle non estimable/non fiable.
    """
    data = stats_df.loc[
        stats_df["scheme"] == scheme
    ].copy()

    if data.empty:
        return

    color_codes = pd.DataFrame(
        0,
        index=PRIMARY_METRICS_ORDER,
        columns=MOMENTS_ORDER,
    )

    annotations = pd.DataFrame(
        "",
        index=PRIMARY_METRICS_ORDER,
        columns=MOMENTS_ORDER,
    )

    for _, row in data.iterrows():
        metric = str(row["metric"])
        moment = str(row["moment"])

        if (
            metric not in PRIMARY_METRICS_ORDER
            or moment not in MOMENTS_ORDER
        ):
            continue

        reliable = bool(row.get("model_reliable", False))
        p_fdr = row.get("p_fdr", float("nan"))
        effect = row.get(
            "effect_size_std_max_contrast",
            float("nan"),
        )

        if not reliable or pd.isna(p_fdr):
            color_codes.loc[metric, moment] = 0
            annotations.loc[metric, moment] = "NA"
            continue

        if float(p_fdr) < cfg.alpha:
            color_codes.loc[metric, moment] = 2
        else:
            color_codes.loc[metric, moment] = 1

        annotations.loc[metric, moment] = (
            f"p={float(p_fdr):.3f}\n"
            f"effet={float(effect):.2f}"
        )

    cmap = ListedColormap(
        [
            "#D1D5DB",  # modèle non estimable : gris clair
            "#1F2937",  # non significatif : gris très foncé
            "#16A34A",  # significatif après FDR : vert
        ]
    )

    fig, ax = plt.subplots(figsize=(15, 7.5))

    sns.heatmap(
        color_codes.astype(float),
        annot=annotations,
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
        f"Résultats pupillométriques — {scheme}\n"
        "Vert : significatif après FDR | "
        "Gris foncé : non significatif | "
        "Gris clair : non estimable",
        fontsize=13,
        pad=16,
    )

    ax.set_xlabel("Moment")
    ax.set_ylabel("Métrique")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"results_table_{scheme}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)