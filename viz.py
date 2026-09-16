"""Visualisations du pipeline FGAT eyetracking."""

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


def _filter_for_x(
    dataframe: pd.DataFrame,
    x_s: float,
) -> pd.DataFrame:
    """
    Garde M1 fixe et les moments dynamiques correspondant à X.

    M1 est toujours inclus car son window_x_s vaut 0.0.
    """
    m1_mask = dataframe["moment"] == "M1_induction"

    dynamic_mask = (
        dataframe["moment"] != "M1_induction"
    ) & np.isclose(
        dataframe["window_x_s"].astype(float),
        x_s,
    )

    return dataframe.loc[m1_mask | dynamic_mask].copy()


def make_subject_condition_summary(
    metrics_long: pd.DataFrame,
    scheme: str,
    moment: str,
    metric: str,
    x_s: float,
) -> tuple[pd.DataFrame, str]:
    """
    Produit des moyennes descriptives sujet × condition.

    Etapes :
    1. moyenne essais -> bloc d'induction ;
    2. moyenne blocs -> sujet × condition.

    Les figures ne traitent donc jamais les essais bruts comme indépendants.
    """
    group_column = SCHEME_TO_GROUP_COLUMN[scheme]

    data = metrics_long.loc[
        metrics_long["moment_valid"]
        & (metrics_long["moment"] == moment)
        & (metrics_long["metric"] == metric)
        & metrics_long["value"].notna()
        & metrics_long[group_column].notna()
    ].copy()

    if moment != "M1_induction":
        data = data.loc[
            np.isclose(
                data["window_x_s"].astype(float),
                x_s,
            )
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
    x_s: float,
    hypotheses_root: Path,
    cfg: Config,
) -> None:
    """Produit une figure descriptive par hypothèse."""
    data, group_column = make_subject_condition_summary(
        metrics_long,
        scheme,
        moment,
        metric,
        x_s,
    )

    if data.empty:
        return

    stat_rows = stats_df.loc[
        (stats_df["scheme"] == scheme)
        & (stats_df["moment"] == moment)
        & (stats_df["metric"] == metric)
    ].copy()

    if moment != "M1_induction":
        stat_rows = stat_rows.loc[
            np.isclose(
                stat_rows["window_x_s"].astype(float),
                x_s,
            )
        ]

    model_reliable = False
    p_fdr = np.nan
    effect = np.nan
    error = ""

    if not stat_rows.empty:
        result = stat_rows.iloc[0]

        model_reliable = bool(
            result.get("model_reliable", False)
        )

        p_fdr = result.get("p_fdr", np.nan)

        effect = result.get(
            "effect_size_std_max_contrast",
            np.nan,
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

    x_dir = hypotheses_root / f"X_{x_s:.1f}s" / scheme / moment
    x_dir.mkdir(parents=True, exist_ok=True)

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

    title_x = "X fixe (M1)" if moment == "M1_induction" else f"X = {x_s:.1f}s"

    ax.set_title(
        f"{scheme} | {moment} | {metric}\n"
        f"{title_x}\n{subtitle}"
    )

    ax.set_xlabel(group_column)
    ax.set_ylabel(f"{metric} — moyenne par sujet")

    fig.tight_layout()

    fig.savefig(
        x_dir / f"{metric}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_results_table_for_x(
    stats_df: pd.DataFrame,
    scheme: str,
    x_s: float,
    output_dir: Path,
    cfg: Config,
) -> None:
    """
    Produit un tableau PNG de résultats pour un schéma et une valeur X.

    Vert :
        p_FDR < .05.

    Gris foncé :
        modèle fiable, résultat non significatif.

    Gris clair :
        modèle non estimable / non fiable.
    """
    data = stats_df.loc[
        stats_df["scheme"] == scheme
    ].copy()

    data = _filter_for_x(data, x_s)

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
        p_fdr = row.get("p_fdr", np.nan)

        effect = row.get(
            "effect_size_std_max_contrast",
            np.nan,
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
            "#D1D5DB",  # NA : gris clair.
            "#1F2937",  # Non significatif : gris foncé.
            "#16A34A",  # Significatif : vert.
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
        f"Résultats pupillométriques — {scheme} — X={x_s:.1f}s\n"
        "Vert : p FDR < .05 | "
        "Gris foncé : non significatif | "
        "Gris clair : non estimable",
        fontsize=13,
        pad=16,
    )

    ax.set_xlabel("Moment")
    ax.set_ylabel("Métrique")

    fig.tight_layout()

    x_dir = output_dir / f"X_{x_s:.1f}s"
    x_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        x_dir / f"results_table_{scheme}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_x_stability(
    stats_df: pd.DataFrame,
    scheme: str,
    moment: str,
    output_dir: Path,
    cfg: Config,
) -> None:
    """
    Produit une heatmap de stabilité des résultats selon X.

    Lignes : métriques.
    Colonnes : valeurs de X.
    Valeur affichée : p-value FDR.

    Vert : significatif après FDR.
    Gris foncé : non significatif.
    Gris clair : modèle non estimable.
    """
    if moment == "M1_induction":
        return

    x_values = list(cfg.dynamic_x_values_s)

    data = stats_df.loc[
        (stats_df["scheme"] == scheme)
        & (stats_df["moment"] == moment)
    ].copy()

    if data.empty:
        return

    color_codes = pd.DataFrame(
        0,
        index=PRIMARY_METRICS_ORDER,
        columns=x_values,
    )

    annotations = pd.DataFrame(
        "",
        index=PRIMARY_METRICS_ORDER,
        columns=x_values,
    )

    for _, row in data.iterrows():
        metric = str(row["metric"])
        x_s = row.get("window_x_s", np.nan)

        if metric not in PRIMARY_METRICS_ORDER or pd.isna(x_s):
            continue

        x_s = round(float(x_s), 1)

        if x_s not in x_values:
            continue

        reliable = bool(row.get("model_reliable", False))
        p_fdr = row.get("p_fdr", np.nan)

        if not reliable or pd.isna(p_fdr):
            color_codes.loc[metric, x_s] = 0
            annotations.loc[metric, x_s] = "NA"

        elif float(p_fdr) < cfg.alpha:
            color_codes.loc[metric, x_s] = 2
            annotations.loc[metric, x_s] = f"{float(p_fdr):.3f}"

        else:
            color_codes.loc[metric, x_s] = 1
            annotations.loc[metric, x_s] = f"{float(p_fdr):.3f}"

    cmap = ListedColormap(
        [
            "#D1D5DB",
            "#1F2937",
            "#16A34A",
        ]
    )

    fig, ax = plt.subplots(figsize=(16, 7))

    sns.heatmap(
        color_codes.astype(float),
        annot=annotations,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 8},
    )

    for text in ax.texts:
        text.set_color("white")

    ax.set_title(
        f"Stabilité des p-values FDR selon X — {scheme} — {moment}\n"
        "Vert : p FDR < .05 | "
        "Gris foncé : non significatif | "
        "Gris clair : NA",
        pad=15,
    )

    ax.set_xlabel("X (secondes)")
    ax.set_ylabel("Métrique")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"x_stability_{scheme}_{moment}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)