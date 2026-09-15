"""Visualisations exploratoires séparées."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import Config


def plot_exclusion_funnel(run_qc: pd.DataFrame, subject_qc: pd.DataFrame,
                          output_dir: Path, cfg: Config) -> None:
    """Produit un graphique synthétique des exclusions."""
    counts = pd.DataFrame({
        "stage": ["Runs chargés", "Runs valides", "Sujets détectés", "Sujets retenus"],
        "count": [
            len(run_qc),
            int(run_qc["run_valid"].sum()),
            len(subject_qc),
            int(subject_qc["subject_valid"].sum()),
        ],
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=counts, x="stage", y="count", ax=ax, color="#4C78A8")
    ax.set_title("Entonnoir d'exclusion")
    ax.set_xlabel("")
    ax.set_ylabel("Nombre")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "exclusion_funnel.png", dpi=cfg.dpi)
    plt.close(fig)

def plot_run_exclusion_reasons(
    run_exclusion_summary: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
) -> None:
    """Produit un barplot des raisons d'exclusion des runs."""
    if run_exclusion_summary.empty:
        return

    data = run_exclusion_summary.sort_values("n_runs", ascending=True)

    fig, ax = plt.subplots(
        figsize=(9, max(4, 0.5 * len(data))),
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



def plot_fdr_heatmap(stats_df: pd.DataFrame, scheme: str,
                     output_dir: Path, cfg: Config) -> None:
    """Heatmap des p-values FDR par moment et métrique."""
    data = stats_df.loc[
        (stats_df["scheme"] == scheme) & stats_df["model_success"]
    ].copy()

    if data.empty:
        return

    pivot = data.pivot(index="metric", columns="moment", values="p_fdr")
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.45)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis_r",
        vmin=0,
        vmax=0.10,
        ax=ax,
    )
    ax.set_title(f"P-values corrigées FDR — {scheme}")
    fig.tight_layout()
    fig.savefig(output_dir / f"fdr_heatmap_{scheme}.png", dpi=cfg.dpi)
    plt.close(fig)


def plot_distribution(
    metrics_long: pd.DataFrame,
    stats_df: pd.DataFrame,
    scheme: str,
    moment: str,
    metric: str,
    hypotheses_root: Path,
    cfg: Config,
) -> None:
    """
    Produit une figure indépendante pour une hypothèse.

    Une hypothèse correspond ici à :
        schéma de groupe × moment × métrique.

    Même si le modèle n'est pas estimable, une figure descriptive peut être
    sauvegardée avec la raison d'échec indiquée dans le titre.
    """
    group_col = {
        "five_categories": "group_5cat",
        "valence": "group_valence",
        "arousal": "group_arousal",
    }[scheme]

    data = metrics_long.loc[
        (metrics_long["moment_valid"])
        & (metrics_long["moment"] == moment)
        & (metrics_long["metric"] == metric)
        & metrics_long["value"].notna()
        & metrics_long[group_col].notna()
    ].copy()

    if data.empty:
        return

    stat_row = stats_df.loc[
        (stats_df["scheme"] == scheme)
        & (stats_df["moment"] == moment)
        & (stats_df["metric"] == metric)
    ]

    p_fdr = float("nan")
    model_reliable = False
    error_message = ""

    if not stat_row.empty:
        result = stat_row.iloc[0]
        p_fdr = result.get("p_fdr", float("nan"))
        model_reliable = bool(result.get("model_reliable", False))
        error_message = str(result.get("error", "") or "")

    if model_reliable and pd.notna(p_fdr):
        subtitle = f"p corrigée FDR = {p_fdr:.4g}"
    elif error_message:
        subtitle = f"Modèle non interprétable : {error_message}"
    else:
        subtitle = "Analyse descriptive : modèle non estimable"

    output_dir = hypotheses_root / scheme / moment
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(
        data=data,
        x=group_col,
        y="value",
        inner=None,
        color="#B8D8E8",
        cut=0,
        ax=ax,
    )

    sns.boxplot(
        data=data,
        x=group_col,
        y="value",
        width=0.25,
        showfliers=False,
        color="white",
        ax=ax,
    )

    sns.stripplot(
        data=data,
        x=group_col,
        y="value",
        color="black",
        alpha=0.30,
        size=3,
        jitter=0.20,
        ax=ax,
    )

    ax.set_title(f"{scheme} | {moment} | {metric}\n{subtitle}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{metric}.png",
        dpi=cfg.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)
