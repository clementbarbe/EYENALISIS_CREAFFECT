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
    output_dir: Path,
    cfg: Config,
) -> None:
    """Distribution par groupe, annotée uniquement avec p-value FDR."""
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
    ].copy()

    if data.empty:
        return

    p_row = stats_df.loc[
        (stats_df["scheme"] == scheme)
        & (stats_df["moment"] == moment)
        & (stats_df["metric"] == metric)
    ]

    p_fdr = float(p_row["p_fdr"].iloc[0]) if not p_row.empty else float("nan")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=data, x=group_col, y="value", inner=None, ax=ax, color="#B8D8E8")
    sns.boxplot(
        data=data, x=group_col, y="value", width=0.25,
        showfliers=False, ax=ax, color="white",
    )
    sns.stripplot(
        data=data, x=group_col, y="value", color="black",
        alpha=0.35, size=3, ax=ax,
    )
    ax.set_title(f"{moment} — {metric}\nFDR p = {p_fdr:.4g}")
    ax.set_xlabel(group_col)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"distribution_{scheme}_{moment}_{metric}.png",
        dpi=cfg.dpi,
    )
    plt.close(fig)
