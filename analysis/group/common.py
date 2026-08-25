from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from config import (
    GROUP_DIR,
    GROUP_QC,
    QC_DIR,
    TRIAL_DIR,
)


# ═════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═════════════════════════════════════════════════════════════

def load_group_data(hypothesis):
    """
    Charge les données essai-par-essai d'une hypothèse.

    Exemple :
        hypothesis="h1"
        -> results/trial_level/h1_trials.csv
    """
    path = TRIAL_DIR / f"{hypothesis}_trials.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Lance d'abord : python cli.py compute-all"
        )

    return pd.read_csv(path)


def load_qc():
    """
    Charge les métriques QC run-par-run.
    """
    path = QC_DIR / "run_qc.csv"

    if not path.exists():
        print(
            "⚠ Fichier QC absent : "
            "aucune exclusion QC appliquée."
        )
        return None

    return pd.read_csv(path)


def get_group_output_dir(hypothesis):
    """
    Crée et retourne le dossier de sortie d'une hypothèse.
    """
    output_dir = GROUP_DIR / hypothesis
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ═════════════════════════════════════════════════════════════
# QC
# ═════════════════════════════════════════════════════════════

def _is_active(value):
    """
    True si un seuil QC est activé.
    None désactive le critère.
    """
    return value is not None and np.isfinite(value)


def build_qc_decision_table(qc_df):
    """
    Crée une table run-par-run indiquant si chaque run est inclus.

    Colonnes ajoutées :
    - include_run
    - exclusion_reasons

    Les critères sont définis dans GROUP_QC de config.py.
    """

    if qc_df is None or len(qc_df) == 0:
        return pd.DataFrame()

    qc = qc_df.copy()

    required_columns = [
        "participant",
        "run_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in qc.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes QC absentes : {missing}"
        )

    optional_columns = [
        "pct_pupil_valid",
        "pupil_data_loss_pct",
        "gaze_data_loss_pct",
        "blink_time_pct",
        "blink_rate_per_min",
        "n_samples_total",
        "duration_s",
    ]

    for column in optional_columns:
        if column not in qc.columns:
            qc[column] = np.nan

    qc["include_run"] = True
    qc["exclusion_reasons"] = ""

    def exclude(mask, reason):
        """
        Exclut les runs correspondant au masque et ajoute la raison.
        """
        qc.loc[mask, "include_run"] = False

        current_reason = qc.loc[
            mask,
            "exclusion_reasons"
        ]

        qc.loc[
            mask,
            "exclusion_reasons"
        ] = np.where(
            current_reason.eq(""),
            reason,
            current_reason + "; " + reason,
        )

    # ─────────────────────────────────────────────────────────
    # Pupille valide après nettoyage
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("min_pupil_valid_pct")

    if _is_active(threshold):
        exclude(
            (
                qc["pct_pupil_valid"].notna()
                &
                (qc["pct_pupil_valid"] < threshold)
            ),
            f"pct_pupil_valid<{threshold}",
        )

    # ─────────────────────────────────────────────────────────
    # Perte brute de signal pupillaire
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("max_pupil_data_loss_pct")

    if _is_active(threshold):
        exclude(
            (
                qc["pupil_data_loss_pct"].notna()
                &
                (qc["pupil_data_loss_pct"] > threshold)
            ),
            f"pupil_data_loss_pct>{threshold}",
        )

    # ─────────────────────────────────────────────────────────
    # Perte brute de signal gaze
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("max_gaze_data_loss_pct")

    if _is_active(threshold):
        exclude(
            (
                qc["gaze_data_loss_pct"].notna()
                &
                (qc["gaze_data_loss_pct"] > threshold)
            ),
            f"gaze_data_loss_pct>{threshold}",
        )

    # ─────────────────────────────────────────────────────────
    # Pourcentage du run couvert par des blinks
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("max_blink_time_pct")

    if _is_active(threshold):
        exclude(
            (
                qc["blink_time_pct"].notna()
                &
                (qc["blink_time_pct"] > threshold)
            ),
            f"blink_time_pct>{threshold}",
        )

    # ─────────────────────────────────────────────────────────
    # Blink rate
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("max_blink_rate_per_min")

    if _is_active(threshold):
        exclude(
            (
                qc["blink_rate_per_min"].notna()
                &
                (qc["blink_rate_per_min"] > threshold)
            ),
            f"blink_rate_per_min>{threshold}",
        )

    # ─────────────────────────────────────────────────────────
    # Nombre de samples
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("min_n_samples_total")

    if _is_active(threshold):
        exclude(
            (
                qc["n_samples_total"].notna()
                &
                (qc["n_samples_total"] < threshold)
            ),
            f"n_samples_total<{threshold}",
        )

    # ─────────────────────────────────────────────────────────
    # Durée du run
    # ─────────────────────────────────────────────────────────

    threshold = GROUP_QC.get("min_run_duration_s")

    if _is_active(threshold):
        exclude(
            (
                qc["duration_s"].notna()
                &
                (qc["duration_s"] < threshold)
            ),
            f"duration_s<{threshold}",
        )

    qc.loc[
        qc["include_run"],
        "exclusion_reasons",
    ] = "included"

    return qc


def apply_group_qc(df, qc_df):
    """
    Applique le QC au niveau participant × run.

    Important :
    - seuls les runs exclus sont retirés ;
    - un participant n'est jamais entièrement exclu automatiquement ;
    - toutes les données de ses runs valides sont conservées.

    Exemple :
        C4X016 a 6 runs
        run 2 exclu pour perte pupillaire élevée
        -> les runs 1, 3, 4, 5 et 6 restent dans l'analyse.
    """

    if not GROUP_QC.get("apply_qc", True):
        print("\nQC groupe désactivé dans config.py.")
        return df.copy()

    if qc_df is None or len(qc_df) == 0:
        print("\n⚠ QC indisponible : données non filtrées.")
        return df.copy()

    qc_decision = build_qc_decision_table(qc_df)

    qc_output_path = GROUP_DIR / "qc_run_inclusion.csv"

    qc_decision.to_csv(
        qc_output_path,
        index=False,
    )

    # Garder uniquement les combinaisons participant × run valides.
    good_runs = qc_decision.loc[
        qc_decision["include_run"],
        ["participant", "run_id"],
    ].drop_duplicates()

    # Inner join : seuls les essais des runs exclus disparaissent.
    output = df.merge(
        good_runs,
        on=["participant", "run_id"],
        how="inner",
    )

    print("\n" + "=" * 60)
    print("QC GROUPE — EXCLUSION AU NIVEAU DES RUNS")
    print("=" * 60)

    print(f"Runs totaux : {len(qc_decision)}")

    print(
        "Runs retenus : "
        f"{int(qc_decision['include_run'].sum())}"
    )

    print(
        "Runs exclus : "
        f"{int((~qc_decision['include_run']).sum())}"
    )

    print(
        "Participants présents avant QC : "
        f"{df['participant'].nunique()}"
    )

    print(
        "Participants avec au moins un run valide après QC : "
        f"{output['participant'].nunique()}"
    )

    excluded_runs = qc_decision.loc[
        ~qc_decision["include_run"],
        [
            "participant",
            "run_id",
            "exclusion_reasons",
        ],
    ]

    if len(excluded_runs):
        print("\nRuns exclus :")
        print(
            excluded_runs.to_string(index=False)
        )

    print(f"\n✓ Tableau QC exporté : {qc_output_path}")

    return output


# ═════════════════════════════════════════════════════════════
# AGRÉGATION SUJET × CONDITION
# ═════════════════════════════════════════════════════════════

def subject_condition_table(
    df,
    metric,
    condition_col="stim_category",
):
    """
    Transforme les données essai-par-essai en tableau sujet × condition.

    Une ligne = participant.
    Une colonne = condition émotionnelle.
    Une cellule = moyenne des essais du sujet dans la condition.
    """

    if metric not in df.columns:
        raise ValueError(
            f"Métrique absente : {metric}"
        )

    summary = (
        df.groupby(
            ["participant", condition_col],
            as_index=False,
        )[metric]
        .mean()
    )

    return summary.pivot(
        index="participant",
        columns=condition_col,
        values=metric,
    )


def print_condition_availability(
    subject_table,
    categories,
):
    """
    Affiche combien de sujets ont une valeur utilisable par condition.
    """

    print("\nDisponibilité des données par condition :")

    for category in categories:
        if category in subject_table.columns:
            n = int(
                subject_table[category]
                .notna()
                .sum()
            )
        else:
            n = 0

        print(f"  {category} : {n} sujets")


# ═════════════════════════════════════════════════════════════
# PLOTS : COMPARAISON ENTRE CONDITIONS
# ═════════════════════════════════════════════════════════════

def plot_condition_boxplot(
    subject_table,
    categories,
    metric,
    title,
    output_path,
    colors=None,
):
    """
    Boxplot avec :

    - une boîte par catégorie ;
    - points individuels jitterés ;
    - moyenne groupe en losange noir ;
    - aucune ligne reliant les sujets.

    C'est le plot principal adapté à H1, ainsi qu'aux analyses
    H2/H4 en mode category.
    """

    available_categories = [
        category
        for category in categories
        if category in subject_table.columns
    ]

    if not available_categories:
        raise ValueError(
            "Aucune catégorie disponible pour le boxplot."
        )

    plot_data = [
        subject_table[category]
        .dropna()
        .values
        for category in available_categories
    ]

    if colors is None:
        colors = [
            "#95a5a6"
            for _ in available_categories
        ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    box = ax.boxplot(
        plot_data,
        labels=available_categories,
        patch_artist=True,
        widths=0.58,
        showmeans=True,

        medianprops={
            "color": "black",
            "linewidth": 2,
        },

        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },

        whiskerprops={
            "color": "black",
            "linewidth": 1.2,
        },

        capprops={
            "color": "black",
            "linewidth": 1.2,
        },

        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markersize": 5,
        },
    )

    for patch, color in zip(
        box["boxes"],
        colors,
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    # Jitter déterministe pour obtenir toujours la même figure.
    rng = np.random.default_rng(42)

    for position, category in enumerate(
        available_categories,
        start=1,
    ):
        values = subject_table[
            category
        ].dropna().values

        jitter = rng.normal(
            loc=0,
            scale=0.045,
            size=len(values),
        )

        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color="black",
            alpha=0.75,
            s=34,
            zorder=3,
        )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel("Condition émotionnelle")
    ax.set_ylabel(metric)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.01,
        0.99,
        "Boîte : Q1–Q3\n"
        "Trait : médiane\n"
        "♦ : moyenne groupe\n"
        "Points : participants",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.95,
        },
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()

    return fig, ax


# ═════════════════════════════════════════════════════════════
# PLOTS : SCATTER PLOTS FACETTÉS PAR SUJET
# ═════════════════════════════════════════════════════════════

def plot_subject_faceted_scatter(
    df,
    x_col,
    y_col,
    title,
    output_path,
    correlation_method="spearman",
    x_label=None,
    y_label=None,
    x_jitter=False,
):
    """
    Scatterplot essai-par-essai facetté par participant.

    Adapté aux relations :
    - H2 : métrique pupillaire × durée d'idéation ;
    - H3 : métrique pupillaire × rating ;
    - H4 : métrique regard × durée d'idéation.

    Paramètres
    ----------
    x_jitter :
        True pour H3 / rating, car le rating est une variable discrète.
    """

    data = df[
        ["participant", x_col, y_col]
    ].dropna().copy()

    participants = sorted(
        data["participant"].unique()
    )

    if not participants:
        raise ValueError(
            "Aucune donnée disponible pour le scatterplot."
        )

    n_subjects = len(participants)
    n_cols = min(3, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.3 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    rng = np.random.default_rng(123)

    for index, participant in enumerate(participants):
        row = index // n_cols
        col = index % n_cols

        ax = axes[row, col]

        participant_data = data.loc[
            data["participant"] == participant
        ].copy()

        x_values = participant_data[x_col].values
        y_values = participant_data[y_col].values

        x_plot = x_values.copy()

        if x_jitter:
            x_plot = x_plot + rng.normal(
                loc=0,
                scale=0.06,
                size=len(x_plot),
            )

        ax.scatter(
            x_plot,
            y_values,
            color="#2c3e50",
            alpha=0.70,
            s=35,
        )

        # Corrélation intra-sujet.
        r_value = np.nan
        p_value = np.nan

        if (
            len(participant_data) >= 4
            and participant_data[x_col].nunique() >= 2
            and participant_data[y_col].nunique() >= 2
        ):
            if correlation_method == "spearman":
                r_value, p_value = stats.spearmanr(
                    x_values,
                    y_values,
                )
                r_label = "ρ"
            else:
                r_value, p_value = stats.pearsonr(
                    x_values,
                    y_values,
                )
                r_label = "r"

            # Droite visuelle de tendance.
            # Elle est affichée même si le test est Spearman,
            # seulement comme guide visuel linéaire.
            slope, intercept, _, _, _ = stats.linregress(
                x_values,
                y_values,
            )

            x_line = np.linspace(
                np.min(x_values),
                np.max(x_values),
                100,
            )

            y_line = intercept + slope * x_line

            ax.plot(
                x_line,
                y_line,
                color="#e74c3c",
                linewidth=2,
            )

            annotation = (
                f"{r_label} = {r_value:.2f}\n"
                f"p = {p_value:.3f}\n"
                f"n = {len(participant_data)}"
            )

        else:
            annotation = (
                f"n = {len(participant_data)}\n"
                "corrélation non calculable"
            )

        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.9,
            },
        )

        ax.set_title(
            str(participant),
            fontweight="bold",
        )

        ax.grid(alpha=0.25)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel(
            x_label if x_label else x_col
        )

        ax.set_ylabel(
            y_label if y_label else y_col
        )

    # Supprime les axes vides.
    for index in range(
        n_subjects,
        n_rows * n_cols,
    ):
        row = index // n_cols
        col = index % n_cols
        axes[row, col].axis("off")

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()

    return fig, axes


# ═════════════════════════════════════════════════════════════
# PLOTS : FOREST PLOT DES CORRÉLATIONS INTRA-SUJETS
# ═════════════════════════════════════════════════════════════

def plot_correlation_forest(
    correlations_df,
    group_result,
    title,
    output_path,
):
    """
    Forest plot des corrélations intra-sujets.

    - une ligne = un participant ;
    - ligne verticale à 0 ;
    - point = r ou rho ;
    - diamant noir = moyenne groupe après Fisher z ;
    - IC 95 % du résultat groupe si disponible.
    """

    if correlations_df is None or len(correlations_df) == 0:
        raise ValueError(
            "Aucune corrélation disponible."
        )

    correlations = correlations_df.copy()

    correlations = correlations.sort_values(
        "participant"
    ).reset_index(drop=True)

    n_subjects = len(correlations)

    labels = correlations[
        "participant"
    ].astype(str).tolist()

    y_positions = np.arange(n_subjects)

    fig, ax = plt.subplots(
        figsize=(9, max(4.5, 0.6 * n_subjects + 2.5))
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )

    ax.scatter(
        correlations["r"],
        y_positions,
        color="#2c3e50",
        s=55,
        zorder=3,
        label="Corrélation intra-sujet",
    )

    # Valeur de r affichée à droite des points.
    for _, row in correlations.iterrows():
        y_position = correlations.index[
            correlations["participant"]
            == row["participant"]
        ][0]

        ax.text(
            row["r"] + 0.03,
            y_position,
            f"r = {row['r']:.2f}",
            va="center",
            fontsize=9,
        )

    # Résultat groupe.
    if (
        group_result is not None
        and np.isfinite(group_result.get("mean_r", np.nan))
    ):
        group_y = n_subjects + 0.9

        mean_r = group_result["mean_r"]

        ci_low = group_result.get(
            "ci_low_r",
            np.nan,
        )

        ci_high = group_result.get(
            "ci_high_r",
            np.nan,
        )

        if np.isfinite(ci_low) and np.isfinite(ci_high):
            ax.hlines(
                group_y,
                ci_low,
                ci_high,
                color="black",
                linewidth=2,
                zorder=2,
            )

        ax.scatter(
            mean_r,
            group_y,
            marker="D",
            color="black",
            s=70,
            zorder=4,
            label="Moyenne groupe",
        )

        labels.append("Groupe (Fisher)")
        y_positions = np.append(
            y_positions,
            group_y,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    ax.set_xlim(-1.05, 1.05)

    ax.set_xlabel(
        "Corrélation intra-sujet (r ou ρ)"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if group_result is not None:
        text = (
            f"n = {group_result.get('n', 0)}\n"
            f"r groupe = "
            f"{group_result.get('mean_r', np.nan):.2f}\n"
            f"p = {group_result.get('p', np.nan):.4f}"
        )

        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.95,
            },
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()

    return fig, ax