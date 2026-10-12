"""
QC approfondi d'un sujet eyetracking FGAT.

Usage
-----
python qc_subject.py data 001

Ou :

python qc_subject.py data C4X001

Choisir un dossier output spécifique :

python qc_subject.py data 001 --output-dir qc_subject_001

Ce script est compatible avec les modules actuels :
- config.py
- parsing_behavior.py
- parsing_eyetracking.py
- preprocessing.py
- sync.py
- moments.py
- exclusion.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DEFAULT_CONFIG, make_output_dirs
from exclusion import evaluate_run_exclusion
from moments import build_moment_windows
from parsing_behavior import parse_behavior_csv
from parsing_eyetracking import parse_asc
from preprocessing import preprocess_pupil
from sync import synchronize_run


sns.set_theme(style="whitegrid")


def normalize_subject_id(subject_argument: str) -> str:
    """
    Normalise un ID sujet.

    Exemples
    --------
    "001" -> "001"
    "C4X001" -> "001"
    """
    subject_id = subject_argument.upper().replace("C4X", "").strip()

    if not re.fullmatch(r"\d{3}", subject_id):
        raise ValueError(
            "ID sujet invalide. Utiliser par exemple : 001 ou C4X001."
        )

    return subject_id


def find_subject_directory(data_root: Path, subject_id: str) -> Path:
    """Trouve le dossier C4X### correspondant au sujet."""
    subject_dir = data_root / f"C4X{subject_id}"

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Dossier sujet introuvable : {subject_dir}"
        )

    return subject_dir


def find_run_files(
    subject_dir: Path,
    subject_id: str,
    run_id: int,
) -> tuple[Path | None, Path | None]:
    """Trouve le CSV comportemental et le .asc correspondant au run."""
    csv_path = subject_dir / (
        f"FGAT_Dist_Run{run_id}_C4X{subject_id}.csv"
    )

    asc_candidates = [
        subject_dir / f"{subject_id}_{run_id}.asc",
        subject_dir / f"C4X{subject_id}_{run_id}.asc",
    ]

    asc_path = next(
        (candidate for candidate in asc_candidates if candidate.exists()),
        None,
    )

    return (
        csv_path if csv_path.exists() else None,
        asc_path,
    )


def downsample_for_plot(
    dataframe: pd.DataFrame,
    max_points: int = 30_000,
) -> pd.DataFrame:
    """Sous-échantillonne uniquement pour les graphiques."""
    if len(dataframe) <= max_points:
        return dataframe.copy()

    step = int(np.ceil(len(dataframe) / max_points))
    return dataframe.iloc[::step].copy()


def make_run_summary(
    subject_id: str,
    run_id: int,
    samples: pd.DataFrame,
    events: pd.DataFrame,
    preprocessed: pd.DataFrame,
    blink_intervals: pd.DataFrame,
    sync,
    run_evaluation: dict,
) -> dict:
    """Construit une ligne QC par run."""
    if preprocessed.empty:
        return {
            "subject_id": subject_id,
            "run_id": run_id,
            "n_samples": 0,
            "duration_s": np.nan,
            "run_valid": False,
            "run_exclusion_reason": run_evaluation.get(
                "run_exclusion_reason",
                "empty_samples",
            ),
        }

    duration_s = (
        preprocessed["timestamp_ms"].max()
        - preprocessed["timestamp_ms"].min()
    ) / 1000.0

    raw_missing = (
        preprocessed["pupil_missing_acquisition"].mean()
        if "pupil_missing_acquisition" in preprocessed.columns
        else np.nan
    )

    qc_missing = (
        preprocessed["pupil_missing_qc"].mean()
        if "pupil_missing_qc" in preprocessed.columns
        else preprocessed["pupil_missing_raw"].mean()
    )

    speed_fraction = (
        preprocessed["speed_artifact_mask"].mean()
        if "speed_artifact_mask" in preprocessed.columns
        else np.nan
    )

    speed_near_missing_fraction = (
        preprocessed["speed_near_missing_mask"].mean()
        if "speed_near_missing_mask" in preprocessed.columns
        else np.nan
    )

    residual_outlier_fraction = (
        preprocessed["residual_outlier_mask"].mean()
        if "residual_outlier_mask" in preprocessed.columns
        else np.nan
    )

    blink_fraction = (
        preprocessed["blink_mask"].mean()
        if "blink_mask" in preprocessed.columns
        else np.nan
    )

    pupil_clean = preprocessed["pupil_clean"].dropna()

    return {
        "subject_id": subject_id,
        "run_id": run_id,
        "n_samples": len(samples),
        "duration_s": duration_s,
        "n_events": len(events),
        "n_blinks": len(blink_intervals),
        "blink_rate_per_min": (
            len(blink_intervals) / duration_s * 60.0
            if duration_s > 0
            else np.nan
        ),
        "mean_blink_duration_ms": (
            blink_intervals["blink_duration_ms"].mean()
            if not blink_intervals.empty
            else 0.0
        ),
        "median_blink_duration_ms": (
            blink_intervals["blink_duration_ms"].median()
            if not blink_intervals.empty
            else 0.0
        ),
        "raw_acquisition_missing_fraction": raw_missing,
        "qc_missing_fraction": qc_missing,
        "pipeline_added_missing_fraction": (
            qc_missing - raw_missing
            if pd.notna(qc_missing) and pd.notna(raw_missing)
            else np.nan
        ),
        "blink_mask_fraction": blink_fraction,
        "speed_artifact_fraction": speed_fraction,
        "speed_near_missing_fraction": speed_near_missing_fraction,
        "residual_outlier_fraction": residual_outlier_fraction,
        "n_unique_clean_pupil": int(pupil_clean.nunique()),
        "pupil_clean_mean": pupil_clean.mean(),
        "pupil_clean_sd": pupil_clean.std(ddof=1),
        "pupil_clean_min": pupil_clean.min(),
        "pupil_clean_max": pupil_clean.max(),
        "sync_success": sync.success,
        "sync_reason": sync.reason,
        "sync_offset_ms": sync.offset_ms,
        "sync_quality_flag": sync.quality_flag,
        "sync_end_difference_s": sync.end_difference_s,
        "run_valid": run_evaluation.get("run_valid", False),
        "run_missing_fraction_used": run_evaluation.get(
            "run_missing_fraction",
            np.nan,
        ),
        "run_exclusion_reason": run_evaluation.get(
            "run_exclusion_reason",
            "",
        ),
    }


def plot_run_overview(
    preprocessed: pd.DataFrame,
    blink_intervals: pd.DataFrame,
    subject_id: str,
    run_id: int,
    output_dir: Path,
) -> None:
    """
    Produit une vue globale du run.

    Panels :
    1. pupille brute et nettoyée ;
    2. masques QC ;
    3. histogrammes de pupille brute/nettoyée ;
    4. fraction de données manquantes par minute.
    """
    if preprocessed.empty:
        return

    plot_df = downsample_for_plot(preprocessed)

    time_s = (
        plot_df["timestamp_ms"]
        - plot_df["timestamp_ms"].iloc[0]
    ) / 1000.0

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(16, 12),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1.5, 2, 2]},
    )

    # --------------------------------------------------------------
    # 1. Signal brut et signal nettoyé.
    # --------------------------------------------------------------
    axes[0].plot(
        time_s,
        plot_df["pupil_size"],
        color="lightgray",
        linewidth=0.4,
        label="Pupille brute",
    )

    axes[0].plot(
        time_s,
        plot_df["pupil_clean"],
        color="#1F77B4",
        linewidth=0.5,
        label="Pupille nettoyée",
    )

    axes[0].set_title(
        f"C4X{subject_id} — Run {run_id} — signal pupillaire global"
    )
    axes[0].set_ylabel("Taille pupillaire")
    axes[0].legend(loc="upper right")

    # --------------------------------------------------------------
    # 2. Masques d'artefacts.
    # --------------------------------------------------------------
    if "pupil_missing_acquisition" in plot_df.columns:
        acquisition_mask = plot_df["pupil_missing_acquisition"].astype(bool)
        axes[1].scatter(
            time_s[acquisition_mask],
            np.ones(acquisition_mask.sum()),
            s=2,
            color="red",
            label="Manquant dans ASC",
        )

    if "blink_mask" in plot_df.columns:
        blink_mask = plot_df["blink_mask"].astype(bool)
        axes[1].scatter(
            time_s[blink_mask],
            np.full(blink_mask.sum(), 0.65),
            s=2,
            color="orange",
            label="Blink / padding",
        )

    if "speed_artifact_mask" in plot_df.columns:
        speed_mask = plot_df["speed_artifact_mask"].astype(bool)
        axes[1].scatter(
            time_s[speed_mask],
            np.full(speed_mask.sum(), 0.30),
            s=2,
            color="purple",
            label="Vitesse pupille élevée",
        )

    if "residual_outlier_mask" in plot_df.columns:
        outlier_mask = plot_df["residual_outlier_mask"].astype(bool)
        axes[1].scatter(
            time_s[outlier_mask],
            np.full(outlier_mask.sum(), 0.05),
            s=2,
            color="black",
            label="Outlier résiduel",
        )

    axes[1].set_ylim(-0.1, 1.2)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Masques")
    axes[1].legend(
        loc="upper right",
        ncol=2,
        fontsize=8,
    )

    # --------------------------------------------------------------
    # 3. Distribution signal brut et nettoyé.
    # --------------------------------------------------------------
    raw_valid = preprocessed["pupil_size"].dropna()
    clean_valid = preprocessed["pupil_clean"].dropna()

    if not raw_valid.empty:
        axes[2].hist(
            raw_valid,
            bins=100,
            density=True,
            alpha=0.45,
            color="lightgray",
            label="Brut",
        )

    if not clean_valid.empty:
        axes[2].hist(
            clean_valid,
            bins=100,
            density=True,
            alpha=0.55,
            color="#1F77B4",
            label="Nettoyé",
        )

    axes[2].set_title("Distribution de la pupille")
    axes[2].set_xlabel("Taille pupillaire")
    axes[2].set_ylabel("Densité")
    axes[2].legend()

    # --------------------------------------------------------------
    # 4. Missingness par minute.
    # --------------------------------------------------------------
    full_df = preprocessed.copy()

    full_df["time_s"] = (
        full_df["timestamp_ms"]
        - full_df["timestamp_ms"].iloc[0]
    ) / 1000.0

    full_df["minute"] = (full_df["time_s"] // 60).astype(int)

    missing_column = (
        "pupil_missing_qc"
        if "pupil_missing_qc" in full_df.columns
        else "pupil_missing_raw"
    )

    by_minute = (
        full_df.groupby("minute")
        .agg(
            acquisition_missing=(
                "pupil_missing_acquisition",
                "mean",
            )
            if "pupil_missing_acquisition" in full_df.columns
            else (missing_column, "mean"),
            qc_missing=(missing_column, "mean"),
        )
        .reset_index()
    )

    axes[3].bar(
        by_minute["minute"],
        by_minute["acquisition_missing"] * 100,
        color="red",
        alpha=0.65,
        label="Manquant ASC",
    )

    axes[3].bar(
        by_minute["minute"],
        (
            by_minute["qc_missing"]
            - by_minute["acquisition_missing"]
        ).clip(lower=0) * 100,
        bottom=by_minute["acquisition_missing"] * 100,
        color="orange",
        alpha=0.75,
        label="Ajout blink/padding",
    )

    axes[3].axhline(
        20,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Seuil historique 20%",
    )

    axes[3].axhline(
        35,
        color="darkred",
        linestyle="--",
        linewidth=1,
        label="Seuil actuel 35%",
    )

    axes[3].set_title("Données manquantes par minute")
    axes[3].set_xlabel("Minute dans le run")
    axes[3].set_ylabel("% échantillons manquants")
    axes[3].legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"run_{run_id}_overview.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def get_worst_segments(
    preprocessed: pd.DataFrame,
    segment_duration_s: float = 10.0,
    n_segments: int = 4,
) -> pd.DataFrame:
    """
    Identifie les segments temporels avec le plus de données manquantes.

    Retourne un DataFrame avec start_ms, end_ms et missing_fraction.
    """
    if preprocessed.empty:
        return pd.DataFrame()

    missing_column = (
        "pupil_missing_qc"
        if "pupil_missing_qc" in preprocessed.columns
        else "pupil_missing_raw"
    )

    start_ms = preprocessed["timestamp_ms"].min()
    end_ms = preprocessed["timestamp_ms"].max()

    segment_ms = segment_duration_s * 1000.0

    rows: list[dict] = []

    current_start = start_ms

    while current_start < end_ms:
        current_end = min(current_start + segment_ms, end_ms)

        segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                current_start,
                current_end,
                inclusive="left",
            )
        ]

        if not segment.empty:
            rows.append(
                {
                    "start_ms": current_start,
                    "end_ms": current_end,
                    "missing_fraction": segment[missing_column].mean(),
                    "n_samples": len(segment),
                }
            )

        current_start += segment_ms

    segments = pd.DataFrame(rows)

    if segments.empty:
        return segments

    return segments.nlargest(
        n_segments,
        "missing_fraction",
    ).sort_values("start_ms")


def plot_worst_segments(
    preprocessed: pd.DataFrame,
    subject_id: str,
    run_id: int,
    output_dir: Path,
) -> None:
    """Affiche les segments temporels les plus dégradés."""
    worst_segments = get_worst_segments(preprocessed)

    if worst_segments.empty:
        return

    n_segments = len(worst_segments)

    fig, axes = plt.subplots(
        nrows=n_segments,
        ncols=1,
        figsize=(15, 3.5 * n_segments),
        squeeze=False,
    )

    for axis, (_, segment_info) in zip(
        axes[:, 0],
        worst_segments.iterrows(),
    ):
        segment = preprocessed.loc[
            preprocessed["timestamp_ms"].between(
                segment_info["start_ms"],
                segment_info["end_ms"],
                inclusive="both",
            )
        ].copy()

        time_s = (
            segment["timestamp_ms"]
            - segment["timestamp_ms"].iloc[0]
        ) / 1000.0

        axis.plot(
            time_s,
            segment["pupil_size"],
            color="lightgray",
            linewidth=0.7,
            label="Brut",
        )

        axis.plot(
            time_s,
            segment["pupil_clean"],
            color="#1F77B4",
            linewidth=1.0,
            label="Nettoyé",
        )

        if "blink_mask" in segment.columns:
            blink_rows = segment.loc[
                segment["blink_mask"].astype(bool)
            ]

            if not blink_rows.empty:
                axis.scatter(
                    (
                        blink_rows["timestamp_ms"]
                        - segment["timestamp_ms"].iloc[0]
                    ) / 1000.0,
                    np.full(len(blink_rows), axis.get_ylim()[0]),
                    s=3,
                    color="orange",
                    alpha=0.7,
                    label="Blink/padding",
                )

        axis.set_title(
            "Segment dégradé : "
            f"missing={segment_info['missing_fraction']:.1%} | "
            f"temps run={segment_info['start_ms'] / 1000:.1f}–"
            f"{segment_info['end_ms'] / 1000:.1f}s"
        )

        axis.set_xlabel("Temps dans le segment (s)")
        axis.set_ylabel("Pupille")
        axis.legend(loc="upper right")

    fig.suptitle(
        f"C4X{subject_id} — Run {run_id} — segments les plus dégradés",
        y=1.01,
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / f"run_{run_id}_worst_segments.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_blink_durations(
    blink_intervals: pd.DataFrame,
    subject_id: str,
    run_id: int,
    output_dir: Path,
) -> None:
    """Histogramme des durées de blink détectées."""
    if blink_intervals.empty:
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5),
    )

    sns.histplot(
        data=blink_intervals,
        x="blink_duration_ms",
        bins=50,
        color="#E67E22",
        ax=axes[0],
    )

    axes[0].axvline(
        blink_intervals["blink_duration_ms"].median(),
        color="black",
        linestyle="--",
        label=(
            "Médiane = "
            f"{blink_intervals['blink_duration_ms'].median():.0f} ms"
        ),
    )

    axes[0].set_title(
        f"C4X{subject_id} — Run {run_id} — durées de blink"
    )
    axes[0].set_xlabel("Durée blink (ms)")
    axes[0].legend()

    sns.boxplot(
        y=blink_intervals["blink_duration_ms"],
        color="#E67E22",
        ax=axes[1],
    )

    axes[1].set_title("Distribution des durées")
    axes[1].set_ylabel("Durée blink (ms)")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"run_{run_id}_blink_durations.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_behavior_windows(
    windows: pd.DataFrame,
    preprocessed: pd.DataFrame,
    subject_id: str,
    run_id: int,
    output_dir: Path,
) -> None:
    """
    Visualise l'emplacement temporel des fenêtres comportementales.

    Cela aide à détecter un problème de synchronisation grossier :
    fenêtres totalement hors acquisition EyeLink, par exemple.
    """
    if windows.empty or preprocessed.empty:
        return

    selected_moments = [
        "M1_early_photometric",
        "M1_late_sustained",
        "H2_full_search",
        "H3_liking_anchor",
        "M2a_search_start",
        "M3a_rating_start",
    ]

    plot_windows = windows.loc[
        windows["moment"].isin(selected_moments)
    ].copy()

    # Une valeur X représentative pour ne pas surcharger la figure.
    plot_windows = plot_windows.loc[
        (plot_windows["moment"].isin(
            [
                "M1_early_photometric",
                "M1_late_sustained",
                "H2_full_search",
                "H3_liking_anchor",
            ]
        ))
        | np.isclose(
            plot_windows["window_x_s"].fillna(0).astype(float),
            1.0,
        )
    ]

    if plot_windows.empty:
        return

    asc_start = preprocessed["timestamp_ms"].min()
    asc_end = preprocessed["timestamp_ms"].max()

    moment_order = {
        "M1_early_photometric": 0,
        "M1_late_sustained": 1,
        "H2_full_search": 2,
        "H3_liking_anchor": 3,
        "M2a_search_start": 4,
        "M3a_rating_start": 5,
    }

    plot_windows["y"] = plot_windows["moment"].map(moment_order)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.axvspan(
        asc_start,
        asc_end,
        color="#D1FAE5",
        alpha=0.3,
        label="Acquisition ASC",
    )

    colors = {
        "M1_early_photometric": "#9CA3AF",
        "M1_late_sustained": "#EF4444",
        "H2_full_search": "#3B82F6",
        "H3_liking_anchor": "#8B5CF6",
        "M2a_search_start": "#10B981",
        "M3a_rating_start": "#F59E0B",
    }

    for _, row in plot_windows.iterrows():
        ax.hlines(
            y=row["y"],
            xmin=row["start_eyelink_ms"],
            xmax=row["end_eyelink_ms"],
            color=colors.get(row["moment"], "black"),
            linewidth=2,
            alpha=0.7,
        )

    ax.set_yticks(
        list(moment_order.values()),
        list(moment_order.keys()),
    )

    ax.set_xlabel("Timestamp EyeLink (ms)")
    ax.set_ylabel("Moment")
    ax.set_title(
        f"C4X{subject_id} — Run {run_id} — position des fenêtres "
        "comportementales dans l'acquisition EyeLink"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / f"run_{run_id}_behavior_windows.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_text_report(
    summary: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
) -> None:
    """Écrit un rapport texte lisible rapidement."""
    report_path = output_dir / "qc_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write(f"QC approfondi — Sujet C4X{subject_id}\n")
        file.write("=" * 72 + "\n\n")

        if summary.empty:
            file.write("Aucun run analysable.\n")
            return

        file.write("Résumé par run\n")
        file.write("-" * 72 + "\n")

        columns = [
            "run_id",
            "run_valid",
            "run_exclusion_reason",
            "raw_acquisition_missing_fraction",
            "qc_missing_fraction",
            "pipeline_added_missing_fraction",
            "n_blinks",
            "blink_rate_per_min",
            "mean_blink_duration_ms",
            "speed_artifact_fraction",
            "residual_outlier_fraction",
        ]

        available = [c for c in columns if c in summary.columns]

        file.write(
            summary[available].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        file.write("\n\n")
        file.write("Interprétation pratique\n")
        file.write("-" * 72 + "\n")
        file.write(
            "- raw_acquisition_missing_fraction : trous réellement présents "
            "dans le fichier ASC.\n"
        )
        file.write(
            "- qc_missing_fraction : trous ASC + périodes considérées "
            "comme blinks/padding par le pipeline.\n"
        )
        file.write(
            "- pipeline_added_missing_fraction : différence entre les deux. "
            "Si très élevée, le masque de blink/padding est probablement "
            "trop agressif.\n"
        )
        file.write(
            "- speed_artifact_fraction : proportion de changements de vitesse "
            "pupillaire anormaux détectés.\n"
        )
        file.write(
            "- residual_outlier_fraction : valeurs retirées après détection "
            "MAD résiduelle.\n"
        )


def process_subject(
    data_root: Path,
    subject_id: str,
    output_root: Path,
) -> None:
    """Exécute le QC complet d'un sujet."""
    cfg = DEFAULT_CONFIG
    subject_dir = find_subject_directory(data_root, subject_id)

    subject_output = output_root / f"C4X{subject_id}"
    subject_output.mkdir(parents=True, exist_ok=True)

    print(f"QC sujet : C4X{subject_id}")
    print(f"Dossier sujet : {subject_dir}")
    print(f"Output QC : {subject_output}")
    print()

    run_summaries: list[dict] = []
    all_blinks: list[pd.DataFrame] = []
    all_windows_qc: list[pd.DataFrame] = []

    for run_id in range(1, 6):
        print(f"[Run {run_id}/5] Chargement...", end="", flush=True)

        csv_path, asc_path = find_run_files(
            subject_dir,
            subject_id,
            run_id,
        )

        if csv_path is None or asc_path is None:
            reason = (
                "missing_behavior_csv"
                if csv_path is None
                else "missing_asc"
            )

            run_summaries.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "run_valid": False,
                    "run_exclusion_reason": reason,
                }
            )

            print(f" EXCLU ({reason})")
            continue

        try:
            trials, timing = parse_behavior_csv(
                csv_path,
                subject_id,
                run_id,
            )

            samples, events = parse_asc(asc_path)

            sync = synchronize_run(
                events,
                timing,
                samples,
                cfg,
            )

            preprocessed, blink_intervals = preprocess_pupil(
                samples,
                events,
                cfg,
            )

            run_evaluation = evaluate_run_exclusion(
                preprocessed,
                sync,
                cfg,
            )

            summary = make_run_summary(
                subject_id=subject_id,
                run_id=run_id,
                samples=samples,
                events=events,
                preprocessed=preprocessed,
                blink_intervals=blink_intervals,
                sync=sync,
                run_evaluation=run_evaluation,
            )

            run_summaries.append(summary)

            if not blink_intervals.empty:
                blink_export = blink_intervals.copy()
                blink_export["subject_id"] = subject_id
                blink_export["run_id"] = run_id
                all_blinks.append(blink_export)

            # Création des fenêtres, même si le run est exclu.
            # Cela permet de vérifier la synchronisation et les fenêtres.
            if sync.success:
                windows = build_moment_windows(
                    trials,
                    timing,
                    sync,
                    cfg,
                )

                if not windows.empty:
                    window_export = windows[
                        [
                            "subject_id",
                            "run_id",
                            "trial_n",
                            "induction_block_id",
                            "moment",
                            "window_x_s",
                            "start_behavior_s",
                            "end_behavior_s",
                            "start_eyelink_ms",
                            "end_eyelink_ms",
                            "baseline_source",
                            "baseline_start_behavior_s",
                            "baseline_end_behavior_s",
                            "baseline_start_eyelink_ms",
                            "baseline_end_eyelink_ms",
                            "actual_duration_s",
                            "duration_valid",
                        ]
                    ].copy()

                    all_windows_qc.append(window_export)

                    plot_behavior_windows(
                        windows=windows,
                        preprocessed=preprocessed,
                        subject_id=subject_id,
                        run_id=run_id,
                        output_dir=subject_output,
                    )

            plot_run_overview(
                preprocessed=preprocessed,
                blink_intervals=blink_intervals,
                subject_id=subject_id,
                run_id=run_id,
                output_dir=subject_output,
            )

            plot_worst_segments(
                preprocessed=preprocessed,
                subject_id=subject_id,
                run_id=run_id,
                output_dir=subject_output,
            )

            plot_blink_durations(
                blink_intervals=blink_intervals,
                subject_id=subject_id,
                run_id=run_id,
                output_dir=subject_output,
            )

            raw_missing = summary.get(
                "raw_acquisition_missing_fraction",
                np.nan,
            )

            qc_missing = summary.get(
                "qc_missing_fraction",
                np.nan,
            )

            added_missing = summary.get(
                "pipeline_added_missing_fraction",
                np.nan,
            )

            status = "OK" if summary["run_valid"] else "EXCLU"

            print(
                f" {status} | "
                f"ASC missing={raw_missing:.1%} | "
                f"QC missing={qc_missing:.1%} | "
                f"pipeline ajouté={added_missing:.1%} | "
                f"blinks={summary.get('n_blinks', 0)}"
            )

        except Exception as exc:
            run_summaries.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "run_valid": False,
                    "run_exclusion_reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            print(f" ERREUR ({type(exc).__name__}: {exc})")

    summary_df = pd.DataFrame(run_summaries)

    summary_df.to_csv(
        subject_output / "qc_summary_runs.csv",
        index=False,
    )

    if all_blinks:
        blinks_df = pd.concat(all_blinks, ignore_index=True)
    else:
        blinks_df = pd.DataFrame(
            columns=[
                "blink_start_ms",
                "blink_end_ms",
                "blink_duration_ms",
                "subject_id",
                "run_id",
            ]
        )

    blinks_df.to_csv(
        subject_output / "qc_blinks_runs.csv",
        index=False,
    )

    if all_windows_qc:
        windows_df = pd.concat(all_windows_qc, ignore_index=True)
    else:
        windows_df = pd.DataFrame()

    windows_df.to_csv(
        subject_output / "qc_windows.csv",
        index=False,
    )

    write_text_report(
        summary=summary_df,
        subject_id=subject_id,
        output_dir=subject_output,
    )

    print()
    print("=" * 72)
    print("QC terminé.")
    print(f"Résultats : {subject_output}")
    print()
    print("Fichiers importants :")
    print("  - qc_summary_runs.csv")
    print("  - qc_report.txt")
    print("  - run_X_overview.png")
    print("  - run_X_worst_segments.png")
    print("  - run_X_behavior_windows.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QC approfondi d'un sujet eyetracking FGAT."
    )

    parser.add_argument(
        "data_root",
        type=Path,
        help="Dossier racine contenant les dossiers C4X###.",
    )

    parser.add_argument(
        "subject_id",
        type=str,
        help="ID sujet : 001 ou C4X001.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier de sortie QC. "
            "Par défaut : <data_root>/output_subject_qc"
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root.resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dossier data_root introuvable : {data_root}"
        )

    subject_id = normalize_subject_id(args.subject_id)

    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_root / "output_subject_qc"
    )

    process_subject(
        data_root=data_root,
        subject_id=subject_id,
        output_root=output_root,
    )