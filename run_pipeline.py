"""Point d'entrée du pipeline FGAT eyetracking confirmatoire."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, make_output_dirs
from exclusion import (
    evaluate_run_exclusion,
    evaluate_subject_exclusion,
    summarize_moment_exclusions,
    summarize_run_exclusions,
    summarize_subject_exclusions,
)
from metrics import (
    extract_h4_gaze_samples,
    extract_metrics,
)
from moments import build_moment_windows
from parsing_behavior import parse_behavior_csv
from parsing_eyetracking import parse_asc
from preprocessing import preprocess_pupil
from stats import run_confirmatory_models
from sync import synchronize_run
from viz import (
    plot_exclusion_funnel,
    plot_gaze_heatmaps,
    plot_hypothesis_distributions,
    plot_liking_scatter,
    plot_results_table,
    plot_run_exclusion_reasons,
)


def discover_subjects(data_root: Path) -> list[Path]:
    """Trouve automatiquement les dossiers C4X###."""
    return sorted(
        directory
        for directory in data_root.iterdir()
        if directory.is_dir()
        and re.fullmatch(r"C4X\d{3}", directory.name)
    )


def find_run_files(
    subject_dir: Path,
    subject_id: str,
    run_id: int,
) -> tuple[Path | None, Path | None]:
    """Trouve le CSV comportemental et l'ASC correspondant."""
    csv_path = subject_dir / (
        f"FGAT_Dist_Run{run_id}_C4X{subject_id}.csv"
    )

    asc_candidates = [
        subject_dir / f"{subject_id}_{run_id}.asc",
        subject_dir / f"C4X{subject_id}_{run_id}.asc",
    ]

    asc_path = next(
        (
            candidate
            for candidate in asc_candidates
            if candidate.exists()
        ),
        None,
    )

    return (
        csv_path if csv_path.exists() else None,
        asc_path,
    )


def process_run(
    subject_dir: Path,
    subject_id: str,
    run_id: int,
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Traite un run complet sans interrompre le pipeline."""
    csv_path, asc_path = find_run_files(
        subject_dir,
        subject_id,
        run_id,
    )

    qc = {
        "subject_id": subject_id,
        "run_id": run_id,
        "run_valid": False,
        "run_exclusion_reason": "",
        "run_missing_fraction": np.nan,
        "run_acquisition_missing_fraction": np.nan,
        "run_unique_pupil_values": np.nan,
        "sync_quality_flag": "not_processed",
        "sync_end_difference_s": np.nan,
        "n_detected_blinks": np.nan,
        "mean_blink_duration_ms": np.nan,
    }

    if csv_path is None:
        qc["run_exclusion_reason"] = "missing_behavior_csv"
        return pd.DataFrame(), pd.DataFrame(), qc

    if asc_path is None:
        qc["run_exclusion_reason"] = "missing_asc"
        return pd.DataFrame(), pd.DataFrame(), qc

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

        qc["sync_quality_flag"] = sync.quality_flag
        qc["sync_end_difference_s"] = sync.end_difference_s

        preprocessed, blink_intervals = preprocess_pupil(
            samples,
            events,
            cfg,
        )

        qc["n_detected_blinks"] = len(blink_intervals)

        if not blink_intervals.empty:
            qc["mean_blink_duration_ms"] = float(
                blink_intervals["blink_duration_ms"].mean()
            )

        qc.update(
            evaluate_run_exclusion(
                preprocessed,
                sync,
                cfg,
            )
        )

        if not qc["run_valid"]:
            return pd.DataFrame(), pd.DataFrame(), qc

        windows = build_moment_windows(
            trials,
            timing,
            sync,
            cfg,
        )

        metrics = extract_metrics(
            windows,
            preprocessed,
            blink_intervals,
            events,
            cfg,
        )

        gaze_samples = extract_h4_gaze_samples(
            windows,
            preprocessed,
            cfg,
        )

        return metrics, gaze_samples, qc

    except Exception as exc:
        qc["run_valid"] = False
        qc["run_exclusion_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return pd.DataFrame(), pd.DataFrame(), qc


def print_stats_summary(stats_df: pd.DataFrame) -> None:
    """Affiche un résumé compact des modèles."""
    reliable = stats_df.loc[
        stats_df["model_reliable"]
    ].copy()

    if reliable.empty:
        print("      Aucun modèle fiable.")
        return

    significant_group = reliable.loc[
        reliable["group_significant_fdr"]
    ]

    significant_rating = reliable.loc[
        reliable["rating_significant_fdr"]
    ]

    print(
        f"      Modèles fiables : {len(reliable)}/{len(stats_df)}"
    )

    print(
        f"      Effets de groupe significatifs FDR : "
        f"{len(significant_group)}"
    )

    print(
        f"      Effets de rating significatifs FDR : "
        f"{len(significant_rating)}"
    )

    for _, row in significant_group.iterrows():
        print(
            f"      - GROUPE | {row['hypothesis']} | "
            f"{row['scheme']} | {row['metric']} | "
            f"p_FDR={row['p_group_fdr']:.4g}"
        )

    for _, row in significant_rating.iterrows():
        print(
            f"      - RATING | {row['hypothesis']} | "
            f"{row['metric']} | "
            f"p_FDR={row['p_rating_fdr']:.4g}"
        )


def main(
    data_root: Path,
    output_root: Path,
) -> None:
    """Exécute le pipeline complet."""
    cfg = DEFAULT_CONFIG
    output_dirs = make_output_dirs(output_root)

    subjects = discover_subjects(data_root)

    print(f"[1/5] Sujets détectés : {len(subjects)}")
    print(f"      Output : {output_dirs['root']}")
    print(
        f"      H2 fenêtre pré-idée fixe : "
        f"{cfg.fixed_pre_idea_window_s:.2f}s"
    )
    print(
        f"      Exclusion run seulement si données ASC manquantes > "
        f"{cfg.max_run_acquisition_missing_fraction:.0%}"
    )

    all_metrics: list[pd.DataFrame] = []
    all_gaze_samples: list[pd.DataFrame] = []
    run_qc_rows: list[dict] = []

    for subject_index, subject_dir in enumerate(subjects, start=1):
        subject_id = subject_dir.name.replace("C4X", "")

        print(
            f"      Sujet {subject_index}/{len(subjects)} "
            f"(C4X{subject_id})",
            flush=True,
        )

        # Run0 exclu : entraînement.
        for run_id in range(1, 6):
            metrics, gaze_samples, qc = process_run(
                subject_dir,
                subject_id,
                run_id,
                cfg,
            )

            run_qc_rows.append(qc)

            if not metrics.empty:
                all_metrics.append(metrics)

            if not gaze_samples.empty:
                all_gaze_samples.append(gaze_samples)

    run_qc = pd.DataFrame(run_qc_rows)

    run_qc.to_csv(
        output_dirs["tables"] / "run_qc.csv",
        index=False,
    )

    run_exclusion_summary = summarize_run_exclusions(run_qc)

    run_exclusion_summary.to_csv(
        output_dirs["tables"] / "run_exclusion_summary.csv",
        index=False,
    )

    print(
        f"[2/5] Runs valides : "
        f"{int(run_qc['run_valid'].sum())}/{len(run_qc)}"
    )

    if not run_exclusion_summary.empty:
        print("      Raisons d'exclusion :")

        for _, row in run_exclusion_summary.iterrows():
            print(
                f"      - {row['reason']}: {int(row['n_runs'])}"
            )

    subject_qc = evaluate_subject_exclusion(run_qc, cfg)

    subject_qc.to_csv(
        output_dirs["tables"] / "subject_qc.csv",
        index=False,
    )

    subject_exclusion_summary = summarize_subject_exclusions(subject_qc)

    subject_exclusion_summary.to_csv(
        output_dirs["tables"] / "subject_exclusion_summary.csv",
        index=False,
    )

    kept_subjects = set(
        subject_qc.loc[
            subject_qc["subject_valid"],
            "subject_id",
        ]
    )

    print(
        f"[3/5] Sujets retenus : "
        f"{len(kept_subjects)}/{len(subject_qc)}"
    )

    if not all_metrics:
        print("[STOP] Aucune métrique exploitable.")
        return

    metrics_long = pd.concat(all_metrics, ignore_index=True)

    metrics_long = metrics_long.loc[
        metrics_long["subject_id"].isin(kept_subjects)
    ].copy()

    metrics_long.to_csv(
        output_dirs["tables"] / "metrics_long.csv",
        index=False,
    )

    moment_exclusion_summary = summarize_moment_exclusions(metrics_long)

    moment_exclusion_summary.to_csv(
        output_dirs["tables"] / "moment_exclusion_summary.csv",
        index=False,
    )

    print("[4/5] Métriques et QC fenêtres sauvegardées.")

    if all_gaze_samples:
        gaze_samples = pd.concat(
            all_gaze_samples,
            ignore_index=True,
        )

        gaze_samples = gaze_samples.loc[
            gaze_samples["subject_id"].isin(kept_subjects)
        ].copy()

        # Ne pas forcément ouvrir ce fichier dans Excel : il peut être grand.
        gaze_samples.to_csv(
            output_dirs["tables"] / "h4_gaze_samples.csv",
            index=False,
        )
    else:
        gaze_samples = pd.DataFrame()

    print("[5/5] Ajustement des modèles confirmatoires...")

    stats_df = run_confirmatory_models(
        metrics_long,
        cfg,
    )

    stats_df.to_csv(
        output_dirs["tables"] / "confirmatory_models_fdr.csv",
        index=False,
    )

    print_stats_summary(stats_df)

    print("[5/5] Création des figures...")

    plot_exclusion_funnel(
        run_qc,
        subject_qc,
        output_dirs["figures_exclusion"],
        cfg,
    )

    plot_run_exclusion_reasons(
        run_exclusion_summary,
        output_dirs["figures_exclusion"],
        cfg,
    )

    plot_results_table(
        stats_df,
        output_dirs["figures_results"],
        cfg,
    )

    plot_hypothesis_distributions(
        metrics_long,
        stats_df,
        output_dirs["figures_distributions"],
        cfg,
    )

    plot_liking_scatter(
        metrics_long,
        stats_df,
        output_dirs["figures_distributions"],
        cfg,
    )

    plot_gaze_heatmaps(
        gaze_samples,
        output_dirs["figures_gaze"],
        cfg,
    )

    print("Pipeline terminé.")
    print(f"Résultats : {output_dirs['root']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline confirmatoire pupillométrie FGAT."
    )

    parser.add_argument(
        "data_root",
        type=Path,
        help="Dossier Data contenant les dossiers C4X###.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier de sortie. "
            "Par défaut : <data_root>/output"
        ),
    )

    args = parser.parse_args()

    data_root = args.data_root.resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {data_root}"
        )

    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_root / "output"
    )

    main(data_root, output_root)