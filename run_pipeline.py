"""Point d'entrée du pipeline FGAT eyetracking."""

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
from metrics import extract_metrics
from moments import build_moment_windows
from parsing_behavior import parse_behavior_csv
from parsing_eyetracking import parse_asc
from preprocessing import preprocess_pupil
from stats import fit_mixed_models
from sync import synchronize_run
from viz import (
    plot_distribution,
    plot_exclusion_funnel,
    plot_fdr_heatmap,
    plot_results_table_png,
    plot_run_exclusion_reasons,
)


def discover_subjects(data_root: Path) -> list[Path]:
    """Trouve tous les dossiers C4X###."""
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
    """Trouve le CSV comportemental et le fichier ASC d'un run."""
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
) -> tuple[pd.DataFrame, dict]:
    """Traite un run sans interrompre tout le pipeline."""
    cfg = DEFAULT_CONFIG

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
        "run_unique_pupil_values": np.nan,
        "valid_trial_fraction": np.nan,
        "sync_quality_flag": "not_processed",
        "sync_end_difference_s": np.nan,
    }

    if csv_path is None:
        qc["run_exclusion_reason"] = "missing_behavior_csv"
        return pd.DataFrame(), qc

    if asc_path is None:
        qc["run_exclusion_reason"] = "missing_asc"
        return pd.DataFrame(), qc

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

        preprocessed = preprocess_pupil(
            samples,
            events,
            cfg,
        )

        qc.update(
            evaluate_run_exclusion(
                preprocessed,
                sync,
                cfg,
            )
        )

        if not qc["run_valid"]:
            return pd.DataFrame(), qc

        windows = build_moment_windows(
            trials,
            timing,
            sync,
            cfg,
        )

        metrics = extract_metrics(
            windows,
            preprocessed,
            cfg.max_moment_missing_fraction,
        )

        valid_trials = metrics.loc[
            metrics["moment_valid"],
            "trial_n",
        ].nunique()

        total_trials = trials["trial_n"].nunique()

        qc["valid_trial_fraction"] = (
            valid_trials / max(1, total_trials)
        )

        if (
            qc["valid_trial_fraction"]
            < cfg.min_valid_trial_fraction_per_run
        ):
            qc["run_valid"] = False

            previous_reason = qc["run_exclusion_reason"]

            qc["run_exclusion_reason"] = ";".join(
                reason
                for reason in [
                    previous_reason,
                    "too_few_usable_trials",
                ]
                if reason
            )

            return pd.DataFrame(), qc

        return metrics, qc

    except Exception as exc:
        qc["run_valid"] = False
        qc["run_exclusion_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return pd.DataFrame(), qc


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

    all_metrics: list[pd.DataFrame] = []
    run_qc_rows: list[dict] = []

    for subject_dir in subjects:
        subject_id = subject_dir.name.replace("C4X", "")

        # Run 0 = entraînement : exclu systématiquement.
        for run_id in range(1, 6):
            metrics, qc = process_run(
                subject_dir,
                subject_id,
                run_id,
            )

            run_qc_rows.append(qc)

            if not metrics.empty:
                all_metrics.append(metrics)

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
        print("      Raisons d'exclusion des runs :")

        for _, row in run_exclusion_summary.iterrows():
            print(
                f"      - {row['reason']}: "
                f"{int(row['n_runs'])}"
            )

    subject_qc = evaluate_subject_exclusion(run_qc, cfg)

    subject_qc.to_csv(
        output_dirs["tables"] / "subject_qc.csv",
        index=False,
    )

    subject_exclusion_summary = summarize_subject_exclusions(
        subject_qc
    )

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
        f"{len(kept_subjects)}/{len(subject_qc)} "
        f"(minimum {cfg.min_valid_runs_per_subject} runs)"
    )

    if not subject_exclusion_summary.empty:
        print("      Raisons d'exclusion des sujets :")

        for _, row in subject_exclusion_summary.iterrows():
            print(
                f"      - {row['reason']}: "
                f"{int(row['n_subjects'])}"
            )

    if not all_metrics:
        print("[STOP] Aucune métrique exploitable.")
        return

    metrics_long = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    metrics_long = metrics_long.loc[
        metrics_long["subject_id"].isin(kept_subjects)
    ].copy()

    metrics_long.to_csv(
        output_dirs["tables"] / "metrics_long.csv",
        index=False,
    )

    moment_exclusion_summary = summarize_moment_exclusions(
        metrics_long
    )

    moment_exclusion_summary.to_csv(
        output_dirs["tables"] / "moment_exclusion_summary.csv",
        index=False,
    )

    # Déduplique car metrics_long a une ligne par métrique.
    window_qc = metrics_long[
        [
            "subject_id",
            "run_id",
            "trial_n",
            "moment",
            "moment_valid",
            "moment_exclusion_reason",
        ]
    ].drop_duplicates()

    valid_by_moment = (
        window_qc.groupby("moment")["moment_valid"]
        .agg(["sum", "count"])
        .rename(
            columns={
                "sum": "valid_windows",
                "count": "all_windows",
            }
        )
    )

    print("[4/5] Fenêtres valides par moment :")

    for moment, row in valid_by_moment.iterrows():
        print(
            f"      - {moment}: "
            f"{int(row['valid_windows'])}/{int(row['all_windows'])}"
        )

    if not moment_exclusion_summary.empty:
        print("      Raisons d'exclusion des fenêtres :")

        for _, row in moment_exclusion_summary.iterrows():
            print(
                f"      - {row['moment']} | "
                f"{row['reason']}: "
                f"{int(row['n_windows'])}"
            )

    stats_df = fit_mixed_models(metrics_long, cfg)

    stats_df.to_csv(
        output_dirs["tables"] / "mixed_models_fdr.csv",
        index=False,
    )

    reliable_models = stats_df.loc[
        stats_df["model_reliable"]
    ].copy()

    print(
        f"[5/5] Modèles fiables : "
        f"{len(reliable_models)}/{len(stats_df)}"
    )

    for _, row in reliable_models.iterrows():
        print(
            f"      {row['scheme']} | "
            f"{row['moment']} | "
            f"{row['metric']} | "
            f"p_FDR={row['p_fdr']:.4g} | "
            f"effet_std={row['effect_size_std_max_contrast']:.3f}"
        )

    # Figures exclusions
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

    schemes = [
        "five_categories",
        "valence",
        "arousal",
    ]

    # Heatmaps classiques + trois tableaux PNG finaux.
    for scheme in schemes:
        plot_fdr_heatmap(
            stats_df,
            scheme,
            output_dirs["figures_heatmaps"],
            cfg,
        )

        plot_results_table_png(
            stats_df,
            scheme,
            output_dirs["figures_heatmaps"],
            cfg,
        )

    # Figures descriptives séparées par hypothèse.
    for _, row in stats_df.iterrows():
        reliable = bool(row.get("model_reliable", False))

        if (
            not cfg.plot_non_estimable_hypotheses
            and not reliable
        ):
            continue

        plot_distribution(
            metrics_long,
            stats_df,
            row["scheme"],
            row["moment"],
            row["metric"],
            output_dirs["figures_hypotheses"],
            cfg,
        )

    print(
        "Pipeline terminé. Résultats : "
        f"{output_dirs['root']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline pupillométrie FGAT_Dist."
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

    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_root / "output"
    )

    main(data_root, output_root)