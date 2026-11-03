"""Pipeline exploratoire CREAFFECT eye-tracking pendant IRMf; runs 1 à 5."""

from __future__ import annotations

import argparse
import re
import json
import importlib.metadata
from dataclasses import asdict
from datetime import datetime, timezone
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
from matching import build_manifest
from timecourse import make_timecourses
from descriptives import export_descriptives
from qc_plots import plot_run_qc
from stats_robust import run_clustered_models
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


def find_run_files(subject_dir: Path, subject_id: str, run_id: int,
                   matched_asc: dict[int, Path] | None = None) -> tuple[Path | None, Path | None]:
    """Recherche CSV par run et ASC UNIQUEMENT via RUN_START/RUN_END."""
    csv_path = subject_dir / f"FGAT_Dist_Run{run_id}_C4X{subject_id}.csv"
    if matched_asc is None:
        matched_asc, _ = build_manifest(subject_dir)
    return (csv_path if csv_path.exists() else None, matched_asc.get(run_id))


def process_run(
    subject_dir: Path,
    subject_id: str,
    run_id: int,
    cfg,
    matched_asc: dict[int, Path] | None = None,
    output_root: Path | None = None,
    collect_timecourses: bool = True,
    make_qc_plot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Traite un run complet sans interrompre le pipeline."""
    csv_path, asc_path = find_run_files(
        subject_dir,
        subject_id,
        run_id,
        matched_asc,
    )

    qc = {
        "subject_id": subject_id,
        "run_id": run_id,
        "asc_path": str(asc_path) if asc_path is not None else "",
        "behavior_csv": str(csv_path) if csv_path is not None else "",
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), qc

    if asc_path is None:
        qc["run_exclusion_reason"] = "missing_asc"
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), qc

    try:
        trials, timing = parse_behavior_csv(
            csv_path,
            subject_id,
            run_id,
        )

        samples, events = parse_asc(asc_path, eye=cfg.eye)

        sync = synchronize_run(
            events,
            timing,
            samples,
            cfg,
            expected_run_id=run_id,
        )

        qc["sync_n_triggers"] = sync.n_triggers
        qc["sync_trigger_interval_median_s"] = sync.median_trigger_interval_s
        qc["sync_trigger_interval_cv"] = sync.trigger_interval_cv
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

        # Diagnostics de dépendance taille de pupille / position du regard.
        subset = preprocessed.loc[preprocessed['pupil_observed_valid'] & preprocessed['gaze_valid']]
        for axis in ('x', 'y'):
            column = f'gaze_{axis}_clean'
            qc[f'pupil_gaze_{axis}_correlation'] = (
                float(subset['pupil_clean'].corr(subset[column]))
                if len(subset) >= 100 and subset[column].std() > 1e-9 else np.nan)
        qc['run_artifact_fraction'] = float(preprocessed['pupil_artifact_mask'].mean())
        qc['run_interpolated_fraction'] = float(preprocessed['pupil_interpolated'].mean())
        qc['run_gaze_valid_fraction'] = float(preprocessed['gaze_valid'].mean())
        if make_qc_plot and output_root is not None:
            plot_run_qc(preprocessed, f'C4X{subject_id}_run{run_id}', output_root / 'figures' / 'qc_traces')
        qc.update(
            evaluate_run_exclusion(
                preprocessed,
                sync,
                cfg,
            )
        )

        if not qc["run_valid"]:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), qc

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
            windows, preprocessed, cfg,
        )
        timecourses = make_timecourses(windows, preprocessed, cfg) if collect_timecourses else pd.DataFrame()
        return metrics, gaze_samples, timecourses, qc

    except Exception as exc:
        qc["run_valid"] = False
        qc["run_exclusion_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), qc


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
    no_stats: bool = False,
    no_figures: bool = False,
    no_timecourses: bool = False,
    qc_plots: bool = False,
    run_mixedlm: bool = False,
) -> None:
    """Exécute le pipeline complet."""
    cfg = DEFAULT_CONFIG
    output_dirs = make_output_dirs(output_root)

    (output_dirs["root"] / "config_used.json").write_text(
        json.dumps({"generated_utc":datetime.now(timezone.utc).isoformat(),
                    "config":asdict(cfg), "analysis_type":"exploratory"},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    deps = ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'seaborn']
    versions = {}
    for dependency in deps:
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = 'not_installed'
    (output_dirs["root"] / "dependency_versions.json").write_text(
        json.dumps(versions, indent=2), encoding='utf-8')
    subjects = discover_subjects(data_root)
    if not subjects:
        raise ValueError(f"Aucun dossier participant C4X### dans {data_root}")

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
    all_timecourses: list[pd.DataFrame] = []
    run_qc_rows: list[dict] = []
    manifest_rows: list[pd.DataFrame] = []

    for subject_index, subject_dir in enumerate(subjects, start=1):
        subject_id = subject_dir.name.replace("C4X", "")

        print(
            f"      Sujet {subject_index}/{len(subjects)} "
            f"(C4X{subject_id})",
            flush=True,
        )

        # Run0 explicitement ignoré, les ASC sont identifiés par marqueurs.
        matched_asc, manifest = build_manifest(subject_dir)
        if not manifest.empty:
            manifest.insert(0, "subject_id", subject_id)
            manifest_rows.append(manifest)
        for run_id in range(1, 6):
            metrics, gaze_samples, timecourses, qc = process_run(
                subject_dir,
                subject_id,
                run_id,
                cfg,
                matched_asc,
                output_root=output_dirs["root"],
                collect_timecourses=not no_timecourses,
                make_qc_plot=qc_plots and not no_figures,
            )

            run_qc_rows.append(qc)

            if not metrics.empty:
                all_metrics.append(metrics)

            if not gaze_samples.empty:
                all_gaze_samples.append(gaze_samples)
            if not timecourses.empty:
                all_timecourses.append(timecourses)

    if manifest_rows:
        pd.concat(manifest_rows, ignore_index=True).to_csv(
            output_dirs["tables"] / "asc_matching_manifest.csv", index=False)
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

    export_descriptives(metrics_long, output_dirs["tables"])
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

    if all_timecourses:
        pupil_timecourses = pd.concat(all_timecourses,ignore_index=True)
        pupil_timecourses = pupil_timecourses.loc[pupil_timecourses['subject_id'].isin(kept_subjects)]
        pupil_timecourses.to_csv(output_dirs["tables"] / "event_locked_pupil_timecourses_20hz.csv",index=False)
    if not no_figures:
        plot_exclusion_funnel(run_qc, subject_qc, output_dirs["figures_exclusion"], cfg)
        plot_run_exclusion_reasons(run_exclusion_summary, output_dirs["figures_exclusion"], cfg)
        plot_gaze_heatmaps(gaze_samples, output_dirs["figures_gaze"], cfg)
    print("[5/5] Modèles exploratoires (non pré-enregistrés)...")
    if no_stats:
        print("Analyses statistiques désactivées (--no-stats).")
        return

    # Modèle principal exploratoire : sujets en effets fixes, SE robustes par
    # participant. L'agrégation par participant couvre les blocs imbriqués.
    stats_df = run_clustered_models(metrics_long, cfg)
    stats_df.to_csv(output_dirs["tables"] / "exploratory_subject_clustered_models_fdr.csv", index=False)
    if run_mixedlm:
        legacy = run_confirmatory_models(metrics_long, cfg)
        legacy.to_csv(output_dirs["tables"] / "exploratory_mixedlm_sensitivity_fdr.csv",index=False)
    if 'model_reliable' in stats_df:
        print_stats_summary(stats_df)
    if no_figures:
        print("Figures désactivées (--no-figures).")
        return

    print("[5/5] Création des figures...")

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

    print("Pipeline terminé.")
    print(f"Résultats : {output_dirs['root']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eye-tracking exploratoire : profil CREAFFECT ou projet déclaratif."
    )

    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        help="Dossier Data contenant les dossiers C4X###.",
    )

    parser.add_argument("--project", "--config", dest="project", type=Path,
                        help="Projet générique version 1 (.json/.yaml), avec manifest et événements")
    parser.add_argument("--list-metrics", action="store_true", help="Afficher le catalogue de métriques")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier de sortie. "
            "Par défaut : <data_root>/output, ou <dossier_projet>/results"
        ),
    )

    parser.add_argument("--no-timecourses",action="store_true",
                        help="Ne pas exporter les courbes pupillaires 20 Hz")
    parser.add_argument("--qc-plots",action="store_true",
                        help="Créer un graphique diagnostic brut/nettoyé par run")
    parser.add_argument("--mixedlm",action="store_true",
                        help="Lancer en complément les anciens modèles mixtes (coûteux)")
    parser.add_argument("--no-stats", action="store_true",
                        help="Calcul QC/métriques et descriptifs, sans modèles statistiques")
    parser.add_argument("--no-figures", action="store_true",
                        help="Conserver les modèles et tableaux, sans graphiques")
    args = parser.parse_args()

    if args.list_metrics:
        from metrics import metric_catalog
        print(metric_catalog().to_csv(index=False))
        raise SystemExit(0)
    if args.project is not None:
        if args.data_root is not None or args.mixedlm:
            parser.error("--project ne se combine pas avec data_root ou --mixedlm")
        from project_pipeline import run_project
        result = run_project(args.project, args.output_dir, no_stats=args.no_stats,
                             no_timecourses=args.no_timecourses,
                             qc_plots=args.qc_plots and not args.no_figures)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2 if result["errors"] else 0)
    if args.data_root is None:
        parser.error("Indiquer data_root (CREAFFECT) ou --project (projet générique)")

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

    main(data_root, output_root, no_stats=args.no_stats,
         no_figures=args.no_figures, no_timecourses=args.no_timecourses,
         qc_plots=args.qc_plots, run_mixedlm=args.mixedlm)
