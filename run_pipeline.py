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
    plot_results_table_for_x,
    plot_run_exclusion_reasons,
    plot_x_stability,
)


def discover_subjects(data_root: Path) -> list[Path]:
    """Trouve automatiquement tous les dossiers C4X###."""
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


def _get_run_qc_trial_fraction(
    metrics: pd.DataFrame,
    cfg,
) -> float:
    """
    Calcule la proportion d'essais dynamiques utilisables pour la QC run.

    La QC utilise une valeur X fixe (par défaut 1.0 s) afin que la décision
    d'exclure un run ne dépende pas de la valeur X ensuite explorée.
    """
    dynamic_moments = [
        "M2a_search_start",
        "M2b_search_end",
        "M3a_rating_start",
        "M3b_rating_end",
    ]

    qc_rows = metrics.loc[
        metrics["moment"].isin(dynamic_moments)
        & np.isclose(
            metrics["window_x_s"].astype(float),
            cfg.run_qc_x_s,
        )
        & metrics["moment_valid"]
    ].copy()

    if qc_rows.empty:
        return 0.0

    valid_trials = qc_rows["trial_n"].nunique()

    total_trials = metrics.loc[
        metrics["moment"].isin(dynamic_moments),
        "trial_n",
    ].nunique()

    return valid_trials / max(1, total_trials)


def process_run(
    subject_dir: Path,
    subject_id: str,
    run_id: int,
) -> tuple[pd.DataFrame, dict]:
    """Traite un run sans interrompre le pipeline si un run échoue."""
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
        "run_qc_x_s": cfg.run_qc_x_s,
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
            cfg,
        )

        qc["valid_trial_fraction"] = _get_run_qc_trial_fraction(
            metrics,
            cfg,
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

    print(f"[1/6] Sujets détectés : {len(subjects)}")
    print(f"      Output : {output_dirs['root']}")
    print(
        "      Exploration X : "
        f"{cfg.dynamic_x_values_s[0]:.1f}s → "
        f"{cfg.dynamic_x_values_s[-1]:.1f}s "
        f"({len(cfg.dynamic_x_values_s)} valeurs)"
    )

    all_metrics: list[pd.DataFrame] = []
    run_qc_rows: list[dict] = []

    for subject_dir in subjects:
        subject_id = subject_dir.name.replace("C4X", "")

        # Run0 = entraînement, exclu systématiquement.
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
        f"[2/6] Runs valides : "
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
        f"[3/6] Sujets retenus : "
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

    # QC fenêtres affichée seulement pour X utilisé dans QC run.
    dynamic_moments = [
        "M2a_search_start",
        "M2b_search_end",
        "M3a_rating_start",
        "M3b_rating_end",
    ]

    window_qc = metrics_long[
        [
            "subject_id",
            "run_id",
            "trial_n",
            "moment",
            "window_x_s",
            "moment_valid",
            "moment_exclusion_reason",
        ]
    ].drop_duplicates()

    qc_m1 = window_qc.loc[
        window_qc["moment"] == "M1_induction"
    ]

    qc_dynamic = window_qc.loc[
        window_qc["moment"].isin(dynamic_moments)
        & np.isclose(
            window_qc["window_x_s"].astype(float),
            cfg.run_qc_x_s,
        )
    ]

    qc_to_display = pd.concat(
        [qc_m1, qc_dynamic],
        ignore_index=True,
    )

    valid_by_moment = (
        qc_to_display.groupby("moment")["moment_valid"]
        .agg(["sum", "count"])
        .rename(
            columns={
                "sum": "valid_windows",
                "count": "all_windows",
            }
        )
    )

    print(
        f"[4/6] Fenêtres valides par moment "
        f"(QC avec X={cfg.run_qc_x_s:.1f}s) :"
    )

    for moment, row in valid_by_moment.iterrows():
        print(
            f"      - {moment}: "
            f"{int(row['valid_windows'])}/{int(row['all_windows'])}"
        )

    if not moment_exclusion_summary.empty:
        print(
            "      Les exclusions détaillées par X sont dans : "
            "moment_exclusion_summary.csv"
        )

    stats_df = fit_mixed_models(metrics_long, cfg)

    stats_df.to_csv(
        output_dirs["tables"] / "mixed_models_fdr.csv",
        index=False,
    )

    reliable_models = stats_df.loc[
        stats_df["model_reliable"]
    ].copy()

    n_significant = int(
        reliable_models["significant_fdr"].sum()
    )

    print(
        f"[5/6] Modèles fiables : "
        f"{len(reliable_models)}/{len(stats_df)} | "
        f"significatifs après FDR : {n_significant}"
    )

    # --------------------------------------------------------------
    # Figures exclusion
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Trois tableaux PNG par X.
    # 16 valeurs X × 3 schémas = 48 tableaux.
    # --------------------------------------------------------------
    for x_s in cfg.dynamic_x_values_s:
        for scheme in schemes:
            plot_results_table_for_x(
                stats_df,
                scheme,
                x_s,
                output_dirs["figures_results_tables"],
                cfg,
            )

    # --------------------------------------------------------------
    # Heatmaps de stabilité selon X.
    # --------------------------------------------------------------
    for scheme in schemes:
        for moment in dynamic_moments:
            plot_x_stability(
                stats_df,
                scheme,
                moment,
                output_dirs["figures_x_stability"],
                cfg,
            )

    # --------------------------------------------------------------
    # Figures descriptives sujet × condition.
    # Une figure par hypothèse, moment, métrique et X.
    # --------------------------------------------------------------
    for _, row in stats_df.iterrows():
        reliable = bool(row.get("model_reliable", False))

        if (
            not cfg.plot_non_estimable_hypotheses
            and not reliable
        ):
            continue

        moment = str(row["moment"])

        if moment == "M1_induction":
            x_for_plot = cfg.dynamic_x_values_s[0]
        else:
            x_for_plot = float(row["window_x_s"])

        plot_distribution(
            metrics_long,
            stats_df,
            str(row["scheme"]),
            moment,
            str(row["metric"]),
            x_for_plot,
            output_dirs["figures_hypotheses"],
            cfg,
        )

    print(
        "[6/6] Pipeline terminé. Résultats : "
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