"""Point d'entrée du pipeline FGAT eyetracking x IRMf."""

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
    plot_run_exclusion_reasons,
)


def discover_subjects(root: Path) -> list[Path]:
    """Trouve automatiquement tous les dossiers C4X###."""
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"C4X\d{3}", path.name)
    )


def find_run_files(subject_dir: Path, subject_id: str, run_id: int) -> tuple[Path | None, Path | None]:
    """Trouve le CSV comportemental et l'ASC correspondant à un run."""
    csv_path = subject_dir / f"FGAT_Dist_Run{run_id}_C4X{subject_id}.csv"

    asc_candidates = [
        subject_dir / f"{subject_id}_{run_id}.asc",
        subject_dir / f"C4X{subject_id}_{run_id}.asc",
    ]
    asc_path = next((p for p in asc_candidates if p.exists()), None)

    return (csv_path if csv_path.exists() else None), asc_path


def process_run(
    subject_dir: Path,
    subject_id: str,
    run_id: int,
) -> tuple[pd.DataFrame, dict]:
    """Traite un run sans interrompre l'analyse globale en cas d'erreur."""
    cfg = DEFAULT_CONFIG
    csv_path, asc_path = find_run_files(subject_dir, subject_id, run_id)

    qc = {
        "subject_id": subject_id,
        "run_id": run_id,
        "run_valid": False,
        "run_exclusion_reason": "",
        "run_missing_fraction": np.nan,
        "run_unique_pupil_values": np.nan,
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
        trials, timing = parse_behavior_csv(csv_path, subject_id, run_id)
        samples, events = parse_asc(asc_path)
        sync = synchronize_run(events, timing, samples, cfg)

        qc["sync_quality_flag"] = sync.quality_flag
        qc["sync_end_difference_s"] = sync.end_difference_s

        preprocessed = preprocess_pupil(samples, events, cfg)
        run_eval = evaluate_run_exclusion(preprocessed, sync, cfg)
        qc.update(run_eval)

        if not qc["run_valid"]:
            return pd.DataFrame(), qc

        windows = build_moment_windows(trials, timing, sync, cfg)
        metrics = extract_metrics(
            windows,
            preprocessed,
            max_missing_fraction=cfg.max_moment_missing_fraction,
        )

        usable_trials = metrics.loc[metrics["moment_valid"], "trial_n"].nunique()
        valid_trial_fraction = usable_trials / max(1, trials["trial_n"].nunique())
        qc["valid_trial_fraction"] = valid_trial_fraction

        if valid_trial_fraction < cfg.min_valid_trial_fraction_per_run:
            qc["run_valid"] = False
            prior = qc["run_exclusion_reason"]
            qc["run_exclusion_reason"] = ";".join(
                x for x in [prior, "too_few_usable_trials"] if x
            )
            return pd.DataFrame(), qc

        return metrics, qc

    except Exception as exc:
        qc["run_exclusion_reason"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), qc


def main(data_root: Path) -> None:
    """Exécute le pipeline complet."""
    cfg = DEFAULT_CONFIG
    output_dirs = make_output_dirs(data_root, cfg)

    subjects = discover_subjects(data_root)
    print(f"[1/5] Sujets détectés : {len(subjects)}")

    all_metrics: list[pd.DataFrame] = []
    run_qc_rows: list[dict] = []

    for subject_dir in subjects:
        subject_id = subject_dir.name.replace("C4X", "")

        # Run0 volontairement absent : entraînement systématiquement exclu.
        for run_id in range(1, 6):
            metrics, qc = process_run(subject_dir, subject_id, run_id)
            run_qc_rows.append(qc)
            if not metrics.empty:
                all_metrics.append(metrics)

    run_qc = pd.DataFrame(run_qc_rows)
    run_qc.to_csv(output_dirs["tables"] / "run_qc.csv", index=False)

    n_valid_runs = int(run_qc["run_valid"].sum())
    print(f"[2/5] Runs valides : {n_valid_runs}/{len(run_qc)}")

    subject_qc = evaluate_subject_exclusion(run_qc, cfg)
    subject_qc.to_csv(output_dirs["tables"] / "subject_qc.csv", index=False)

    kept_subjects = set(
        subject_qc.loc[subject_qc["subject_valid"], "subject_id"]
    )
    print(
        f"[3/5] Sujets retenus : {len(kept_subjects)}/{len(subject_qc)} "
        f"(minimum {cfg.min_valid_runs_per_subject} runs valides)"
    )

    if not all_metrics:
        print("[STOP] Aucune métrique exploitable.")
        return

    metrics_long = pd.concat(all_metrics, ignore_index=True)
    metrics_long = metrics_long.loc[
        metrics_long["subject_id"].isin(kept_subjects)
    ].copy()

    metrics_long.to_csv(output_dirs["tables"] / "metrics_long.csv", index=False)

    valid_by_moment = (
        metrics_long.groupby("moment")["moment_valid"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "valid_windows", "count": "all_windows"})
    )
    print("[4/5] Fenêtres valides par moment :")
    for moment, row in valid_by_moment.iterrows():
        print(f"  - {moment}: {int(row['valid_windows'])}/{int(row['all_windows'])}")

    stats_df = fit_mixed_models(metrics_long, cfg)
    stats_df.to_csv(output_dirs["tables"] / "mixed_models_fdr.csv", index=False)

    successful = stats_df.loc[stats_df["model_success"]]
    print(f"[5/5] Modèles ajustés : {len(successful)}/{len(stats_df)}")
    for _, row in successful.iterrows():
        print(
            f"  {row['scheme']} | {row['moment']} | {row['metric']} | "
            f"p_FDR={row['p_fdr']:.4g} | "
            f"effect_std={row['effect_size_std_max_contrast']:.3f}"
        )

    plot_exclusion_funnel(run_qc, subject_qc, output_dirs["figures"], cfg)

    for scheme in ["five_categories", "valence", "arousal"]:
        plot_fdr_heatmap(stats_df, scheme, output_dirs["figures"], cfg)

    # Un graphique de distribution par test effectivement ajusté.
    for _, row in successful.iterrows():
        plot_distribution(
            metrics_long=metrics_long,
            stats_df=stats_df,
            scheme=row["scheme"],
            moment=row["moment"],
            metric=row["metric"],
            output_dir=output_dirs["figures"],
            cfg=cfg,
        )

    print(f"Sorties écrites dans : {output_dirs['root']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline eyetracking pupillométrie FGAT_Dist."
    )

    parser.add_argument(
        "data_root",
        type=Path,
        help="Dossier racine contenant les dossiers C4X###.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dossier parent dans lequel créer output/. "
            "Par défaut : le dossier data_root."
        ),
    )

    args = parser.parse_args()

    output_parent = args.output_dir if args.output_dir is not None else args.data_root
    main(args.data_root, output_parent)
