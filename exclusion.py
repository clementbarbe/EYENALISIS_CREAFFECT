"""Règles et résumés d'exclusion permissifs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from sync import SyncResult


def evaluate_run_exclusion(
    preprocessed: pd.DataFrame,
    sync: SyncResult,
    cfg: Config,
) -> dict:
    """
    Exclut uniquement les runs réellement inutilisables.

    Le taux utilisé est le taux de données réellement absentes dans le fichier
    ASC : pupil_missing_acquisition.

    Les blinks, padding et outliers du pipeline ne doivent pas à eux seuls
    exclure un run entier.
    """
    reasons: list[str] = []

    if not sync.success:
        reasons.append(sync.reason or "sync_failed")

    if (
        cfg.exclude_bad_sync_end_check
        and sync.quality_flag == "poor_end_consistency"
    ):
        reasons.append("poor_sync_end_consistency")

    if preprocessed.empty:
        reasons.append("empty_samples")

        return {
            "run_valid": False,
            "run_missing_fraction": np.nan,
            "run_acquisition_missing_fraction": np.nan,
            "run_unique_pupil_values": 0,
            "run_exclusion_reason": ";".join(reasons),
        }

    missing_column = (
        "pupil_missing_acquisition"
        if "pupil_missing_acquisition" in preprocessed.columns
        else "pupil_missing_raw"
    )

    acquisition_missing_fraction = float(
        preprocessed[missing_column].mean()
    )

    clean_signal = preprocessed["pupil_clean"].dropna()
    unique_values = int(clean_signal.nunique())

    if (
        acquisition_missing_fraction
        > cfg.max_run_acquisition_missing_fraction
    ):
        reasons.append("run_acquisition_missing_above_threshold")

    if clean_signal.empty:
        reasons.append("no_usable_pupil_signal")

    elif unique_values < cfg.min_unique_pupil_values:
        reasons.append("implausible_or_constant_pupil")

    return {
        "run_valid": len(reasons) == 0,
        "run_missing_fraction": acquisition_missing_fraction,
        "run_acquisition_missing_fraction": acquisition_missing_fraction,
        "run_unique_pupil_values": unique_values,
        "run_exclusion_reason": ";".join(reasons),
    }


def evaluate_subject_exclusion(
    run_qc: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Exclut les sujets ayant moins de N runs valides."""
    subject_qc = (
        run_qc.groupby("subject_id", as_index=False)["run_valid"]
        .sum()
        .rename(columns={"run_valid": "n_valid_runs"})
    )

    subject_qc["subject_valid"] = (
        subject_qc["n_valid_runs"]
        >= cfg.min_valid_runs_per_subject
    )

    subject_qc["subject_exclusion_reason"] = np.where(
        subject_qc["subject_valid"],
        "",
        f"fewer_than_{cfg.min_valid_runs_per_subject}_valid_runs",
    )

    return subject_qc


def summarize_run_exclusions(
    run_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Compte les raisons d'exclusion des runs."""
    excluded = run_qc.loc[
        ~run_qc["run_valid"].fillna(False)
    ].copy()

    if excluded.empty:
        return pd.DataFrame(columns=["reason", "n_runs"])

    excluded["run_exclusion_reason"] = (
        excluded["run_exclusion_reason"]
        .fillna("unknown")
        .replace("", "unknown")
    )

    exploded = (
        excluded.assign(
            reason=excluded["run_exclusion_reason"].str.split(";")
        )
        .explode("reason")
    )

    return (
        exploded.groupby("reason", as_index=False)
        .size()
        .rename(columns={"size": "n_runs"})
        .sort_values("n_runs", ascending=False)
    )


def summarize_subject_exclusions(
    subject_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Compte les raisons d'exclusion des sujets."""
    excluded = subject_qc.loc[
        ~subject_qc["subject_valid"].fillna(False)
    ].copy()

    if excluded.empty:
        return pd.DataFrame(columns=["reason", "n_subjects"])

    return (
        excluded.groupby(
            "subject_exclusion_reason",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "subject_exclusion_reason": "reason",
                "size": "n_subjects",
            }
        )
        .sort_values("n_subjects", ascending=False)
    )


def summarize_moment_exclusions(metrics_long: pd.DataFrame) -> pd.DataFrame:
    """Exclusions *par métrique* (et non pseudo-compte des fenêtres uniques).

    La même fenêtre peut être valide pour gaze mais invalide pour pupillométrie.
    ROI non configurée ou microsaccades non validées apparaissent séparément.
    """
    cols = ['moment','metric','reason','n_metric_windows','n_subjects']
    if metrics_long.empty:
        return pd.DataFrame(columns=cols)
    keys = ['subject_id','run_id','trial_n','moment','metric','moment_valid',
            'moment_exclusion_reason']
    a = metrics_long.loc[~metrics_long['moment_valid'].fillna(False), keys].drop_duplicates().copy()
    if a.empty:
        return pd.DataFrame(columns=cols)
    a['reason'] = a['moment_exclusion_reason'].fillna('').replace('','unclassified').str.split(';')
    a = a.explode('reason')
    return (a.groupby(['moment','metric','reason'],as_index=False)
             .agg(n_metric_windows=('trial_n','size'),
                  n_subjects=('subject_id','nunique'))
             .sort_values(['moment','metric','n_metric_windows'],ascending=[True,True,False]))
