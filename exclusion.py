"""Règles et résumés d'exclusion."""

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
    """Détermine si un run est valide."""
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
            "run_unique_pupil_values": 0,
            "run_exclusion_reason": ";".join(reasons),
        }

    missing_fraction = float(
        preprocessed["pupil_missing_raw"].mean()
    )

    valid_pupil = preprocessed["pupil_clean"].dropna()
    unique_pupil_values = int(valid_pupil.nunique())

    if missing_fraction > cfg.max_run_missing_fraction:
        reasons.append("run_missing_fraction_above_threshold")

    if (
        valid_pupil.empty
        or unique_pupil_values < cfg.min_unique_pupil_values
    ):
        reasons.append("implausible_or_constant_pupil")

    return {
        "run_valid": len(reasons) == 0,
        "run_missing_fraction": missing_fraction,
        "run_unique_pupil_values": unique_pupil_values,
        "run_exclusion_reason": ";".join(reasons),
    }


def evaluate_subject_exclusion(
    run_qc: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Exclut les sujets avec moins de N runs valides."""
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

    excluded["subject_exclusion_reason"] = (
        excluded["subject_exclusion_reason"]
        .fillna("unknown")
        .replace("", "unknown")
    )

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


def summarize_moment_exclusions(
    metrics_long: pd.DataFrame,
) -> pd.DataFrame:
    """Compte les exclusions au niveau essai x moment."""
    keys = [
        "subject_id",
        "run_id",
        "trial_n",
        "moment",
        "moment_valid",
        "moment_exclusion_reason",
    ]

    windows = metrics_long[keys].drop_duplicates()

    excluded = windows.loc[
        ~windows["moment_valid"].fillna(False)
    ].copy()

    if excluded.empty:
        return pd.DataFrame(
            columns=["moment", "reason", "n_windows"]
        )

    excluded["moment_exclusion_reason"] = (
        excluded["moment_exclusion_reason"]
        .fillna("unknown")
        .replace("", "unknown")
    )

    return (
        excluded.groupby(
            ["moment", "moment_exclusion_reason"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "moment_exclusion_reason": "reason",
                "size": "n_windows",
            }
        )
        .sort_values(
            ["moment", "n_windows"],
            ascending=[True, False],
        )
    )