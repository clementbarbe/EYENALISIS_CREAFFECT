"""Règles d'exclusion aux niveaux run, moment et sujet."""

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
    """Évalue la validité globale d'un run."""
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

    missing_fraction = float(preprocessed["pupil_missing_raw"].mean())
    valid_pupil = preprocessed["pupil_clean"].dropna()
    n_unique = int(valid_pupil.nunique())

    if missing_fraction > cfg.max_run_missing_fraction:
        reasons.append("run_missing_fraction_above_threshold")

    if valid_pupil.empty or n_unique < cfg.min_unique_pupil_values:
        reasons.append("implausible_or_constant_pupil")

    return {
        "run_valid": len(reasons) == 0,
        "run_missing_fraction": missing_fraction,
        "run_unique_pupil_values": n_unique,
        "run_exclusion_reason": ";".join(reasons) if reasons else "",
    }


def evaluate_subject_exclusion(run_qc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Détermine quels sujets ont au moins le nombre requis de runs valides."""
    summary = (
        run_qc.groupby("subject_id", as_index=False)["run_valid"]
        .sum()
        .rename(columns={"run_valid": "n_valid_runs"})
    )
    summary["subject_valid"] = summary["n_valid_runs"] >= cfg.min_valid_runs_per_subject
    summary["subject_exclusion_reason"] = np.where(
        summary["subject_valid"],
        "",
        f"fewer_than_{cfg.min_valid_runs_per_subject}_valid_runs",
    )
    return summary
