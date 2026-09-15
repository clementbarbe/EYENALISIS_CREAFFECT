"""Synchronisation comportementale ↔ EyeLink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import Config


@dataclass
class SyncResult:
    """Résultat de la synchronisation d'un run."""

    success: bool
    reason: Optional[str]
    first_button_ms: Optional[float]
    offset_ms: Optional[float]
    end_difference_s: Optional[float]
    quality_flag: str


def synchronize_run(
    events: pd.DataFrame,
    timing: pd.Series,
    samples: pd.DataFrame,
    cfg: Config,
) -> SyncResult:
    """Calcule l'offset EyeLink à partir du premier BUTTON 1 1."""
    if "t0_fMRI" not in timing.index or pd.isna(timing["t0_fMRI"]):
        return SyncResult(False, "missing_t0_fMRI", None, None, None, "failed")

    buttons = events.loc[
        (events["event_type"] == "BUTTON")
        & (events["button_code"] == cfg.sync_button_code)
        & (events["button_state"] == cfg.sync_button_state)
    ]

    if buttons.empty:
        return SyncResult(False, "missing_BUTTON_1_1", None, None, None, "failed")

    first_button_ms = float(buttons["timestamp_ms"].iloc[0])
    offset_ms = first_button_ms - float(timing["t0_fMRI"]) * 1000.0

    end_difference_s = np.nan
    quality = "unknown"

    if not samples.empty and "t_run_end" in timing.index and pd.notna(timing["t_run_end"]):
        asc_end_ms = float(samples["timestamp_ms"].max())
        expected_end_ms = first_button_ms + (
            float(timing["t_run_end"]) - float(timing["t0_fMRI"])
        ) * 1000.0
        end_difference_s = abs(asc_end_ms - expected_end_ms) / 1000.0
        quality = (
            "good"
            if end_difference_s <= cfg.max_sync_end_difference_s
            else "poor_end_consistency"
        )

    return SyncResult(
        success=True,
        reason=None,
        first_button_ms=first_button_ms,
        offset_ms=offset_ms,
        end_difference_s=end_difference_s,
        quality_flag=quality,
    )


def behavioral_to_eyelink_ms(
    behavior_time_s: float,
    sync: SyncResult,
    timing: pd.Series,
) -> float:
    """Convertit un temps comportemental relatif en timestamp EyeLink."""
    if not sync.success or sync.first_button_ms is None:
        return np.nan
    if pd.isna(behavior_time_s) or pd.isna(timing["t0_fMRI"]):
        return np.nan

    return sync.first_button_ms + (
        float(behavior_time_s) - float(timing["t0_fMRI"])
    ) * 1000.0
