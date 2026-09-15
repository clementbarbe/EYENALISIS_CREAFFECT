"""Lecture et normalisation des CSV comportementaux FGAT_Dist."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CATEGORY_TO_VALENCE = {
    "LVLA": "Negative",
    "LVHA": "Negative",
    "MVMA": "Neutral",
    "HVLA": "Positive",
    "HVHA": "Positive",
}

CATEGORY_TO_AROUSAL = {
    "LVLA": "Low",
    "HVLA": "Low",
    "MVMA": "Neutral",
    "LVHA": "High",
    "HVHA": "High",
}


def _to_bool(series: pd.Series) -> pd.Series:
    """Convertit robustement une colonne textuelle/booléenne en bool."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def _as_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convertit des colonnes en numérique si elles existent."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_behavior_csv(
    csv_path: Path,
    subject_id: str,
    run_id: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Lit un CSV comportemental.

    Returns
    -------
    trials:
        Une ligne par essai FGAT_Dist, avec catégories propagées par bloc.
    timing:
        Ligne TIMING unique utilisée pour synchronisation et QC.
    """
    raw = pd.read_csv(csv_path)

    if "task" not in raw.columns:
        raise ValueError(f"Colonne 'task' absente : {csv_path}")

    timing_df = raw.loc[raw["task"].astype(str) == "TIMING"].copy()
    if timing_df.empty:
        raise ValueError(f"Aucune ligne TIMING : {csv_path}")
    timing = timing_df.iloc[0]

    trials = raw.loc[raw["task"].astype(str) == "FGAT_Dist"].copy()
    if trials.empty:
        raise ValueError(f"Aucun essai FGAT_Dist : {csv_path}")

    required = [
        "trial_n", "has_induction", "stim_category",
        "emo_stim_onset", "fgat_cue_onset", "fgat_input_time",
        "LTask_onset", "LTask_validation", "timeout", "is_resumed",
    ]
    missing = [c for c in required if c not in trials.columns]
    if missing:
        raise ValueError(f"Colonnes comportementales manquantes : {missing}")

    numeric_cols = [
        "trial_n", "trial_n_global", "asso_id", "emo_stim_onset",
        "emo_grid_valence", "emo_grid_arousal", "fgat_cue_onset",
        "fgat_input_time", "LTask_onset", "LTask_first_movement",
        "LTask_validation", "LTask_rating",
    ]
    trials = _as_numeric(trials, numeric_cols)
    trials["trial_n"] = trials["trial_n"].astype("Int64")
    trials = trials.sort_values("trial_n").reset_index(drop=True)

    trials["has_induction"] = _to_bool(trials["has_induction"])
    trials["timeout"] = _to_bool(trials["timeout"])
    trials["is_resumed"] = _to_bool(trials["is_resumed"])

    # Regroupement explicite par blocs de trois essais.
    trials["induction_block_id"] = (trials["trial_n"] // 3).astype("Int64")

    block_values = (
        trials.loc[trials["has_induction"]]
        .groupby("induction_block_id", dropna=False)
        [["stim_category", "emo_grid_valence", "emo_grid_arousal", "emo_stim_onset"]]
        .first()
        .rename(columns={
            "stim_category": "block_stim_category",
            "emo_grid_valence": "block_emo_grid_valence",
            "emo_grid_arousal": "block_emo_grid_arousal",
            "emo_stim_onset": "block_emo_stim_onset",
        })
    )

    trials = trials.merge(
        block_values,
        on="induction_block_id",
        how="left",
        validate="many_to_one",
    )

    trials["stim_category"] = trials["block_stim_category"]
    trials["emo_grid_valence"] = trials["block_emo_grid_valence"]
    trials["emo_grid_arousal"] = trials["block_emo_grid_arousal"]
    trials["emo_stim_onset"] = trials["block_emo_stim_onset"]

    trials["group_5cat"] = trials["stim_category"]
    trials["group_valence"] = trials["stim_category"].map(CATEGORY_TO_VALENCE)
    trials["group_arousal"] = trials["stim_category"].map(CATEGORY_TO_AROUSAL)

    trials["subject_id"] = subject_id
    trials["run_id"] = run_id
    trials["source_csv"] = str(csv_path)

    return trials, timing
