"""Lecture et préparation des CSV comportementaux FGAT_Dist."""

from __future__ import annotations

from pathlib import Path

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
    """Convertit une colonne hétérogène en booléen."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def _convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convertit les colonnes existantes en numérique."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_behavior_csv(
    csv_path: Path,
    subject_id: str,
    run_id: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge un CSV comportemental.

    Returns
    -------
    trials:
        Une ligne par essai FGAT_Dist avec catégories propagées.
    timing:
        Ligne TIMING unique du run.
    """
    raw = pd.read_csv(csv_path)

    if "task" not in raw.columns:
        raise ValueError(f"Colonne task absente dans {csv_path}")

    timing_df = raw.loc[raw["task"].astype(str) == "TIMING"].copy()
    if timing_df.empty:
        raise ValueError(f"Ligne TIMING absente dans {csv_path}")

    timing = timing_df.iloc[0]

    trials = raw.loc[raw["task"].astype(str) == "FGAT_Dist"].copy()
    if trials.empty:
        raise ValueError(f"Aucun essai FGAT_Dist dans {csv_path}")

    required_columns = [
        "trial_n",
        "has_induction",
        "stim_category",
        "emo_stim_onset",
        "fgat_cue_onset",
        "fgat_input_time",
        "LTask_onset",
        "LTask_validation",
        "timeout",
        "is_resumed",
    ]

    missing_columns = [c for c in required_columns if c not in trials.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing_columns}")

    numeric_columns = [
        "trial_n",
        "trial_n_global",
        "asso_id",
        "emo_stim_onset",
        "emo_grid_valence",
        "emo_grid_arousal",
        "fgat_cue_onset",
        "fgat_input_time",
        "LTask_onset",
        "LTask_first_movement",
        "LTask_validation",
        "LTask_rating",
    ]

    trials = _convert_numeric_columns(trials, numeric_columns)

    trials["trial_n"] = trials["trial_n"].astype("Int64")
    trials = trials.sort_values("trial_n").reset_index(drop=True)

    trials["has_induction"] = _to_bool(trials["has_induction"])
    trials["timeout"] = _to_bool(trials["timeout"])
    trials["is_resumed"] = _to_bool(trials["is_resumed"])

    # Les blocs sont 0-2, 3-5, 6-8, 9-11, 12-14.
    trials["induction_block_id"] = (trials["trial_n"] // 3).astype("Int64")

    induction_values = (
        trials.loc[trials["has_induction"]]
        .groupby("induction_block_id", dropna=False)
        [
            [
                "stim_category",
                "emo_stim_onset",
                "emo_grid_valence",
                "emo_grid_arousal",
            ]
        ]
        .first()
        .rename(
            columns={
                "stim_category": "block_stim_category",
                "emo_stim_onset": "block_emo_stim_onset",
                "emo_grid_valence": "block_emo_grid_valence",
                "emo_grid_arousal": "block_emo_grid_arousal",
            }
        )
    )

    trials = trials.merge(
        induction_values,
        on="induction_block_id",
        how="left",
        validate="many_to_one",
    )

    # Propagation aux trois essais du bloc.
    trials["stim_category"] = trials["block_stim_category"]
    trials["emo_stim_onset"] = trials["block_emo_stim_onset"]
    trials["emo_grid_valence"] = trials["block_emo_grid_valence"]
    trials["emo_grid_arousal"] = trials["block_emo_grid_arousal"]

    trials["group_5cat"] = trials["stim_category"]
    trials["group_valence"] = trials["stim_category"].map(CATEGORY_TO_VALENCE)
    trials["group_arousal"] = trials["stim_category"].map(CATEGORY_TO_AROUSAL)

    trials["subject_id"] = subject_id
    trials["run_id"] = run_id
    trials["source_csv"] = str(csv_path)

    return trials, timing