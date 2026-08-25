from pathlib import Path
import pandas as pd

from config import TRIAL_DIR, SUBJECT_DIR


def load_hypothesis_trials(hypothesis):
    path = TRIAL_DIR / f"{hypothesis}_trials.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Lance d'abord l'extraction."
        )

    return pd.read_csv(path)


def get_subject_data(hypothesis, participant_id):
    df = load_hypothesis_trials(hypothesis)

    df = df.loc[
        df["participant"] == participant_id
    ].copy()

    if df.empty:
        raise ValueError(
            f"Aucune donnée pour {participant_id}, hypothèse {hypothesis}."
        )

    return df


def subject_output_dir(participant_id, hypothesis):
    directory = SUBJECT_DIR / participant_id / hypothesis
    directory.mkdir(parents=True, exist_ok=True)
    return directory