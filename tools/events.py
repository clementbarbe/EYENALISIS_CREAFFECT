import pandas as pd


def events_in_window(
    df,
    t_start,
    t_end,
    start_col="start",
    end_col="end",
):
    """
    Retourne les événements qui chevauchent la fenêtre temporelle.

    Un événement est retenu s'il commence avant la fin de la fenêtre
    ET se termine après le début de la fenêtre.

    Exemple :
        fenêtre : [1000, 2000]
        fixation : [900, 1200]

    Cette fixation est retenue car elle chevauche la fenêtre.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    required_columns = {start_col, end_col}

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    overlap_mask = (
        (df[end_col] >= t_start)
        &
        (df[start_col] <= t_end)
    )

    return df.loc[overlap_mask].copy()