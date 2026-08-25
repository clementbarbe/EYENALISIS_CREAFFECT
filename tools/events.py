import pandas as pd


def events_in_window(
    df,
    t_start,
    t_end,
    start_col="start",
    end_col="end",
):
    """
    Sélectionne les événements qui chevauchent la fenêtre temporelle.

    Un événement est conservé si :

        event_end >= window_start
        ET
        event_start <= window_end

    Cela inclut par exemple une fixation ayant commencé juste avant
    le début de la fenêtre mais se poursuivant dans cette fenêtre.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    required_columns = {start_col, end_col}

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    overlap = (
        (df[end_col] >= t_start)
        &
        (df[start_col] <= t_end)
    )

    return df.loc[overlap].copy()