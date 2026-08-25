import pandas as pd

from config import TRIAL_DIR, QC_DIR, GROUP_QC


def load_group_data(hypothesis):
    path = TRIAL_DIR / f"{hypothesis}_trials.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier absent : {path}. Lance extract-all."
        )

    return pd.read_csv(path)


def load_qc():
    path = QC_DIR / "run_qc.csv"

    if not path.exists():
        print("⚠ Aucun fichier QC trouvé : aucune exclusion QC appliquée.")
        return None

    return pd.read_csv(path)


def apply_group_qc(df, qc_df=None):
    if qc_df is None:
        return df.copy()

    good_runs = qc_df.loc[
        (qc_df["pupil_data_loss_pct"]
         <= GROUP_QC["max_pupil_data_loss_pct"])
        &
        (qc_df["gaze_data_loss_pct"]
         <= GROUP_QC["max_gaze_data_loss_pct"])
        &
        (qc_df["pct_pupil_valid"]
         >= GROUP_QC["min_pupil_valid_pct"]),
        ["participant", "run_id"],
    ].drop_duplicates()

    output = df.merge(
        good_runs,
        on=["participant", "run_id"],
        how="inner",
    )

    return output


def subject_condition_table(
    df,
    metric,
    condition_col="stim_category",
):
    summary = (
        df.groupby(
            ["participant", condition_col],
            as_index=False,
        )[metric]
        .mean()
    )

    return summary.pivot(
        index="participant",
        columns=condition_col,
        values=metric,
    )