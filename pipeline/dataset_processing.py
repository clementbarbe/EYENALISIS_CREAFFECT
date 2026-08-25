import pandas as pd

from config import TRIAL_DIR
from pipeline.participant_processing import process_participant


def process_dataset(
    participants,
    params,
    analysis_windows,
    selected_participants=None,
    verbose=True,
):
    if selected_participants is None:
        selected_participants = list(participants.keys())

    all_h1 = []
    all_h2 = []
    all_h3 = []
    all_h4 = []

    all_results = {}

    for participant_id in selected_participants:

        if participant_id not in participants:
            print(f"⚠ Participant inconnu : {participant_id}")
            continue

        print(f"\n{'=' * 70}")
        print(f"TRAITEMENT : {participant_id}")
        print(f"{'=' * 70}")

        result = process_participant(
            participant_id=participant_id,
            participant_info=participants[participant_id],
            params=params,
            analysis_windows=analysis_windows,
            verbose=verbose,
        )

        all_results[participant_id] = result

        for hypothesis, collector in [
            ("h1", all_h1),
            ("h2", all_h2),
            ("h3", all_h3),
            ("h4", all_h4),
        ]:
            df = result["concat"][hypothesis]

            if len(df):
                collector.append(df)

    exports = {
        "h1": pd.concat(all_h1, ignore_index=True)
        if all_h1 else pd.DataFrame(),

        "h2": pd.concat(all_h2, ignore_index=True)
        if all_h2 else pd.DataFrame(),

        "h3": pd.concat(all_h3, ignore_index=True)
        if all_h3 else pd.DataFrame(),

        "h4": pd.concat(all_h4, ignore_index=True)
        if all_h4 else pd.DataFrame(),
    }

    for hypothesis, df in exports.items():
        output_path = TRIAL_DIR / f"{hypothesis}_trials.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Export : {output_path} ({len(df)} lignes)")

    return all_results