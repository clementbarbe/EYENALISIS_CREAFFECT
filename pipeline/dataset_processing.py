import pandas as pd

from config import QC_DIR, TRIAL_DIR
from pipeline.participant_processing import (
    process_participant,
)


def _update_global_csv(
    new_df,
    output_path,
    updated_participants,
):
    """
    Met à jour un CSV global sans supprimer les autres participants.

    - Si le fichier n'existe pas : création.
    - Si le fichier existe : suppression des anciennes lignes des
      participants retraités, puis ajout de leurs nouvelles données.

    Cela permet d'utiliser :

        python cli.py extract-subject --participant C4X016

    sans écraser les résultats de tous les autres participants.
    """

    if output_path.exists():
        old_df = pd.read_csv(output_path)

        if (
            "participant" in old_df.columns
            and updated_participants
        ):
            old_df = old_df.loc[
                ~old_df["participant"].isin(
                    updated_participants
                )
            ].copy()

        output_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
            sort=False,
        )

    else:
        output_df = new_df.copy()

    output_df.to_csv(
        output_path,
        index=False,
    )

    return output_df


def process_dataset(
    participants,
    params,
    analysis_windows,
    selected_participants=None,
    verbose=True,
):
    """
    Traite tout le dataset ou une liste de participants.

    Les CSV globaux trial-level et QC sont mis à jour automatiquement.
    """

    if selected_participants is None:
        selected_participants = sorted(
            participants.keys()
        )

    results = {}

    all_h1 = []
    all_h2 = []
    all_h3 = []
    all_h4 = []
    all_qc = []

    processed_participants = []

    for participant_id in selected_participants:
        if participant_id not in participants:
            print(
                f"⚠ Participant inconnu : "
                f"{participant_id}"
            )
            continue

        print("\n" + "=" * 70)
        print(f"TRAITEMENT DATASET : {participant_id}")
        print("=" * 70)

        participant_result = process_participant(
            participant_id=participant_id,
            participant_info=participants[participant_id],
            params=params,
            analysis_windows=analysis_windows,
            verbose=verbose,
        )

        results[participant_id] = participant_result
        processed_participants.append(participant_id)

        concat = participant_result["concat"]

        if len(concat["h1"]):
            all_h1.append(concat["h1"])

        if len(concat["h2"]):
            all_h2.append(concat["h2"])

        if len(concat["h3"]):
            all_h3.append(concat["h3"])

        if len(concat["h4"]):
            all_h4.append(concat["h4"])

        if len(participant_result["qc"]):
            all_qc.append(participant_result["qc"])

    exports = {
        "h1": (
            pd.concat(all_h1, ignore_index=True)
            if all_h1
            else pd.DataFrame()
        ),

        "h2": (
            pd.concat(all_h2, ignore_index=True)
            if all_h2
            else pd.DataFrame()
        ),

        "h3": (
            pd.concat(all_h3, ignore_index=True)
            if all_h3
            else pd.DataFrame()
        ),

        "h4": (
            pd.concat(all_h4, ignore_index=True)
            if all_h4
            else pd.DataFrame()
        ),
    }

    # ─────────────────────────────────────────────────────────
    # Exports trial-level
    # ─────────────────────────────────────────────────────────

    for hypothesis, new_df in exports.items():
        if new_df.empty:
            continue

        output_path = TRIAL_DIR / f"{hypothesis}_trials.csv"

        updated_df = _update_global_csv(
            new_df=new_df,
            output_path=output_path,
            updated_participants=processed_participants,
        )

        print(
            f"✓ Export : {output_path} "
            f"({len(updated_df)} lignes au total)"
        )

    # ─────────────────────────────────────────────────────────
    # Export QC
    # ─────────────────────────────────────────────────────────

    if all_qc:
        qc_new_df = pd.concat(
            all_qc,
            ignore_index=True,
        )

        qc_path = QC_DIR / "run_qc.csv"

        qc_updated_df = _update_global_csv(
            new_df=qc_new_df,
            output_path=qc_path,
            updated_participants=processed_participants,
        )

        print(
            f"✓ Export QC : {qc_path} "
            f"({len(qc_updated_df)} lignes au total)"
        )

    return results