import traceback

import numpy as np
import pandas as pd

from pipeline.run_processing import process_one_run
from tools.behavior import (
    get_trials_for_run,
    parse_behavioral,
)


def process_participant(
    participant_id,
    participant_info,
    params,
    analysis_windows,
    verbose=True,
):
    """
    Traite tous les runs disponibles pour un participant.
    """

    print("\n" + "#" * 70)
    print(f"# PARTICIPANT : {participant_id}")
    print(
        f"# Runs : "
        f"{list(participant_info['runs'].keys())}"
    )
    print("#" * 70)

    data_dir = participant_info["data_dir"]

    run_results = {}

    all_h1 = []
    all_h2 = []
    all_h3 = []
    all_h4 = []

    qc_rows = []

    for run_id, run_info in sorted(
        participant_info["runs"].items()
    ):
        asc_path = data_dir / run_info["asc"]

        section_id = run_info.get(
            "section_id",
            None,
        )

        if not asc_path.exists():
            print(
                f"  ⚠ ASC absent : {asc_path} "
                "→ run ignoré."
            )
            continue

        if run_id not in participant_info.get(
            "csv_per_run",
            {},
        ):
            print(
                f"  ⚠ CSV comportemental absent : "
                f"{participant_id}, run {run_id}"
            )
            continue

        csv_path = (
            data_dir
            /
            participant_info["csv_per_run"][run_id]
        )

        try:
            behavior_df = parse_behavioral(csv_path)

            trials_df, t0_fmri = get_trials_for_run(
            behav_all=behavior_df,
            participant_id=participant_id,
            run_id=run_id,
            section_id=section_id,

            # Important : chaque CSV est déjà un fichier
            # spécifique à un sujet et à un run.
            already_one_run_file=True,
)

            if trials_df.empty:
                print(
                    f"  ⚠ Aucun essai : "
                    f"{participant_id}, run {run_id}"
                )
                continue

            if not np.isfinite(t0_fmri):
                print(
                    f"  ⚠ t0_fMRI introuvable : "
                    f"{participant_id}, run {run_id}"
                )
                continue

            print(
                f"\n  Run {run_id} : "
                f"{len(trials_df)} essais | "
                f"t0_fMRI = {t0_fmri:.3f} s"
            )

            run_result = process_one_run(
                participant_id=participant_id,
                run_id=run_id,
                asc_path=asc_path,
                trials_df=trials_df,
                t0_fmri_behav=t0_fmri,
                params=params,
                analysis_windows=analysis_windows,
                verbose=verbose,
            )

            run_results[run_id] = run_result

            if run_result["qc"] is not None:
                qc_rows.append(run_result["qc"])

            if len(run_result["h1"]):
                all_h1.append(run_result["h1"])

            if len(run_result["h2"]):
                all_h2.append(run_result["h2"])

            if len(run_result["h3"]):
                all_h3.append(run_result["h3"])

            if len(run_result["h4"]):
                all_h4.append(run_result["h4"])

        except Exception as error:
            print(
                f"  ✗ ERREUR : "
                f"{participant_id}, run {run_id}"
            )
            print(f"    {error}")
            traceback.print_exc()

    concat = {
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

    qc_df = (
        pd.DataFrame(qc_rows)
        if qc_rows
        else pd.DataFrame()
    )

    print("\n" + "=" * 60)
    print(f"BILAN {participant_id}")
    print("=" * 60)

    for hypothesis in ["h1", "h2", "h3", "h4"]:
        df = concat[hypothesis]

        n_trials = len(df)

        n_runs = (
            df["run_id"].nunique()
            if n_trials and "run_id" in df.columns
            else 0
        )

        print(
            f"  {hypothesis.upper()} : "
            f"{n_trials} essais sur {n_runs} runs"
        )

    return {
        "participant": participant_id,
        "run_results": run_results,
        "concat": concat,
        "qc": qc_df,
    }