import pandas as pd

from tools.behavior import parse_behavioral, get_trials_for_run
from pipeline.run_processing import process_one_run


def process_participant(
    participant_id,
    participant_info,
    params,
    analysis_windows,
    verbose=True,
):
    run_results = {}

    h1_all = []
    h2_all = []
    h3_all = []
    h4_all = []

    for run_id, run_info in sorted(participant_info["runs"].items()):

        if run_id not in participant_info["csv_per_run"]:
            print(f"⚠ CSV manquant : {participant_id}, run {run_id}")
            continue

        asc_path = participant_info["data_dir"] / run_info["asc"]
        csv_path = (
            participant_info["data_dir"]
            / participant_info["csv_per_run"][run_id]
        )

        behavior = parse_behavioral(csv_path)

        trials, t0_fmri = get_trials_for_run(
            behavior,
            participant_id=participant_id,
            run_id=run_id,
            section_id=run_info.get("section_id"),
        )

        if trials.empty:
            print(f"⚠ Aucun essai : {participant_id}, run {run_id}")
            continue

        if pd.isna(t0_fmri):
            print(f"⚠ t0 fMRI absent : {participant_id}, run {run_id}")
            continue

        result = process_one_run(
            participant_id=participant_id,
            run_id=run_id,
            asc_path=asc_path,
            trials_df=trials,
            t0_fmri_behav=t0_fmri,
            params=params,
            analysis_windows=analysis_windows,
            verbose=verbose,
        )

        run_results[run_id] = result

        for hypothesis, target in [
            ("h1", h1_all),
            ("h2", h2_all),
            ("h3", h3_all),
            ("h4", h4_all),
        ]:
            if len(result[hypothesis]):
                target.append(result[hypothesis])

    return {
        "participant": participant_id,
        "run_results": run_results,
        "concat": {
            "h1": pd.concat(h1_all, ignore_index=True)
            if h1_all else pd.DataFrame(),

            "h2": pd.concat(h2_all, ignore_index=True)
            if h2_all else pd.DataFrame(),

            "h3": pd.concat(h3_all, ignore_index=True)
            if h3_all else pd.DataFrame(),

            "h4": pd.concat(h4_all, ignore_index=True)
            if h4_all else pd.DataFrame(),
        },
    }