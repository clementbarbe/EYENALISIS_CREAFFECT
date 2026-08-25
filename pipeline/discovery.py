from pathlib import Path


def discover_participants(data_dir: Path):
    """
    Détecte automatiquement :
    - les dossiers participants ;
    - les fichiers ASC ;
    - les CSV comportementaux par run.

    Convention attendue :
        data/C4X016/016_0.asc
        data/C4X016/FGAT_Dist_Run0_C4X016.csv
    """

    participants = {}

    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        participant_id = subject_dir.name
        subject_number = participant_id[-3:]

        runs = {}
        csv_per_run = {}

        for run_index in range(6):
            run_id = run_index + 1

            asc_filename = (
                f"{subject_number}_{run_index}.asc"
            )

            asc_path = subject_dir / asc_filename

            if asc_path.exists():
                runs[run_id] = {
                    "asc": asc_filename,

                    # Garde la convention de ton notebook initial.
                    "section_id": run_index + 3,
                }

            csv_filename = (
                f"FGAT_Dist_Run{run_index}_"
                f"{participant_id}.csv"
            )

            csv_path = subject_dir / csv_filename

            if csv_path.exists():
                csv_per_run[run_id] = csv_filename

        participants[participant_id] = {
            "data_dir": subject_dir,
            "runs": runs,
            "csv_per_run": csv_per_run,
        }

    return participants