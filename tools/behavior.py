import numpy as np
import pandas as pd


def parse_behavioral(filepath):
    """
    Lecture robuste d'un fichier comportemental.
    """

    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(filepath, sep=sep)

            if len(df.columns) > 1:
                df.columns = [
                    str(column).strip()
                    for column in df.columns
                ]
                return df

        except Exception:
            continue

    records = []
    current = {}

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:

        for line in file:
            line = line.strip()

            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            key = key.strip()
            value = value.strip()

            if key == "participant" and key in current:
                records.append(current)
                current = {}

            if value in ("", "None", "nan", "NaN"):
                current[key] = np.nan

            elif value in ("True", "False"):
                current[key] = value == "True"

            else:
                try:
                    current[key] = float(value)
                except ValueError:
                    current[key] = value

    if current:
        records.append(current)

    return pd.DataFrame(records)


def _clean_category(series):
    return (
        series.astype(str)
        .str.strip()
        .replace({
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "?": np.nan,
        })
    )


def get_trials_for_run(
    behav_all,
    participant_id=None,
    run_id=None,
    section_id=None,
    already_one_run_file=False,
):
    """
    Extrait les essais d'un run.

    Parameters
    ----------
    behav_all : pd.DataFrame
        Données comportementales.

    participant_id : str | None
        Utilisé seulement si already_one_run_file=False.

    run_id : int | None
        Utilisé seulement si already_one_run_file=False.

    section_id : int | None
        Utilisé seulement si already_one_run_file=False.

    already_one_run_file : bool
        True si le fichier est déjà spécifique à un participant et un run,
        par exemple :

            FGAT_Dist_Run0_C4X016.csv

        Dans ce cas, aucun filtre participant/run/section n'est appliqué.
    """

    if behav_all is None or len(behav_all) == 0:
        return pd.DataFrame(), np.nan

    df = behav_all.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print("\n    Diagnostic comportemental :")
    print(f"    Colonnes : {list(df.columns)}")
    print(f"    Nombre de lignes brutes : {len(df)}")

    # ─────────────────────────────────────────────────────────
    # IMPORTANT :
    # Avec un fichier déjà spécifique à un run, on ne filtre pas.
    # ─────────────────────────────────────────────────────────

    if already_one_run_file:
        print(
            "    CSV déjà spécifique au run : "
            "aucun filtrage participant/run/section."
        )

    else:
        # ─────────────────────────────────────────────────────
        # Participant
        # ─────────────────────────────────────────────────────

        if participant_id is not None and "participant" in df.columns:
            participant_values = (
                df["participant"]
                .astype(str)
                .str.strip()
            )

            mask = participant_values == str(participant_id)

            if not mask.any():
                print(
                    f"    ⚠ Aucun participant trouvé : "
                    f"{participant_id}"
                )
                print(
                    "    Valeurs participant disponibles :",
                    participant_values.unique().tolist()
                )
                return pd.DataFrame(), np.nan

            df = df.loc[mask].copy()

        # ─────────────────────────────────────────────────────
        # Run
        # ─────────────────────────────────────────────────────

        if run_id is not None and "run_id" in df.columns:
            run_values = pd.to_numeric(
                df["run_id"],
                errors="coerce",
            )

            mask = run_values == float(run_id)

            if not mask.any():
                print(
                    f"    ⚠ Aucun run trouvé : {run_id}"
                )
                print(
                    "    Valeurs run_id disponibles :",
                    sorted(
                        run_values.dropna()
                        .unique()
                        .tolist()
                    )
                )
                return pd.DataFrame(), np.nan

            df = df.loc[mask].copy()

        # ─────────────────────────────────────────────────────
        # Section
        # ─────────────────────────────────────────────────────

        if section_id is not None and "section_id" in df.columns:
            section_values = pd.to_numeric(
                df["section_id"],
                errors="coerce",
            )

            mask = section_values == float(section_id)

            if not mask.any():
                print(
                    f"    ⚠ Aucune section trouvée : "
                    f"{section_id}"
                )
                print(
                    "    Valeurs section_id disponibles :",
                    sorted(
                        section_values.dropna()
                        .unique()
                        .tolist()
                    )
                )
                return pd.DataFrame(), np.nan

            df = df.loc[mask].copy()

    # ─────────────────────────────────────────────────────────
    # Détection des lignes TIMING
    # ─────────────────────────────────────────────────────────

    if "task" in df.columns:
        timing_mask = (
            df["task"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("TIMING")
        )
    else:
        timing_mask = pd.Series(
            False,
            index=df.index,
        )

    timing = df.loc[timing_mask].copy()

    # ─────────────────────────────────────────────────────────
    # t0 fMRI
    # ─────────────────────────────────────────────────────────

    t0 = np.nan

    if "t0_fMRI" in timing.columns:
        values = pd.to_numeric(
            timing["t0_fMRI"],
            errors="coerce",
        ).dropna()

        if len(values):
            t0 = float(values.iloc[0])

    # Fallback sur toutes les lignes du CSV.
    if not np.isfinite(t0) and "t0_fMRI" in df.columns:
        values = pd.to_numeric(
            df["t0_fMRI"],
            errors="coerce",
        ).dropna()

        if len(values):
            t0 = float(values.iloc[0])

    # ─────────────────────────────────────────────────────────
    # Extraction essais
    # ─────────────────────────────────────────────────────────

    if "trial_n" in df.columns:
        trial_values = pd.to_numeric(
            df["trial_n"],
            errors="coerce",
        )

        trial_mask = (
            ~timing_mask
            &
            trial_values.notna()
        )

        trials = df.loc[trial_mask].copy()

    else:
        # Si trial_n est absent, toutes les lignes non-TIMING sont
        # considérées comme essais potentiels.
        trials = df.loc[~timing_mask].copy()

    trials.reset_index(drop=True, inplace=True)

    # ─────────────────────────────────────────────────────────
    # Nettoyage catégorie émotionnelle
    # ─────────────────────────────────────────────────────────

    if "stim_category" in trials.columns:
        trials["stim_category"] = _clean_category(
            trials["stim_category"]
        )

    print(
        f"    Extraction : {len(trials)} essais | "
        f"t0={t0}"
    )

    if "stim_category" in trials.columns:
        print(
            "    Catégories :",
            trials["stim_category"]
            .value_counts(dropna=False)
            .to_dict(),
        )

    return trials, t0