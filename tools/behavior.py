import numpy as np
import pandas as pd


def parse_behavioral(filepath):
    """
    Lecture robuste d'un fichier comportemental.

    Accepte :
    - CSV virgule
    - CSV point-virgule
    - TSV
    - format texte clé: valeur
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


def _strict_filter(df, mask, label):
    """
    Filtrage strict.

    Si aucune ligne n'est trouvée, retourne un dataframe vide.
    Cela empêche d'analyser involontairement les données d'un autre
    participant ou d'un autre run.
    """
    filtered = df.loc[mask].copy()

    if filtered.empty:
        print(f"    ⚠ Aucun enregistrement après filtrage : {label}")

    return filtered


def get_trials_for_run(
    behav_all,
    participant_id,
    run_id,
    section_id=None,
):
    """
    Extrait les essais d'un participant et d'un run.

    Retourne
    --------
    trials : pd.DataFrame
        Essais valides du run.

    t0 : float
        Temps t0_fMRI en secondes, ou np.nan si introuvable.
    """

    if behav_all is None or len(behav_all) == 0:
        return pd.DataFrame(), np.nan

    df = behav_all.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ─────────────────────────────────────────────────────────
    # Participant
    # ─────────────────────────────────────────────────────────

    if "participant" in df.columns:
        participant_values = (
            df["participant"]
            .astype(str)
            .str.strip()
        )

        df = _strict_filter(
            df,
            participant_values == str(participant_id),
            label=f"participant={participant_id}",
        )

        if df.empty:
            return pd.DataFrame(), np.nan

    # ─────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────

    if "run_id" in df.columns:
        run_values = pd.to_numeric(
            df["run_id"],
            errors="coerce",
        )

        df = _strict_filter(
            df,
            run_values == float(run_id),
            label=f"run_id={run_id}",
        )

        if df.empty:
            return pd.DataFrame(), np.nan

    # ─────────────────────────────────────────────────────────
    # Section
    # ─────────────────────────────────────────────────────────

    if section_id is not None and "section_id" in df.columns:
        section_values = pd.to_numeric(
            df["section_id"],
            errors="coerce",
        )

        df = _strict_filter(
            df,
            section_values == float(section_id),
            label=f"section_id={section_id}",
        )

        if df.empty:
            return pd.DataFrame(), np.nan

    # ─────────────────────────────────────────────────────────
    # Ligne TIMING
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

    # Fallback : chercher dans l'ensemble du fichier du run.
    if not np.isfinite(t0) and "t0_fMRI" in df.columns:
        values = pd.to_numeric(
            df["t0_fMRI"],
            errors="coerce",
        ).dropna()

        if len(values):
            t0 = float(values.iloc[0])

    # ─────────────────────────────────────────────────────────
    # Essais
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
        trials = df.loc[~timing_mask].copy()

    trials.reset_index(drop=True, inplace=True)

    # ─────────────────────────────────────────────────────────
    # Nettoyage catégories
    # ─────────────────────────────────────────────────────────

    if "stim_category" in trials.columns:
        trials["stim_category"] = (
            trials["stim_category"]
            .astype(str)
            .str.strip()
            .replace({
                "": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "?": np.nan,
            })
        )

    print(
        f"    extraction run {run_id}: "
        f"{len(trials)} essais | "
        f"t0={t0}"
    )

    if "stim_category" in trials.columns:
        print(
            "    catégories :",
            trials["stim_category"]
            .value_counts(dropna=False)
            .to_dict(),
        )

    return trials, t0