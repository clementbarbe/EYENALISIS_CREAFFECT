import numpy as np
import pandas as pd

from tools.asc_parser import parse_asc
from tools.events import events_in_window
from tools.metrics import (
    baseline_correct_global,
    gaze_entropy,
    gaze_metrics,
    pupil_metrics,
)
from tools.preprocessing import (
    preprocess_gaze,
    preprocess_pupil,
)
from tools.qc import (
    compute_blink_qc,
    compute_data_loss_qc,
    compute_pupil_qc,
)


def process_one_run(
    participant_id,
    run_id,
    asc_path,
    trials_df,
    t0_fmri_behav,
    params,
    analysis_windows,
    verbose=True,
):
    """
    Traite un run complet et produit les tables H1, H2, H3, H4.

    Paramètres
    ----------
    participant_id : str
    run_id : int
    asc_path : pathlib.Path | str
    trials_df : pd.DataFrame
    t0_fmri_behav : float
        t0 fMRI dans l'unité du fichier comportemental : secondes.
    params : dict
    analysis_windows : dict
    verbose : bool
    """

    tag = f"[{participant_id} run{run_id}]"

    result = {
        "participant": participant_id,
        "run_id": run_id,

        "info": {},

        "h1": pd.DataFrame(),
        "h2": pd.DataFrame(),
        "h3": pd.DataFrame(),
        "h4": pd.DataFrame(),

        "qc": None,
        "eye_data": None,
    }

    if verbose:
        print("\n" + "=" * 65)
        print(f"{tag} DÉBUT")
        print("=" * 65)
        print(f"  Chargement ASC : {asc_path}")

    # ─────────────────────────────────────────────────────────
    # Lecture ASC
    # ─────────────────────────────────────────────────────────

    eye = parse_asc(asc_path)

    time_raw = eye["samples"]["time"]
    x_raw = eye["samples"]["x"]
    y_raw = eye["samples"]["y"]
    pupil_raw = eye["samples"]["pupil"]

    blinks_df = eye["blinks"]

    if len(time_raw) == 0:
        print(f"  ⚠ {tag} Aucun sample dans le fichier ASC.")
        return result

    duration_s = (
        time_raw[-1] - time_raw[0]
    ) / 1000.0

    result["info"]["n_samples"] = len(time_raw)
    result["info"]["duration_s"] = float(duration_s)
    result["info"]["n_blinks"] = int(len(blinks_df))

    # ─────────────────────────────────────────────────────────
    # Synchronisation
    # ─────────────────────────────────────────────────────────
    #
    # Ici, le premier trigger BUTTON 1 est conservé car il a été
    # vérifié comme correspondant au trigger fMRI.
    # ─────────────────────────────────────────────────────────

    buttons_df = eye["buttons"]

    if len(buttons_df):
        trigger_rows = buttons_df.loc[
            (buttons_df["button"] == 1)
            &
            (buttons_df["state"] == 1)
        ].copy()

        if trigger_rows.empty:
            trigger_rows = buttons_df.copy()

        t0_eyelink = int(
            trigger_rows.iloc[0]["time"]
        )

    else:
        t0_eyelink = int(time_raw[0])

        if verbose:
            print(
                "  ⚠ Aucun BUTTON : "
                "t0 EyeLink = début enregistrement."
            )

    result["info"]["t0_eyelink"] = t0_eyelink
    result["info"]["t0_fmri_behav"] = float(t0_fmri_behav)

    # Temps EyeLink en secondes relatives au trigger.
    time_rel = (
        time_raw - t0_eyelink
    ) / 1000.0

    # ─────────────────────────────────────────────────────────
    # Prétraitement
    # ─────────────────────────────────────────────────────────

    pupil_clean = preprocess_pupil(
        time=time_raw,
        pupil=pupil_raw,
        blinks_df=blinks_df,
        params=params,
    )

    x_clean, y_clean = preprocess_gaze(
        x=x_raw,
        y=y_raw,
        time=time_raw,
        blinks_df=blinks_df,
        params=params,
    )

    n_pupil_valid = int(
        np.sum(np.isfinite(pupil_clean))
    )

    pct_pupil_valid = (
        100 * n_pupil_valid / len(time_raw)
    )

    result["info"]["pct_pupil_valid"] = (
        float(pct_pupil_valid)
    )

    if verbose:
        print(
            f"  Samples : {len(time_raw):,} | "
            f"Durée : {duration_s:.1f} s | "
            f"Blinks : {len(blinks_df)}"
        )

        print(
            "  Pupille valide après nettoyage : "
            f"{pct_pupil_valid:.1f}%"
        )

    # ─────────────────────────────────────────────────────────
    # QC
    # ─────────────────────────────────────────────────────────

    blink_qc = compute_blink_qc(
        blinks_df,
        time_raw,
    )

    loss_qc = compute_data_loss_qc(
        time_raw=time_raw,
        x_raw=x_raw,
        y_raw=y_raw,
        pupil_raw=pupil_raw,
        params=params,
    )

    pupil_qc = compute_pupil_qc(
        pupil_raw=pupil_raw,
        pupil_clean=pupil_clean,
        sampling_rate=params["sampling_rate"],
    )

    result["qc"] = {
        "participant": participant_id,
        "run_id": run_id,
        "pct_pupil_valid": float(pct_pupil_valid),
        **blink_qc,
        **loss_qc,
        **pupil_qc,
    }

    # ─────────────────────────────────────────────────────────
    # Données utiles pour QC / inspection
    # ─────────────────────────────────────────────────────────

    result["eye_data"] = {
        "time_raw": time_raw,
        "time_rel": time_rel,

        "x_raw": x_raw,
        "y_raw": y_raw,
        "pupil_raw": pupil_raw,

        "x_clean": x_clean,
        "y_clean": y_clean,
        "pupil_clean": pupil_clean,

        "eye": eye,
    }

    # ─────────────────────────────────────────────────────────
    # Fonctions locales
    # ─────────────────────────────────────────────────────────

    def clean_category(value):
        """
        Les valeurs vides ne doivent jamais être exportées comme ''.
        """
        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        if value in ("", "nan", "NaN", "None", "?"):
            return np.nan

        return value

    def get_valence_arousal(category):
        if category in ["HVHA", "HVLA", "LVHA", "LVLA"]:
            valence = (
                "High"
                if category.startswith("HV")
                else "Low"
            )

            arousal = (
                "High"
                if category.endswith("HA")
                else "Low"
            )

            return valence, arousal

        if category == "MVMA":
            return "Neutral", "Neutral"

        return np.nan, np.nan

    def behav_to_eyelink(time_behav):
        """
        Temps comportemental en secondes -> temps EyeLink en ms.
        """
        return (
            t0_eyelink
            +
            (float(time_behav) - t0_fmri_behav)
            * 1000.0
        )

    def extract_window(
        start_rel,
        end_rel,
        pre_margin_s=0.0,
    ):
        """
        Extrait une fenêtre de signal en temps relatif au trigger.

        tfs = temps relatif au début de l'événement analysé.
        """
        window_start = start_rel - pre_margin_s

        mask = (
            (time_rel >= window_start)
            &
            (time_rel <= end_rel)
        )

        return {
            "mask": mask,
            "time_rel": time_rel[mask],
            "tfs": time_rel[mask] - start_rel,

            "pupil": pupil_clean[mask],
            "x": x_clean[mask],
            "y": y_clean[mask],
        }

    def get_pupil_window_metrics(
        window,
        event_time_rel,
    ):
        """
        Calcule les métriques post-événement avec correction baseline.
        """
        baseline = baseline_correct_global(
            pupil=pupil_clean,
            time_rel=time_rel,
            event_time_rel=event_time_rel,
            pre_ms=params["baseline_pre_ms"],
        )

        post_mask = window["tfs"] >= 0

        pupil_post = window["pupil"][post_mask]
        time_post = window["tfs"][post_mask]

        pupil_baseline_corrected = (
            pupil_post - baseline
        )

        metrics_bc = pupil_metrics(
            pupil_baseline_corrected,
            time_post,
        )

        metrics_raw = pupil_metrics(
            pupil_post,
            time_post,
        )

        return metrics_bc, metrics_raw, baseline

    def common_trial_columns(trial):
        return {
            "participant": participant_id,
            "run_id": run_id,

            "trial_n": trial.get(
                "trial_n",
                np.nan,
            ),

            "trial_n_global": trial.get(
                "trial_n_global",
                np.nan,
            ),

            "stim_category": clean_category(
                trial.get(
                    "stim_category",
                    np.nan,
                )
            ),

            "condition": trial.get(
                "condition",
                np.nan,
            ),
        }

    # ─────────────────────────────────────────────────────────
    # H1 — Induction émotionnelle
    # ─────────────────────────────────────────────────────────

    h1_rows = []

    h1_duration_s = float(
        analysis_windows
        .get("h1", {})
        .get("main", 10.0)
    )

    for _, trial in trials_df.iterrows():
        onset = trial.get("emo_stim_onset", np.nan)
        category = clean_category(
            trial.get("stim_category", np.nan)
        )

        if pd.isna(onset) or pd.isna(category):
            continue

        onset_rel = float(onset) - t0_fmri_behav
        offset_rel = onset_rel + h1_duration_s

        window = extract_window(
            start_rel=onset_rel,
            end_rel=offset_rel,
            pre_margin_s=(
                params["baseline_pre_ms"] / 1000.0
            ),
        )

        metrics_bc, metrics_raw, baseline = (
            get_pupil_window_metrics(
                window,
                onset_rel,
            )
        )

        post_mask = window["tfs"] >= 0

        gaze_x = window["x"][post_mask]
        gaze_y = window["y"][post_mask]

        gaze = gaze_metrics(
            gaze_x,
            gaze_y,
            duration_s=h1_duration_s,
        )

        entropy = gaze_entropy(
            gaze_x,
            gaze_y,
            params,
        )

        onset_el = behav_to_eyelink(onset)
        offset_el = onset_el + h1_duration_s * 1000.0

        saccades = events_in_window(
            eye["saccades"],
            onset_el,
            offset_el,
        )

        fixations = events_in_window(
            eye["fixations"],
            onset_el,
            offset_el,
        )

        valence, arousal = get_valence_arousal(
            category
        )

        h1_rows.append({
            **common_trial_columns(trial),

            "window_label": (
                f"h1_0_{h1_duration_s:g}s"
            ),

            "window_duration_s": h1_duration_s,

            "valence": valence,
            "arousal": arousal,

            "baseline": baseline,

            **{
                f"pup_{key}": value
                for key, value
                in metrics_bc.items()
            },

            **{
                f"pup_raw_{key}": value
                for key, value
                in metrics_raw.items()
            },

            **{
                f"gaze_{key}": value
                for key, value
                in gaze.items()
            },

            "gaze_entropy": entropy,

            "n_saccades": len(saccades),

            "sac_freq": (
                len(saccades) / h1_duration_s
                if h1_duration_s > 0
                else np.nan
            ),

            "n_fixations": len(fixations),

            "fix_freq": (
                len(fixations) / h1_duration_s
                if h1_duration_s > 0
                else np.nan
            ),

            "mean_fix_dur": (
                float(fixations["duration"].mean())
                if len(fixations)
                else np.nan
            ),
        })

    result["h1"] = pd.DataFrame(h1_rows)

    # ─────────────────────────────────────────────────────────
    # H2 — Génération d'idée
    # ─────────────────────────────────────────────────────────

    h2_rows = []

    max_ideation_duration_s = float(
        analysis_windows
        .get("h2", {})
        .get("max_ideation_duration_s", 60.0)
    )

    for _, trial in trials_df.iterrows():
        cue = trial.get("fgat_cue_onset", np.nan)
        input_time = trial.get(
            "fgat_input_time",
            np.nan,
        )

        if pd.isna(cue) or pd.isna(input_time):
            continue

        cue_rel = float(cue) - t0_fmri_behav
        input_rel = float(input_time) - t0_fmri_behav

        ideation_duration = input_rel - cue_rel

        if (
            ideation_duration <= 0
            or ideation_duration > max_ideation_duration_s
        ):
            continue

        window = extract_window(
            start_rel=cue_rel,
            end_rel=input_rel,
            pre_margin_s=(
                params["baseline_pre_ms"] / 1000.0
            ),
        )

        metrics_bc, metrics_raw, baseline = (
            get_pupil_window_metrics(
                window,
                cue_rel,
            )
        )

        post_mask = window["tfs"] >= 0

        gaze_x = window["x"][post_mask]
        gaze_y = window["y"][post_mask]

        gaze = gaze_metrics(
            gaze_x,
            gaze_y,
            duration_s=ideation_duration,
        )

        entropy = gaze_entropy(
            gaze_x,
            gaze_y,
            params,
        )

        h2_rows.append({
            **common_trial_columns(trial),

            "window_label": "h2_full_ideation",

            "ideation_dur": ideation_duration,

            "baseline": baseline,

            **{
                f"pup_{key}": value
                for key, value
                in metrics_bc.items()
            },

            **{
                f"pup_raw_{key}": value
                for key, value
                in metrics_raw.items()
            },

            **{
                f"gaze_{key}": value
                for key, value
                in gaze.items()
            },

            "gaze_entropy": entropy,
        })

    result["h2"] = pd.DataFrame(h2_rows)

    # ─────────────────────────────────────────────────────────
    # H3 — Évaluation
    # ─────────────────────────────────────────────────────────

    h3_rows = []

    for _, trial in trials_df.iterrows():
        onset = trial.get("LTask_onset", np.nan)

        if pd.isna(onset):
            continue

        onset_rel = float(onset) - t0_fmri_behav

        first_movement = trial.get(
            "LTask_first_movement",
            np.nan,
        )

        validation = trial.get(
            "LTask_validation",
            np.nan,
        )

        movement_rel = (
            float(first_movement) - t0_fmri_behav
            if pd.notna(first_movement)
            else np.nan
        )

        validation_rel = (
            float(validation) - t0_fmri_behav
            if pd.notna(validation)
            else np.nan
        )

        row = {
            **common_trial_columns(trial),

            "rating": trial.get(
                "LTask_rating",
                np.nan,
            ),
        }

        # ── Onset -> premier mouvement ──
        if (
            np.isfinite(movement_rel)
            and movement_rel > onset_rel
        ):
            duration_to_move = movement_rel - onset_rel

            window_movement = extract_window(
                start_rel=onset_rel,
                end_rel=movement_rel,
                pre_margin_s=(
                    params["baseline_pre_ms"] / 1000.0
                ),
            )

            metrics_bc, metrics_raw, baseline = (
                get_pupil_window_metrics(
                    window_movement,
                    onset_rel,
                )
            )

            row.update({
                "window_label_m": (
                    "h3_onset_to_movement"
                ),

                "dur_to_move": duration_to_move,

                "baseline_m": baseline,

                **{
                    f"pup_{key}_m": value
                    for key, value
                    in metrics_bc.items()
                },

                **{
                    f"pup_raw_{key}_m": value
                    for key, value
                    in metrics_raw.items()
                },
            })

        # ── Onset -> validation ──
        if (
            np.isfinite(validation_rel)
            and validation_rel > onset_rel
        ):
            duration_to_valid = validation_rel - onset_rel

            window_validation = extract_window(
                start_rel=onset_rel,
                end_rel=validation_rel,
                pre_margin_s=(
                    params["baseline_pre_ms"] / 1000.0
                ),
            )

            metrics_bc, metrics_raw, baseline = (
                get_pupil_window_metrics(
                    window_validation,
                    onset_rel,
                )
            )

            row.update({
                "window_label_v": (
                    "h3_onset_to_validation"
                ),

                "dur_to_valid": duration_to_valid,

                "baseline_v": baseline,

                **{
                    f"pup_{key}_v": value
                    for key, value
                    in metrics_bc.items()
                },

                **{
                    f"pup_raw_{key}_v": value
                    for key, value
                    in metrics_raw.items()
                },
            })

        h3_rows.append(row)

    result["h3"] = pd.DataFrame(h3_rows)

    # ─────────────────────────────────────────────────────────
    # H4 — Dynamique spatiale du regard
    # ─────────────────────────────────────────────────────────

    h4_rows = []

    for _, trial in trials_df.iterrows():
        cue = trial.get("fgat_cue_onset", np.nan)
        input_time = trial.get(
            "fgat_input_time",
            np.nan,
        )

        if pd.isna(cue) or pd.isna(input_time):
            continue

        cue_rel = float(cue) - t0_fmri_behav
        input_rel = float(input_time) - t0_fmri_behav

        ideation_duration = input_rel - cue_rel

        if (
            ideation_duration <= 0
            or ideation_duration > max_ideation_duration_s
        ):
            continue

        window = extract_window(
            start_rel=cue_rel,
            end_rel=input_rel,
            pre_margin_s=0.0,
        )

        gaze = gaze_metrics(
            window["x"],
            window["y"],
            duration_s=ideation_duration,
        )

        entropy = gaze_entropy(
            window["x"],
            window["y"],
            params,
        )

        cue_el = behav_to_eyelink(cue)
        input_el = behav_to_eyelink(input_time)

        fixations = events_in_window(
            eye["fixations"],
            cue_el,
            input_el,
        )

        saccades = events_in_window(
            eye["saccades"],
            cue_el,
            input_el,
        )

        h4_rows.append({
            **common_trial_columns(trial),

            "window_label": "h4_full_ideation",

            "ideation_dur": ideation_duration,

            **{
                f"gaze_{key}": value
                for key, value
                in gaze.items()
            },

            "gaze_entropy": entropy,

            "n_fix": len(fixations),
            "n_sac": len(saccades),

            "mean_fix_dur": (
                float(fixations["duration"].mean())
                if len(fixations)
                else np.nan
            ),

            "fix_rate": (
                len(fixations) / ideation_duration
                if ideation_duration > 0
                else np.nan
            ),

            "sac_rate": (
                len(saccades) / ideation_duration
                if ideation_duration > 0
                else np.nan
            ),
        })

    result["h4"] = pd.DataFrame(h4_rows)

    # ─────────────────────────────────────────────────────────
    # Résumé
    # ─────────────────────────────────────────────────────────

    if verbose:
        print(f"\n  {tag} Résumé :")

        for hypothesis in ["h1", "h2", "h3", "h4"]:
            print(
                f"    {hypothesis.upper()} : "
                f"{len(result[hypothesis])} essais"
            )

        print("=" * 65)

    return result