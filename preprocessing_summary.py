"""Durées mesurées après prétraitement et bilans descriptifs de cohorte.

Les temps portent sur tout l'ASC, avant les exclusions de fenêtres/analyses.
Un sample couvre au plus 1/fs seconde, sans combler les trous d'acquisition.
Les catégories de la pupille forment une partition ; gaze et blinks se recouvrent.
"""
from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


DURATION_COLUMNS = (
    "recording_span_s", "acquired_duration_s", "unrecorded_duration_s",
    "pupil_observed_duration_s", "pupil_interpolated_duration_s",
    "pupil_unusable_recorded_duration_s", "pupil_available_duration_s",
    "gaze_valid_duration_s", "raw_pupil_missing_duration_s", "blink_mask_duration_s",
)
PARTITION_COLUMNS = ("pupil_observed_duration_s", "pupil_interpolated_duration_s",
                     "pupil_unusable_recorded_duration_s", "unrecorded_duration_s")
LABELS = {
    "recording_span_s": "Durée totale ASC",
    "acquired_duration_s": "Temps couvert par l'acquisition",
    "unrecorded_duration_s": "Trous sans échantillon",
    "pupil_observed_duration_s": "Pupille observée conservée",
    "pupil_interpolated_duration_s": "Pupille reconstruite",
    "pupil_unusable_recorded_duration_s": "Pupille inutilisable malgré acquisition",
    "pupil_available_duration_s": "Pupille disponible après nettoyage",
    "gaze_valid_duration_s": "Regard valide",
    "raw_pupil_missing_duration_s": "Pupille absente/invalide à l'acquisition",
    "blink_mask_duration_s": "Samples couverts par EBLINK, sans padding",
}
REFERENCES = [
    ("SR Research — format ASC et événements EyeLink", "https://www.sr-research.com/download/dispdoc/page25.html"),
    ("Kret et Sjak-Shie (2019) — artefacts et vitesse robuste", "https://doi.org/10.3758/s13428-018-1075-y"),
    ("Mathôt et al. (2018) — données invalides et baseline", "https://doi.org/10.3758/s13428-017-1007-2"),
    ("Winn et al. (2018) — blinks, interpolation et filtrage", "https://doi.org/10.1177/2331216518800869"),
    ("Mathôt et Vilotijević (2023) — workflow et QC", "https://doi.org/10.3758/s13428-022-01957-7"),
    ("Fink et al. (2024) — prétraitement selon l'analyse", "https://doi.org/10.3758/s13428-023-02098-1"),
]


def summarize_recording(samples, cfg, events=None):
    """Faits de durée sur un run prétraité, même s'il sera ensuite exclu."""
    metadata = samples.attrs.get("eyelink_metadata", {})
    fs = metadata.get("sampling_rate_hz") or cfg.sampling_rate_hz
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("A positive sampling frequency is required for durations")
    result = {key: 0.0 for key in DURATION_COLUMNS}
    result.update(preprocessing_completed=True, duration_scope="full_asc_recording",
                  sampling_rate_hz=float(fs), n_samples=len(samples),
                  recorded_eye=metadata.get('selected_eye', cfg.eye),
                  screen_width_px=cfg.gaze_screen_width_px, screen_height_px=cfg.gaze_screen_height_px,
                  first_sample_ms=np.nan, last_sample_ms=np.nan,
                  marker_run_duration_s=np.nan, n_acquisition_segments=0,
                  longest_acquisition_gap_s=0.0)
    if samples.empty:
        return result
    time = samples.timestamp_ms.to_numpy(dtype=float)
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError("Duration summary requires increasing finite timestamps")
    dt_ms = 1000.0 / fs
    # Supports non recouvrants : [t_i, min(t_i + dt, t_suivant)[.
    weights = np.r_[np.minimum(np.diff(time), dt_ms), dt_ms] / 1000.0
    span = float((time[-1] - time[0] + dt_ms) / 1000.0)
    finite = np.isfinite(samples.pupil_clean.to_numpy(dtype=float))
    observed = samples.pupil_observed_valid.to_numpy(dtype=bool) & finite
    interpolated = samples.pupil_interpolated.to_numpy(dtype=bool) & finite
    if (observed & interpolated).any():
        raise ValueError("Observed and interpolated pupil masks must be disjoint")
    unusable = ~(observed | interpolated)
    acquired = float(weights.sum())
    result.update(
        recording_span_s=span, acquired_duration_s=acquired,
        unrecorded_duration_s=max(0.0, span - acquired),
        pupil_observed_duration_s=float(weights[observed].sum()),
        pupil_interpolated_duration_s=float(weights[interpolated].sum()),
        pupil_unusable_recorded_duration_s=float(weights[unusable].sum()),
        pupil_available_duration_s=float(weights[observed | interpolated].sum()),
        gaze_valid_duration_s=float(weights[samples.gaze_valid.to_numpy(dtype=bool)].sum()),
        raw_pupil_missing_duration_s=float(weights[samples.pupil_missing_acquisition.to_numpy(dtype=bool)].sum()),
        blink_mask_duration_s=float(weights[samples.event_blink_mask.to_numpy(dtype=bool)].sum()),
        first_sample_ms=float(time[0]), last_sample_ms=float(time[-1]),
        n_acquisition_segments=int(samples.segment_id.nunique()) if "segment_id" in samples else 1,
        longest_acquisition_gap_s=float(max(0.0, np.max(np.diff(time) - dt_ms))) if len(time) > 1 else 0.0,
    )
    if events is not None and not events.empty:
        messages = events.loc[events.event_type.eq("MSG")]
        markers = []
        for name in ("RUN_START", "RUN_END"):
            match = messages.raw_line.str.extract(rf"\b{name}\s+(\d+)\b", expand=False)
            selected = messages.loc[match.notna()]
            markers.append((selected, match.loc[selected.index]))
        (start, start_ids), (end, end_ids) = markers
        if (len(start) == len(end) == 1 and start_ids.iloc[0] == end_ids.iloc[0]
                and end.timestamp_ms.iloc[0] > start.timestamp_ms.iloc[0]):
            result["marker_run_duration_s"] = float((end.timestamp_ms.iloc[0] - start.timestamp_ms.iloc[0]) / 1000.0)
    return result


def _boolean(values):
    """Booléen nullable : une valeur absente n'est ni une exclusion ni un succès."""
    mapping = {"true": True, "1": True, "false": False, "0": False}
    return values.map(lambda x: mapping.get(str(x).strip().lower(), pd.NA)).astype("boolean")


def normalize_runs(runs, *, profile="creaffect", subject_qc=None):
    runs = runs.copy()
    if "preprocessing_completed" not in runs or "recording_span_s" not in runs:
        raise ValueError("Les durées de prétraitement manquent : relancer run_pipeline.py avec --preprocessing-only dans un nouveau dossier.")
    if "participant_id" not in runs:
        runs["participant_id"] = runs["subject_id"].astype(str)
        if profile == "creaffect":
            runs["participant_id"] = "C4X" + runs.participant_id.str.zfill(3)
    if "session_id" not in runs:
        runs["session_id"] = "1"
    for key in ("participant_id", "session_id", "run_id"):
        runs[key] = runs[key].astype(str)
    if runs.duplicated(["participant_id", "session_id", "run_id"]).any():
        raise ValueError("Plusieurs lignes pour une même acquisition dans le bilan")
    runs["preprocessing_completed"] = _boolean(runs.preprocessing_completed).fillna(False)
    for key in DURATION_COLUMNS:
        runs[key] = pd.to_numeric(runs.get(key, pd.Series(np.nan, index=runs.index)), errors="coerce")
        # Un fichier indisponible ne devient jamais un run de zéro seconde.
        runs.loc[~runs.preprocessing_completed, key] = np.nan
    run_flag = next((key for key in ("run_qc_passed", "run_valid") if key in runs), None)
    runs["run_eligibility"] = "not_evaluated"
    if run_flag is not None:
        flag = _boolean(runs[run_flag])
        runs.loc[flag.eq(True).fillna(False), "run_eligibility"] = "retained"
        runs.loc[flag.eq(False).fillna(False), "run_eligibility"] = "excluded"
    runs["subject_eligibility"] = "not_evaluated"
    if subject_qc is not None and not subject_qc.empty:
        subjects = subject_qc.copy()
        if "participant_id" not in subjects:
            subjects["participant_id"] = subjects.subject_id.astype(str)
            if profile == "creaffect":
                subjects["participant_id"] = "C4X" + subjects.participant_id.str.zfill(3)
        if subjects.participant_id.duplicated().any():
            raise ValueError("Duplicate participant in subject QC")
        flags = pd.Series(_boolean(subjects.subject_valid).to_numpy(), index=subjects.participant_id)
        flags = _boolean(runs.participant_id.map(flags))
        runs.loc[flags.eq(True).fillna(False), "subject_eligibility"] = "retained"
        runs.loc[flags.eq(False).fillna(False), "subject_eligibility"] = "excluded"
    for key in ("run_exclusion_reason", "error"):
        if key not in runs:
            runs[key] = ""
        runs[key] = runs[key].fillna("")
    for key in DURATION_COLUMNS:
        runs[key.removesuffix("_s") + "_min"] = runs[key] / 60.0
    runs["pupil_observed_fraction_span"] = runs.pupil_observed_duration_s / runs.recording_span_s.replace(0, np.nan)
    runs["sampling_coverage_fraction"] = (runs.acquired_duration_s / runs.recording_span_s.replace(0, np.nan)).clip(0, 1)
    return runs


def _aggregate(group):
    measured = group.loc[group.preprocessing_completed]
    values = measured.recording_span_s.dropna()
    row = dict(n_runs_expected=len(group), n_runs_preprocessed=len(measured),
               n_runs_without_duration=int(group.recording_span_s.isna().sum()),
               n_participants=int(group.participant_id.nunique()),
               n_runs_retained=int(group.run_eligibility.eq("retained").sum()),
               n_runs_excluded=int(group.run_eligibility.eq("excluded").sum()),
               n_runs_eligibility_unknown=int(group.run_eligibility.eq("not_evaluated").sum()))
    for key in DURATION_COLUMNS:
        row["total_" + key] = measured[key].sum(min_count=1)
        row["total_" + key.removesuffix('_s') + '_min'] = row["total_" + key] / 60.0
    for stat, value in (("mean", values.mean()), ("sd", values.std(ddof=1)),
                        ("median", values.median()), ("q25", values.quantile(.25)),
                        ("q75", values.quantile(.75)), ("min", values.min()), ("max", values.max())):
        row[f"run_duration_{stat}_s"] = value
        row[f"run_duration_{stat}_min"] = value / 60.0
    total = row["total_recording_span_s"]
    row["pupil_observed_fraction_span"] = row["total_pupil_observed_duration_s"] / total if total > 0 else np.nan
    row["sampling_coverage_fraction"] = row["total_acquired_duration_s"] / total if total > 0 else np.nan
    return row


def summary_tables(runs):
    scopes = {"all_expected_runs": runs,
              "retained_runs": runs.loc[runs.run_eligibility.eq("retained")],
              "retained_subjects_and_runs": runs.loc[runs.run_eligibility.eq("retained")
                                                       & runs.subject_eligibility.eq("retained")]}
    applied = {'all_expected_runs': True,
               'retained_runs': bool(runs.run_eligibility.ne('not_evaluated').any()),
               'retained_subjects_and_runs': bool(runs.subject_eligibility.ne('not_evaluated').any())}
    cohort = pd.DataFrame([{"scope": name, 'selection_evaluated': applied[name], **_aggregate(frame)}
                           for name, frame in scopes.items()])
    participants = pd.DataFrame([{ "participant_id": key, "subject_eligibility": group.subject_eligibility.iloc[0],
                                  **_aggregate(group)} for key, group in runs.groupby("participant_id", sort=True)])
    by_run = pd.DataFrame([{ "session_id": session, "run_id": run, **_aggregate(group)}
                          for (session, run), group in runs.groupby(["session_id", "run_id"], sort=True)])
    return cohort, participants, by_run


def preprocessing_steps(parameters):
    """Table pédagogique générée à partir des paramètres effectivement utilisés."""
    p = parameters
    return pd.DataFrame([
        ("1. Lire et vérifier", "Choisir l'œil, lire timestamps, unités et métadonnées ; repérer les segments.",
         f"Œil demandé : {p['eye']} ; fréquence de référence : {p['sampling_rate_hz']:g} Hz.", "SR Research ; Mathôt et Vilotijević 2023"),
        ("2. Marquer la pupille absente", "Une valeur absente ou impossible devient inutilisable, jamais une petite pupille réelle.",
         f"Non finie ou ≤ {p['min_plausible_pupil']:g} unité EyeLink.", "Kret et Sjak-Shie 2019"),
        ("3. Repérer les artefacts", "Détecter les changements trop rapides et les valeurs extrêmes ; conserver un masque explicite.",
         f"Vitesse : médiane + {p['pupil_speed_mad_threshold']:g} × 1,4826 × MAD ; rejet global activé : {p['pupil_reject_speed_outliers']} ; amplitude : {p['pupil_outlier_mad_threshold']:g} MAD normalisées.",
         "Vitesse inspirée de Kret et Sjak-Shie 2019 ; contrôle global d'amplitude propre au code"),
        ("4. Écarter les bords des pertes", "La paupière peut déformer la mesure avant et après un blink. Le padding est du nettoyage, pas du temps de blink.",
         f"Padding : {p['blink_padding_before_ms']} ms avant / {p['blink_padding_after_ms']} ms après blinks, pertes et vitesses rejetées.", "Kret et Sjak-Shie 2019 ; Winn et al. 2018"),
        ("5. Reconstruire les petits trous", "Relier linéairement deux points valides seulement si le trou autorisé est court ; conserver le masque interpolé.",
         f"Plafond après padding : {p['interpolation_max_gap_ms']} ms ; jamais entre segments ni à travers une interruption d'acquisition.", "Winn et al. 2018 ; Fink et al. 2024"),
        ("6. Lisser la pupille", "Atténuer les fluctuations rapides ; le filtrage ne remplit pas les trous restants.",
         f"Butterworth ordre {p['lowpass_order']}, {p['lowpass_cutoff_hz']:g} Hz, zéro phase ; activé : {p['lowpass_enabled']} ; sous-segments ≥40 samples seulement.", "Winn et al. 2018 ; Fink et al. 2024"),
        ("7. Contrôler le regard séparément", "Garder les positions plausibles sur l'écran, hors EBLINK et pertes de pupille ; aucun filtre pupillaire sur le gaze.",
         f"Écran de référence : {p['gaze_screen_width_px']:g} × {p['gaze_screen_height_px']:g} px.", "Choix opérationnel du pipeline"),
        ("8. Évaluer les données disponibles", "Mesurer les durées et pertes avant d'analyser les événements ; inspecter visuellement les traces.",
         "Les runs indisponibles restent recensés ; les interpolations sont séparées des mesures observées.", "Mathôt et Vilotijević 2023"),
    ], columns=["step", "explanation", "effective_setting", "methodological_basis"])


def exclusion_criteria(parameters, profile):
    p = parameters
    rows = []
    if profile == "creaffect":
        rows.extend([
            ("Run", "Fichiers ou marqueurs non appariables, synchronisation échouée", "Exclut le run", "Règle technique du projet"),
            ("Run", f"Écart absolu de fin > {p['max_sync_end_difference_s']:g} s", "Exclut le run ; ne valide pas à lui seul l'alignement des événements", "Seuil du projet"),
            ("Run", f"Pupille absente/invalide dans les samples acquis > {100*p['max_run_acquisition_missing_fraction']:g} %", "Exclut le run ; inclut certaines pertes pendant les blinks, sans padding", "Seuil du projet"),
            ("Run", f"Aucune pupille nettoyée finie, ou < {p['min_unique_pupil_values']} valeurs uniques après nettoyage", "Exclut le run", "Heuristique technique du projet, pas un test physiologique"),
            ("Participant", f"Moins de {p['min_valid_runs_per_subject']} runs retenus", "Exclut le participant du profil CREAFFECT", "Choix du projet ; pas une norme universelle"),
        ])
    else:
        rows.append(("Acquisition", "Erreur d'import ou de structure", "Échec déclaré ; pas d'exclusion globale par taux de pupille ou nombre de runs", "Comportement du mode projet"))
    rows.extend([
        ("Fenêtre — étape suivante", f"Durée < {p['minimum_usable_window_s']:g} s par défaut, timeout ou reprise", "Invalide la fenêtre ; seuil de durée surchargeable en mode projet", "Choix analytique"),
        ("Fenêtre — étape suivante", f"Couverture temporelle < {100*(1-p['max_window_missing_sampling_fraction']):g} %", "Invalide la fenêtre", "Choix QC du projet"),
        ("Pupille — étape suivante", f"Observé < {100*p['min_observed_fraction_window']:g} %, interpolé > {100*p['max_interpolated_fraction_window']:g} %, ou trou > {p['max_longest_gap_window_ms']:g} ms", "Invalide les métriques pupillaires de la fenêtre", "Choix QC du projet"),
        ("Baseline — étape suivante", f"Durée < {p['minimum_baseline_duration_s']:g} s, observé < {100*p['min_observed_fraction_baseline']:g} %, absence d'acquisition > {100*p['max_baseline_missing_fraction']:g} %, ou couverture < {100*(1-p['max_window_missing_sampling_fraction']):g} %", "Invalide les métriques dépendant de la baseline", "Principe : Mathôt et al. 2018 ; seuils du projet"),
        ("Gaze — étape suivante", f"Regard valide < {100*p['min_gaze_valid_fraction']:g} %", "Invalide les métriques gaze/fixation/saccade concernées", "Choix QC du projet"),
    ])
    return pd.DataFrame(rows, columns=["level", "criterion", "consequence", "status"])


def _plots(runs, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    known = runs.loc[runs.preprocessing_completed & runs.recording_span_s.notna()]
    if known.empty:
        return []
    groups = [(key, group.loc[group.preprocessing_completed & group.recording_span_s.notna()])
              for key, group in runs.groupby(["session_id", "run_id"], sort=True)]
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * .9), 4.5))
    positions = [i for i, (_, group) in enumerate(groups, start=1) if not group.empty]
    ax.boxplot([group.recording_span_s.to_numpy()/60 for _, group in groups if not group.empty],
               positions=positions, showfliers=False)
    ax.set_xticks(range(1, len(groups)+1), [f"{session}/{run}\nn={len(group)}" for (session, run), group in groups])
    ax.set_xlim(.5, len(groups)+.5)
    rng = np.random.default_rng(0)
    for i, (_, group) in enumerate(groups, start=1):
        if group.empty:
            ax.text(i, .04, 'Non mesuré', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#9a4141')
            continue
        ax.scatter(i + rng.uniform(-.12, .12, len(group)), group.recording_span_s / 60,
                   s=10, alpha=.5, color="#315e80")
    ax.set(xlabel="Session / run — un point = une acquisition", ylabel="Durée totale ASC (min)",
           title="Durées avant exclusions — fichiers effectivement prétraités")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(destination / "duration_by_run.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    groups = [("Tous les ASC prétraités", known),
              ("Runs retenus par QC", known.loc[known.run_eligibility.eq("retained")]),
              ("Runs et participants retenus", known.loc[known.run_eligibility.eq("retained")
                                                          & known.subject_eligibility.eq("retained")])]
    if not known.subject_eligibility.ne('not_evaluated').any():
        groups = groups[:2]
    if not known.run_eligibility.ne('not_evaluated').any():
        groups = groups[:1]
    colors = ["#36876b", "#e1ad32", "#b76564", "#a4abb6"]
    for i, (label, group) in enumerate(groups):
        left = 0.0
        for key, color in zip(PARTITION_COLUMNS, colors):
            value = float(group[key].sum()) / 3600
            ax.barh(i, value, left=left, color=color, label=LABELS[key] if i == 0 else None)
            left += value
    ax.set(yticks=range(len(groups)), yticklabels=[f"{label} (n={len(group)})" for label, group in groups],
           xlabel="Heures cumulées — cohortes imbriquées, ne pas additionner les lignes",
           title="Budget temporel de la pupille")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "cohort_time_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ["duration_by_run.png", "cohort_time_budget.png"]


def export_preprocessing_report(run_qc, output_root, parameters, *, profile="creaffect", subject_qc=None, figures=True):
    """Exporte les bilans après prétraitement, sans utiliser de métrique d'hypothèse."""
    root = Path(output_root)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    runs = normalize_runs(run_qc, profile=profile, subject_qc=subject_qc)
    cohort, participants, by_run = summary_tables(runs)
    effective = parameters.copy()
    for column, key in (("sampling_rate_hz", "sampling_rate_hz"), ("recorded_eye", "eye"),
                        ("screen_width_px", "gaze_screen_width_px"), ("screen_height_px", "gaze_screen_height_px")):
        if column in runs:
            values = runs.loc[runs.preprocessing_completed, column].dropna().unique()
            if len(values) == 1:
                effective[key] = values[0].item() if hasattr(values[0], 'item') else values[0]
    steps = preprocessing_steps(effective)
    criteria = exclusion_criteria(parameters, profile)
    for name, frame in (("run_summary", runs), ("participant_summary", participants),
                        ("cohort_summary", cohort), ("by_run", by_run),
                        ("steps", steps), ("exclusion_criteria", criteria)):
        frame.to_csv(tables / f"preprocessing_{name}.csv", index=False)
    files = _plots(runs, root / "figures" / "preprocessing") if figures else []
    metadata = dict(generated_utc=datetime.now(timezone.utc).isoformat(), profile=profile,
                    parameters=parameters, duration_scope="full_asc_recording",
                    missing_duration_policy="unknown_not_zero", expected_runs=len(runs),
                    measured_runs=int(runs.preprocessing_completed.sum()))
    (root / "preprocessing_report_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    total = cohort.iloc[0]
    absent_groups = by_run.loc[by_run.n_runs_preprocessed.eq(0)]
    absence_notice = ''
    if not absent_groups.empty:
        names = ', '.join(f"{row.session_id}/{row.run_id}" for row in absent_groups.itertuples())
        absence_notice = ("<p class='note'><strong>Aucune durée mesurable pour les sessions/runs : "
                          + html.escape(names) + ".</strong> Consulter le manifest d'appariement et les erreurs. "
                          "Ce constat ne démontre pas qu'aucune acquisition expérimentale ou IRM n'a eu lieu.</p>")
    run_view = by_run[[key for key in (
        'session_id', 'run_id', 'n_runs_expected', 'n_runs_preprocessed', 'n_runs_without_duration',
        'run_duration_mean_min', 'run_duration_sd_min', 'run_duration_median_min',
        'run_duration_q25_min', 'run_duration_q75_min', 'run_duration_min_min', 'run_duration_max_min') if key in by_run]].rename(columns={
            'session_id':'Session', 'run_id':'Run', 'n_runs_expected':'N attendus',
            'n_runs_preprocessed':'N prétraités', 'n_runs_without_duration':'Sans durée',
            'run_duration_mean_min':'Moyenne (min)', 'run_duration_sd_min':'SD (min)',
            'run_duration_median_min':'Médiane (min)', 'run_duration_q25_min':'Q1 (min)',
            'run_duration_q75_min':'Q3 (min)', 'run_duration_min_min':'Minimum (min)', 'run_duration_max_min':'Maximum (min)'})
    participant_view = participants[[key for key in (
        'participant_id', 'subject_eligibility', 'n_runs_expected', 'n_runs_preprocessed', 'n_runs_retained',
        'total_recording_span_min', 'total_pupil_observed_duration_min', 'total_pupil_interpolated_duration_min',
        'total_gaze_valid_duration_min') if key in participants]].rename(columns={
            'participant_id':'Participant', 'subject_eligibility':'Statut participant',
            'n_runs_expected':'N attendus', 'n_runs_preprocessed':'N prétraités', 'n_runs_retained':'N retenus',
            'total_recording_span_min':'ASC (min)', 'total_pupil_observed_duration_min':'Pupille observée (min)',
            'total_pupil_interpolated_duration_min':'Pupille interpolée (min)',
            'total_gaze_valid_duration_min':'Gaze valide (min)'})
    fragments = ["<!doctype html><html lang='fr'><meta charset='utf-8'><title>Bilan de prétraitement</title>",
                 "<style>body{font-family:system-ui;max-width:1350px;margin:2rem auto;padding:0 1rem;color:#203040}"
                 "table{border-collapse:collapse;font-size:.82rem}td,th{border:1px solid #ddd;padding:.4rem}"
                 ".scroll{overflow:auto;max-height:600px}img{max-width:100%}.note{background:#edf3f6;padding:1rem}</style>",
                 "<h1>Eye-tracking : bilan des durées et du prétraitement</h1>",
                 f"<p>Profil : {html.escape(profile)}. {int(total.n_participants)} participants, {len(runs)} runs attendus, "
                 f"{int(total.n_runs_preprocessed)} prétraités, {int(total.n_runs_without_duration)} sans durée mesurable.</p>",
                 absence_notice,
                 "<p class='note'>Les durées portent sur tout le fichier ASC, avant les exclusions de fenêtres. "
                 "Un run indisponible a une durée inconnue, pas zéro. Une durée de signal conservé ne prouve pas "
                 "que tous les essais sont analysables. Le temps de regard valide et les blinks recouvrent la pupille : ne pas les additionner.</p>",
                 "<h2>Les durées à distinguer</h2><ul>"]
    for key, label in LABELS.items():
        value = total["total_" + key]
        formatted = f"{value/60:.2f} min" if pd.notna(value) else "inconnue"
        fragments.append(f"<li>{html.escape(label)} : <strong>{formatted}</strong></li>")
    fragments.extend(["</ul><p>Durée ASC = pupille observée + pupille interpolée + pupille inutilisable acquise + temps sans échantillon. "
                      "L'interpolation ne rajoute pas de temps d'enregistrement. Les durées RUN_START→RUN_END, si disponibles, "
                      "sont exportées séparément et ne définissent pas le périmètre des masques ci-dessus.</p>",
                      "<h2>Étapes et recommandations</h2>", steps.to_html(index=False, escape=True),
                      "<p>Les références soutiennent les principes. Les seuils numériques sont des choix du projet à valider visuellement ; "
                      "ce pipeline n'implémente pas intégralement toutes les méthodes de chaque article. Avec ±100 ms de padding et un plafond "
                      "de reconstruction de 200 ms, la plupart des blinks restent des trous. La baseline et le QC par fenêtre interviennent ensuite.</p>",
                      "<h2>Critères d'exclusion réellement appliqués</h2>", criteria.to_html(index=False, escape=True),
                      "<h2>Durées par numéro de run</h2><p>Les sommes portent sur les fichiers mesurés. Moyenne, médiane, quartiles et écart-type "
                      "décrivent les durées de runs, pas une moyenne pondérée par participant.</p><div class='scroll'>",
                      run_view.round(2).to_html(index=False, escape=True), "</div>"])
    example = root / 'figures' / 'preprocessing' / 'preprocessing_steps_example.png'
    if figures and example.is_file():
        fragments.append("<h2>Le même signal à quatre étapes</h2>"
                         "<p>Extrait illustratif du premier ASC non vide prétraité, sans sélection selon un effet de condition. "
                         "Les axes verticaux sont ajustés à chaque panneau. Ce seul extrait ne valide pas la cohorte.</p>"
                         "<img src='figures/preprocessing/preprocessing_steps_example.png' alt='Étapes du nettoyage de la pupille'>")
    for name in files:
        fragments.append(f"<img src='figures/preprocessing/{name}' alt='Bilan des durées'>")
    fragments.extend(["<h2>Participants</h2><p><code>not_evaluated</code> signifie qu'aucune exclusion globale de ce niveau "
                      "n'est évaluée, et non que le participant ou le run est exclu.</p><div class='scroll'>",
                      participant_view.round(2).to_html(index=False, escape=True), "</div>",
                      "<h2>Acquisitions : durées et exclusions</h2><p>Tous les runs, y compris absents et exclus, figurent dans le CSV. "
                      "Les colonnes sont en secondes (_s) et minutes (_min).</p><div class='scroll'>"])
    columns = [key for key in ("participant_id", "session_id", "run_id", "preprocessing_completed", "run_eligibility",
                               "subject_eligibility", "recording_span_min", "acquired_duration_min",
                               "pupil_observed_duration_min", "pupil_interpolated_duration_min", "gaze_valid_duration_min",
                               "marker_run_duration_s", "run_exclusion_reason", "error") if key in runs]
    fragments.extend(["<label>Rechercher un participant, run ou motif : <input id='run-search' type='search'></label>",
                      runs[columns].round(3).to_html(index=False, escape=True, table_id='run-table'),
                      "</div><script>document.getElementById('run-search').addEventListener('input',function(){"
                      "const q=this.value.toLowerCase();document.querySelectorAll('#run-table tbody tr').forEach("
                      "r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none');});</script>",
                      "<h2>Références</h2><ul>"])
    for title, url in REFERENCES:
        fragments.append(f"<li><a href='{html.escape(url)}'>{html.escape(title)}</a></li>")
    fragments.append("</ul></html>")
    (root / "preprocessing_report.html").write_text("\n".join(fragments), encoding="utf-8")
    return {"runs": runs, "cohort": cohort, "participants": participants, "by_run": by_run}
