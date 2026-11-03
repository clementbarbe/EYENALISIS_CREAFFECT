"""Reproductions synthétiques de l'audit du 3 novembre 2026.

Usage depuis la racine : python docs/audit/reproduce_findings.py
Ne lit aucune donnée de participant et ne modifie aucune sortie du pipeline.
Les valeurs décrivent le comportement présent ; ce ne sont pas des assertions
exigeant que les défauts persistent après leur correction.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DEFAULT_CONFIG
from metrics import _qc_segment, blink_rate, extract_metrics, pupil_peak_latency
from parsing_eyetracking import EVENT_COLUMNS, parse_asc
from preprocessing import preprocess_pupil
from stats_robust import run_clustered_models
from timecourse import make_timecourses


def samples_at(timestamps, pupil=1000.0, gaze_x=960.0, gaze_y=540.0):
    return pd.DataFrame(dict(timestamp_ms=timestamps, pupil_size=pupil,
                             gaze_x=gaze_x, gaze_y=gaze_y))


def window_metadata():
    return dict(
        subject_id="synthetic", run_id=1, trial_n=0, induction_block_id=0,
        moment="H2_pre_idea", window_x_s=1.0,
        start_eyelink_ms=1000.0, end_eyelink_ms=2000.0,
        baseline_start_eyelink_ms=0.0, baseline_end_eyelink_ms=500.0,
        timeout=False, is_resumed=False, duration_valid=True,
        actual_duration_s=1.0, group_5cat="A", group_valence="A",
        group_arousal="A",
    )


def reproduce():
    # Désactiver le filtre permet d'isoler le rejet des points avant lissage.
    cfg = replace(DEFAULT_CONFIG, lowpass_enabled=False)
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    results = {}

    t = np.arange(4000, dtype=float)
    pupil = 1000 + 100 * np.sin(t * 2 * np.pi / 2000)
    pupil[2000] += 50
    clean, _ = preprocess_pupil(samples_at(t, pupil), events, cfg)
    results["speed_outlier_away_from_missing"] = {
        key: bool(clean.loc[2000, key])
        for key in ("speed_artifact_mask", "speed_near_missing_mask",
                    "pupil_observed_valid")
    }

    with TemporaryDirectory(prefix="eyetrack-audit-") as folder:
        asc = Path(folder) / "binocular.asc"
        asc.write_text(
            "START 1000 LEFT RIGHT SAMPLES EVENTS\n"
            "SAMPLES GAZE LEFT RIGHT RATE 1000\n"
            "1000 200 300 1000 700 800 2000\n", encoding="latin-1")
        parsed, _ = parse_asc(asc)
        results["binocular_eye_selection"] = {
            "configured_eye": cfg.eye,
            "parsed_pupil": float(parsed.iloc[0]["pupil_size"]),
            "left_pupil_in_file": 1000.0,
            "right_pupil_in_file": 2000.0,
        }

    # 100 échantillons sur les 500 attendus pour la baseline ; 5/50 par bin.
    t = np.r_[np.arange(400.0, 500.0), np.arange(1000.0, 1005.0)]
    clean, blinks = preprocess_pupil(samples_at(t), events, cfg)
    meta = window_metadata()
    tc = make_timecourses(pd.DataFrame([meta]), clean, cfg)
    results["timecourse_partial_acquisition"] = {
        "baseline_observed_fraction_metrics_qc": _qc_segment(
            clean.loc[clean.timestamp_ms.lt(500)], 0, 500, cfg
        )["observed_valid_fraction"],
        "timecourse_baseline_valid": bool(tc.iloc[0]["baseline_valid"]),
        "timecourse_observed_fraction_bin": float(tc.iloc[0]["observed_fraction_bin"]),
        "actual_samples_in_50ms_bin": 5,
        "expected_samples_in_50ms_bin": 50,
    }

    # Contrôler séparément les exclusions comportementales avec acquisition complète.
    clean, blinks = preprocess_pupil(samples_at(np.arange(2500.0)), events, cfg)
    meta["timeout"] = True
    tc = make_timecourses(pd.DataFrame([meta]), clean, cfg)
    results["timecourse_timeout"] = {
        "input_timeout": True, "exported_bins": len(tc),
        "finite_deltas": int(tc["pupil_delta"].notna().sum()),
    }

    peak_window = pd.DataFrame(dict(
        timestamp_ms=[0.0, 100.0, 200.0, 300.0],
        pupil_clean=[np.nan, np.nan, 4.0, 5.0]))
    results["peak_latency_origin"] = {
        "reported_latency_s": pupil_peak_latency(peak_window, 0, blinks, events, cfg),
        "latency_from_nominal_window_start_s": 0.3,
    }

    blink = pd.DataFrame(dict(blink_start_ms=[800.0], blink_end_ms=[1200.0],
                              blink_duration_ms=[400.0]))
    rates = []
    for start, stop in ((0, 1000), (1000, 2000)):
        segment = pd.DataFrame(dict(timestamp_ms=np.arange(float(start), float(stop))))
        rates.append(blink_rate(segment, 0, blink, events, cfg))
    results["blink_overlapping_adjacent_windows"] = {
        "rates_per_s": rates,
        "physical_events_across_both_windows": 1,
        "onsets_per_window": [1, 0],
    }

    clean, blinks = preprocess_pupil(
        samples_at(np.arange(2500.0), gaze_x=-100, gaze_y=-100), events, cfg)
    meta = window_metadata()
    measures = extract_metrics(pd.DataFrame([meta]), clean, blinks, events, cfg)
    chosen = measures.loc[measures.metric.isin(
        ["all_saccade_rate_per_s", "blink_rate_per_s", "gaze_dispersion_x"]),
        ["metric", "value", "moment_valid", "gaze_valid_fraction"]]
    results["unverified_event_stream_and_invalid_gaze"] = [
        {"metric": r.metric, "value": float(r.value) if pd.notna(r.value) else None,
         "moment_valid": bool(r.moment_valid),
         "gaze_valid_fraction": float(r.gaze_valid_fraction)}
        for r in chosen.itertuples()
    ]

    # 24 sujets avant retrait des ratings absents ; seulement 2 après retrait.
    rng = np.random.default_rng(8)
    rows = []
    for subject in range(24):
        for trial in range(30):
            rows.append(dict(
                subject_id=str(subject), run_id=1, trial_n=trial,
                moment="H3_liking", metric="pupil_baseline_corrected",
                value=rng.normal(), moment_valid=True,
                group_5cat="A" if trial % 2 else "B",
                group_valence="A", group_arousal="A",
                LTask_rating=rng.normal() if subject < 2 else np.nan))
    model = run_clustered_models(pd.DataFrame(rows), cfg).iloc[0]
    results["model_counts_after_rating_exclusion"] = {
        "reported_subjects": int(model.n_subjects),
        "reported_rows": int(model.n_rows),
        "actual_subjects_with_rating": 2,
        "actual_rows_with_rating": 60,
        "minimum_subjects_configured": cfg.min_subjects_for_model,
        "model_reliable": bool(model.model_reliable),
    }
    return results


if __name__ == "__main__":
    print(json.dumps(reproduce(), ensure_ascii=False, indent=2, allow_nan=False))
