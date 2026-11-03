"""Contrats scientifiques et second projet sans conventions CREAFFECT."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_CONFIG
from event_windows import ClockMap, build_project_windows, fit_clock, read_task_events
from examples.generate_demo import generate_demo
from metrics import blink_rate, extract_metrics, pupil_peak_latency
from parsing_eyetracking import EVENT_COLUMNS, parse_asc
from preprocessing import _interpolate_short_gaps, preprocess_pupil
from project_config import load_project, validate_project
from project_models import run_project_models
from project_pipeline import run_project
from timecourse import make_timecourses

CFG = replace(DEFAULT_CONFIG, lowpass_enabled=False)


def raw_samples(time, pupil=1000.0):
    return pd.DataFrame(dict(timestamp_ms=time, pupil_size=pupil, gaze_x=960.0, gaze_y=540.0))


def meta(**updates):
    result = dict(subject_id="one", run_id=1, trial_n=0, induction_block_id=0,
                  moment="H2_pre_idea", start_eyelink_ms=1000.0, end_eyelink_ms=2000.0,
                  baseline_start_eyelink_ms=0.0, baseline_end_eyelink_ms=500.0,
                  duration_valid=True, timeout=False, is_resumed=False)
    return {**result, **updates}


def test_binocular_fractional_timestamps_and_eye_events(tmp_path):
    path = tmp_path / "eye.asc"
    path.write_text("START 1000 LEFT RIGHT SAMPLES EVENTS\nPUPIL AREA\n"
                    "SAMPLES GAZE LEFT RIGHT RATE 2000 FILTER 2\n"
                    "EVENTS GAZE LEFT RIGHT RATE 2000 FILTER 2\n"
                    "1000.0 200 300 1000 700 800 2000\n"
                    "1000.5 201 301 1001 701 801 2001\n"
                    "EBLINK L 1000 1001 1\nEBLINK R 1000 1001 1\nEND 1001\n")
    right, events = parse_asc(path, eye="R", strict=True)
    left, _ = parse_asc(path, eye="L", strict=True)
    assert right.pupil_size.tolist() == [2000, 2001]
    assert left.gaze_x.tolist() == [200, 201]
    assert events.eye.tolist() == ["R"]
    assert right.attrs["eyelink_metadata"]["pupil_measure"] == "area"
    assert right.attrs["eyelink_metadata"]["sampling_rate_hz"] == 2000


@pytest.mark.parametrize("text", [
    "START 0 RIGHT SAMPLES\nPUPIL DIAMETER\nEVENTS GAZE RIGHT RATE 1000\n0 1 2 3\n",
    "SAMPLES HREF RIGHT RATE 1000\n0 1 2 3\n",
    "0 1 2 3\n0 4 5 6\n",
])
def test_unsupported_or_ambiguous_asc_fails_explicitly(tmp_path, text):
    path = tmp_path / "bad.asc"
    path.write_text(text)
    with pytest.raises(ValueError):
        parse_asc(path, strict=True)


def test_speed_spike_rejected_away_from_missing():
    time = np.arange(4000.0)
    pupil = 1000 + 100 * np.sin(time * 2 * np.pi / 2000)
    pupil[2000] += 50
    clean, _ = preprocess_pupil(raw_samples(time, pupil), pd.DataFrame(columns=EVENT_COLUMNS), CFG)
    assert clean.loc[2000, "speed_artifact_mask"]
    assert not clean.loc[2000, "pupil_observed_valid"]


def test_no_interpolation_across_acquisition_segment():
    signal = pd.Series([1.0, 1.0, np.nan, 2.0, 2.0])
    result, interpolated = _interpolate_short_gaps(signal, np.arange(5.0), 200, np.array([0, 0, 0, 1, 1]))
    assert np.isnan(result.iloc[2]) and not interpolated.any()


def test_timecourse_coverage_and_empty_bins_match_expected_time():
    samples, _ = preprocess_pupil(raw_samples(np.r_[np.arange(400.0, 500), np.arange(1000.0, 1005)]),
                                  pd.DataFrame(columns=EVENT_COLUMNS), CFG)
    result = make_timecourses(pd.DataFrame([meta()]), samples, CFG)
    assert len(result) == 20
    assert result.iloc[0].observed_fraction_bin == pytest.approx(0.1)
    assert not result.baseline_valid.any()
    assert result.pupil_delta.isna().all()
    assert result.iloc[1:].n_samples_bin.eq(0).all()


def test_timecourse_does_not_reinclude_timeout():
    samples, _ = preprocess_pupil(raw_samples(np.arange(2500.0)), pd.DataFrame(columns=EVENT_COLUMNS), CFG)
    result = make_timecourses(pd.DataFrame([meta(timeout=True)]), samples, CFG)
    assert len(result) == 20 and result.pupil_delta.isna().all()
    assert not result.window_valid.any()


def test_latency_uses_declared_anchor_and_blinks_have_unique_onset():
    data = pd.DataFrame(dict(timestamp_ms=[1200.0, 1300, 1400, 1500], pupil_clean=[np.nan, np.nan, 4, 5]))
    data.attrs.update(window_start_ms=1200.0, window_end_ms=1600.0, anchor_eyelink_ms=1000.0)
    assert pupil_peak_latency(data, 0, None, None, CFG) == pytest.approx(0.5)
    blinks = pd.DataFrame(dict(blink_start_ms=[800.0], blink_end_ms=[1200.0], blink_duration_ms=[400.0]))
    rates = [blink_rate(pd.DataFrame(dict(timestamp_ms=np.arange(start, start + 1000.0))), 0, blinks, None, CFG)
             for start in [0.0, 1000.0]]
    assert rates == [1.0, 0.0]


def test_unknown_event_stream_missing_differs_from_observed_zero():
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    samples, blinks = preprocess_pupil(raw_samples(np.arange(2500.0)), events, CFG)
    windows = pd.DataFrame([meta(metrics=["blink_count", "blink_mean_duration_ms", "all_saccade_rate_per_s"])])
    missing = extract_metrics(windows, samples, blinks, events, CFG)
    assert missing.value.isna().all()
    assert missing.moment_exclusion_reason.str.contains("event_stream_unavailable").all()
    events.attrs["eyelink_metadata"] = {"event_streams": ["EBLINK", "ESACC"]}
    observed = extract_metrics(windows, samples, blinks, events, CFG).set_index("metric")
    assert observed.loc["blink_count", "value"] == 0
    assert observed.loc["all_saccade_rate_per_s", "value"] == 0
    assert not observed.loc["blink_mean_duration_ms", "moment_valid"]


def test_invalid_baseline_only_blocks_metrics_that_require_it():
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    samples, blinks = preprocess_pupil(raw_samples(np.arange(2500.0)), events, CFG)
    windows = pd.DataFrame([meta(baseline_start_eyelink_ms=np.nan, baseline_end_eyelink_ms=np.nan,
                                metrics=["pupil_mean", "pupil_baseline_corrected", "gaze_dispersion_x"])])
    result = extract_metrics(windows, samples, blinks, events, CFG).set_index("metric")
    assert result.loc["pupil_mean", "moment_valid"]
    assert result.loc["gaze_dispersion_x", "moment_valid"]
    assert not result.loc["pupil_baseline_corrected", "moment_valid"]


def basic_project():
    return dict(version=1, project_id="arbitrary", recordings_manifest="manifest.tsv",
                synchronization={"mode": "eyelink_clock"}, windows=[
                    dict(id="decision", anchor={"event": "stim"}, start={"anchor_offset_s": 0},
                         end={"event": "response"}, metrics=["pupil_mean"])])


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(typo=True),
    lambda p: p.update(parameters={"invented_threshold": .7}),
    lambda p: p["windows"][0].update(metrics=["unknown_metric"]),
    lambda p: p["windows"][0].update(end={"event": "response", "edge": "middle"}),
    lambda p: p["windows"][0].update(baseline="unknown_baseline"),
])
def test_project_rejects_unsupported_configuration(mutation):
    project = basic_project()
    mutation(project)
    with pytest.raises(ValueError):
        validate_project(project)


def test_rate_inference_is_not_silently_fitted_as_continuous_ols():
    project = basic_project()
    project["windows"][0]["metrics"] = ["all_saccade_rate_per_s"]
    project["hypotheses"] = [dict(id="h", window="decision", metric="all_saccade_rate_per_s",
                                  predictor="condition", contrast=["B", "A"])]
    with pytest.raises(ValueError, match="continuous outcomes"):
        validate_project(project)


def test_variable_windows_missing_repeated_and_truncation():
    project = basic_project()
    project["windows"].append(dict(id="before", anchor={"event": "response"},
                                   start={"anchor_offset_s": -1.2}, end={"anchor_offset_s": -.2},
                                   lower_bound={"event": "stim"}, clipping_policy="truncate", metrics=["pupil_mean"]))
    events = pd.DataFrame([
        dict(trial_id="a", event_id="s1", event_type="stim", onset=1.0, duration=0),
        dict(trial_id="a", event_id="r1", event_type="response", onset=2.0, duration=0),
        dict(trial_id="b", event_id="s2", event_type="stim", onset=4.0, duration=0),
        dict(trial_id="c", event_id="s3", event_type="stim", onset=7.0, duration=0),
        dict(trial_id="c", event_id="s4", event_type="stim", onset=7.1, duration=0),
        dict(trial_id="c", event_id="r3", event_type="response", onset=9.0, duration=0),
    ])
    windows = build_project_windows(events, project, dict(participant_id="x", session_id="s", run_id="one"), ClockMap(100), CFG)
    good = windows.loc[windows.trial_id.eq("a") & windows.window_id.eq("before")].iloc[0]
    assert good.truncated and good.actual_duration_s == pytest.approx(.8)
    assert good.start_eyelink_ms == 1100
    assert windows.loc[windows.trial_id.eq("b"), "window_exclusion_reason"].str.contains("missing_event:response").all()
    assert windows.loc[windows.trial_id.eq("c"), "window_exclusion_reason"].str.contains("ambiguous_event:stim").all()


def test_clock_drift_is_estimated_only_from_paired_anchors(tmp_path):
    pd.DataFrame(dict(reference_s=[0, 100, 200], eyelink_ms=[500, 100510, 200520])).to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    clock = fit_clock({"sync_path": "a.tsv"}, {"mode": "paired_anchors"}, tmp_path)
    assert clock.slope == pytest.approx(1.0001)
    assert clock.to_ms(10) == pytest.approx(10501)
    pd.DataFrame(dict(reference_s=[20], eyelink_ms=[20500])).to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    single = fit_clock({"sync_path": "a.tsv"}, {"mode": "paired_anchors"}, tmp_path)
    assert single.n_anchors == 1 and "drift_unverified" in single.quality


def test_declared_model_complete_cases_and_within_subject_contrast():
    rng = np.random.default_rng(9)
    rows = []
    for subject in range(16):
        for trial in range(12):
            condition = "A" if trial % 2 == 0 else "B"
            rows.append(dict(participant_id=str(subject), session_id="one", run_id="one",
                             window_spec_id="w", metric="pupil_mean", moment_valid=True,
                             condition=condition, value=100 + subject + 2 * (condition == "B") + rng.normal(0, .2),
                             covariate=rng.normal() if subject < 2 else np.nan))
    hypothesis = dict(id="h", window="w", metric="pupil_mean", predictor="condition", contrast=["B", "A"])
    result = run_project_models(pd.DataFrame(rows), [hypothesis], CFG).iloc[0]
    assert result.model_reliable and result.estimate == pytest.approx(2, abs=.15)
    incomplete = run_project_models(pd.DataFrame(rows), [{**hypothesis, "numeric_covariates": ["covariate"]}], CFG).iloc[0]
    assert incomplete.n_subjects == 2 and incomplete.n_rows == 24
    assert not incomplete.model_reliable
    mixed = pd.DataFrame(rows)
    mixed["pupil_measure"] = np.where(mixed.participant_id.eq("0"), "area", "diameter")
    result = run_project_models(mixed, [hypothesis], CFG).iloc[0]
    assert not result.model_reliable and result.error == "mixed_pupil_measures_require_harmonization"


def test_second_project_end_to_end_and_cli(tmp_path):
    project = generate_demo(tmp_path / "demo", subjects=3, sampling_hz=500)
    output = tmp_path / "results"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "run_pipeline.py"), "--project", str(project),
                                "--output-dir", str(output), "--qc-plots"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    windows = pd.read_csv(output / "tables" / "windows.csv")
    metrics = pd.read_csv(output / "tables" / "metrics_long.csv")
    courses = pd.read_csv(output / "tables" / "timecourses.csv")
    assert len(windows) == 216 and len(metrics) == 540
    assert windows.participant_id.str.startswith("volunteer-").all()
    assert windows.run_id.eq("task-one").all()
    assert windows.sampling_rate_hz.eq(500).all()
    assert windows.truncated.any()
    assert windows.window_exclusion_reason.fillna("").str.contains("missing_event:first_movement").any()
    assert courses.loc[courses.trial_id.eq("trial-6"), "pupil_delta"].isna().all()
    fixed = courses.loc[courses.original_window.eq("stimulus_late")]
    assert fixed.groupby(["participant_id", "trial_id"]).size().eq(16).all()
    assert (output / "report.html").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 3
    provenance = json.loads((output / "provenance.json").read_text())
    assert len(provenance["recordings"]) == 3
    assert all(len(r["asc_metadata"]["sha256"]) == 64 for r in provenance["recordings"])
    with pytest.raises(ValueError, match="not empty"):
        run_project(project, output)


def test_yaml_configuration_roundtrip(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(basic_project()), encoding="utf-8")
    project, cfg = load_project(path)
    assert project["project_id"] == "arbitrary" and cfg.eye == "R"


def test_failed_recording_is_reported_and_not_partially_exported(tmp_path):
    project = generate_demo(tmp_path / "demo", subjects=2)
    # Une table non monotone : erreur de synchronisation structurelle.
    bad = tmp_path / "demo" / "volunteer-02" / "anchors.tsv"
    pd.DataFrame(dict(reference_s=[0, 1, 2], eyelink_ms=[1000, 999, 2000])).to_csv(bad, sep="\t", index=False)
    output = tmp_path / "results"
    result = run_project(project, output, no_stats=True)
    assert result["processed"] == 1 and result["errors"] == 1
    metrics = pd.read_csv(output / "tables" / "metrics_long.csv")
    assert set(metrics.participant_id) == {"volunteer-01"}
    qc = pd.read_csv(output / "tables" / "recording_qc.csv")
    assert qc.loc[1, "error"].startswith("ValueError: Paired anchors")
    assert (output / "logs" / "recording-0002.log").is_file()


def test_repeated_anchor_requires_explicit_occurrence_and_preserves_instances():
    project = basic_project()
    window = project["windows"][0]
    window.update(anchor={"event": "stim", "occurrence": "each"},
                  start={"anchor_offset_s": 0}, end={"anchor_offset_s": .6})
    validate_project(project)
    events = pd.DataFrame([dict(trial_id="one", event_id=str(i), event_type="stim", onset=float(i), duration=0)
                           for i in [1, 2]])
    windows = build_project_windows(events, project, dict(participant_id="p", session_id="s", run_id="r"), ClockMap(0), CFG)
    assert len(windows) == 2 and windows.anchor_event_id.nunique() == 2
    assert windows.window_exclusion_reason.eq("").all()
