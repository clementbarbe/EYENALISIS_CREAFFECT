"""Contrats du bilan : temps, trous, exclusions et indépendance des analyses."""
from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import pytest

from config import DEFAULT_CONFIG
from examples.generate_demo import generate_demo
from preprocessing_summary import (summarize_recording, normalize_runs, summary_tables,
                                    export_preprocessing_report, PARTITION_COLUMNS)
from project_pipeline import run_project
from summarize_preprocessing import summarize_existing


def cleaned(time):
    n = len(time)
    return pd.DataFrame(dict(timestamp_ms=time, pupil_clean=np.ones(n),
                             pupil_observed_valid=np.ones(n, bool), pupil_interpolated=np.zeros(n, bool),
                             gaze_valid=np.ones(n, bool), pupil_missing_acquisition=np.zeros(n, bool),
                             event_blink_mask=np.zeros(n, bool)))


def test_durations_partition_with_real_acquisition_gap():
    data = cleaned(np.array([0., 1., 2., 10., 11.]))
    data.loc[1, 'pupil_observed_valid'] = False
    data.loc[1, 'pupil_interpolated'] = True
    data.loc[2, 'pupil_observed_valid'] = False
    data.loc[2, 'pupil_clean'] = np.nan
    summary = summarize_recording(data, DEFAULT_CONFIG)
    assert summary['recording_span_s'] == pytest.approx(.012)
    assert summary['acquired_duration_s'] == pytest.approx(.005)
    assert summary['unrecorded_duration_s'] == pytest.approx(.007)
    assert summary['pupil_observed_duration_s'] == pytest.approx(.003)
    assert summary['pupil_interpolated_duration_s'] == pytest.approx(.001)
    assert summary['pupil_unusable_recorded_duration_s'] == pytest.approx(.001)
    assert sum(summary[key] for key in PARTITION_COLUMNS) == pytest.approx(summary['recording_span_s'])


def test_header_sampling_frequency_is_used_for_binocular_or_other_rates():
    data = cleaned(np.arange(0., 10., .5))
    data.attrs['eyelink_metadata'] = {'sampling_rate_hz': 2000, 'selected_eye': 'L'}
    summary = summarize_recording(data, DEFAULT_CONFIG)
    assert summary['acquired_duration_s'] == pytest.approx(.01)
    assert summary['recording_span_s'] == pytest.approx(.01)
    assert summary['recorded_eye'] == 'L'


def test_unknown_run_is_not_zero_and_excluded_run_remains_in_cohort():
    first = {**summarize_recording(cleaned(np.arange(1000.)), DEFAULT_CONFIG),
             'subject_id': '001', 'run_id': 1, 'run_qc_passed': True}
    excluded = {**first, 'run_id': 2, 'run_qc_passed': False}
    missing = {'subject_id': '002', 'run_id': 1, 'preprocessing_completed': False,
               'run_qc_passed': False, 'recording_span_s': 0}
    subjects = pd.DataFrame(dict(subject_id=['001','002'], subject_valid=[True,False]))
    runs = normalize_runs(pd.DataFrame([first, excluded, missing]), subject_qc=subjects)
    assert np.isnan(runs.iloc[2].recording_span_s)
    assert runs.iloc[0].participant_id == 'C4X001'
    cohort, participants, by_run = summary_tables(runs)
    all_runs = cohort.iloc[0]
    assert all_runs.n_runs_expected == 3 and all_runs.n_runs_preprocessed == 2
    assert all_runs.n_runs_without_duration == 1
    assert all_runs.total_recording_span_s == pytest.approx(2)
    assert cohort.iloc[1].total_recording_span_s == pytest.approx(1)
    assert participants.loc[participants.participant_id.eq('C4X002'), 'total_recording_span_s'].isna().all()


def test_standalone_report_roundtrip_keeps_ids_and_exclusions(tmp_path):
    row = {**summarize_recording(cleaned(np.arange(1000.)), DEFAULT_CONFIG),
           'subject_id':'001','run_id':'1','run_qc_passed':False,'run_exclusion_reason':'bad_sync'}
    root = tmp_path/'original'
    export_preprocessing_report(pd.DataFrame([row]), root, asdict(DEFAULT_CONFIG), figures=False)
    report = summarize_existing(root, tmp_path/'copy', figures=False)
    assert report['runs'].iloc[0].participant_id == 'C4X001'
    assert report['runs'].iloc[0].run_eligibility == 'excluded'
    assert report['runs'].iloc[0].run_exclusion_reason == 'bad_sync'
    assert report['runs'].iloc[0].recording_span_s == pytest.approx(1)
    assert (tmp_path/'copy/preprocessing_report.html').is_file()


def test_preprocessing_only_project_does_not_need_events_or_clock_files(tmp_path, monkeypatch):
    project = generate_demo(tmp_path/'input', subjects=1)
    (project.parent/'volunteer-01/events.tsv').unlink()
    (project.parent/'volunteer-01/anchors.tsv').unlink()
    def analysis_must_not_run(*args, **kwargs):
        pytest.fail('Event analysis was called during preprocessing-only mode')
    monkeypatch.setattr('project_pipeline.extract_metrics', analysis_must_not_run)
    result = run_project(project, tmp_path/'output', preprocessing_only=True, figures=False)
    assert result['processed'] == 1 and result['windows'] == 0 and result['metric_rows'] == 0
    runs = pd.read_csv(tmp_path/'output/tables/preprocessing_run_summary.csv')
    assert runs.preprocessing_completed.all() and runs.run_eligibility.eq('not_evaluated').all()
    assert runs.recording_span_s.iloc[0] == pytest.approx(24.)
    assert (tmp_path/'output/preprocessing_report.html').is_file()


def test_preprocessing_only_does_not_overwrite_existing_research_results(tmp_path):
    from run_pipeline import main
    output = tmp_path/'existing'
    output.mkdir()
    sentinel = output/'metrics_long.csv'
    sentinel.write_text('previous results\n')
    with pytest.raises(ValueError, match="pas vide"):
        main(tmp_path/'data', output, preprocessing_only=True)
    assert sentinel.read_text() == 'previous results\n'
