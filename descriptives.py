"""Résumés QC et descriptifs conditionnels sans pseudo-réplication."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def export_descriptives(metrics: pd.DataFrame, out: Path):
    if metrics.empty:
        return
    dims = ['subject_id','run_id','trial_n','moment']
    q = metrics.drop_duplicates(subset=dims).copy()
    cols = ['observed_valid_fraction','interpolated_fraction',
            'gaze_valid_fraction','baseline_observed_valid_fraction',
            'longest_gap_ms', 'sampling_coverage_fraction']
    qc = q.groupby(['moment','group_5cat'],dropna=False)[cols].agg(['mean','median','count'])
    qc.to_csv(out/'qc_by_condition.csv')
    q.groupby(['subject_id','run_id','moment'])[cols].median().to_csv(out/'qc_by_participant_run.csv')
    valid = metrics.loc[metrics['moment_valid'] & metrics['value'].notna()].copy()
    valid.groupby(['subject_id','moment','metric','group_5cat'],dropna=False)['value'].agg(
        mean='mean', n_trials='count', std='std').to_csv(out/'participant_condition_descriptives.csv')
    valid.groupby(['moment','metric','group_5cat'],dropna=False).agg(
        n_valid_trials=('value','size'), n_participants=('subject_id','nunique')).to_csv(
            out/'n_valid_by_metric_condition.csv')
