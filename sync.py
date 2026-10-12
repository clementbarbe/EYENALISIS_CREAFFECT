"""Synchronisation comportement / EyeLink avec vérifications indépendantes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from config import Config

@dataclass
class SyncResult:
    success: bool
    reason: Optional[str]
    first_button_ms: Optional[float]
    offset_ms: Optional[float]
    end_difference_s: Optional[float]
    quality_flag: str
    n_triggers: int = 0
    median_trigger_interval_s: float = np.nan
    trigger_interval_cv: float = np.nan
    run_marker_id: Optional[int] = None


def synchronize_run(events: pd.DataFrame, timing: pd.Series,
                    samples: pd.DataFrame, cfg: Config, expected_run_id: int | None = None) -> SyncResult:
    def failed(why):
        return SyncResult(False, why, None, None, np.nan, 'failed')
    if samples.empty or not np.all(np.diff(samples['timestamp_ms'].to_numpy(dtype=float)) > 0):
        return failed('empty_or_nonmonotonic_samples')
    interval = np.diff(samples['timestamp_ms'].to_numpy(dtype=float))
    effective_fs = 1000. / np.median(interval) if len(interval) else np.nan
    if np.isfinite(effective_fs) and abs(effective_fs - cfg.sampling_rate_hz) > 0.1 * cfg.sampling_rate_hz:
        return failed('sampling_rate_mismatch_check_config')
    if 't0_fMRI' not in timing or pd.isna(timing['t0_fMRI']):
        return failed('missing_t0_fMRI')
    if 't_run_end' not in timing or pd.isna(timing['t_run_end']):
        return failed('missing_t_run_end')
    messages = events.loc[events['event_type'].eq('MSG')]
    run_messages = messages.loc[messages['raw_line'].str.contains(r'\bRUN_START\s+\d+\b', case=False, regex=True, na=False)]
    end_messages = messages.loc[messages['raw_line'].str.contains(r'\bRUN_END\s+\d+\b', case=False, regex=True, na=False)]
    if len(run_messages) != 1 or len(end_messages) != 1:
        return failed('missing_or_multiple_run_markers')
    import re
    found_run = int(re.search(r'\bRUN_START\s+(\d+)\b', run_messages.iloc[0]['raw_line'], re.I).group(1))
    found_end = int(re.search(r'\bRUN_END\s+(\d+)\b', end_messages.iloc[0]['raw_line'], re.I).group(1))
    if found_run != found_end or (expected_run_id is not None and found_run != expected_run_id):
        return failed('run_marker_mismatch')
    run_start = float(run_messages.iloc[0]['timestamp_ms'])
    run_end = float(end_messages.iloc[0]['timestamp_ms'])
    pulses = events.loc[(events['event_type'] == 'BUTTON')
                       & (events['button_code'] == cfg.sync_button_code)
                       & (events['button_state'] == cfg.sync_button_state)
                       & events['timestamp_ms'].between(run_start, run_end)]['timestamp_ms'].to_numpy(dtype=float)
    if not len(pulses):
        return failed('missing_scanner_triggers')
    first = float(pulses[0])
    offset = first - 1000 * float(timing['t0_fMRI'])
    expected_last = first + (float(timing['t_run_end']) - float(timing['t0_fMRI'])) * 1000
    # Le dernier sample ET RUN_END ont été inspectés, mais le fin du behavior
    # correspond à la fin d'acquisition enregistrée, pas nécessairement RUN_END.
    actual_last = float(samples['timestamp_ms'].iloc[-1])
    delta = abs(actual_last - expected_last) / 1000.0
    diff = np.diff(pulses) / 1000.0
    med = float(np.median(diff)) if len(diff) else np.nan
    cv = float(np.std(diff) / np.mean(diff)) if len(diff) and np.mean(diff) > 0 else np.nan
    if delta > cfg.max_sync_end_difference_s:
        return SyncResult(False, 'poor_sync_end_consistency', first, offset, delta,
                          'failed', len(pulses), med, cv, found_run)
    quality = ('trigger_irregularity_warning' if np.isfinite(cv)
               and cv > cfg.max_sync_pulse_interval_cv else 'good')
    return SyncResult(True, None, first, offset, delta, quality, len(pulses), med, cv, found_run)


def behavioral_to_eyelink_ms(behavioral_time_s: float, timing: pd.Series, sync: SyncResult) -> float:
    if not sync.success or sync.first_button_ms is None or pd.isna(behavioral_time_s):
        return np.nan
    return float(sync.first_button_ms + (float(behavioral_time_s)-float(timing['t0_fMRI']))*1000.0)
