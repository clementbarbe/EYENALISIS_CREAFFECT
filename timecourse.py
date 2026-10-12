"""Courbes évènementielles à 20 Hz pour visualisation et modèles temporels.

Le calcul n'extrapole pas les périodes sans acquisition. Moyenne d'échantillons
observés seulement par bin; une baseline invalide donne un delta manquant.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import Config


def make_timecourses(windows: pd.DataFrame, samples: pd.DataFrame,
                     cfg: Config, target_hz: int = 20) -> pd.DataFrame:
    if windows.empty or samples.empty:
        return pd.DataFrame()
    time = samples['timestamp_ms'].to_numpy(dtype=float)
    rows = []
    selection = windows.loc[windows['moment'].isin(['M1_early_photometric',
                            'H1_induction_late','H2_pre_idea'])]
    width_ms = 1000. / target_hz
    for _, meta in selection.iterrows():
        beg = meta['start_eyelink_ms']; end = meta['end_eyelink_ms']
        bl = meta['baseline_start_eyelink_ms']; br = meta['baseline_end_eyelink_ms']
        if any(pd.isna(x) for x in (beg,end,bl,br)) or end <= beg:
            continue
        a, b = np.searchsorted(time,[beg,end],side='left')
        chunk = samples.iloc[a:b]
        if chunk.empty:
            continue
        i,j = np.searchsorted(time,[bl,br],side='left')
        baseline = samples.iloc[i:j]
        valid_base = (len(baseline) > 0 and
                      baseline['pupil_observed_valid'].mean() >= cfg.min_observed_fraction_baseline)
        base = float(baseline.loc[baseline['pupil_observed_valid'],'pupil_clean'].median()) if valid_base else np.nan
        t = chunk['timestamp_ms'].to_numpy(dtype=float)
        y = chunk['pupil_clean'].to_numpy(dtype=float)
        obs = chunk['pupil_observed_valid'].to_numpy(dtype=bool)
        bins = np.floor((t-beg)/width_ms).astype(int)
        for k in np.unique(bins):
            which = bins == k
            usable = which & obs & np.isfinite(y)
            frac = np.count_nonzero(usable) / max(1, np.count_nonzero(which))
            val = float(np.mean(y[usable])) if usable.any() and frac >= cfg.min_observed_fraction_window else np.nan
            if meta['moment'] == 'M1_early_photometric':
                relative = (k+0.5) / target_hz
                name = 'induction'
            elif meta['moment'] == 'H1_induction_late':
                relative = cfg.induction_late_start_s + (k+0.5)/target_hz
                name = 'induction'
            else:
                relative = (float(beg) + (k+.5)*width_ms - float(end))/1000.
                name = 'pre_idea'
            rows.append(dict(subject_id=meta['subject_id'],run_id=meta['run_id'],
                             trial_n=meta['trial_n'],moment=name,
                             original_window=meta['moment'],
                             group_5cat=meta['group_5cat'],group_valence=meta['group_valence'],
                             group_arousal=meta['group_arousal'],
                             relative_time_s=round(relative,4),
                             pupil_observed_mean=val,
                             pupil_delta=val-base if np.isfinite(base) and np.isfinite(val) else np.nan,
                             baseline_valid=valid_base,observed_fraction_bin=frac))
    return pd.DataFrame(rows)
