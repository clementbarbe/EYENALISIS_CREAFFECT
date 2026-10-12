"""Appariement non ambigu sur les messages EyeLink, jamais sur l'indice ASC.

Un run sans RUN_START explicite n'est pas apparié automatiquement.
Les doublons de conversion EDF sont dédoublonnés uniquement si les marqueurs
RUN_START/RUN_END ET les bornes des échantillons coïncident.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

START = re.compile(r'\bRUN_START\s+(\d+)\b', re.I)
END = re.compile(r'\bRUN_END\s+(\d+)\b', re.I)
EDF = re.compile(r'\bCONVERTED FROM\s+(\S+\.edf)\b', re.I)


def scan_asc_metadata(path: Path) -> dict:
    row = dict(path=str(path), filename=path.name, run_id=np.nan,
               start_run_ms=np.nan, end_run_ms=np.nan,
               first_sample_ms=np.nan, last_sample_ms=np.nan,
               edf_source='', status='ok', acquisition_hz=np.nan, gaze_coords='')
    starts, ends = [], []
    with path.open('r', encoding='latin-1', errors='replace') as stream:
        for lineno, line in enumerate(stream):
            if lineno < 20:
                origin = EDF.search(line)
                if origin:
                    row['edf_source'] = Path(origin.group(1)).name
            if line.startswith('MSG'):
                if ' RECCFG ' in line:
                    match = re.search(r'\bRECCFG\s+\w+\s+(\d+)\b', line)
                    if match: row['acquisition_hz'] = float(match.group(1))
                if ' GAZE_COORDS ' in line:
                    match = re.search(r'\bGAZE_COORDS\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)', line)
                    if match: row['gaze_coords'] = ','.join(match.groups())
                a = START.search(line); b = END.search(line)
                if a:
                    tokens = line.split()
                    if len(tokens) >= 3:
                        starts.append((int(a.group(1)), float(tokens[1])))
                if b:
                    tokens = line.split()
                    if len(tokens) >= 3:
                        ends.append((int(b.group(1)), float(tokens[1])))
            elif line and line[0].isdigit():
                sample = line.split(maxsplit=1)[0]
                if sample.isdigit():
                    t = float(sample)
                    if not np.isfinite(row['first_sample_ms']):
                        row['first_sample_ms'] = t
                    row['last_sample_ms'] = t
    if len(starts) != 1 or len(ends) != 1:
        row['status'] = 'missing_or_multiple_run_markers'
    elif starts[0][0] != ends[0][0] or ends[0][1] <= starts[0][1]:
        row['status'] = 'inconsistent_run_markers'
    else:
        row.update(run_id=starts[0][0], start_run_ms=starts[0][1], end_run_ms=ends[0][1])
        if starts[0][0] == 0:
            row['status'] = 'training_run_ignored'
    return row


def build_manifest(subject_dir: Path, runs=(1, 2, 3, 4, 5)) -> tuple[dict[int, Path], pd.DataFrame]:
    records = [scan_asc_metadata(f) for f in sorted(subject_dir.glob('*.asc'))]
    rows = pd.DataFrame(records)
    selected = {}
    if rows.empty:
        return selected, pd.DataFrame(columns=['path', 'run_id', 'status', 'selected'])
    rows['selected'] = False
    for run in runs:
        group = rows.loc[rows['run_id'].eq(run) & rows['status'].eq('ok')]
        if group.empty:
            continue
        if len(group) == 1:
            chosen = group.index[0]
        else:
            # Deux conversions de la même acquisition ont les mêmes bornes + marqueurs.
            cols = ['start_run_ms', 'end_run_ms', 'first_sample_ms', 'last_sample_ms']
            if any(group[c].nunique(dropna=False) != 1 for c in cols):
                rows.loc[group.index, 'status'] = 'ambiguous_duplicate_run'
                continue
            # Préférer ASC de même stem que le fichier EDF source.
            matching = group.loc[group.apply(
                lambda r: Path(r['filename']).stem == Path(r['edf_source']).stem, axis=1)]
            chosen = (matching.index[0] if len(matching) else group.index[0])
            rows.loc[group.index.difference([chosen]), 'status'] = 'duplicate_conversion_ignored'
        selected[run] = Path(rows.loc[chosen, 'path'])
        rows.loc[chosen, 'selected'] = True
    return selected, rows
