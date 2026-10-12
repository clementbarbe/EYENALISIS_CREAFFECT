"""Tests de non-régression indépendants des données des participants."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from dataclasses import replace
from config import DEFAULT_CONFIG
from matching import build_manifest
from preprocessing import preprocess_pupil, _interpolate_short_gaps
from metrics import pupil_auc, _qc_segment


def test_match_run0_ignored(tmp_path):
    def write(name, start, end, first, last):
        (tmp_path / name).write_text(f'** CONVERTED FROM path/001_{start}.edf\n'
            f'MSG {first} RUN_START {start}\n{first+1} 300 500 800\n'
            f'{last} 300 500 800\nMSG {last} RUN_END {end}\n',encoding='latin-1')
    write('001_1.asc', 0, 0, 1000, 2000)
    write('001_2.asc', 1, 1, 3000, 5000)
    found, m = build_manifest(tmp_path)
    assert found[1].name == '001_2.asc'
    assert 0 not in found
    assert m.loc[m['filename'].eq('001_1.asc'), 'status'].iloc[0] == 'training_run_ignored'


def test_interpolation_not_across_long_gap():
    t = np.arange(20)*10.
    signal = pd.Series(np.r_[np.ones(5), [np.nan]*10, np.ones(5)*2])
    x, mask = _interpolate_short_gaps(signal,t,50)
    assert x.iloc[5:15].isna().all()
    assert not mask.any()
    x, mask = _interpolate_short_gaps(signal,t,150)
    assert x.iloc[5:15].notna().all()
    assert mask.sum() == 10


def test_blinks_not_padded_in_metrics():
    t = np.arange(1000, dtype=float)
    samples = pd.DataFrame({'timestamp_ms':t,'pupil_size':np.full(1000, 850.),
                            'gaze_x':np.full(1000,960.), 'gaze_y':np.full(1000,540.)})
    events = pd.DataFrame({'event_type':['EBLINK'], 'start_ms':[400.], 'end_ms':[470.],
                           'duration_ms':[70.]})
    clean, blinks = preprocess_pupil(samples, events, DEFAULT_CONFIG)
    assert len(blinks) == 1
    assert blinks.iloc[0]['blink_duration_ms'] == 70
    assert clean['pupil_artifact_mask'].sum() > 71
    assert clean['blink_mask'].sum() == 71


def test_auc_no_bridge_across_nan():
    s = pd.DataFrame({'timestamp_ms':[0.,1.,2.,1000.,1001.],
                      'pupil_clean':[5.,5.,5.,np.nan,6.]})
    assert abs(pupil_auc(s, 4.,pd.DataFrame(),pd.DataFrame(),DEFAULT_CONFIG) - 0.002) < 1e-8


def test_no_coverage_without_samples():
    q = _qc_segment(pd.DataFrame(columns=['timestamp_ms']), 0, 1000, DEFAULT_CONFIG)
    assert q['observed_valid_fraction'] == 0
    assert q['longest_gap_ms'] == 1000


def test_clustered_model_not_pseudoreplication():
    from stats_robust import run_clustered_models
    rows = []
    rng = np.random.default_rng(11)
    for subject in range(24):
        intercept = rng.normal(0, 0.35)
        for run in range(1, 6):
            for i, group in enumerate(('LVLA','LVHA','MVMA','HVLA','HVHA')):
                for trial in range(3):
                    rows.append(dict(subject_id=f'C4X{subject:03}',run_id=run,
                        trial_n=i*3+trial,group_5cat=group,
                        group_valence={'LVLA':'Negative','LVHA':'Negative','MVMA':'Neutral',
                                       'HVLA':'Positive','HVHA':'Positive'}[group],
                        group_arousal={'LVLA':'Low','HVLA':'Low','MVMA':'Neutral',
                                      'LVHA':'High','HVHA':'High'}[group],
                        moment='H2_pre_idea',metric='pupil_baseline_corrected',
                        value=intercept+ .25*(group == 'LVHA') + rng.normal(0,.2),
                        moment_valid=True,fgat_search_duration_s=2.+rng.random(),
                        LTask_rating=np.nan))
    x=run_clustered_models(pd.DataFrame(rows),DEFAULT_CONFIG)
    main=x.loc[(x['metric']=='pupil_baseline_corrected') &
                (x['scheme']=='five_categories')]
    assert len(main)==1
    assert bool(main.iloc[0]['converged'])
    assert 0 <= main.iloc[0]['p_group_raw'] <= 1
    assert main.iloc[0]['n_subjects'] == 24


def test_reject_ambiguous_duplicate_run(tmp_path):
    (tmp_path/'a.asc').write_text('MSG 1000 RUN_START 2\n1001 40 40 600\n'
        'MSG 2000 RUN_END 2\n2000 40 40 600\n')
    (tmp_path/'b.asc').write_text('MSG 3000 RUN_START 2\n3001 40 40 600\n'
        'MSG 4000 RUN_END 2\n4000 40 40 600\n')
    selected,manifest=build_manifest(tmp_path)
    assert 2 not in selected
    assert set(manifest['status']) == {'ambiguous_duplicate_run'}


def test_end_of_run_sync_error_is_blocking():
    from sync import synchronize_run
    events=pd.DataFrame({'event_type':['MSG','BUTTON','MSG'],
                         'raw_line':['MSG 1000 RUN_START 1','BUTTON 1100 1 1','MSG 5000 RUN_END 1'],
                         'timestamp_ms':[1000.,1100.,5000.],
                         'button_code':[np.nan,1.,np.nan],
                         'button_state':[np.nan,1.,np.nan]})
    samples=pd.DataFrame({'timestamp_ms':np.arange(1000.,5001.,1.)})
    timing=pd.Series({'t0_fMRI':0.,'t_run_end':15.})
    sync=synchronize_run(events,timing,samples,DEFAULT_CONFIG,expected_run_id=1)
    assert not sync.success
    assert sync.reason == 'poor_sync_end_consistency'
