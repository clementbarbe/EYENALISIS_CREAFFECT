"""Contrôle qualité visuel des traces. Le downsampling n'affecte pas les métriques."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_run_qc(samples: pd.DataFrame, name: str, root: Path):
    if samples.empty:
        return
    root.mkdir(parents=True,exist_ok=True)
    step = max(1, len(samples)//20000)
    s = samples.iloc[::step]
    t=(s['timestamp_ms']-float(s['timestamp_ms'].iloc[0]))/1000
    fig, (ax, ay) = plt.subplots(2,1,figsize=(15,6),sharex=True)
    ax.plot(t, s['pupil_size'], linewidth=.5,alpha=.45,label='brut')
    ax.plot(t, s['pupil_clean'], linewidth=.7,label='nettoyé')
    ax.set(ylabel='Pupille (unités EyeLink)',title=f'QC {name}')
    ax.legend(loc='upper right')
    ay.plot(t,s['pupil_artifact_mask'].astype(int),linewidth=.55,label='artefact rejeté')
    ay.plot(t,s['pupil_interpolated'].astype(int)*.65,linewidth=.5,label='interpolé')
    ay.plot(t,s['gaze_valid'].astype(int)*.3,linewidth=.5,label='gaze valide ×0.3')
    ay.set(xlabel='Secondes depuis début ASC',ylabel='Masques (0/1)',ylim=(-.05,1.1))
    ay.legend(loc='upper right',ncol=3)
    fig.tight_layout()
    fig.savefig(root/f'{name}_pupil_QC.png',dpi=130)
    plt.close(fig)


def plot_preprocessing_steps(samples: pd.DataFrame, cfg, path: Path):
    """Extrait pédagogique du même signal à quatre étapes ; aucune correction ajoutée."""
    if samples.empty:
        return
    time = samples.timestamp_ms.to_numpy(dtype=float)
    blink = samples.event_blink_mask.to_numpy(dtype=bool)
    candidates = np.flatnonzero(blink & np.r_[True, ~blink[:-1]])
    center = time[candidates[0]] if len(candidates) else time[len(time)//2]
    left, right = max(time[0], center - 2500), min(time[-1], center + 3500)
    s = samples.loc[samples.timestamp_ms.between(left, right)].copy()
    x = (s.timestamp_ms.to_numpy(dtype=float) - center) / 1000
    raw = s.pupil_size.to_numpy(dtype=float)
    rejected = raw.copy()
    rejected[s.pupil_artifact_mask.to_numpy(dtype=bool)] = np.nan
    before = s.pupil_before_filter.to_numpy(dtype=float)
    signals = [raw, rejected, before, s.pupil_clean.to_numpy(dtype=float)]
    titles = ['1. Pupille brute', '2. Après rejet des artefacts et padding',
              '3. Après interpolation autorisée',
              '4. Après filtre passe-bas' if cfg.lowpass_enabled else '4. Signal final — filtre désactivé']
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    # Les interruptions d'acquisition doivent rester visibles sur les courbes.
    gaps = np.flatnonzero(np.diff(s.timestamp_ms) > 1.5 * 1000/cfg.sampling_rate_hz) + 1
    for ax, signal, title in zip(axes, signals, titles):
        plot_x, plot_y = x.copy(), signal.copy()
        for index in gaps[::-1]:
            plot_x = np.insert(plot_x, index, (x[index-1] + x[index])/2)
            plot_y = np.insert(plot_y, index, np.nan)
        ax.plot(plot_x, plot_y, lw=1, color='#315e80')
        ax.set(ylabel='Unités EyeLink', title=title)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Secondes relatives au premier EBLINK (ou au milieu de l’enregistrement)')
    fig.suptitle(f'Un même extrait, quatre étapes — padding {cfg.blink_padding_before_ms}/'
                 f'{cfg.blink_padding_after_ms} ms ; interpolation ≤{cfg.interpolation_max_gap_ms} ms ; '
                 f'filtre {cfg.lowpass_cutoff_hz:g} Hz\n'
                 'Un trou non reconstruit reste absent ; le filtre ne recrée pas les données.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
