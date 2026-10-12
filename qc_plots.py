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
