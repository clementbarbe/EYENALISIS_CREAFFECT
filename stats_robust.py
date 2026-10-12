"""Analyse de sensibilité rapide: effets sujet fixes et SE robustes par sujet.

Le regroupement au niveau du participant couvre toutes les dépendances
intra-participant, y compris les 3 essais imbriqués dans un bloc émotionnel.
Ne pas considérer comme inférence confirmatoire (analyses exploratoires).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from stats import HYPOTHESES, GROUP_SCHEMES
from config import Config


def run_clustered_models(metrics_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if metrics_long.empty:
        return pd.DataFrame()
    valid = metrics_long.loc[metrics_long['moment_valid'] & metrics_long['value'].notna()].copy()
    rows = []
    for name, spec in HYPOTHESES.items():
        for scheme in spec['schemes']:
            group = GROUP_SCHEMES[scheme]
            data = valid.loc[(valid['moment'] == spec['moment']) & valid['metric'].isin(spec['metrics'])]
            for metric, d in data.groupby('metric'):
                d = d.dropna(subset=['value', 'subject_id', group, 'run_id']).copy()
                row = dict(hypothesis=name, moment=spec['moment'], metric=metric,
                           scheme=scheme, group_column=group, n_rows=len(d),
                           n_subjects=d['subject_id'].nunique(), p_group_raw=np.nan,
                           p_rating_raw=np.nan, p_group_fdr=np.nan, p_rating_fdr=np.nan,
                           group_significant_fdr=False, rating_significant_fdr=False,
                           model_reliable=False, group_effect_std=np.nan, rating_effect_std=np.nan,
                           converged=False, formula='', error='')
                if (len(d) < cfg.min_rows_for_model or
                    d['subject_id'].nunique() < cfg.min_subjects_for_model or
                    d[group].nunique() < 2 or d['value'].std() < 1e-10):
                    row['error'] = 'insufficient_data_or_levels'
                    rows.append(row)
                    continue
                d['subject_id'] = d['subject_id'].astype(str)
                d['value_z'] = (d['value']-d['value'].mean())/d['value'].std()
                rhs = f'C({group}) + C(run_id) + C(subject_id)'
                if spec['rating_model']:
                    d = d.dropna(subset=['LTask_rating'])
                    if d['LTask_rating'].std() < 1e-10:
                        row['error'] = 'invalid_rating'; rows.append(row); continue
                    d['rating_z'] = (d['LTask_rating']-d['LTask_rating'].mean())/d['LTask_rating'].std()
                    rhs += ' + rating_z'
                # Variables de durée susceptibles de confondre l'effet, fixées avant résultats.
                rt = 'fgat_search_duration_s' if name.startswith('H2') else ('LTask_response_duration_s' if spec['rating_model'] else None)
                if rt is not None and rt in d and d[rt].notna().sum() == len(d) and d[rt].nunique() > 2:
                    d['rt_log'] = np.log1p(d[rt].clip(lower=0))
                    rhs += ' + rt_log'
                row['formula'] = 'value_z ~ ' + rhs
                try:
                    fit = smf.ols(row['formula'], data=d).fit(
                        cov_type='cluster', cov_kwds={'groups':d['subject_id'],
                                                     'use_correction':True}, use_t=True)
                    terms = [x for x in fit.params.index if x.startswith(f'C({group})')]
                    if not terms:
                        row['error'] = 'non_estimable_group'; rows.append(row); continue
                    hyp = ' = 0, '.join(terms) + ' = 0'
                    row['p_group_raw'] = float(fit.f_test(hyp).pvalue)
                    row['converged'] = True
                    row['model_reliable'] = True
                    row['group_effect_std'] = float(fit.params[terms].loc[fit.params[terms].abs().idxmax()])
                    row['max_abs_group_beta_z'] = float(fit.params[terms].abs().max())
                    if spec['rating_model'] and 'rating_z' in fit.pvalues:
                        row['p_rating_raw'] = float(fit.pvalues['rating_z'])
                        row['rating_beta_z'] = float(fit.params['rating_z'])
                        row['rating_effect_std'] = row['rating_beta_z']
                        row['rating_ci_low_z'], row['rating_ci_high_z'] = map(float, fit.conf_int().loc['rating_z'])
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for family in ('group','rating'):
        p = f'p_{family}_raw'; fdr = f'p_{family}_fdr'
        mask = result[p].notna() & result['converged']
        if mask.any():
            result.loc[mask, fdr] = multipletests(result.loc[mask, p],
                                                  method=cfg.fdr_method)[1]
            result.loc[mask, f'{family}_significant_fdr'] = result.loc[mask, fdr] < cfg.alpha
    return result
