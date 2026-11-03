"""Contrastes déclarés : OLS exploratoire intra-sujet, SE clusterisées par sujet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from metrics import METRIC_DEFINITIONS


def run_project_models(metrics, hypotheses, cfg):
    rows = []
    for hypothesis in hypotheses:
        row = dict(hypothesis=hypothesis["id"], window=hypothesis["window"],
                   metric=hypothesis["metric"], family=hypothesis.get("family", "default"),
                   predictor=hypothesis["predictor"], contrast=" - ".join(hypothesis["contrast"]),
                   n_rows=0, n_subjects=0, n_within_subjects=0, estimate=np.nan,
                   se=np.nan, ci_low=np.nan, ci_high=np.nan, p_raw=np.nan, p_adjusted=np.nan,
                   significant=False, model_reliable=False, error="", formula="")
        definition = METRIC_DEFINITIONS[hypothesis["metric"]]
        row["unit"] = definition.unit
        rows.append(row)
        if metrics.empty:
            row["error"] = "no_metrics"
            continue
        subset = metrics.loc[metrics.window_spec_id.eq(hypothesis["window"])
                             & metrics.metric.eq(hypothesis["metric"])
                             & metrics.moment_valid].copy()
        if definition.family == "pupil" and "pupil_measure" in subset and subset.pupil_measure.nunique() > 1:
            row["error"] = "mixed_pupil_measures_require_harmonization"
            continue
        predictor = hypothesis["predictor"]
        numeric = hypothesis.get("numeric_covariates", [])
        categorical = hypothesis.get("categorical_covariates", [])
        required = ["value", "participant_id", predictor, *numeric, *categorical]
        if set(required) - set(subset):
            row["error"] = "missing_model_columns:" + ",".join(sorted(set(required) - set(subset)))
            continue
        positive, reference = hypothesis["contrast"]
        subset = subset.loc[subset[predictor].isin([positive, reference])].copy()
        for column in ["value", *numeric]:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
            subset.loc[~np.isfinite(subset[column]), column] = np.nan
        for column in ["participant_id", predictor, *categorical]:
            subset[column] = subset[column].replace("", np.nan)
        subset = subset.dropna(subset=required).reset_index(drop=True)
        row.update(n_rows=len(subset), n_subjects=int(subset.participant_id.nunique()))
        row["n_within_subjects"] = int(subset.groupby("participant_id")[predictor].nunique().gt(1).sum())
        if (len(subset) < cfg.min_rows_for_model or row["n_subjects"] < cfg.min_subjects_for_model
                or subset[predictor].nunique() != 2):
            row["error"] = "insufficient_complete_data_or_levels"
            continue
        if row["n_within_subjects"] < cfg.min_subjects_for_model:
            row["error"] = "insufficient_within_subject_variation"
            continue
        if subset.value.std() < 1e-12:
            row["error"] = "constant_outcome"
            continue
        data = pd.DataFrame({"_value": subset.value, "_subject": subset.participant_id,
                             "_condition": pd.Categorical(subset[predictor], categories=[reference, positive])})
        rhs = ["C(_condition)", "C(_subject)"]
        if {"session_id", "run_id"}.issubset(subset):
            data["_run"] = subset.session_id.astype(str) + "/" + subset.run_id.astype(str)
            if data._run.nunique() > 1:
                rhs.append("C(_run)")
        for i, column in enumerate(numeric):
            name = f"_numeric_{i}"
            data[name] = subset[column].to_numpy()
            rhs.append(name)
        for i, column in enumerate(categorical):
            name = f"_categorical_{i}"
            data[name] = subset[column].to_numpy()
            if data[name].nunique() > 1:
                rhs.append(f"C({name})")
        row["formula"] = "_value ~ " + " + ".join(rhs)
        try:
            model = smf.ols(row["formula"], data, missing="raise")
            if np.linalg.matrix_rank(model.exog) < model.exog.shape[1] or len(data) <= model.exog.shape[1]:
                row["error"] = "rank_deficient_or_saturated_design"
                continue
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": data._subject, "use_correction": True}, use_t=True)
            term = next(name for name in fit.params.index if name.startswith("C(_condition)"))
            row.update(estimate=float(fit.params[term]), se=float(fit.bse[term]),
                       p_raw=float(fit.pvalues[term]), model_reliable=True)
            row["ci_low"], row["ci_high"] = map(float, fit.conf_int().loc[term])
            if not np.isfinite([row[k] for k in ("estimate", "se", "p_raw", "ci_low", "ci_high")]).all() or row["se"] <= 0:
                row.update(model_reliable=False, error="nonfinite_or_degenerate_inference")
        except (ValueError, np.linalg.LinAlgError, StopIteration) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for _, indices in result.groupby("family").groups.items():
        eligible = [i for i in indices if result.at[i, "model_reliable"]]
        if eligible:
            corrected = multipletests(result.loc[eligible, "p_raw"], method=cfg.fdr_method)[1]
            result.loc[eligible, "p_adjusted"] = corrected
            result.loc[eligible, "significant"] = corrected < cfg.alpha
    return result
