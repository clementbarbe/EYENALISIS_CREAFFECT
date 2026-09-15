"""Modèles mixtes exploratoires et correction FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from config import Config


GROUP_SCHEMES = {
    "five_categories": "group_5cat",
    "valence": "group_valence",
    "arousal": "group_arousal",
}


def fit_mixed_models(metrics_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Ajuste un MixedLM par schéma x moment x métrique.

    Modèle : value ~ C(group), intercept aléatoire sujet.
    Le run est inclus comme effet fixe de contrôle lorsque possible.
    """
    results: list[dict] = []

    valid = metrics_long.loc[
        metrics_long["moment_valid"] & metrics_long["value"].notna()
    ].copy()

    for scheme_name, group_col in GROUP_SCHEMES.items():
        for (moment, metric), data in valid.groupby(["moment", "metric"]):
            data = data.dropna(subset=[group_col, "value", "subject_id"])
            n_rows = len(data)
            n_subjects = data["subject_id"].nunique()
            n_levels = data[group_col].nunique()

            base = {
                "scheme": scheme_name,
                "group_column": group_col,
                "moment": moment,
                "metric": metric,
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "n_levels": n_levels,
                "model_success": False,
                "p_raw": np.nan,
                "effect_size_std_max_contrast": np.nan,
                "ci_low_std": np.nan,
                "ci_high_std": np.nan,
                "error": "",
            }

            if (
                n_rows < cfg.min_rows_for_model
                or n_subjects < cfg.min_subjects_for_model
                or n_levels < cfg.min_levels_for_model
            ):
                base["error"] = "insufficient_data"
                results.append(base)
                continue

            try:
                formula = f"value ~ C({group_col}) + C(run_id)"
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data["subject_id"],
                    re_formula="1",
                )
                fit = model.fit(reml=False, method="lbfgs", disp=False)

                group_terms = [
                    name for name in fit.params.index
                    if name.startswith(f"C({group_col})")
                ]

                if not group_terms:
                    base["error"] = "no_estimable_group_contrast"
                    results.append(base)
                    continue

                params = fit.params[group_terms]
                pvalues = fit.pvalues[group_terms]

                # Test global Wald sur les contrastes de groupe.
                indices = [fit.params.index.get_loc(term) for term in group_terms]
                r = np.zeros((len(indices), len(fit.params)))
                for i, idx in enumerate(indices):
                    r[i, idx] = 1.0

                wald = fit.wald_test(r)
                p_raw = float(np.asarray(wald.pvalue).squeeze())

                residual_sd = float(np.sqrt(fit.scale))
                strongest = params.abs().idxmax()
                estimate = float(params[strongest])
                se = float(fit.bse[strongest])

                base.update({
                    "model_success": True,
                    "p_raw": p_raw,
                    "strongest_contrast": strongest,
                    "effect_size_std_max_contrast": estimate / residual_sd,
                    "ci_low_std": (estimate - 1.96 * se) / residual_sd,
                    "ci_high_std": (estimate + 1.96 * se) / residual_sd,
                    "aic": float(fit.aic),
                    "converged": bool(fit.converged),
                    "n_obs": int(fit.nobs),
                })

            except Exception as exc:
                base["error"] = f"{type(exc).__name__}: {exc}"

            results.append(base)

    result_df = pd.DataFrame(results)

    # Une famille FDR = un schéma de comparaison.
    result_df["p_fdr"] = np.nan
    result_df["significant_fdr"] = False

    for scheme, index in result_df.groupby("scheme").groups.items():
        subset = result_df.loc[index]
        ok = subset["model_success"] & subset["p_raw"].notna()

        if ok.any():
            rejected, corrected, _, _ = multipletests(
                subset.loc[ok, "p_raw"],
                alpha=cfg.alpha,
                method=cfg.fdr_method,
            )
            result_df.loc[subset.loc[ok].index, "p_fdr"] = corrected
            result_df.loc[subset.loc[ok].index, "significant_fdr"] = rejected

    return result_df
