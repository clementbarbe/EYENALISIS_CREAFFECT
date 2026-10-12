"""Modèles mixtes historiques : sensibilité exploratoire, non confirmatoire."""

from __future__ import annotations

import warnings

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


HYPOTHESES = {
    "H1_induction_emotion": {
        "moment": "H1_induction_late",
        "metrics": [
            "pupil_baseline_corrected",
            "pupil_mean",
            "saccade_rate_per_s",
            "all_saccade_rate_per_s",
        ],
        "schemes": [
            "five_categories",
            "valence",
            "arousal",
        ],
        "rating_model": False,
    },
    "H2_pre_idea": {
        "moment": "H2_pre_idea",
        "metrics": [
            "pupil_baseline_corrected",
            "pupil_mean",
        ],
        "schemes": [
            "five_categories",
            "valence",
            "arousal",
        ],
        "rating_model": False,
    },
    "H3_liking": {
        "moment": "H3_liking",
        "metrics": [
            "pupil_baseline_corrected",
            "pupil_mean",
        ],
        "schemes": [
            "five_categories",
        ],
        "rating_model": True,
    },
}


def _fit_mixed_model(
    formula: str,
    data: pd.DataFrame,
) -> tuple[object | None, str, list[str]]:
    """Ajuste un MixedLM avec plusieurs optimiseurs."""
    optimizers = ["lbfgs", "bfgs", "cg", "powell"]

    for optimizer in optimizers:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                model = smf.mixedlm(
                    formula=formula,
                    data=data,
                    groups=data["subject_id"],
                    re_formula="1",
                )

                fit = model.fit(
                    reml=False,
                    method=optimizer,
                    maxiter=500,
                    disp=False,
                )

            messages = [str(w.message) for w in caught]

            if bool(fit.converged):
                return fit, optimizer, messages

        except Exception:
            continue

    return None, "", []


def _has_critical_warning(messages: list[str]) -> bool:
    """Détecte les warnings rendant les p-values incertaines."""
    critical_patterns = [
        "hessian matrix",
        "optimization failed",
        "failed to converge",
        "gradient optimization failed",
    ]

    return any(
        any(pattern in message.lower() for pattern in critical_patterns)
        for message in messages
    )


def _global_factor_pvalue(
    fit: object,
    factor_name: str,
) -> tuple[float, str, float, float]:
    """Retourne p-value globale Wald et contraste le plus fort."""
    terms = [
        name
        for name in fit.params.index
        if name.startswith(f"C({factor_name})")
    ]

    if not terms:
        return np.nan, "", np.nan, np.nan

    indices = [
        fit.params.index.get_loc(term)
        for term in terms
    ]

    restriction = np.zeros(
        (len(indices), len(fit.params))
    )

    for i, parameter_index in enumerate(indices):
        restriction[i, parameter_index] = 1.0

    try:
        test = fit.wald_test(restriction, scalar=True)
    except TypeError:
        test = fit.wald_test(restriction)

    strongest_term = fit.params[terms].abs().idxmax()

    return (
        float(np.asarray(test.pvalue).squeeze()),
        strongest_term,
        float(fit.params[strongest_term]),
        float(fit.bse[strongest_term]),
    )


def _rating_pvalue(
    fit: object,
) -> tuple[float, float, float]:
    """Retourne p-value, coefficient et SE du rating liking."""
    if "LTask_rating_z" not in fit.params.index:
        return np.nan, np.nan, np.nan

    return (
        float(fit.pvalues["LTask_rating_z"]),
        float(fit.params["LTask_rating_z"]),
        float(fit.bse["LTask_rating_z"]),
    )


def run_confirmatory_models(
    metrics_long: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Lance les modèles mixtes de sensibilité H1, H2 et H3.

    H1:
        Catégorie émotionnelle -> pupille et saccades pendant induction tardive.

    H2:
        Catégorie émotionnelle -> pupille juste avant idée.

    H3:
        Pupille pendant évaluation liking ~ rating continu + catégorie.
    """
    rows: list[dict] = []

    valid = metrics_long.loc[
        metrics_long["moment_valid"]
        & metrics_long["value"].notna()
    ].copy()

    for hypothesis_name, spec in HYPOTHESES.items():
        for scheme_name in spec["schemes"]:
            group_column = GROUP_SCHEMES[scheme_name]

            subset = valid.loc[
                (valid["moment"] == spec["moment"])
                & valid["metric"].isin(spec["metrics"])
            ].copy()

            for metric_name, data in subset.groupby("metric"):
                required = [
                    "subject_id",
                    "run_id",
                    group_column,
                    "value",
                ]

                if spec["rating_model"]:
                    required.append("LTask_rating")

                data = data.dropna(subset=required).copy()

                result = {
                    "hypothesis": hypothesis_name,
                    "scheme": scheme_name,
                    "group_column": group_column,
                    "moment": spec["moment"],
                    "metric": metric_name,
                    "n_rows": len(data),
                    "n_subjects": data["subject_id"].nunique(),
                    "n_levels": data[group_column].nunique(),
                    "model_reliable": False,
                    "optimizer": "",
                    "p_group_raw": np.nan,
                    "p_group_fdr": np.nan,
                    "group_significant_fdr": False,
                    "strongest_group_contrast": "",
                    "group_effect_std": np.nan,
                    "group_ci_low_std": np.nan,
                    "group_ci_high_std": np.nan,
                    "p_rating_raw": np.nan,
                    "p_rating_fdr": np.nan,
                    "rating_significant_fdr": False,
                    "rating_effect_std": np.nan,
                    "rating_ci_low_std": np.nan,
                    "warning_messages": "",
                    "error": "",
                }

                if len(data) < cfg.min_rows_for_model:
                    result["error"] = "insufficient_rows"
                    rows.append(result)
                    continue

                if data["subject_id"].nunique() < cfg.min_subjects_for_model:
                    result["error"] = "insufficient_subjects"
                    rows.append(result)
                    continue

                if data[group_column].nunique() < cfg.min_levels_for_model:
                    result["error"] = "insufficient_group_levels"
                    rows.append(result)
                    continue

                value_sd = float(data["value"].std(ddof=1))

                if not np.isfinite(value_sd) or value_sd <= 1e-12:
                    result["error"] = "near_constant_metric"
                    rows.append(result)
                    continue

                data["value_z"] = (
                    data["value"] - data["value"].mean()
                ) / value_sd

                if spec["rating_model"]:
                    rating_sd = float(
                        data["LTask_rating"].std(ddof=1)
                    )

                    if (
                        not np.isfinite(rating_sd)
                        or rating_sd <= 1e-12
                    ):
                        result["error"] = "near_constant_rating"
                        rows.append(result)
                        continue

                    data["LTask_rating_z"] = (
                        data["LTask_rating"]
                        - data["LTask_rating"].mean()
                    ) / rating_sd

                    formula = (
                        f"value_z ~ C({group_column}) "
                        "+ LTask_rating_z + C(run_id)"
                    )
                else:
                    formula = (
                        f"value_z ~ C({group_column}) + C(run_id)"
                    )

                fit, optimizer, messages = _fit_mixed_model(
                    formula,
                    data,
                )

                result["optimizer"] = optimizer
                result["warning_messages"] = " | ".join(messages)

                if fit is None:
                    result["error"] = "mixedlm_failed"
                    rows.append(result)
                    continue

                if _has_critical_warning(messages):
                    result["error"] = "unreliable_hessian_or_convergence"
                    rows.append(result)
                    continue

                try:
                    (
                        p_group,
                        contrast,
                        estimate,
                        standard_error,
                    ) = _global_factor_pvalue(
                        fit,
                        group_column,
                    )

                    result.update(
                        {
                            "model_reliable": True,
                            "p_group_raw": p_group,
                            "strongest_group_contrast": contrast,
                            "group_effect_std": estimate,
                            "group_ci_low_std": (
                                estimate - 1.96 * standard_error
                            ),
                            "group_ci_high_std": (
                                estimate + 1.96 * standard_error
                            ),
                        }
                    )

                    if spec["rating_model"]:
                        (
                            p_rating,
                            rating_estimate,
                            rating_standard_error,
                        ) = _rating_pvalue(fit)

                        result.update(
                            {
                                "p_rating_raw": p_rating,
                                "rating_effect_std": rating_estimate,
                                "rating_ci_low_std": (
                                    rating_estimate
                                    - 1.96 * rating_standard_error
                                ),
                                "rating_ci_high_std": (
                                    rating_estimate
                                    + 1.96 * rating_standard_error
                                ),
                            }
                        )

                except Exception as exc:
                    result["model_reliable"] = False
                    result["error"] = (
                        f"post_fit_error: {type(exc).__name__}: {exc}"
                    )

                rows.append(result)

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    # Correction FDR pour les tests de catégorie.
    eligible_group = (
        results["model_reliable"]
        & results["p_group_raw"].notna()
    )

    if eligible_group.any():
        rejected, corrected, _, _ = multipletests(
            results.loc[eligible_group, "p_group_raw"],
            alpha=cfg.alpha,
            method=cfg.fdr_method,
        )

        indices = results.loc[eligible_group].index

        results.loc[indices, "p_group_fdr"] = corrected
        results.loc[indices, "group_significant_fdr"] = rejected

    # Correction FDR séparée pour les tests de rating H3.
    eligible_rating = (
        results["model_reliable"]
        & results["p_rating_raw"].notna()
    )

    if eligible_rating.any():
        rejected, corrected, _, _ = multipletests(
            results.loc[eligible_rating, "p_rating_raw"],
            alpha=cfg.alpha,
            method=cfg.fdr_method,
        )

        indices = results.loc[eligible_rating].index

        results.loc[indices, "p_rating_fdr"] = corrected
        results.loc[indices, "rating_significant_fdr"] = rejected

    return results