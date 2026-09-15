"""Modèles mixtes exploratoires robustes et correction FDR."""

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


def _count_subjects_with_multiple_levels(
    data: pd.DataFrame,
    group_col: str,
) -> int:
    """Compte les sujets possédant au moins deux niveaux de groupe."""
    levels_per_subject = data.groupby("subject_id")[group_col].nunique()
    return int((levels_per_subject >= 2).sum())


def _fit_mixedlm_with_retries(
    formula: str,
    data: pd.DataFrame,
) -> tuple[object | None, list[str], str | None]:
    """
    Ajuste MixedLM avec plusieurs optimiseurs.

    Les warnings retournés correspondent uniquement au modèle retenu,
    pas aux essais précédents échoués.
    """
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
                    disp=False,
                    maxiter=500,
                )

            warning_messages = [str(w.message) for w in caught]

            if bool(fit.converged):
                return fit, warning_messages, optimizer

        except Exception:
            continue

    return None, [], None


def _has_critical_warning(warning_messages: list[str]) -> bool:
    """Détermine si les warnings invalident les p-values Wald."""
    critical_patterns = [
        "Hessian matrix at the estimated parameter values is not positive definite",
        "Maximum Likelihood optimization failed to converge",
        "MixedLM optimization failed",
        "Gradient optimization failed",
    ]

    return any(
        any(pattern.lower() in message.lower() for pattern in critical_patterns)
        for message in warning_messages
    )


def _is_singular_random_effect(
    fit: object,
    tolerance: float = 1e-8,
) -> bool:
    """Indique si la variance aléatoire sujet est quasi nulle."""
    try:
        covariance = np.asarray(fit.cov_re, dtype=float)

        if covariance.size == 0:
            return True

        diagonal = np.diag(covariance)

        return bool(
            np.any(~np.isfinite(diagonal))
            or np.any(diagonal < tolerance)
        )

    except Exception:
        return True


def fit_mixed_models(
    metrics_long: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Ajuste un modèle mixte pour chaque schéma x moment x métrique.

    Modèle:
        value_z ~ C(group) + C(run_id) + (1 | subject_id)

    La standardisation est seulement utilisée pour la stabilité numérique.
    Les tailles d'effet sont exprimées en unités d'écart-type.
    """
    results: list[dict] = []

    valid = metrics_long.loc[
        metrics_long["moment_valid"]
        & metrics_long["value"].notna()
    ].copy()

    for scheme_name, group_col in GROUP_SCHEMES.items():
        for (moment, metric), data in valid.groupby(["moment", "metric"]):
            data = data.dropna(
                subset=["subject_id", "run_id", group_col, "value"]
            ).copy()

            n_rows = len(data)
            n_subjects = data["subject_id"].nunique()
            n_levels = data[group_col].nunique()

            n_subjects_multilevel = _count_subjects_with_multiple_levels(
                data,
                group_col,
            )

            base = {
                "scheme": scheme_name,
                "group_column": group_col,
                "moment": moment,
                "metric": metric,
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "n_levels": n_levels,
                "n_subjects_with_multiple_group_levels": n_subjects_multilevel,
                "model_success": False,
                "model_reliable": False,
                "optimizer": "",
                "random_effect_singular": np.nan,
                "p_raw": np.nan,
                "p_fdr": np.nan,
                "significant_fdr": False,
                "effect_size_std_max_contrast": np.nan,
                "ci_low_std": np.nan,
                "ci_high_std": np.nan,
                "aic": np.nan,
                "n_obs": np.nan,
                "warning_messages": "",
                "error": "",
            }

            if n_rows < cfg.min_rows_for_model:
                base["error"] = "insufficient_rows"
                results.append(base)
                continue

            if n_subjects < cfg.min_subjects_for_model:
                base["error"] = "insufficient_subjects"
                results.append(base)
                continue

            if n_levels < cfg.min_levels_for_model:
                base["error"] = "insufficient_group_levels"
                results.append(base)
                continue

            if n_subjects_multilevel < 2:
                base["error"] = "insufficient_within_subject_group_variation"
                results.append(base)
                continue

            value_sd = float(data["value"].std(ddof=1))

            if not np.isfinite(value_sd) or value_sd <= 1e-12:
                base["error"] = "near_constant_metric"
                results.append(base)
                continue

            data["value_z"] = (
                data["value"] - data["value"].mean()
            ) / value_sd

            formula = f"value_z ~ C({group_col}) + C(run_id)"

            fit, warning_messages, optimizer = _fit_mixedlm_with_retries(
                formula=formula,
                data=data,
            )

            base["optimizer"] = optimizer or ""
            base["warning_messages"] = " | ".join(warning_messages)

            if fit is None:
                base["error"] = "mixedlm_failed_to_converge"
                results.append(base)
                continue

            if _has_critical_warning(warning_messages):
                base["error"] = "unreliable_hessian_or_convergence_warning"
                results.append(base)
                continue

            if not bool(fit.converged):
                base["error"] = "fit_not_converged"
                results.append(base)
                continue

            try:
                group_terms = [
                    name
                    for name in fit.params.index
                    if name.startswith(f"C({group_col})")
                ]

                if not group_terms:
                    base["error"] = "no_estimable_group_contrast"
                    results.append(base)
                    continue

                parameter_indices = [
                    fit.params.index.get_loc(term)
                    for term in group_terms
                ]

                restriction_matrix = np.zeros(
                    (len(parameter_indices), len(fit.params))
                )

                for row_index, parameter_index in enumerate(parameter_indices):
                    restriction_matrix[row_index, parameter_index] = 1.0

                try:
                    wald = fit.wald_test(
                        restriction_matrix,
                        scalar=True,
                    )
                except TypeError:
                    wald = fit.wald_test(restriction_matrix)

                p_raw = float(np.asarray(wald.pvalue).squeeze())

                parameters = fit.params[group_terms]
                standard_errors = fit.bse[group_terms]

                strongest_term = parameters.abs().idxmax()
                estimate_z = float(parameters[strongest_term])
                se_z = float(standard_errors[strongest_term])

                base.update(
                    {
                        "model_success": True,
                        "model_reliable": True,
                        "random_effect_singular": _is_singular_random_effect(fit),
                        "p_raw": p_raw,
                        "strongest_contrast": strongest_term,
                        "effect_size_std_max_contrast": estimate_z,
                        "ci_low_std": estimate_z - 1.96 * se_z,
                        "ci_high_std": estimate_z + 1.96 * se_z,
                        "aic": float(fit.aic),
                        "n_obs": int(fit.nobs),
                    }
                )

            except Exception as exc:
                base["error"] = (
                    f"post_fit_error: {type(exc).__name__}: {exc}"
                )
                base["model_success"] = False
                base["model_reliable"] = False

            results.append(base)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # FDR par schéma : 5 catégories / valence / arousal.
    for _, indices in result_df.groupby("scheme").groups.items():
        subset = result_df.loc[indices].copy()

        eligible = (
            subset["model_success"]
            & subset["model_reliable"]
            & subset["p_raw"].notna()
        )

        if not eligible.any():
            continue

        rejected, corrected_pvalues, _, _ = multipletests(
            subset.loc[eligible, "p_raw"].to_numpy(),
            alpha=cfg.alpha,
            method=cfg.fdr_method,
        )

        eligible_indices = subset.loc[eligible].index

        result_df.loc[eligible_indices, "p_fdr"] = corrected_pvalues
        result_df.loc[eligible_indices, "significant_fdr"] = rejected

    return result_df