"""Analyses statistiques : blocs d'induction, modèles mixtes et FDR."""

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

# Les métriques QC / regard ne participent pas à la FDR pupillométrique.
PRIMARY_PUPIL_METRICS = {
    "pupil_mean",
    "pupil_sd",
    "pupil_baseline_corrected",
    "pupil_slope_per_s",
    "pupil_auc",
    "pupil_peak",
    "pupil_peak_latency_s",
}


def aggregate_to_induction_blocks(
    metrics_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrège les essais d'un même bloc d'induction.

    Les trois essais d'un bloc partagent la même induction émotionnelle.
    Pour M2/M3, ils sont donc moyennés avant l'analyse statistique.

    Niveau final :
        sujet × run × induction_block × moment × X × métrique.
    """
    valid = metrics_long.loc[
        metrics_long["moment_valid"]
        & metrics_long["value"].notna()
    ].copy()

    grouping_columns = [
        "subject_id",
        "run_id",
        "induction_block_id",
        "moment",
        "window_x_s",
        "metric",
        "group_5cat",
        "group_valence",
        "group_arousal",
    ]

    aggregation = {
        "value": "mean",
        "trial_n": "count",
        "missing_fraction_window": "mean",
        "actual_duration_s": "mean",
        "emo_grid_valence": "mean",
        "emo_grid_arousal": "mean",
        "LTask_rating": "mean",
    }

    block_data = (
        valid.groupby(
            grouping_columns,
            as_index=False,
            dropna=False,
        )
        .agg(aggregation)
        .rename(
            columns={
                "value": "value_block_mean",
                "trial_n": "n_valid_trials_in_block",
                "missing_fraction_window": "mean_missing_fraction",
                "actual_duration_s": "mean_window_duration_s",
            }
        )
    )

    block_data["value"] = block_data["value_block_mean"]

    return block_data


def _count_subjects_with_multiple_levels(
    data: pd.DataFrame,
    group_column: str,
) -> int:
    """Compte les sujets ayant au moins deux niveaux de condition."""
    levels = data.groupby("subject_id")[group_column].nunique()
    return int((levels >= 2).sum())


def _fit_mixedlm_with_retries(
    formula: str,
    data: pd.DataFrame,
) -> tuple[object | None, list[str], str | None]:
    """Ajuste MixedLM avec plusieurs optimiseurs."""
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
                return fit, messages, optimizer

        except Exception:
            continue

    return None, [], None


def _has_critical_warning(
    warning_messages: list[str],
) -> bool:
    """Détecte les warnings invalidant l'inférence Wald."""
    critical_patterns = [
        "Hessian matrix at the estimated parameter values is not positive definite",
        "Maximum Likelihood optimization failed to converge",
        "MixedLM optimization failed",
        "Gradient optimization failed",
    ]

    return any(
        any(
            pattern.lower() in message.lower()
            for pattern in critical_patterns
        )
        for message in warning_messages
    )


def _is_singular_random_effect(
    fit: object,
    tolerance: float = 1e-8,
) -> bool:
    """Indique si la variance inter-sujet est quasi nulle."""
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
    Ajuste un modèle mixte par schéma × moment × métrique × X.

    Modèle :
        value_z ~ C(condition) + C(run_id) + (1 | subject_id)

    Les analyses sont faites au niveau des blocs d'induction,
    pas au niveau des essais bruts.
    """
    results: list[dict] = []

    block_metrics = aggregate_to_induction_blocks(metrics_long)

    valid = block_metrics.loc[
        block_metrics["metric"].isin(PRIMARY_PUPIL_METRICS)
        & block_metrics["value"].notna()
    ].copy()

    for scheme_name, group_column in GROUP_SCHEMES.items():
        grouped = valid.groupby(
            ["moment", "metric", "window_x_s"],
            dropna=False,
        )

        for (moment, metric, window_x_s), data in grouped:
            data = data.dropna(
                subset=[
                    "subject_id",
                    "run_id",
                    group_column,
                    "value",
                ]
            ).copy()

            n_rows = len(data)
            n_subjects = data["subject_id"].nunique()
            n_levels = data[group_column].nunique()

            n_subjects_multilevel = (
                _count_subjects_with_multiple_levels(
                    data,
                    group_column,
                )
            )

            base = {
                "scheme": scheme_name,
                "group_column": group_column,
                "moment": moment,
                "metric": metric,
                "window_x_s": window_x_s,
                "n_induction_blocks": n_rows,
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
                "warning_messages": "",
                "error": "",
            }

            if n_rows < cfg.min_rows_for_model:
                base["error"] = "insufficient_induction_blocks"
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

            # Standardisation uniquement pour améliorer la stabilité numérique.
            data["value_z"] = (
                data["value"] - data["value"].mean()
            ) / value_sd

            formula = f"value_z ~ C({group_column}) + C(run_id)"

            fit, messages, optimizer = _fit_mixedlm_with_retries(
                formula,
                data,
            )

            base["optimizer"] = optimizer or ""
            base["warning_messages"] = " | ".join(messages)

            if fit is None:
                base["error"] = "mixedlm_failed_to_converge"
                results.append(base)
                continue

            if _has_critical_warning(messages):
                base["error"] = "unreliable_hessian_or_convergence_warning"
                results.append(base)
                continue

            try:
                group_terms = [
                    term
                    for term in fit.params.index
                    if term.startswith(f"C({group_column})")
                ]

                if not group_terms:
                    base["error"] = "no_estimable_group_contrast"
                    results.append(base)
                    continue

                parameter_indices = [
                    fit.params.index.get_loc(term)
                    for term in group_terms
                ]

                restriction = np.zeros(
                    (len(parameter_indices), len(fit.params))
                )

                for row_index, parameter_index in enumerate(
                    parameter_indices
                ):
                    restriction[row_index, parameter_index] = 1.0

                try:
                    wald = fit.wald_test(
                        restriction,
                        scalar=True,
                    )
                except TypeError:
                    wald = fit.wald_test(restriction)

                p_raw = float(np.asarray(wald.pvalue).squeeze())

                parameters = fit.params[group_terms]
                standard_errors = fit.bse[group_terms]

                strongest_term = parameters.abs().idxmax()

                estimate = float(parameters[strongest_term])
                standard_error = float(
                    standard_errors[strongest_term]
                )

                base.update(
                    {
                        "model_success": True,
                        "model_reliable": True,
                        "random_effect_singular": (
                            _is_singular_random_effect(fit)
                        ),
                        "p_raw": p_raw,
                        "strongest_contrast": strongest_term,
                        "effect_size_std_max_contrast": estimate,
                        "ci_low_std": (
                            estimate - 1.96 * standard_error
                        ),
                        "ci_high_std": (
                            estimate + 1.96 * standard_error
                        ),
                        "aic": float(fit.aic),
                    }
                )

            except Exception as exc:
                base["error"] = (
                    f"post_fit_error: {type(exc).__name__}: {exc}"
                )
                base["model_success"] = False
                base["model_reliable"] = False

            results.append(base)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    # FDR par schéma.
    # Chaque famille inclut :
    # moments × métriques × toutes les valeurs X.
    for _, indices in results_df.groupby("scheme").groups.items():
        subset = results_df.loc[indices].copy()

        eligible = (
            subset["model_success"]
            & subset["model_reliable"]
            & subset["p_raw"].notna()
        )

        if not eligible.any():
            continue

        rejected, p_corrected, _, _ = multipletests(
            subset.loc[eligible, "p_raw"].to_numpy(),
            alpha=cfg.alpha,
            method=cfg.fdr_method,
        )

        eligible_indices = subset.loc[eligible].index

        results_df.loc[
            eligible_indices,
            "p_fdr",
        ] = p_corrected

        results_df.loc[
            eligible_indices,
            "significant_fdr",
        ] = rejected

    return results_df