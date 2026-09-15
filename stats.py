"""Modèles mixtes exploratoires robustes et correction FDR."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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
    """
    Nombre de sujets ayant au moins deux niveaux du facteur de groupe.

    Un effet de catégorie dans un modèle à intercept aléatoire sujet est
    beaucoup plus informatif si les comparaisons existent effectivement
    à l'intérieur des sujets.
    """
    levels_per_subject = data.groupby("subject_id")[group_col].nunique()
    return int((levels_per_subject >= 2).sum())


def _fit_mixedlm_with_retries(
    formula: str,
    data: pd.DataFrame,
) -> tuple[object | None, list[str], str | None]:
    """
    Ajuste un MixedLM avec plusieurs optimiseurs.

    Returns
    -------
    fit:
        Résultat Statsmodels ou None si tous les essais échouent.
    warning_messages:
        Liste des warnings capturés pendant l'ajustement.
    optimizer_used:
        Optimiseur ayant produit le résultat retenu.
    """
    optimizers = ["lbfgs", "bfgs", "cg", "powell"]
    all_warning_messages: list[str] = []

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

            current_warnings = [str(w.message) for w in caught]
            all_warning_messages.extend(current_warnings)

            if bool(fit.converged):
                return fit, all_warning_messages, optimizer

        except Exception as exc:
            all_warning_messages.append(
                f"{optimizer}: {type(exc).__name__}: {exc}"
            )

    return None, all_warning_messages, None


def _has_unreliable_hessian_or_convergence_warning(
    warning_messages: list[str],
) -> bool:
    """
    Détermine si les warnings invalident l'inférence Wald.

    Une covariance singulière seule est conservée comme drapeau de QC :
    elle peut indiquer une variance inter-sujet estimée à zéro.
    En revanche, Hessienne non positive définie ou absence de convergence
    invalident les SE/p-values et entraînent l'exclusion du modèle.
    """
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


def _is_singular_random_effect(fit: object, tolerance: float = 1e-8) -> bool:
    """
    Indique si la variance de l'intercept aléatoire est quasi nulle.

    Ce statut est rapporté en QC mais n'invalide pas systématiquement le modèle
    si celui-ci a convergé sans problème de Hessienne.
    """
    try:
        covariance = np.asarray(fit.cov_re, dtype=float)
        if covariance.size == 0:
            return True

        diagonal = np.diag(covariance)
        return bool(np.any(~np.isfinite(diagonal)) or np.any(diagonal < tolerance))

    except Exception:
        return True


def fit_mixed_models(metrics_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Ajuste un modèle mixte pour chaque schéma x moment x métrique.

    Modèle principal :
        standardized_value ~ C(group) + C(run_id) + (1 | subject_id)

    Notes
    -----
    - La standardisation est effectuée séparément pour chaque combinaison
      moment x métrique x schéma, uniquement pour améliorer la stabilité
      numérique de l'optimisation.
    - Les p-values reportées correspondent au test Wald global du facteur
      de groupe.
    - Les modèles avec Hessienne non positive définie ou non-convergence
      sont marqués invalides : aucune p-value ne leur est attribuée.
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

            # Critères minimaux avant de lancer un modèle.
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

            # Recommandé : ne pas interpréter une comparaison qui n'existe
            # presque jamais intra-sujet.
            if n_subjects_multilevel < 2:
                base["error"] = "insufficient_within_subject_group_variation"
                results.append(base)
                continue

            # Une métrique constante rend le modèle non identifiable.
            value_sd = float(data["value"].std(ddof=1))
            if not np.isfinite(value_sd) or value_sd <= 1e-12:
                base["error"] = "near_constant_metric"
                results.append(base)
                continue

            # Standardisation pour optimiser numériquement le MixedLM.
            data["value_z"] = (
                data["value"] - data["value"].mean()
            ) / value_sd

            # C(run_id) est conservé comme covariable de contrôle.
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

            if _has_unreliable_hessian_or_convergence_warning(warning_messages):
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

                for row_idx, parameter_idx in enumerate(parameter_indices):
                    restriction_matrix[row_idx, parameter_idx] = 1.0

                # scalar=True évite le FutureWarning Statsmodels.
                wald = fit.wald_test(restriction_matrix, scalar=True)
                p_raw = float(wald.pvalue)

                params = fit.params[group_terms]
                ses = fit.bse[group_terms]

                strongest_term = params.abs().idxmax()
                estimate_z = float(params[strongest_term])
                se_z = float(ses[strongest_term])

                base.update({
                    "model_success": True,
                    "model_reliable": True,
                    "random_effect_singular": _is_singular_random_effect(fit),
                    "p_raw": p_raw,
                    "strongest_contrast": strongest_term,
                    # Comme la variable dépendante est standardisée,
                    # le contraste est directement une taille d'effet standardisée.
                    "effect_size_std_max_contrast": estimate_z,
                    "ci_low_std": estimate_z - 1.96 * se_z,
                    "ci_high_std": estimate_z + 1.96 * se_z,
                    "aic": float(fit.aic),
                    "n_obs": int(fit.nobs),
                })

            except Exception as exc:
                base["error"] = f"post_fit_error: {type(exc).__name__}: {exc}"
                base["model_success"] = False
                base["model_reliable"] = False

            results.append(base)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # Correction FDR : une famille par schéma de comparaison.
    for scheme_name, indices in result_df.groupby("scheme").groups.items():
        subset = result_df.loc[indices].copy()

        eligible = (
            subset["model_success"]
            & subset["model_reliable"]
            & subset["p_raw"].notna()
        )

        if not eligible.any():
            continue

        rejected, corrected, _, _ = multipletests(
            subset.loc[eligible, "p_raw"].to_numpy(),
            alpha=cfg.alpha,
            method=cfg.fdr_method,
        )

        eligible_indices = subset.loc[eligible].index
        result_df.loc[eligible_indices, "p_fdr"] = corrected
        result_df.loc[eligible_indices, "significant_fdr"] = rejected

    return result_df