import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def fdr_bh(pvalues, alpha=0.05):
    pvalues = np.asarray(pvalues, dtype=float)

    valid = np.isfinite(pvalues)

    corrected = np.full_like(pvalues, np.nan, dtype=float)
    reject = np.full(len(pvalues), False, dtype=bool)

    if valid.sum() == 0:
        return corrected, reject

    p = pvalues[valid]
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    corrected_valid = np.empty(n)
    corrected_valid[order] = adjusted

    corrected[valid] = corrected_valid
    reject[valid] = corrected_valid < alpha

    return corrected, reject


def paired_friedman(subject_table, categories, value_col=None):
    available = [
        category for category in categories
        if category in subject_table.columns
    ]

    if len(available) < 2:
        return np.nan, np.nan, 0

    complete = subject_table[available].dropna()

    if len(complete) < 2:
        return np.nan, np.nan, len(complete)

    values = [
        complete[column].values
        for column in available
    ]

    try:
        statistic, pvalue = stats.friedmanchisquare(*values)
    except ValueError:
        statistic, pvalue = np.nan, np.nan

    return statistic, pvalue, len(complete)


def pairwise_wilcoxon(subject_table, categories, alpha=0.05):
    rows = []

    for condition_1, condition_2 in combinations(categories, 2):

        if (
            condition_1 not in subject_table.columns
            or condition_2 not in subject_table.columns
        ):
            continue

        pair = subject_table[
            [condition_1, condition_2]
        ].dropna()

        if len(pair) < 3:
            continue

        try:
            statistic, pvalue = stats.wilcoxon(
                pair[condition_1],
                pair[condition_2],
                alternative="two-sided",
            )
        except ValueError:
            statistic, pvalue = np.nan, np.nan

        rows.append({
            "condition_1": condition_1,
            "condition_2": condition_2,
            "n": len(pair),
            "stat": statistic,
            "p": pvalue,
        })

    output = pd.DataFrame(rows)

    if len(output):
        output["p_fdr"], output["significant_fdr"] = fdr_bh(
            output["p"].values,
            alpha=alpha,
        )

    return output


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def inverse_fisher_z(z):
    return np.tanh(z)


def within_subject_correlations(
    df,
    x_col,
    y_col,
    method="spearman",
):
    rows = []

    for participant, subject_df in df.groupby("participant"):

        values = subject_df[
            [x_col, y_col]
        ].dropna()

        if len(values) < 4:
            continue

        if (
            values[x_col].nunique() < 2
            or values[y_col].nunique() < 2
        ):
            continue

        if method == "spearman":
            r, pvalue = stats.spearmanr(
                values[x_col],
                values[y_col],
            )
        else:
            r, pvalue = stats.pearsonr(
                values[x_col],
                values[y_col],
            )

        rows.append({
            "participant": participant,
            "n_trials": len(values),
            "r": r,
            "p": pvalue,
            "z": fisher_z(r),
        })

    return pd.DataFrame(rows)


def group_test_fisher_correlations(corr_df):
    if corr_df is None or len(corr_df) < 3:
        return {
            "n": 0,
            "mean_r": np.nan,
            "mean_z": np.nan,
            "t": np.nan,
            "p": np.nan,
        }

    z = corr_df["z"].dropna().values

    t_stat, pvalue = stats.ttest_1samp(z, 0)

    mean_z = np.mean(z)

    return {
        "n": len(z),
        "mean_z": mean_z,
        "mean_r": inverse_fisher_z(mean_z),
        "t": t_stat,
        "p": pvalue,
    }