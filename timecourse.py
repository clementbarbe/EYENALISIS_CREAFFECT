"""Courbes alignées : bins attendus complets et QC partagé avec les métriques."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quality import evaluate_window, qc_segment


def make_timecourses(windows, samples, cfg, target_hz=20, moments=None):
    if not np.isfinite(target_hz) or target_hz <= 0 or target_hz > cfg.sampling_rate_hz:
        raise ValueError("Invalid timecourse frequency")
    if windows.empty or samples.empty:
        return pd.DataFrame()
    if moments is None:
        moments = ["M1_early_photometric", "H1_induction_late", "H2_pre_idea"]
    selection = windows.loc[windows.moment.isin(moments)]
    rows = []
    for _, meta in selection.iterrows():
        assessed = evaluate_window(meta, samples, cfg)
        beg, end = meta.start_eyelink_ms, meta.end_eyelink_ms
        if not np.isfinite(beg) or not np.isfinite(end) or end <= beg:
            continue
        chunk = assessed["window"]
        time = chunk.timestamp_ms.to_numpy(dtype=float)
        clock_slope = float(meta.get("clock_slope", 1.0))
        width = 1000.0 / target_hz * clock_slope
        if "anchor_eyelink_ms" in meta:
            anchor, name = meta.anchor_eyelink_ms, meta.moment
        elif meta.moment in {"M1_early_photometric", "H1_induction_late"}:
            offset = cfg.induction_early_start_s if meta.moment.startswith("M1") else cfg.induction_late_start_s
            anchor, name = beg - offset * 1000.0, "induction"
        else:
            anchor, name = end, "pre_idea"
        for k in range(int(np.ceil((end - beg) / width - 1e-10))):
            left, right = beg + k * width, min(end, beg + (k + 1) * width)
            a, b = np.searchsorted(time, [left, right], side="left")
            data = chunk.iloc[a:b]
            q = qc_segment(data, left, right, cfg)
            reasons = assessed["pupil_reasons"].copy()
            if q["observed_valid_fraction"] < cfg.min_observed_fraction_window:
                reasons.append("low_observed_bin_fraction")
            if q["sampling_coverage_fraction"] < 1 - cfg.max_window_missing_sampling_fraction:
                reasons.append("missing_bin_sampling_coverage")
            val = (float(data.loc[data.pupil_observed_valid, "pupil_clean"].mean())
                   if not reasons else np.nan)
            baseline_valid = not assessed["baseline_reasons"]
            delta = val - assessed["baseline_value"] if baseline_valid else np.nan
            context = {key: meta.get(key, np.nan) for key in (
                "project_id", "participant_id", "subject_id", "session_id", "run_id", "trial_id",
                "trial_n", "window_id", "window_spec_id", "analysis_variant_id",
                "condition", "group_5cat", "group_valence", "group_arousal")}
            rows.append({**context, "moment": name, "original_window": meta.moment,
                         "relative_time_s": round(((left + right) / 2 - anchor) / (1000.0 * clock_slope), 9),
                         "bin_start_eyelink_ms": left, "bin_end_eyelink_ms": right,
                         "pupil_observed_mean": val, "pupil_delta": delta,
                         "baseline_valid": baseline_valid,
                         "baseline_exclusion_reason": ";".join(assessed["baseline_reasons"]),
                         "observed_fraction_bin": q["observed_valid_fraction"],
                         "sampling_coverage_fraction_bin": q["sampling_coverage_fraction"],
                         "n_samples_bin": q["n_samples"], "expected_samples_bin": q["expected_samples"],
                         "window_valid": not assessed["pupil_reasons"],
                         "bin_valid": not reasons,
                         "exclusion_reason": ";".join(dict.fromkeys(reasons))})
    return pd.DataFrame(rows)
