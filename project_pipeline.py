"""Exécution d'un projet déclaratif, sans conventions de nommage CREAFFECT."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from event_windows import build_project_windows, fit_clock, read_table, read_task_events
from metrics import extract_metrics, metric_catalog
from parsing_eyetracking import parse_asc
from preprocessing import preprocess_pupil
from project_config import load_project
from project_models import run_project_models
from qc_plots import plot_run_qc
from timecourse import make_timecourses


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path):
    manifest = read_table(path)
    required = {"participant_id", "run_id", "asc_path", "events_path"}
    if manifest.empty or not required.issubset(manifest):
        raise ValueError("Manifest requires participant_id, run_id, asc_path, events_path and at least one row")
    if "session_id" not in manifest:
        manifest["session_id"] = "1"
    for key in [*required, "session_id"]:
        if manifest[key].eq("").any():
            raise ValueError(f"Empty manifest {key}")
    if manifest.duplicated(["participant_id", "session_id", "run_id"]).any():
        raise ValueError("Duplicate participant/session/run in manifest")
    for field in ("asc_path", "events_path"):
        for value in manifest[field]:
            if not (path.parent / value).is_file():
                raise FileNotFoundError(f"Missing {field}: {path.parent / value}")
    return manifest


def _recording_config(cfg, parameters, recording, samples):
    metadata = samples.attrs["eyelink_metadata"]
    fs = metadata["sampling_rate_hz"]
    if samples.empty or not np.isfinite(fs) or fs <= 0:
        raise ValueError("No samples or invalid ASC sampling rate")
    if "sampling_rate_hz" in parameters and not np.isclose(fs, cfg.sampling_rate_hz):
        raise ValueError("Configured sampling frequency differs from ASC header")
    if len(samples) > 1 and not np.isclose(1000 / np.median(np.diff(samples.timestamp_ms)), fs, rtol=0.02):
        raise ValueError("Sample timestamps differ from ASC sampling frequency")
    updates = {"sampling_rate_hz": fs, "eye": metadata["selected_eye"]}
    coords = metadata.get("gaze_coords")
    if coords is not None:
        if coords[0] != 0 or coords[1] != 0 or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError("Only screen coordinates with origin (0, 0) are supported")
        for key, value in (("gaze_screen_width_px", coords[2] + 1), ("gaze_screen_height_px", coords[3] + 1)):
            if key in parameters and not np.isclose(value, getattr(cfg, key)):
                raise ValueError(f"Configured {key} differs from ASC GAZE_COORDS")
            updates[key] = value
    elif not {"gaze_screen_width_px", "gaze_screen_height_px"}.issubset(parameters):
        raise ValueError("ASC needs GAZE_COORDS or explicit screen width and height")
    local = replace(cfg, **updates)
    if local.lowpass_enabled and local.lowpass_cutoff_hz >= fs / 2:
        raise ValueError("Pupil lowpass cutoff must be below Nyquist frequency")
    return local


def _write_report(root, project, recording_qc, metrics, models, figures):
    summary = (metrics.groupby(["window_spec_id", "metric"], dropna=False)
               .agg(n_windows=("value", "size"), n_valid=("moment_valid", "sum"),
                    n_participants=("participant_id", "nunique"))) if not metrics.empty else pd.DataFrame()
    contents = ["<!doctype html><html lang='fr'><meta charset='utf-8'>",
                "<title>Rapport eye-tracking</title><style>body{font-family:system-ui;margin:2rem;max-width:1300px}"
                "table{border-collapse:collapse;font-size:.85rem}td,th{border:1px solid #ddd;padding:.4rem}"
                "img{max-width:100%}details{margin:1rem 0}</style>",
                f"<h1>{html.escape(project['project_id'])}</h1>",
                "<p>Rapport exploratoire. Les exclusions et prérequis sont conservés dans les tableaux. "
                "La configuration de prétraitement doit être validée pour chaque protocole.</p>",
                "<h2>Acquisitions</h2>", recording_qc.to_html(index=False, escape=True),
                "<h2>Fenêtres et métriques</h2>", summary.to_html(escape=True)]
    if not models.empty:
        contents.extend(["<h2>Contrastes déclarés</h2>", models.to_html(index=False, escape=True)])
    for filename in figures:
        contents.append(f"<details><summary>{html.escape(filename)}</summary>"
                        f"<img src='figures/{html.escape(filename, quote=True)}' alt='Trace QC'></details>")
    contents.append("</html>")
    (root / "report.html").write_text("\n".join(contents), encoding="utf-8")


def run_project(project_path, output_root=None, *, no_stats=False, no_timecourses=False, qc_plots=False):
    project_path = Path(project_path).resolve()
    project, cfg = load_project(project_path)
    manifest_path = (project_path.parent / project["recordings_manifest"]).resolve()
    manifest = _read_manifest(manifest_path)
    root = Path(output_root).resolve() if output_root is not None else project_path.parent / "results"
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new directory: {root}")
    for directory in (root, root / "tables", root / "logs", root / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "seaborn", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    provenance = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                      project_sha256=_hash(project_path), manifest_sha256=_hash(manifest_path),
                      dependency_versions=versions,
                      code_sha256={p.name: _hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                      recordings=[])
    (root / "config_used.json").write_text(json.dumps({"project": project, "defaults": asdict(cfg)},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.to_csv(root / "tables" / "recordings_manifest.csv", index=False)
    metric_catalog().to_csv(root / "tables" / "metric_catalog.csv", index=False)
    all_windows, all_metrics, all_timecourses, qc_rows, figures = [], [], [], [], []
    for index, recording in enumerate(manifest.to_dict("records"), start=1):
        print(f"[{index}/{len(manifest)}] {recording['participant_id']} / {recording['session_id']} / {recording['run_id']}", flush=True)
        qc = {key: recording[key] for key in ("participant_id", "session_id", "run_id")}
        qc.update(recording_processed=False, error="")
        record_provenance = {**qc}
        try:
            asc_path = (manifest_path.parent / recording["asc_path"]).resolve()
            event_path = (manifest_path.parent / recording["events_path"]).resolve()
            samples, ocular = parse_asc(asc_path, eye=recording.get("eye") or cfg.eye, strict=True)
            local_cfg = _recording_config(cfg, project.get("parameters", {}), recording, samples)
            clock = fit_clock(recording, project["synchronization"], manifest_path.parent)
            task = read_task_events(event_path, project.get("events", {}), recording)
            windows = build_project_windows(task, project, recording, clock, local_cfg)
            cleaned, blinks = preprocess_pupil(samples, ocular, local_cfg)
            metrics = extract_metrics(windows, cleaned, blinks, ocular, local_cfg)
            metadata = samples.attrs["eyelink_metadata"]
            for frame in (windows, metrics):
                frame["recorded_eye"] = metadata["selected_eye"]
                frame["sampling_rate_hz"] = local_cfg.sampling_rate_hz
                frame["pupil_measure"] = metadata["pupil_measure"]
                frame["pupil_unit"] = "EyeLink_arbitrary_unit"
            window_keys = ["participant_id", "session_id", "run_id", "trial_id", "window_id", "anchor_event_id"]
            window_quality = ["pupil_window_valid", "gaze_window_valid", "baseline_valid",
                              "observed_valid_fraction", "sampling_coverage_fraction", "longest_gap_ms"]
            windows = windows.merge(metrics[window_keys + window_quality].drop_duplicates(window_keys),
                                    on=window_keys, how="left", validate="one_to_one")
            timecourses = pd.DataFrame()
            selection = project.get("timecourses", {}).get("windows", [])
            if selection and not no_timecourses:
                selected = windows.loc[windows.window_spec_id.isin(selection)]
                width = project.get("timecourses", {}).get("bin_width_s", 0.05)
                timecourses = make_timecourses(selected, cleaned, local_cfg,
                                               target_hz=1.0 / width, moments=selected.moment.unique())
            if qc_plots:
                name = f"recording-{index:04}"
                plot_run_qc(cleaned, name, root / "figures")
                figures.append(f"{name}_pupil_QC.png")
            qc.update(recording_processed=True, n_windows=len(windows), n_metric_rows=len(metrics),
                      n_valid_metric_rows=int(metrics.moment_valid.sum()),
                      observed_pupil_fraction=float(cleaned.pupil_observed_valid.mean()),
                      interpolated_fraction=float(cleaned.pupil_interpolated.mean()),
                      gaze_valid_fraction=float(cleaned.gaze_valid.mean()),
                      clock_offset_ms=clock.offset_ms, clock_slope=clock.slope,
                      clock_drift_ppm=(clock.slope - 1) * 1e6 if clock.n_anchors >= 2 else np.nan,
                      clock_quality=clock.quality,
                      n_clock_anchors=clock.n_anchors, max_clock_residual_ms=clock.max_residual_ms)
            record_provenance.update(asc=str(asc_path), asc_metadata=metadata,
                                     events=str(event_path), events_sha256=_hash(event_path),
                                     clock=asdict(clock), parameters=asdict(local_cfg))
            if recording.get("sync_path"):
                record_provenance["sync_sha256"] = _hash(manifest_path.parent / recording["sync_path"])
            all_windows.append(windows)
            all_metrics.append(metrics)
            if not timecourses.empty:
                all_timecourses.append(timecourses)
        except Exception as exc:
            qc["recording_processed"] = False
            qc["error"] = f"{type(exc).__name__}: {exc}"
            (root / "logs" / f"recording-{index:04}.log").write_text(traceback.format_exc(), encoding="utf-8")
        record_provenance.update(recording_processed=qc["recording_processed"], error=qc["error"])
        provenance["recordings"].append(record_provenance)
        qc_rows.append(qc)
    windows = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    courses = pd.concat(all_timecourses, ignore_index=True) if all_timecourses else pd.DataFrame()
    recording_qc = pd.DataFrame(qc_rows)
    models = run_project_models(metrics, project.get("hypotheses", []), cfg) if not no_stats else pd.DataFrame()
    for filename, frame in (("windows.csv", windows.drop(columns="metrics", errors="ignore")),
                            ("metrics_long.csv", metrics), ("recording_qc.csv", recording_qc),
                            ("timecourses.csv", courses), ("hypothesis_models.csv", models)):
        if not len(frame.columns):
            frame = pd.DataFrame(columns={"windows.csv": ["participant_id", "window_id"],
                                          "metrics_long.csv": ["participant_id", "window_id", "metric", "value"],
                                          "timecourses.csv": ["participant_id", "window_id", "relative_time_s", "pupil_delta"],
                                          "hypothesis_models.csv": ["hypothesis", "metric", "n_subjects", "p_raw", "error"]}[filename])
        frame.to_csv(root / "tables" / filename, index=False)
    if not metrics.empty:
        metrics.groupby(["window_spec_id", "metric", "moment_exclusion_reason"], dropna=False).size().rename(
            "n_metric_windows").to_csv(root / "tables" / "metric_qc_summary.csv")
        dims = ["participant_id", "window_spec_id", "window_id", "analysis_variant_id", "metric", "unit", "pupil_measure"] + (["condition"] if "condition" in metrics else [])
        metrics.loc[metrics.moment_valid].groupby(dims, dropna=False).value.agg(
            ["count", "mean", "std"]).to_csv(root / "tables" / "descriptives.csv")
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(root, project, recording_qc, metrics, models, figures)
    return dict(output_dir=str(root), recordings=len(manifest),
                processed=int(recording_qc.recording_processed.sum()),
                errors=int((~recording_qc.recording_processed).sum()),
                windows=len(windows), metric_rows=len(metrics), timecourse_bins=len(courses))
