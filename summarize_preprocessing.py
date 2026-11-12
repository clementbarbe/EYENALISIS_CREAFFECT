"""Reconstruit un bilan de cohorte à partir des sorties de prétraitement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from preprocessing_summary import export_preprocessing_report


def summarize_existing(source, destination=None, *, figures=True):
    source = Path(source).resolve()
    if source.is_file():
        table = source
        root = source.parent.parent if source.parent.name == "tables" else source.parent
    else:
        root = source
        table = root / "tables" / "preprocessing_run_summary.csv"
    metadata_path = root / "preprocessing_report_config.json"
    if not metadata_path.is_file():
        raise ValueError("Bilan de prétraitement absent : relancer run_pipeline.py --preprocessing-only dans un nouveau dossier.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ids = {"participant_id": str, "subject_id": str, "session_id": str, "run_id": str}
    runs = pd.read_csv(table, dtype=ids)
    subject_path = root / "tables" / "subject_qc.csv"
    subject_qc = pd.read_csv(subject_path, dtype=ids) if subject_path.is_file() else None
    # Les statuts déjà exportés doivent survivre à une reconstruction sans subject_qc.
    if subject_qc is None and "subject_eligibility" in runs:
        statuses = runs[["participant_id", "subject_eligibility"]].drop_duplicates()
        statuses["subject_valid"] = statuses.subject_eligibility.map({"retained": True, "excluded": False})
        subject_qc = statuses
    if destination is not None and figures and Path(destination).resolve() != root:
        source_example = root / 'figures' / 'preprocessing' / 'preprocessing_steps_example.png'
        if source_example.is_file():
            target = Path(destination) / 'figures' / 'preprocessing' / source_example.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_example, target)
    return export_preprocessing_report(runs, destination or root, metadata["parameters"],
                                       profile=metadata["profile"], subject_qc=subject_qc, figures=figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Dossier de résultats ou preprocessing_run_summary.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = summarize_existing(args.results, args.output_dir, figures=not args.no_figures)
    row = result["cohort"].iloc[0]
    print(f"Bilan : {row.n_participants:g} participants, {row.n_runs_preprocessed:g}/{row.n_runs_expected:g} runs prétraités.")
    destination = args.output_dir or (args.results.parent.parent if args.results.is_file() and args.results.parent.name == 'tables'
                                      else args.results.parent if args.results.is_file() else args.results)
    print(f"Rapport : {destination / 'preprocessing_report.html'}")
