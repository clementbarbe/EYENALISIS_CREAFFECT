import argparse

from config import PARAMS, ANALYSIS_WINDOWS, DATA_DIR
from pipeline.discovery import discover_participants
from pipeline.dataset_processing import process_dataset

from analysis.h1 import analyse_subject_h1
from analysis.h2 import analyse_subject_h2
from analysis.h3 import analyse_subject_h3
from analysis.h4 import analyse_subject_h4

from analysis.group.h1_group import analyse_group_h1
from analysis.group.h2_group import analyse_group_h2
from analysis.group.h3_group import analyse_group_h3
from analysis.group.h4_group import analyse_group_h4


def main():
    parser = argparse.ArgumentParser(
        description="GOAT Eye-Tracking Pipeline"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ─────────────────────────────────────────────────────────
    # EXTRACTION
    # ─────────────────────────────────────────────────────────

    extract_all = subparsers.add_parser("extract-all")
    extract_all.add_argument(
        "--quiet",
        action="store_true",
    )

    extract_subject = subparsers.add_parser("extract-subject")
    extract_subject.add_argument(
        "--participant",
        required=True,
    )
    extract_subject.add_argument(
        "--quiet",
        action="store_true",
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSES INDIVIDUELLES
    # ─────────────────────────────────────────────────────────

    subject_h1 = subparsers.add_parser("subject-h1")
    subject_h1.add_argument(
        "--participant",
        required=True,
    )
    subject_h1.add_argument(
        "--metric",
        default="pup_mean",
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSES INDIVIDUELLES H2
    # ─────────────────────────────────────────────────────────

    subject_h2 = subparsers.add_parser("subject-h2")
    subject_h2.add_argument(
        "--participant",
        required=True,
    )
    subject_h2.add_argument(
        "--metric",
        default="pup_mean",
    )
    subject_h2.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSES INDIVIDUELLES H3
    # ─────────────────────────────────────────────────────────

    subject_h3 = subparsers.add_parser("subject-h3")
    subject_h3.add_argument(
        "--participant",
        required=True,
    )
    subject_h3.add_argument(
        "--metric",
        default="pup_mean_v",
    )
    subject_h3.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSES INDIVIDUELLES H4
    # ─────────────────────────────────────────────────────────

    subject_h4 = subparsers.add_parser("subject-h4")
    subject_h4.add_argument(
        "--participant",
        required=True,
    )
    subject_h4.add_argument(
        "--metric",
        default="gaze_entropy",
    )
    subject_h4.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSES GROUPE
    # ─────────────────────────────────────────────────────────

    group_h1 = subparsers.add_parser("group-h1")
    group_h1.add_argument(
        "--metric",
        default="pup_mean",
    )

    # Même logique à ajouter pour group-h2, group-h3, group-h4.

    args = parser.parse_args()

    participants = discover_participants(DATA_DIR)

    if args.command == "extract-all":
        process_dataset(
            participants=participants,
            params=PARAMS,
            analysis_windows=ANALYSIS_WINDOWS,
            verbose=not args.quiet,
        )

    elif args.command == "extract-subject":
        process_dataset(
            participants=participants,
            params=PARAMS,
            analysis_windows=ANALYSIS_WINDOWS,
            selected_participants=[args.participant],
            verbose=not args.quiet,
        )

    elif args.command == "subject-h1":
        analyse_subject_h1(
            participant_id=args.participant,
            metric=args.metric,
        )

    elif args.command == "group-h1":
        analyse_group_h1(
            metric=args.metric,
        )

    elif args.command == "subject-h2":
        analyse_subject_h2(
            participant_id=args.participant,
            metric=args.metric,
            correlation_method=args.method,
        )

    elif args.command == "subject-h3":
        analyse_subject_h3(
            participant_id=args.participant,
            metric=args.metric,
            correlation_method=args.method,
        )

    elif args.command == "subject-h4":
        analyse_subject_h4(
            participant_id=args.participant,
            metric=args.metric,
            correlation_method=args.method,
        )

    elif args.command == "group-h2":
        analyse_group_h2(
            metric=args.metric,
            mode=args.mode,
            correlation_method=args.method,
        )

    elif args.command == "group-h3":
        analyse_group_h3(
            metric=args.metric,
            correlation_method=args.method,
        )

    elif args.command == "group-h4":
        analyse_group_h4(
            metric=args.metric,
            mode=args.mode,
            correlation_method=args.method,
        )

    # ─────────────────────────────────────────────────────────
    # ANALYSE GROUPE H2
    # ─────────────────────────────────────────────────────────

    group_h2 = subparsers.add_parser("group-h2")
    group_h2.add_argument(
        "--metric",
        default="pup_mean",
    )
    group_h2.add_argument(
        "--mode",
        default="duration",
        choices=["duration", "category"],
    )
    group_h2.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSE GROUPE H3
    # ─────────────────────────────────────────────────────────

    group_h3 = subparsers.add_parser("group-h3")
    group_h3.add_argument(
        "--metric",
        default="pup_mean_v",
    )
    group_h3.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )

    # ─────────────────────────────────────────────────────────
    # ANALYSE GROUPE H4
    # ─────────────────────────────────────────────────────────

    group_h4 = subparsers.add_parser("group-h4")
    group_h4.add_argument(
        "--metric",
        default="gaze_entropy",
    )
    group_h4.add_argument(
        "--mode",
        default="duration",
        choices=["duration", "category"],
    )
    group_h4.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
    )


if __name__ == "__main__":
    main()