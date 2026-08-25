import argparse

from config import (
    ANALYSIS_WINDOWS,
    DATA_DIR,
    PARAMS,
)

from pipeline.discovery import discover_participants

from pipeline.dataset_processing import (
    compute_all_metrics_clean,
    process_dataset,
)

# ─────────────────────────────────────────────────────────────
# ANALYSES INDIVIDUELLES
# ─────────────────────────────────────────────────────────────

from analysis.h1 import analyse_subject_h1
from analysis.h2 import analyse_subject_h2
from analysis.h3 import analyse_subject_h3
from analysis.h4 import analyse_subject_h4

# ─────────────────────────────────────────────────────────────
# ANALYSES DE GROUPE
# ─────────────────────────────────────────────────────────────

from analysis.group.h1_group import analyse_group_h1
from analysis.group.h2_group import analyse_group_h2
from analysis.group.h3_group import analyse_group_h3
from analysis.group.h4_group import analyse_group_h4


def main():
    parser = argparse.ArgumentParser(
        description=(
            "GOAT Eye-Tracking Pipeline — "
            "extraction, QC et analyses H1-H4."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="Commandes disponibles",
    )

    # ══════════════════════════════════════════════════════════
    # EXTRACTION / CALCUL DES MÉTRIQUES
    # ══════════════════════════════════════════════════════════

    extract_all = subparsers.add_parser(
        "extract-all",
        help=(
            "Traite tous les participants et met à jour les CSV "
            "existants sans supprimer les autres participants."
        ),
    )

    extract_all.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les messages pendant le traitement.",
    )

    extract_subject = subparsers.add_parser(
        "extract-subject",
        help=(
            "Traite un seul participant et met à jour ses lignes "
            "dans les exports globaux."
        ),
    )

    extract_subject.add_argument(
        "--participant",
        required=True,
        help="Identifiant du participant, par exemple C4X016.",
    )

    extract_subject.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les messages pendant le traitement.",
    )

    compute_all = subparsers.add_parser(
        "compute-all",
        help=(
            "Supprime les anciens exports trial-level/QC puis "
            "recalcule toutes les métriques H1-H4 pour tous les sujets."
        ),
    )

    compute_all.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les messages pendant le traitement.",
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES INDIVIDUELLES — H1
    # ══════════════════════════════════════════════════════════

    subject_h1 = subparsers.add_parser(
        "subject-h1",
        help="Analyse individuelle H1 : induction émotionnelle.",
    )

    subject_h1.add_argument(
        "--participant",
        required=True,
        help="Identifiant du participant, par exemple C4X016.",
    )

    subject_h1.add_argument(
        "--metric",
        default="pup_mean",
        help=(
            "Métrique H1. Exemples : pup_mean, pup_max, pup_slope, "
            "sac_freq, gaze_disp, gaze_entropy."
        ),
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES INDIVIDUELLES — H2
    # ══════════════════════════════════════════════════════════

    subject_h2 = subparsers.add_parser(
        "subject-h2",
        help="Analyse individuelle H2 : génération d'idée.",
    )

    subject_h2.add_argument(
        "--participant",
        required=True,
        help="Identifiant du participant, par exemple C4X016.",
    )

    subject_h2.add_argument(
        "--metric",
        default="pup_mean",
        help=(
            "Métrique H2. Exemples : pup_mean, pup_max, pup_slope, "
            "pup_auc_per_s, gaze_entropy."
        ),
    )

    subject_h2.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Méthode de corrélation avec la durée d'idéation.",
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES INDIVIDUELLES — H3
    # ══════════════════════════════════════════════════════════

    subject_h3 = subparsers.add_parser(
        "subject-h3",
        help="Analyse individuelle H3 : évaluation et rating.",
    )

    subject_h3.add_argument(
        "--participant",
        required=True,
        help="Identifiant du participant, par exemple C4X016.",
    )

    subject_h3.add_argument(
        "--metric",
        default="pup_mean_v",
        help=(
            "Métrique H3. Validation : pup_mean_v, pup_max_v, "
            "pup_slope_v, pup_auc_v. Mouvement : pup_mean_m, "
            "pup_max_m, pup_slope_m, pup_auc_m."
        ),
    )

    subject_h3.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Méthode de corrélation avec le rating.",
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES INDIVIDUELLES — H4
    # ══════════════════════════════════════════════════════════

    subject_h4 = subparsers.add_parser(
        "subject-h4",
        help="Analyse individuelle H4 : dynamique spatiale du regard.",
    )

    subject_h4.add_argument(
        "--participant",
        required=True,
        help="Identifiant du participant, par exemple C4X016.",
    )

    subject_h4.add_argument(
        "--metric",
        default="gaze_entropy",
        help=(
            "Métrique H4. Exemples : gaze_entropy, gaze_disp, "
            "gaze_path_len_per_s, fix_rate, sac_rate, mean_fix_dur."
        ),
    )

    subject_h4.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Méthode de corrélation avec la durée d'idéation.",
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES GROUPE — H1
    # ══════════════════════════════════════════════════════════

    group_h1 = subparsers.add_parser(
        "group-h1",
        help=(
            "Analyse groupe H1 : Friedman par catégorie émotionnelle "
            "et post-hoc Wilcoxon corrigés FDR."
        ),
    )

    group_h1.add_argument(
        "--metric",
        default="pup_mean",
        help=(
            "Métrique H1. Exemples : pup_mean, pup_max, pup_slope, "
            "sac_freq, gaze_disp, gaze_entropy."
        ),
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES GROUPE — H2
    # ══════════════════════════════════════════════════════════

    group_h2 = subparsers.add_parser(
        "group-h2",
        help=(
            "Analyse groupe H2 : catégories émotionnelles ou relation "
            "métrique × durée d'idéation."
        ),
    )

    group_h2.add_argument(
        "--metric",
        default="pup_mean",
        help=(
            "Métrique H2. Exemples : pup_mean, pup_max, pup_slope, "
            "pup_auc_per_s, gaze_entropy."
        ),
    )

    group_h2.add_argument(
        "--mode",
        default="duration",
        choices=["duration", "category"],
        help=(
            "duration : corrélation intra-sujet avec ideation_dur. "
            "category : Friedman entre catégories émotionnelles."
        ),
    )

    group_h2.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Méthode de corrélation si --mode duration.",
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES GROUPE — H3
    # ══════════════════════════════════════════════════════════

    group_h3 = subparsers.add_parser(
        "group-h3",
        help=(
            "Analyse groupe H3 : corrélations intra-sujet "
            "métrique pupillaire × rating."
        ),
    )

    group_h3.add_argument(
        "--metric",
        default="pup_mean_v",
        help=(
            "Métrique H3. Validation : pup_mean_v, pup_max_v, "
            "pup_slope_v, pup_auc_v. Mouvement : pup_mean_m, "
            "pup_max_m, pup_slope_m, pup_auc_m."
        ),
    )

    group_h3.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Méthode de corrélation avec le rating.",
    )

    # ══════════════════════════════════════════════════════════
    # ANALYSES GROUPE — H4
    # ══════════════════════════════════════════════════════════

    group_h4 = subparsers.add_parser(
        "group-h4",
        help=(
            "Analyse groupe H4 : catégories émotionnelles ou relation "
            "dynamique spatiale × durée d'idéation."
        ),
    )

    group_h4.add_argument(
        "--metric",
        default="gaze_entropy",
        help=(
            "Métrique H4. Exemples : gaze_entropy, gaze_disp, "
            "gaze_path_len_per_s, fix_rate, sac_rate, mean_fix_dur."
        ),
    )

    group_h4.add_argument(
        "--mode",
        default="duration",
        choices=["duration", "category"],
        help=(
            "duration : corrélation intra-sujet avec ideation_dur. "
            "category : Friedman entre catégories émotionnelles."
        ),
    )

    group_h4.add_argument(
        "--method",
        default="spearman",
        choices=["spearman", "pearson"],
        help="Méthode de corrélation si --mode duration.",
    )

    # ══════════════════════════════════════════════════════════
    # PARSING DES ARGUMENTS
    # ══════════════════════════════════════════════════════════

    args = parser.parse_args()

    participants = discover_participants(DATA_DIR)

    if not participants:
        raise RuntimeError(
            f"Aucun participant trouvé dans : {DATA_DIR}"
        )

    # ══════════════════════════════════════════════════════════
    # EXÉCUTION DES COMMANDES
    # ══════════════════════════════════════════════════════════

    if args.command == "extract-all":
        process_dataset(
            participants=participants,
            params=PARAMS,
            analysis_windows=ANALYSIS_WINDOWS,
            selected_participants=None,
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

    elif args.command == "compute-all":
        compute_all_metrics_clean(
            participants=participants,
            params=PARAMS,
            analysis_windows=ANALYSIS_WINDOWS,
            verbose=not args.quiet,
        )

    elif args.command == "subject-h1":
        analyse_subject_h1(
            participant_id=args.participant,
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

    elif args.command == "group-h1":
        analyse_group_h1(
            metric=args.metric,
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


if __name__ == "__main__":
    main()
