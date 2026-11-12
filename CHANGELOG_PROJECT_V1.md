# Mode projet version 1 — 3 novembre 2026

## Bilan du prétraitement — 12 novembre 2026

- Nouveau `--preprocessing-only` pour arrêter après le nettoyage/QC, sans
  extraction événementielle ni modèle ; disponible pour CREAFFECT et les projets.
- Bilan par run, participant, numéro de run et cohorte, avec durées totales,
  acquises, observées, interpolées, inutilisables et trous d'acquisition.
  Les durées inconnues restent manquantes, les runs exclus restent présents.
- Rapport HTML et figures de présentation, dont un même extrait à quatre
  étapes. Paramètres et critères exportés avec leur statut méthodologique.
- Script `summarize_preprocessing.py` pour refaire ce bilan à partir des sorties.
- Guide M2 pour expliquer les étapes et les exclusions avec références.
- Les algorithmes et seuils de nettoyage restent ceux du mode projet version 1.

## Première version

Première implémentation issue de l'audit scientifique et technique.

- Nouveau lancement `run_pipeline.py --project projet.json` ou `.yaml`,
  avec validation des champs et refus des options inconnues.
- Manifest indépendant des identifiants, sessions, runs et noms de fichiers.
  Une seule acquisition par participant est suffisante pour l'extraction.
- Table longue d'événements, mapping des noms/colonnes, métadonnées d'essai,
  fenêtres fixes, entre événements, avant réponse, bornées, glissantes,
  ancres répétées explicites et baselines configurables.
- Horloge EyeLink commune déclarée, offset fourni ou ancres appariées :
  estimation affine, résidus et dérive ; dérive non vérifiée avec une seule ancre.
- Lecture ASC mono/binoculaire par œil, fréquence/aire/diamètre/repère/filtre,
  timestamps fractionnaires et segments d'acquisition. Les formats ambigus,
  changements de format et réinitialisations de timestamps sont refusés.
- Catalogue de 30 entrées avec unités, familles et prérequis. Ajout de médiane,
  pic corrigé, AUC par durée intégrée, comptes/durées oculaires, EFIX et ROI.
  L'entrée microsaccades reste non validée et désactivée.
- Correction du rejet des vitesses pupillaires, interpolation et filtrage
  indépendants des interruptions d'acquisition.
- QC commun aux métriques et courbes, couverture rapportée au temps attendu,
  bins absents exportés et baselines de durée minimale contrôlée.
- Comptage des onsets dans des fenêtres semi-ouvertes, durée moyenne sans
  événement rendue manquante, latence depuis l'ancre, exposition explicite
  pour les taux et contrôle de présence du flux oculaire.
- Effectifs des modèles historiques recomptés après retrait des ratings.
  Le nouveau moteur statistique traite des contrastes intra-sujet continus
  avec SE clusterisées, IC et correction par famille ; il refuse les demandes
  d'inférence sur comptes, proportions, taux et fenêtres glissantes/répétées.
- Tables conservant les exclusions, rapport HTML, traces QC optionnelles,
  configuration et SHA-256 du code/données, métadonnées effectives par acquisition.
- Jeu synthétique reproductible, tests de contrats et validation d'un run réel.

Le lancement CREAFFECT reste disponible. Les corrections communes changent
volontairement certaines mesures : les résultats historiques ne doivent pas
être considérés comme ceux de cette version. Les anciennes sorties n'ont pas
été écrasées pendant le développement.

Les AOI dynamiques/polygones, modèles mixtes génériques, GAMM, déconvolution,
corrections PFE/photométriques et export BIDS complet restent hors de cette
première implémentation. Le fichier `examples/project_spec.proposed.yaml`
conserve le cahier des charges plus large ; le contrat exécutable est dans
`examples/project_example.json` et `docs/MODE_PROJET.md`.
