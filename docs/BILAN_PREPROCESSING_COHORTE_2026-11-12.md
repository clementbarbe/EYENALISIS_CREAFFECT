# Chiffres pour présenter la cohorte — 12 novembre 2026

Bilan recalculé avec `run_pipeline.py data --preprocessing-only`, sans extraction
de fenêtres ni tests de conditions. Les chiffres décrivent les fichiers ASC
appariables dans le dossier de données à cette date. Ils portent sur toute
la durée de chaque ASC, y compris d'éventuelles marges ou attentes du scanner.

## Effectifs et volumes de données

| Périmètre | Participants avec données dans ce périmètre | Runs prétraités | Durée totale ASC | Pupille observée conservée | Pupille interpolée |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tous les ASC appariables, avant exclusions | 52 | 206 | 28,84 h | 17,62 h | 17,32 s |
| Runs retenus par le QC de run | 48 | 181 | 25,22 h | 17,36 h | 17,25 s |
| Runs et participants retenus | 46 | 179 | 24,93 h | 17,22 h | 17,25 s |

Le plan attend 260 runs (52 × 5). **54 runs n'ont pas d'ASC appariable** ; leur
durée reste inconnue. Les trois lignes du tableau sont des ensembles imbriqués :
ne pas additionner leurs durées. Deux participants ont chacun un seul run retenu,
ce qui explique le passage de 181 à 179 runs après le critère participant.

Les timestamps présents couvrent pratiquement toute la durée des ASC mesurés.
Les pertes de pupille et les masquages d'artefacts se produisent donc ici dans
des samples enregistrés ; ils ne doivent pas être confondus avec des trous
sans échantillon. La très faible reconstruction est cohérente avec le plafond
de 200 ms appliqué après un padding de 100+100 ms.

## Durée des fichiers par numéro de run, avant exclusions

| Run | ASC mesurés / attendus | Moyenne ± SD (min) | Médiane (min) | Q1–Q3 (min) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 51 / 52 | 8,44 ± 1,00 | 8,33 | 7,66–9,16 |
| 2 | 52 / 52 | 8,47 ± 1,09 | 8,39 | 7,60–9,34 |
| 3 | 0 / 52 | Inconnue | Inconnue | Inconnus |
| 4 | 52 / 52 | 8,37 ± 1,03 | 8,10 | 7,69–9,15 |
| 5 | 51 / 52 | 8,32 ± 1,18 | 8,07 | 7,55–9,10 |

**Le run 3 n'a aucun ASC appariable dans les fichiers disponibles.** Cela ne
prouve pas qu'aucun run comportemental/IRM correspondant n'a été acquis :
il faut contrôler la provenance EDF et les marqueurs. Les 54 lignes absentes
comprennent ces 52 runs 3, un run 1 et un run 5.

## Exclusions de runs et participants

- 54 runs : `missing_asc`, c'est-à-dire pas d'ASC sélectionné sans ambiguïté.
- 25 runs : pupille absente/invalide à l'acquisition au-delà du seuil de 45 %.
- Parmi ces runs techniquement dégradés, 11 n'ont aucun signal pupillaire
  utilisable après prétraitement. Les motifs se recouvrent : ne pas additionner
  54 + 25 + 11 comme des runs distincts.
- 6 participants : moins de 2 runs retenus.

Ces chiffres ne sont pas les nombres d'essais utilisables pour chaque métrique.
Les fenêtres et baselines seront contrôlées à l'étape suivante.

Le [guide M2](PRETRAITEMENT_GUIDE_M2.md) explique les méthodes et critères avec
références. Le rapport à ouvrir pour la présentation est
`output/bilan_preprocessing_2026-11-12/preprocessing_report.html` ; les graphiques
et les tables sources se trouvent dans les sous-dossiers `figures/preprocessing`
et `tables` de ce même dossier.
