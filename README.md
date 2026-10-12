# CREAFFECT · Eye-tracking simultané à l'IRMf — Pipeline exploratoire V2

**Statut :** pipeline exploratoire, auditable et reproductible. Il ne s'agit **pas** d'un protocole universellement validé ni d'une analyse confirmatoire pré-enregistrée. Les seuils QC proposés sont des points de départ à examiner **sans regarder les effets des conditions**.

## 1. Prérequis et usage

Python >= 3.10. Installer les dépendances :

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/
```

Organisation des fichiers (sans copier les données dans le code) :

```text
data/
  C4X001/
    FGAT_Dist_Run1_C4X001.csv
    FGAT_Dist_Run2_C4X001.csv
    ...
    001_2.asc             # le nom ne détermine PAS le run
    001_3.asc
    ...
  C4X002/
    ...
  ...
```

Commande :

```bash
python run_pipeline.py /CHEMIN/VERS/data --output-dir /CHEMIN/VERS/results
```

Pour une première passe QC sans statistiques (conseillée avant de fixer la stratégie d'analyse) :

```bash
python run_pipeline.py /CHEMIN/VERS/data --output-dir /CHEMIN/VERS/results_QC --no-stats --qc-plots
```

Options : `--no-stats` (QC et métriques uniquement), `--no-figures` (résultats tabulaires), `--no-timecourses` (ne produit pas les séries à 20 Hz), `--qc-plots` (traces brut/nettoyé par run), `--mixedlm` (modèles mixtes supplémentaires, plus lents et exploratoires).

**Runs :** 1, 2, 3, 4, 5 seulement. Le **run 0 (entraînement) est ignoré** systématiquement. Ne pas placer les données dans un répertoire `output/` nommé `C4X###`.

## 2. Corrections techniques fondamentales

### Identification ASC ↔ run comportemental

- Scan des messages EyeLink `MSG ... RUN_START N` et `MSG ... RUN_END N`.
- Aucune déduction de `N` à partir du nom ASC : `001_2.asc` peut être le run 1.
- Une conversion en double n'est dédoublonnée que si marqueurs et bornes temporelles sont identiques ; les collisions incohérentes sont marquées *ambiguës* et restent exclues.
- Le run 0 est enregistré dans le manifest (`training_run_ignored`) sans traitement.
- Tout run manquant, ambigu ou incohérent est recensé ; aucune substitution silencieuse.
- Rapport obligatoire : `tables/asc_matching_manifest.csv`.

### Synchronisation IRMf

- Le premier `BUTTON 1 1` **après RUN_START** et avant `RUN_END` est l'ancre temporelle, compte tenu du protocole actuellement observé.
- Décalage calculé depuis `t0_fMRI` ; vérification de fin `t_run_end` vs dernier échantillon EyeLink (tolérance **2 s**, conservatrice par rapport à l'ancienne valeur de 10 s).
- Statistiques des impulsions (`n_triggers`, intervalle médian, CV) exportées ; irrégularité = avertissement, non correction d'horloge par interpolation.
- **Attention :** une différence de quelques ms en fin de run ne prouve pas l'alignement événement-par-événement. Si les triggers du scanner ou des timestamps IRM indépendants sont disponibles, les utiliser pour vérifier l'offset et une éventuelle dérive linéaire. Le code n'invente pas de correction de dérive sans paire d'horloges.

### Pipelines pupille, blinks et gaze distincts

- `pupil_missing_acquisition` : impossible/absent dans l'enregistrement.
- `event_blink_mask` : uniquement les événements `EBLINK` d'EyeLink (physiologiques possibles, à valider sur la vidéo).
- `pupil_artifact_mask` : événements, pertes de tracking, changements rapides proches des pertes, et padding ±100 ms ; **ne représente pas un comptage de blinks**.
- `pupil_observed_valid` : fraction non reconstruite après retrait des artefacts.
- `pupil_interpolated` : fraction interpolée, séparée du signal observé.
- `gaze_valid`, `gaze_x_clean`, `gaze_y_clean` : coordonnées écran plausibles avec blinks et pertes de tracking exclus.
- Les trous courts sont interpolés seulement lorsqu'il existe un point valide de part et d'autre et que la durée temporelle est <= 200 ms.
- Filtre Butterworth passe-bas 6 Hz zéro phase sur sous-segments continus ; sans rétablir artificiellement les grands trous. Vérifier les effets de lissage sur quelques sujets.
- Les mesures de pupille restent en **unités EyeLink de l'acquisition**, pas en millimètres physiques en l'absence d'étalonnage.
- `pupil_auc` = AUC **soustractive par rapport à la baseline**, intégrée sans relier les trous longs ; pour des fenêtres de durée variable, interpréter avec la durée observée.

### Critères QC (*choix analytiques à éprouver*)

| Niveau | Valeur initiale | Interprétation |
|---|---:|---|
| Synchronisation fin run | max 2 s | Rejeter les runs non alignés |
| Manquants pupillaires à l'acquisition / run | max 45 % | Rejeter run techniquement dégradé ; seuil volontairement permissif en IRM |
| Proportion pupille réellement observée / fenêtre | min 70 % | Ne pas faire confiance à trop de reconstructions |
| Proportion interpolée / fenêtre | max 20 % | Contrôle distinct du signal observé |
| Plus long artefact / fenêtre | max 500 ms | Éviter un trou central qui occupe une grande partie de H2 |
| Couverture des timestamps / fenêtre | min 95 % | Éviter une acquisition partielle |
| Proportion observée / baseline | min 75 % | Baseline valide, estimée par médiane |
| Gaze valide / fenêtre | min 70 % | Métriques gaze seulement |
| Durée de fenêtre | min 500 ms | Fenêtre sous ce seuil non analysée |

Les critères sont appliqués **par famille de métriques**. L'absence d'une baseline fiable invalide `pupil_baseline_corrected` et `pupil_auc`, sans invalider automatiquement la pupille brute, le blink rate ou les saccades. Une métrique délibérément non configurée donne `NaN` et une raison (`roi_not_configured`). Les fenêtres avec `timeout` ou `is_resumed` sont exclues (comportement du code d'origine).

### Inductions, recherche et liking

Les moments conservés :

- `M1_early_photometric` : induction 0–2 s, contrôle descriptif visuel/photométrique.
- `H1_induction_late` : induction 2–10 s ; ne signifie **pas** « arousal pur ».
- `H2_pre_idea` : 1 s avant clic « idée trouvée », exposition possible à préparation motrice.
- `H3_liking` : présentation de l'évaluation → premier mouvement ; fenêtre de durée variable.
- `H4_full_search_gaze` : apparition du cue → clic.

Sensibilités temporelles distinctes, **non incluses par défaut dans les tests statistiques** :

- `H2_pre_idea_without_motor` : 1 s précédant les 200 dernières ms avant le clic.
- `H3_liking_first_1s_pre_move` : jusqu'à 1 s après onset et avant premier mouvement ; exclue si trop courte.

Les baselines sont les dernières 500 ms de la croix précédant l'événement, médiane de points **réellement observés** (correction soustractive). La baseline peut porter un état physiologique résiduel de l'événement précédent : vérifier la chronologie de tâche.

**Nouveaux exports :** courbes alignées sur l'induction (0–10 s) et le clic pré-idée en bins de 50 ms, QC des bins, sans interpolation entre des trous longs. Ces données ne sont pas « significatives » au point temporel individuel sans un test pour comparaisons multiples temporelles (modèles fonctionnels / test par permutations en grappes avec unité sujet).

### Analyses oculomotrices

- `EBLINK` utilisés comme événements indépendants des masques de nettoyage.
- `ESACC` = saccades détectées selon la configuration EyeLink (vérifier seuils dans la configuration de l'acquisition).
- Les événements <1° ne sont **pas automatiquement des microsaccades**. La métrique est conservée dans les tableaux comme **non validée et non testable**, avec `exploratory_small_saccades_not_validated`.
- Gaze heatmaps H4 : chaque essai pondéré à parts égales puis chaque participant à parts égales par condition, représentation par histogramme 2D lissé; pas un test de différences de fixations.
- ROI `gaze_aversion_fraction` **désactivée** tant que les coordonnées réelles de la cible dans le repère EyeLink ne sont pas spécifiées dans `config.py`.

## 3. Statistiques exploratoires

- **Défaut :** modèles avec `C(subject_id)` (intercepts propres aux participants), `C(run_id)` et covariance robuste **clusterisée par sujet** (`stats_robust.py`). Les trois essais d'un même bloc ne deviennent pas artificiellement des sujets indépendants puisque la covariance autorise une corrélation arbitraire des essais **au sein du même participant**. Les nombres de participants réellement observés sont fournis pour chaque modèle.
- Modèles de liking : association des ratings au signal, en ajustant les différences de moyenne inter-sujets via les effets fixes participants ; covariable optionnelle de temps de réponse log-transformée lorsque complète.
- Analyse H2 : covariable de durée de recherche log-transformée si disponible sur les observations.
- FDR Benjamini–Hochberg au niveau de la famille globale des tests de groupes, et séparément des tests des ratings, **dans chaque table de modèles**.
- Coefficients standardisés *globalement* pour comparaison descriptive. Pas de causalité ni d'identification d'effets purs de valence/arousal.
- Option `--mixedlm` : modèles mixtes d'origine comme **sensibilité**, plus lents et parfois numériquement fragiles. Leurs p-values sont asymptotiques et ne sont pas une validation confirmatoire.
- La littérature recommande d'éviter de considérer l'équivalence des essais d'un bloc comme des réplications indépendantes. Le regroupement par participant dans les erreurs standards traite cet aspect statistique, même sans ajouter explicitement un effet aléatoire de bloc.
- Rien n'autorise à annoncer un résultat confirmatoire sans plan préalable : choix de fenêtres, famille de tests, spécification du modèle et QC doivent être figés avant la dernière analyse.

## 4. Livrables générés automatiquement

`tables/` :

- `asc_matching_manifest.csv` : provenance EDF, marqueurs, copies et choix de run.
- `run_qc.csv`, `run_exclusion_summary.csv`, `subject_qc.csv`, `subject_exclusion_summary.csv`.
- `metrics_long.csv` : **une ligne par fenêtre × métrique × essai**, avec raisons d'exclusion spécifiques, fraction interpolée/observée, groupe, image/son/ID des stimuli.
- `qc_by_condition.csv`, `qc_by_participant_run.csv` : contrôle des pertes différentielles pouvant biaiser les comparaisons.
- `participant_condition_descriptives.csv`, `n_valid_by_metric_condition.csv`.
- `h4_gaze_samples.csv` : échantillonnage gaze (gros fichier, n'ouvrir que dans Python/R).
- `event_locked_pupil_timecourses_20hz.csv` : séries temporelles décalées par baseline.
- `exploratory_subject_clustered_models_fdr.csv` : résultats statistiques exploratoires.
- `exploratory_mixedlm_sensitivity_fdr.csv` si `--mixedlm`.

`figures/` : distribution des mesures, carte gaze, entonnoir QC et, si `--qc-plots`, **traces brutes et nettoyées** (inspection visuelle essentielle). `config_used.json` fige les choix numériques et la date d'analyse.

**Validation sur l'exemple de l'archive originale (un seul participant) :** appariements `001_2.asc → run 1`, `001_3.asc → run 2`, `001_4.asc → run 4`, `001_5.asc → run 5`; conversions du run 0 ignorées, `C4X001.asc` reconnu comme deuxième export du run 5; **run 3 manquant**, aucun appariement inventé. **4/5 runs** valides sur cet exemple. Tests Python unitaires et lancement `--no-stats` puis `--no-figures` réussis. Cela **ne valide pas** empiriquement la cohorte de 52 participants (non fournie).

## 5. Points qui demandent encore une intervention humaine

1. **Valider physiquement l'appariement trigger scanner ↔ marqueur `BUTTON 1 1`.** Si le scanner fournit des timestamps indépendants, examiner la dérive sur tout le run et les différences d'horloge. Il n'y a pas d'intégration au BOLD/volume NIfTI dans cette version.
2. **Contrôler la calibration EyeLink** et la précision géométrique par run, idéalement avec les logs de validation et les cibles de fixation connues. L'archive ne donne pas de données de calibration quantitatives fiables. Le contrôle de précision gaze « en degrés » ne peut donc pas être inventé.
3. **Renseigner `cue_roi_center_x/y` et dimensions réelles du stimulus**, valider l'orientation, l'échelle et les pixels d'écran via les logs d'acquisition.
4. **Quantifier la luminance/contraste des images et éventuellement du son**, ou joindre une table de stimuli via `stim_id`, `stim_image`, `stim_sound`. Sans assets visuels originaux et protocole de présentation, impossible de corriger l'effet photomoteur automatiquement.
5. **Pupil foreshortening error** : évaluer la corrélation entre pupil et gaze (`pupil_gaze_x/y_correlation` dans le QC), mais **ne pas utiliser une régression gaze→pupille comme correction automatique** sans vérifier qu'elle n'élimine pas un vrai effet psychophysiologique. Une correction géométrique suppose la géométrie caméra-œil-écran.
6. **Blinks et petites saccades** : valider visuellement un échantillon représentatif des pertes et événements, idéalement sur vidéo de l'œil. Une séparation « ESACC < 1° » n'est pas une validation des microsaccades.
7. **Inspections QC avant résultats** : résumer la couverture par condition, la baseline et les exclusions, et documenter tout ajustement de seuil. Tester 100/200/500 ms de gap et 60/70/80 % d'échantillons observés sans choisir le seuil par p-value.
8. **Pré-enregistrement** et contrôle des effets de tâche (ordre, image, recherche, contexte du run) avant toute inférence confirmatoire.

## 6. Bibliographie vérifiée (sélection méthodologique)

- Kret, M. E., & Sjak-Shie, E. E. (2019). *Preprocessing pupil size data: Guidelines and code*. Behavior Research Methods. DOI: [10.3758/s13428-018-1075-y](https://doi.org/10.3758/s13428-018-1075-y).
- Mathôt, S., Fabius, J., Van Heusden, E., & Van der Stigchel, S. (2018). *Safe and sensible preprocessing and baseline correction of pupil-size data*. Behavior Research Methods. DOI: [10.3758/s13428-017-1007-2](https://doi.org/10.3758/s13428-017-1007-2).
- Mathôt, S. (2018). *Pupillometry: Psychology, Physiology, and Function*. Journal of Cognition. DOI: [10.5334/joc.18](https://doi.org/10.5334/joc.18).
- *Methods in cognitive pupillometry: Design, preprocessing, and statistical analysis*. Behavior Research Methods (2023). DOI: [10.3758/s13428-022-01957-7](https://doi.org/10.3758/s13428-022-01957-7).
- *Publication guidelines and recommendations for pupillary measurement in psychophysiological studies*. Psychophysiology (2022). [Texte en libre accès](https://pmc.ncbi.nlm.nih.gov/articles/PMC9272460/).
- *From pre-processing to advanced dynamic modeling of pupil data* (2024). [Texte en libre accès](https://pmc.ncbi.nlm.nih.gov/articles/PMC10991010/).
- Hayes, T. R., & Petrov, A. A. (2016). *Mapping and correcting the influence of gaze position on pupil size measurements*. DOI: [10.3758/s13428-015-0588-x](https://doi.org/10.3758/s13428-015-0588-x).
- Shoenfelt, A., Isaacowitz, D. M., & Ebner, N. C. (2025). *Eye Tracking in Aging: Challenges, Best Practices, and Novel Frontiers* (inclut IRM simultanée). DOI: [10.1145/3733834](https://doi.org/10.1145/3733834).
- Amiot, V., et al. (2024). *PupilMetrics: a support system for preprocessing of pupillometric data and extraction of outcome measures*. Scientific Reports. DOI: [10.1038/s41598-024-79920-z](https://doi.org/10.1038/s41598-024-79920-z).
- SR Research, documentation [EyeLink / EDF2ASC](https://www.sr-research.com/download/dispdoc/page25.html).

**Ce qui n'est pas fourni** : acquisition/lecture directe EDF, correction photométrique automatique (images absentes), correction PFE géométrique sans configuration caméra, recalibration rétrospective, correction effective de dérive sans timestamps externes, détection des microsaccades validée, modèle spatio-temporel IRMf au niveau des volumes, ou inférence statistique confirmatoire. Ce sont des limites de données/protocole, pas des fonctions « à deviner » par le logiciel.
