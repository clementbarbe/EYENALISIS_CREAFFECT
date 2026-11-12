# Mode projet configurable — première implémentation

Le lancement historique de CREAFFECT reste disponible. Le nouveau mode
`--project` accepte un projet **JSON ou YAML version 1**, un manifest
d'acquisitions et des événements en table longue. Il n'impose ni noms C4X/FGAT,
ni cinq runs, ni blocs de trois essais, ni IRM.

Le [grand schéma proposé pendant l'audit](../examples/project_spec.proposed.yaml)
reste un cahier des charges. Le contrat effectivement pris en charge est celui
de [project_example.json](../examples/project_example.json), décrit ici.
Les paramètres inconnus sont refusés pour éviter des options silencieusement
ignorées.

## Essayer sur un projet synthétique complet

Après installation des dépendances dans votre environnement Python :

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/
python examples/generate_demo.py /tmp/eyetrack_demo
python run_pipeline.py --project /tmp/eyetrack_demo/project.json \
  --output-dir /tmp/eyetrack_demo_results --qc-plots
```

Les répertoires de démonstration et de résultats doivent être nouveaux ou vides.
Le mode projet refuse d'écraser un dossier de résultats existant. Pour refaire
un essai, choisir d'autres chemins.

La démonstration génère trois participants, un run chacun, des samples à 500 Hz,
deux conditions, des réponses de durées variables, un blink, une interruption
d'acquisition, un mouvement absent et un essai en timeout. Elle fournit des
ancres d'horloge avec une petite dérive connue. Les événements, signaux et
étiquettes oculaires sont synthétiques : ils vérifient le logiciel, pas un
détecteur de mouvements ni une hypothèse physiologique.

Les résultats comprennent `report.html`, les tableaux et les traces de QC.
Avec trois participants, l'hypothèse de démonstration est décrite comme non
estimable au seuil par défaut de douze participants ; aucun résultat de groupe
fiable n'est revendiqué.

Le Python par défaut observé pendant le développement est celui de MRtrix3
et ne contient pas toutes les dépendances. Utiliser un environnement dédié.
La validation locale a employé Python FSL 3.12.3 avec `PYTHONNOUSERSITE=1` pour
éviter le mélange des packages système et utilisateur.

## Manifest d'acquisitions

`recordings_manifest` est un chemin relatif au fichier de projet. Le manifest
peut être un TSV ou CSV. Ses chemins de données sont relatifs au manifest.
Les identifiants restent des chaînes, zéros initiaux compris.

```text
participant_id  session_id  run_id    asc_path            events_path        sync_path
person-001      visit-a     task-one  person-001/eye.asc  person-001/log.tsv person-001/anchors.tsv
```

Colonnes obligatoires : `participant_id`, `run_id`, `asc_path`, `events_path`.
`session_id` vaut `1` s'il est absent. Une combinaison participant/session/run
doit être unique. Une colonne facultative `eye` sélectionne L/R pour cette
acquisition ; sinon `parameters.eye` s'applique.

Le parseur exige des en-têtes ASC `SAMPLES GAZE`, œil et fréquence, ainsi que
`PUPIL AREA` ou `PUPIL DIAMETER`. Les samples et événements de l'œil choisi
sont sélectionnés, y compris dans un export binoculaire. Les timestamps
fractionnaires sont acceptés. La fréquence est lue dans l'en-tête et comparée
aux timestamps ; une fréquence explicitement configurée doit correspondre.

Les dimensions d'écran viennent de `MSG ... GAZE_COORDS`. À défaut, définir
`gaze_screen_width_px` et `gaze_screen_height_px` dans `parameters`. Le repère
doit avoir une origine `(0, 0)`. Les formats RAW/HREF, un changement de fréquence
ou de format dans un même ASC, et les horloges réinitialisées sont refusés.
Des segments START/END successifs avec horloge croissante sont conservés ; le
filtre et l'interpolation ne traversent pas leurs frontières.

## Table longue d'événements

Chaque ligne représente un événement de tâche, avec onset et durée en
**secondes dans l'horloge de référence déclarée** :

```text
trial_id  event_id  event_type  onset  duration  condition
trial-1   fix-1     FIX         0.1    0.7       A
trial-1   stim-1    STIM        0.8    0.15      A
trial-1   resp-1    RESP        2.0    0         A
```

`trial_id`, `event_type` et `onset` sont obligatoires. Sans colonne `duration`,
la durée vaut zéro. Sans `event_id`, des identifiants de ligne sont générés ;
des identifiants explicites sont préférables pour la provenance.
Les durées doivent être finies et non négatives.

Les autres colonnes constituent des métadonnées d'essai : condition,
stimulus, bloc, temps de réponse, etc. Une même métadonnée doit avoir au plus
une valeur non vide par essai. `timeout` et `is_resumed` acceptent
`true/false`, `1/0`, `yes/no` ; des cellules vides peuvent être complétées par
une valeur sur une autre ligne du même essai. Une métadonnée variant entre
événements d'un même essai doit être préparée séparément dans cette version.
Les identifiants participant/session/run présents dans la table doivent
correspondre au manifest.

Les noms de colonnes et événements peuvent être adaptés sans modifier Python :

```json
"events": {
  "columns": {"trial_id": "essai", "event_type": "type", "onset": "temps"},
  "mapping": {"stimulus": "STIM", "response": "RESP", "fixation": "FIX"}
}
```

`columns` associe nom canonique → nom dans le fichier ; `mapping` associe
nom canonique d'événement → libellé dans les données. Les associations doivent
être univoques. Les jointures de fenêtres se font uniquement dans le même essai.
Les imports comportementaux spécifiques, par exemple les CSV larges FGAT,
nécessitent encore un adaptateur vers cette table.

## Horloges

Choisir explicitement un mode dans `synchronization` :

| Mode | Contrat |
| --- | --- |
| `eyelink_clock` | Les onsets de tâche sont déjà les timestamps EyeLink divisés par 1 000. Aucune translation implicite par le premier sample. |
| `offset` | Chaque ligne du manifest fournit `offset_ms`. Conversion `t_EyeLink_ms = offset_ms + 1000 × onset_s`. Dérive non vérifiée. |
| `paired_anchors` | `sync_path` désigne un TSV/CSV de paires `reference_s`, `eyelink_ms`. Ajustement affine si au moins deux paires, offset seul pour une paire. |

Les ancres doivent être finies et strictement croissantes. Le contrôle des
résidus utilise `max_residual_ms` (5 ms par défaut). `drift_warning_ppm`
(150 par défaut) qualifie une pente éloignée de 1. Ces seuils sont des choix
configurables, pas des normes de mesure. La dérive estimée reste manquante
quand elle n'est pas identifiable. Deux points définissent une droite ; leur
résidu nul ne prouve pas à lui seul la qualité de synchronisation.

Les ancres doivent représenter les mêmes événements physiques sur les deux
horloges. Le logiciel ne les apparie pas automatiquement à des volumes IRM.
Les bins temporels sont réguliers dans l'horloge de référence et leurs bornes
sont converties vers EyeLink avec la même pente.

## Fenêtres et baselines

Une fenêtre déclare son ancre, ses bornes, ses métriques et éventuellement une
baseline, une ROI et un seuil de durée :

```json
{
  "id": "avant_reponse",
  "anchor": {"event": "response"},
  "start": {"anchor_offset_s": -1.2},
  "end": {"anchor_offset_s": -0.2},
  "lower_bound": {"event": "stimulus"},
  "clipping_policy": "truncate",
  "min_duration_s": 0.5,
  "baseline": "fixation_before",
  "metrics": ["pupil_baseline_corrected", "pupil_slope_per_s"]
}
```

Une borne peut être un offset de l'ancre, un événement avec `edge` (`onset`
ou `offset`) et `offset_s`, ou une expression `min`/`max` contenant d'autres
bornes. On peut donc décrire cue → réponse, onset → premier mouvement,
onset → min(onset +1 s, mouvement), ou des bornes avant un clic.

`lower_bound` et `upper_bound` contraignent la phase analysée.
`clipping_policy` vaut `reject` par défaut, ou `truncate` avec export des
bornes demandées, des bornes réelles et de l'indicateur `truncated`.
Une acquisition partielle ne déclenche aucune troncature implicite : le QC
de couverture la détecte.

Si un événement est répété, le défaut `occurrence: only` refuse l'ambiguïté
pour la fenêtre concernée. Choix disponibles : `first`, `last`, entier à partir
de 1, ou `last_before_anchor` pour une référence secondaire.
`anchor.occurrence: each` crée une fenêtre par événement d'ancrage, avec
`anchor_event_id` distinct. Une référence sans occurrence à ce même type
d'événement conserve l'occurrence d'ancrage sélectionnée.

Une définition glissante remplace start/end :

```json
"sliding": {"range_s": [0, 4], "widths_s": [0.5, 1, 2], "step_s": 0.1}
```

Les fenêtres glissantes restent descriptives dans cette version. Leur largeur
et position sont conservées dans `analysis_variant_id` et `window_id`.
`role` accepte `primary`, `exploratory` ou `sensitivity` ; cette étiquette ne
transforme pas une analyse déjà explorée en analyse confirmatoire.

Une baseline se définit ainsi :

```json
"baselines": {
  "fixation_before": {
    "event": "fixation", "occurrence": "last_before_anchor",
    "duration_s": 0.5, "end_edge": "offset", "clipping_policy": "reject"
  }
}
```

L'estimation est la médiane des samples observés du signal nettoyé, suivie
d'une correction soustractive. Le QC impose aussi la durée minimale
`minimum_baseline_duration_s`. Une baseline invalide bloque les métriques
qui en dépendent, sans bloquer automatiquement moyenne absolue, gaze ou blinks.
Une baseline postérieure à l'ancre est refusée. Il n'y a aucun remplacement
silencieux d'un événement manquant.

## Prétraitement, QC et métriques

`parameters` expose les paramètres de prétraitement et QC réellement appliqués.
La liste acceptée figure dans `project_config.PARAMETERS` ; le projet d'exemple
montre les réglages essentiels. Les seuils doivent être validés sur les données
du protocole. Le plafond d'interpolation porte sur le trou **après padding**.

Les règles communes de couverture, baseline et validité s'appliquent aux
métriques et aux courbes. La couverture se calcule sur les samples attendus,
y compris dans les bins vides. Les moyennes temporelles utilisent les samples
observés du signal nettoyé ; les métriques pupillaires peuvent inclure les
interpolations autorisées. Les deux moyennes ne sont donc pas forcément égales.

```bash
python run_pipeline.py --list-metrics
```

Le catalogue comprend 30 entrées avec famille, unité, description et prérequis :

| Famille | Principales mesures |
| --- | --- |
| Pupille | Moyenne, médiane, SD, moyenne corrigée, pente, pic absolu/corrigé, latence du pic, AUC, durée intégrée, AUC par seconde intégrée |
| Blinks | Nombre d'onsets, taux par exposition, durée moyenne, fraction de fermeture |
| Saccades | Nombre/taux total, taux au-dessus du seuil, amplitude moyenne, durée moyenne, moyenne des pics de vitesse |
| Fixations | Nombre d'onsets, durée moyenne, fraction de la fenêtre couverte par EFIX |
| Gaze/ROI | SD X/Y, fraction hors ROI, temps dans ROI |
| QC | Fraction d'acquisition manquante |

L'entrée microsaccades reste désactivée comme mesure validée. Les événements
oculaires proviennent d'EBLINK/ESACC/EFIX ; cette version n'ajoute pas de nouveau
détecteur en vitesse. Un en-tête `EVENTS` atteste le flux d'événements exporté,
pas la validité empirique des réglages de détection.

Les comptes utilisent les onsets dans `[début, fin)` ; les fractions de durée
utilisent les intersections avec la fenêtre. Une durée moyenne sans événement
vaut `NaN`, avec une raison, tandis qu'un compte nul peut être valide quand le
flux a bien été exporté. Le taux de saccades est rapporté aux secondes de gaze
valide ; le taux de blinks à l'exposition de tracking, fermetures EBLINK comprises.
Les dénominateurs sont exportés. Ce sont des définitions différentes des taux
historiques rapportés à la simple différence premier–dernier sample.

La latence est relative à l'ancre déclarée, qui peut précéder le début de la
fenêtre. Les unités pupillaires restent arbitraires ; le type aire/diamètre est
exporté. Aucune conversion en millimètres physiques n'est inventée.

Les ROI prises en charge sont des rectangles statiques nommés dans `rois`, puis
référencés par `window.roi`. Le projet d'exemple en fournit un. Les AOI dynamiques,
polygones et tables d'AOI par stimulus ne sont pas encore implémentés.

## Hypothèses et statistiques

Chaque hypothèse déclare une fenêtre, une métrique, un prédicteur catégoriel,
un contraste `[niveau_positif, niveau_référence]`, des covariables numériques
ou catégorielles et une famille de comparaisons. Les autres niveaux de ce
prédicteur sont exclus de ce contraste dans cette version.

Le modèle implémenté est une OLS exploratoire avec effets fixes participant,
effets session/run si nécessaires et covariance clusterisée par participant.
La variable dépendante conserve son unité ; aucune standardisation globale
n'est imposée. Les effectifs sont recomptés après toutes les exclusions et
valeurs manquantes du modèle. Le rang, le nombre de participants et la variation
intra-sujet sont contrôlés avant l'inférence. Les p-values sont corrigées dans
chaque `family` avec `fdr_method` et `alpha`.
Un mélange de mesures pupillaires en aire et en diamètre est refusé par ce
moteur statistique ; les données doivent être harmonisées explicitement.

Ce premier moteur statistique traite des mesures continues. Les modèles de
comptes, proportions et taux, les effets purement inter-sujets, les effets
aléatoires de stimuli, les GAMM, la déconvolution et les tests temporels ne sont
pas implémentés. Les demandes d'inférence sur comptes/proportions/taux ou sur
fenêtres glissantes/ancres répétées sont refusées par la validation. Les mesures
descriptives correspondantes restent calculables et exportables.

## Sorties et erreurs

| Fichier | Contenu |
| --- | --- |
| `tables/windows.csv` | Toutes les fenêtres, ancres, bornes, troncatures, sources de baseline et QC |
| `tables/metrics_long.csv` | Une ligne fenêtre × métrique, y compris valeurs exclues et raisons |
| `tables/timecourses.csv` | Tous les bins attendus des fenêtres sélectionnées, avec QC et `NaN` explicites |
| `tables/recording_qc.csv` | Acquisitions traitées/échouées, fractions de signal et contrôle d'horloge |
| `tables/hypothesis_models.csv` | Contrastes, effectifs exacts, estimations, IC, p-values et raisons de non-estimabilité |
| `tables/metric_catalog.csv` | Définitions et prérequis du catalogue |
| `tables/descriptives.csv` | Descriptifs par participant, fenêtre/variante, métrique et condition si disponible |
| `tables/metric_qc_summary.csv` | Comptages des raisons QC par famille de fenêtres/métrique |
| `config_used.json`, `provenance.json` | Projet, paramètres effectifs par acquisition, versions et SHA-256 des sources/code |
| `logs/` | Traceback complet des acquisitions échouées |
| `report.html`, `figures/` | Bilan consultable et traces QC avec `--qc-plots` |

Une acquisition échouée reste visible dans le QC, et les autres acquisitions
continuent. Le CLI retourne le code **2** si au moins une acquisition échoue.
Une erreur de configuration ou de manifest arrête le lancement explicitement.
Les essais manquants/ambigus ne disparaissent pas : leurs fenêtres sont conservées
avec une raison d'exclusion. Il n'y a pas d'exclusion globale de participant
fondée sur « au moins deux runs » dans ce mode.

Options applicables : `--no-stats`, `--no-timecourses`, `--qc-plots`,
`--no-figures`. Le rapport tabulaire HTML reste produit. `--config` est un alias
de `--project`. `--mixedlm` reste réservé au lancement historique CREAFFECT.

`--preprocessing-only` arrête le mode projet avant la lecture des événements
et des ancres d'horloge : seuls les ASC sont prétraités. Il exporte
`preprocessing_report.html` et les tableaux de durées, avec synchronisation et
exclusions globales déclarées non évaluées. Le manifest peut alors omettre les
chemins d'événements. Les fichiers de projet et manifest restent validés.
Le [guide M2](PRETRAITEMENT_GUIDE_M2.md) explique les durées, les étapes et les
différents niveaux d'exclusion. Le bilan de prétraitement est aussi généré lors
d'une exécution complète.

## Changements scientifiques apportés au chemin CREAFFECT

Les corrections partagées modifient volontairement certains résultats :
rejet de vitesses aberrantes hors des pertes, sélection d'œil, couverture des
baselines/bins, latence depuis le début nominal, comptage par onset, durée
moyenne sans événement rendue manquante, contrôle des flux et exposition des
événements. Les heatmaps H4 utilisent maintenant le QC gaze de fenêtre.
La durée moyenne des blinks privilégie la durée fournie par EyeLink.

Sur le run 1 de C4X001 contrôlé après correction, le traitement reste réussi
avec 1 445 lignes de métriques et 1 300 bins. Le masque d'artefacts passe de
3,49 % à 8,90 %, principalement du fait du rejet des vitesses aberrantes, et
la fraction interpolée de 0,023 % à zéro. La synchronisation reste à 6,525 ms
d'écart de fin. Ce contrôle technique ne valide pas les effets de la cohorte.
Les anciens résultats du dossier `output/` n'ont pas été écrasés.

Validation exécutée : **32 tests réussis**, ainsi qu'un lancement CLI complet
sur C4X001 avec les quatre runs appariables (4/5), 5 780 lignes de métriques,
5 200 bins temporels et 11 figures. Un lancement générique séparé produit
216 fenêtres, 540 lignes de métriques et un rapport sur les trois participants
synthétiques.

La prochaine étape scientifique est une comparaison visuelle et une sensibilité
documentée sur plusieurs runs représentatifs, avant une nouvelle analyse de
cohorte. Les corrections PFE, photométriques, la calibration rétrospective et
les liens avec les volumes BOLD restent dépendants des données du protocole.
