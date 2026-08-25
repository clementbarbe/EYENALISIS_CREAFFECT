EYE-TRACKING — COMMANDES ET ORDRE D'EXÉCUTION
==================================================

1. Vérifier que Python voit les commandes disponibles
------------------------------------------------------

python cli.py --help


2. Vérifier que les données sont dans data/
--------------------------------------------

Structure attendue :

data/
├── C4X001/
│   ├── 001_0.asc
│   ├── 001_1.asc
│   ├── ...
│   ├── FGAT_Dist_Run0_C4X001.csv
│   ├── FGAT_Dist_Run1_C4X001.csv
│   └── ...
│
├── C4X016/
│   ├── 016_0.asc
│   ├── ...
│   └── FGAT_Dist_Run0_C4X016.csv
│
└── ...


3. Traiter un seul participant pour vérifier la pipeline
---------------------------------------------------------

python cli.py extract-subject --participant C4X016


4. Vérifier les exports créés
-----------------------------

results/qc/run_qc.csv

results/trial_level/h1_trials.csv
results/trial_level/h2_trials.csv
results/trial_level/h3_trials.csv
results/trial_level/h4_trials.csv


5. Traiter tous les participants
--------------------------------

python cli.py extract-all


6. Traiter tous les participants sans afficher les détails
----------------------------------------------------------

python cli.py extract-all --quiet


7. Refaire seulement un participant après une correction
----------------------------------------------------------

python cli.py extract-subject --participant C4X016


8. Analyse individuelle H1
---------------------------

python cli.py subject-h1 --participant C4X016

python cli.py subject-h1 --participant C4X016 --metric pup_mean
python cli.py subject-h1 --participant C4X016 --metric pup_max
python cli.py subject-h1 --participant C4X016 --metric pup_slope
python cli.py subject-h1 --participant C4X016 --metric sac_freq
python cli.py subject-h1 --participant C4X016 --metric gaze_entropy


9. Analyse individuelle H2
---------------------------

python cli.py subject-h2 --participant C4X016

python cli.py subject-h2 --participant C4X016 --metric pup_mean
python cli.py subject-h2 --participant C4X016 --metric pup_max
python cli.py subject-h2 --participant C4X016 --metric pup_slope
python cli.py subject-h2 --participant C4X016 --metric pup_auc_per_s
python cli.py subject-h2 --participant C4X016 --metric gaze_entropy


10. Analyse individuelle H3
---------------------------

Fenêtre : LTask_onset -> LTask_validation

python cli.py subject-h3 --participant C4X016 --metric pup_mean_v
python cli.py subject-h3 --participant C4X016 --metric pup_max_v
python cli.py subject-h3 --participant C4X016 --metric pup_slope_v
python cli.py subject-h3 --participant C4X016 --metric pup_auc_v

Fenêtre : LTask_onset -> LTask_first_movement

python cli.py subject-h3 --participant C4X016 --metric pup_mean_m
python cli.py subject-h3 --participant C4X016 --metric pup_max_m
python cli.py subject-h3 --participant C4X016 --metric pup_slope_m
python cli.py subject-h3 --participant C4X016 --metric pup_auc_m


11. Analyse individuelle H4
---------------------------

python cli.py subject-h4 --participant C4X016

python cli.py subject-h4 --participant C4X016 --metric gaze_entropy
python cli.py subject-h4 --participant C4X016 --metric gaze_disp
python cli.py subject-h4 --participant C4X016 --metric gaze_path_len_per_s
python cli.py subject-h4 --participant C4X016 --metric fix_rate
python cli.py subject-h4 --participant C4X016 --metric sac_rate
python cli.py subject-h4 --participant C4X016 --metric mean_fix_dur


12. Analyse de groupe H1
------------------------

python cli.py group-h1 --metric pup_mean
python cli.py group-h1 --metric pup_max
python cli.py group-h1 --metric pup_slope
python cli.py group-h1 --metric sac_freq
python cli.py group-h1 --metric gaze_entropy


13. Analyse de groupe H2
------------------------

Comparaison entre catégories émotionnelles :

python cli.py group-h2 --metric pup_mean --mode category
python cli.py group-h2 --metric pup_max --mode category
python cli.py group-h2 --metric gaze_entropy --mode category

Relation avec la durée d'idéation :

python cli.py group-h2 --metric pup_mean --mode duration
python cli.py group-h2 --metric pup_slope --mode duration
python cli.py group-h2 --metric pup_auc_per_s --mode duration
python cli.py group-h2 --metric gaze_entropy --mode duration


14. Analyse de groupe H3
------------------------

python cli.py group-h3 --metric pup_mean_v
python cli.py group-h3 --metric pup_max_v
python cli.py group-h3 --metric pup_slope_v
python cli.py group-h3 --metric pup_auc_v

python cli.py group-h3 --metric pup_mean_m
python cli.py group-h3 --metric pup_max_m
python cli.py group-h3 --metric pup_slope_m
python cli.py group-h3 --metric pup_auc_m


15. Analyse de groupe H4
------------------------

Comparaison entre catégories émotionnelles :

python cli.py group-h4 --metric gaze_entropy --mode category
python cli.py group-h4 --metric gaze_disp --mode category
python cli.py group-h4 --metric gaze_path_len_per_s --mode category
python cli.py group-h4 --metric fix_rate --mode category
python cli.py group-h4 --metric sac_rate --mode category

Relation avec la durée d'idéation :

python cli.py group-h4 --metric gaze_entropy --mode duration
python cli.py group-h4 --metric gaze_disp --mode duration
python cli.py group-h4 --metric gaze_path_len_per_s --mode duration
python cli.py group-h4 --metric fix_rate --mode duration
python cli.py group-h4 --metric sac_rate --mode duration


16. Modifier les paramètres
---------------------------

Modifier config.py :

- durée H1 :
  ANALYSIS_WINDOWS["h1"]["main"]

- baseline pupille :
  PARAMS["baseline_pre_ms"]

- marge autour des blinks :
  PARAMS["blink_margin_ms"]

- seuil MAD :
  PARAMS["mad_threshold"]

- taille de grille pour l'entropie :
  PARAMS["entropy_bins_x"]
  PARAMS["entropy_bins_y"]

- exclusions QC groupe :
  GROUP_QC["max_pupil_data_loss_pct"]
  GROUP_QC["max_gaze_data_loss_pct"]


17. Après modification de config.py
-----------------------------------

Relancer l'extraction pour recalculer les métriques :

python cli.py extract-all

Ou pour seulement un participant :

python cli.py extract-subject --participant C4X016
