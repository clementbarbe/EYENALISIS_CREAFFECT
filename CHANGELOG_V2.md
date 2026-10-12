# Changements par rapport à EYENALISIS_CREAFFECT_GPT

- Nouveau `matching.py` : run 0 ignoré ; appariements ASC aux `RUN_START/RUN_END` ; collisions/doublons reportés, refus de substitution par nom.
- `sync.py` : run explicitement identifié ; plusieurs triggers caractérisés ; désaccord fin > 2 s bloquant.
- `preprocessing.py` : accélération des masques de vitesse (intervalles vectorisés), blinks EBLINK distincts du padding, transparence des interpolations, QC gaze et pupil séparés, filtre SOS sur segments continus.
- `metrics.py` : contrôle qualité par fenêtre et par métrique, AUC soustractive ne traversant pas les longues lacunes, données de stimuli conservées.
- `moments.py` : fenêtres H1/H2/H3/H4 originales et deux sensibilités H2/H3.
- `timecourse.py` : pupille à 20 Hz alignée sur induction/clic.
- `stats_robust.py` : modèles exploratoires avec effets fixes participants et erreurs standards clusterisées par participant, p-values corrigées FDR.
- `descriptives.py` : QC et distributions par sujet, condition et run.
- `qc_plots.py` : visualisation par run facultative.
- `viz.py` : heatmaps par histogrammes lissés, pondération par sujet et essai pour éviter qu'un run long domine.
- `run_pipeline.py` : sorties de provenance et paramètres, flags CLI, ne dépend plus d'une numérotation ASC trompeuse.
- `tests/` : tests automatisés et validation sur fichiers exemples.

Les fichiers sources privés (EDF, ASC, CSV de participants, NIfTI) **ne sont pas redistribués** dans cette archive de code.
