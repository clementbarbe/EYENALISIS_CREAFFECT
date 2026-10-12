# Contrôle technique exécuté sur l'exemple fourni

Date d'audit : 12 octobre 2026. Archive de référence : `EYENALISIS_CREAFFECT_GPT.zip`, contenant **un seul participant C4X001**.

| Run comportemental | ASC trouvé par RUN_START | Synchronisation de fin (abs.) | Statut |
| --- | --- | ---: | --- |
| 0 | `001_0.asc` / `001_1.asc` | non calculée | entraînement, ignoré |
| 1 | `001_2.asc` | 6,5 ms | utilisable |
| 2 | `001_3.asc` | 2,5 ms | utilisable |
| 3 | aucun | non calculée | manquant : exclusion explicite |
| 4 | `001_4.asc` | 3,0 ms | utilisable |
| 5 | `001_5.asc` | 0,7 ms | utilisable |

`C4X001.asc` correspond à une deuxième conversion du run 5 : marqué comme doublon dans le rapport de provenance, non analysé deux fois.

**Vérifications effectuées :** 8 tests unitaires, découverte automatique de fichiers ASC, extraction de métriques, qualité par fenêtre, exports de courbes évènementielles à 20 Hz, modèles exploratoires (les données d'un participant ne suffisent pas aux modèles de groupe), création de figures de QC et de cartes gaze. **Aucune preuve de validité des modèles à 52 participants** n'est revendiquée. Les seuils d'exclusion doivent être comparés à la distribution de QC de la cohorte complète.

**Attention au contenu de l'archive corrigée :** elle contient du code et des documents, mais **pas les enregistrements privés** du participant. Utiliser le répertoire contenant vos propres données à l'exécution.
