# Données

Ce dossier n'est **pas versionné** (voir `.gitignore`). Il décrit comment reconstituer les données à l'identique.

| Sous-dossier | Contenu |
| --- | --- |
| `raw/` | données Amazon telles que téléchargées, jamais modifiées |
| `external/` | ressources tierces (métadonnées produits, taxonomies) |
| `interim/` | étapes intermédiaires de traitement, reproductibles |
| `processed/` | tables Parquet prêtes pour l'entraînement et l'évaluation |

## Reconstitution

1. Télécharger le jeu de données source (voir `docs/data_card.md` pour l'URL exacte, la version et la licence).
2. Placer les fichiers dans `raw/`.
3. Lancer `python -m drift_reco.pipelines.build_dataset`.

Chaque exécution du pipeline écrit les empreintes SHA-256 des fichiers produits dans le manifeste du run correspondant, ce qui permet de vérifier qu'un résultat a bien été calculé sur la version attendue des données.
