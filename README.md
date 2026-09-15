# Drift Reco

> Méthodologie de prise en compte du *concept drift* pour les systèmes de recommandation à fusion long terme / court terme, évaluée sur des données Amazon.

## Question de recherche

Une méthodologie unifiée de détection de dérive peut-elle maintenir la qualité de recommandation dans une marge définie par rapport à une baseline pré-dérive, tout en gardant une latence de détection faible ?

## Vue d'ensemble de la méthodologie

| Brique | Rôle |
| --- | --- |
| Détection ensembliste | PUDD, divergence KL et dégradation de performance, vote 2 sur 3 |
| DTD | Détermination dynamique des seuils |
| Réponse adaptative | α-shift, ré-entraînement en arrière-plan avec hot-swap, mise à jour incrémentale des embeddings |

### Périmètre assumé

La méthodologie n'est **pas entièrement agnostique au modèle** : elle requiert trois points d'accroche dans le recommandeur hôte — des embeddings utilisateur exposés, un poids de fusion α ajustable et un signal de retour en ligne. Le cadrage exact est documenté dans `docs/scope.md`. Le prototype constitue une **preuve de la méthodologie**, pas le livrable en soi ; l'absence de déploiement en production est une limite de périmètre choisie, pas un travail inachevé.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Utilisation

```bash
make help          # lister les cibles disponibles
make check         # contrôle de conformité complet
make test          # suite de tests
python -m drift_reco.pipelines.run_ablation --config configs/ablation_c.yaml
```

## Structure du dépôt

```
docs/         cadrage, architecture, protocole expérimental, data card, ADR
src/          code source (seule source de vérité de la logique)
tests/        tests unitaires, miroir de src/
scripts/      points d'entrée en ligne de commande
notebooks/    exploration uniquement
data/         données locales, non versionnées (voir data/README.md)
artifacts/    un dossier horodaté par exécution + registre index.csv
reports/      rapports de conformité générés
JOURNAL.md    journal chronologique des runs
```

## Traçabilité

Chaque exécution produit un dossier `artifacts/runs/<RUN_ID>/` contenant la configuration validée, les métriques (`metrics.csv`), la trace d'événements (`events.csv`), le log complet (`run.log`), un résumé lisible (`results.md`), les notes libres et un manifeste de provenance (commit Git, versions, empreintes des fichiers). Aucun chiffre publié n'existe sans le run qui l'a produit.

## Résultats

Les résultats consolidés des configurations d'ablation A–E et des baselines sont dans `docs/results.md`. Les valeurs absolues de NDCG@10 sont faibles : c'est le comportement attendu sur des données Amazon très éparses, et l'analyse porte sur les écarts relatifs à la baseline pré-dérive.

## Licence

Voir `LICENSE`.
