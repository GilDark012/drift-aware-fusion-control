# Protocole expérimental

## Conditions

| Configuration | Détecteurs actifs | Seuils | Réponse |
| --- | --- | --- | --- |
| Baseline 0 | aucun | — | aucune |
| Baseline 1 | — | — | ré-entraînement périodique |
| A | PUDD seul | statique | α-shift |
| B | KL seul | statique | α-shift |
| C | vote 2 sur 3 | dynamique (DTD) | α-shift |
| D | vote 2 sur 3 | dynamique (DTD) | ré-entraînement en arrière-plan + hot-swap |
| E | vote 2 sur 3 | dynamique (DTD) | mise à jour incrémentale des embeddings |

## Métriques

| Métrique | Définition | Rôle |
| --- | --- | --- |
| NDCG@10 | qualité du classement sur la fenêtre d'évaluation | qualité |
| Recall@10 | rappel sur la fenêtre d'évaluation | qualité |
| Latence de détection | nombre de fenêtres entre le point de dérive injecté et le vote positif | réactivité |
| Taux de faux positifs | votes positifs hors dérive | robustesse |
| Coût d'adaptation | temps de calcul et nombre de mises à jour de paramètres | coût |

## Plan d'exécution

1. Fixer les seeds (au minimum 5 répétitions par condition).
2. Exécuter chaque condition via `python -m drift_reco.pipelines.run_ablation --config configs/<condition>.yaml`.
3. Consigner chaque exécution dans son propre `RUN_ID`.
4. Agréger les répétitions : moyenne et écart-type, puis test statistique face à la baseline.
5. Consolider dans `docs/results.md` avec le tableau comparatif et les figures.

## Critère de succès

La qualité post-dérive reste dans la marge définie par rapport à la baseline pré-dérive, avec une latence de détection inférieure au seuil fixé dans le cadrage, et un écart significatif face aux baselines sur au moins une configuration.

## Conditions d'invalidation

Énoncer à l'avance ce qui ferait conclure à l'échec de l'hypothèse : par exemple une latence non améliorée par le vote ensembliste par rapport au meilleur détecteur seul, ou un taux de faux positifs supérieur à celui d'une détection unique.
