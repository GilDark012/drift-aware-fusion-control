# Architecture

## Vue d'ensemble

```
flux d'interactions
        │
        ▼
  [ DataStream ]  ──►  [ FeatureBuilder ]
        │                     │
        ▼                     ▼
  [ DriftDetectionEnsemble ]  ◄── [ DynamicThresholdEstimator ]
        │  PUDD · KL · dégradation de performance (vote 2 sur 3)
        ▼
  [ AdaptationController ]
        │  α-shift · ré-entraînement en arrière-plan + hot-swap · MAJ incrémentale
        ▼
  [ HostRecommender ]  ──►  [ Evaluator ]  ──►  [ LabJournal ]
```

## Composants

| Composant | Responsabilité | Classe de base |
| --- | --- | --- |
| Détecteurs | Émettre un signal binaire de dérive assorti d'une confiance | `BaseDetector(ABC)` |
| Ensemble | Agréger les votes selon la règle 2 sur 3 | `EnsembleVoter` |
| DTD | Estimer les seuils par détecteur à partir des fenêtres récentes | `DynamicThresholdEstimator` |
| Adaptation | Choisir et appliquer une stratégie de réponse | `BaseAdaptationStrategy(ABC)` |
| Évaluation | Calculer NDCG@k, Recall@k, latence de détection, coût | `Evaluator` |
| Journalisation | Enregistrer configuration, métriques, événements, artefacts | `LabJournal` |

## Points d'accroche requis dans le recommandeur hôte

1. Embeddings utilisateur exposés en lecture et en écriture.
2. Poids de fusion α modifiable à chaud.
3. Signal de retour en ligne (clics ou notes) accessible par fenêtre.

Ces interfaces sont formalisées par le protocole `HostRecommender` ; tout recommandeur les implémentant est compatible.

## Flux de données

Décrire ici le format d'entrée (Parquet Amazon), les fenêtres temporelles, la définition du point de dérive et la séparation entraînement / évaluation.
