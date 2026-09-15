# Cadrage du projet

## Question de recherche

Une méthodologie unifiée de détection de dérive peut-elle maintenir la qualité de recommandation dans une marge définie par rapport à une baseline pré-dérive, tout en gardant une latence de détection faible ?

## Ce que le projet livre

- Une méthodologie de prise en compte de la dérive combinant détection ensembliste, détermination dynamique de seuils et réponse adaptative.
- Un prototype qui sert de **preuve de la méthodologie**, et non de produit déployé.
- Une étude d'ablation sur les configurations A à E, comparée à des baselines.

## Ce que le projet ne livre pas, et pourquoi

| Hors périmètre | Justification |
| --- | --- |
| Déploiement en production, service en ligne | Le prototype démontre la méthodologie ; l'ingénierie de mise en service relève d'un travail distinct et ne conditionne pas la validité des résultats. |
| Agnosticisme complet au modèle | La méthodologie exige trois points d'accroche dans le recommandeur hôte (voir ci-dessous). Le cadrage exact est « méthodologie de prise en compte de la dérive pour recommandeurs à fusion long terme / court terme ». |
| Reproduction exhaustive d'AdaMoE | Risque de crédibilité disproportionné par rapport à l'apport ; la comparaison se fait sur des baselines maîtrisées et documentées. |

## Hypothèses d'intégration

La méthodologie suppose que le recommandeur hôte expose :

1. des embeddings utilisateur accessibles,
2. un poids de fusion α ajustable entre composantes long terme et court terme,
3. un signal de retour en ligne exploitable.

Ces trois exigences sont des conditions d'application explicites, pas des limitations découvertes après coup.

## Attentes sur les métriques

Les valeurs absolues de NDCG@10 seront faibles sur des données Amazon très éparses. C'est un comportement attendu du jeu de données et non un défaut de la méthode : l'analyse porte sur les écarts relatifs à la baseline pré-dérive et sur la latence de détection.

## Risques identifiés

| Risque | Impact | Atténuation |
| --- | --- | --- |
| Latence de génération des candidats DTD | Pression sur le calendrier | Fixer une enveloppe de calcul par expérience, préparer une grille réduite de repli |
| Crédibilité de la reproduction d'AdaMoE | Contestation en soutenance | Hors périmètre, remplacé par des baselines maîtrisées et documentées |
| Métriques absolues faibles mal interprétées | Malentendu en soutenance | Contextualisation explicite dans le README et les résultats |
