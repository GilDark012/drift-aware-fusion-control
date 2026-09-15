# Data card — Amazon Reviews'23

Fiche descriptive du jeu de données, renseignée à partir des artefacts effectivement
produits par le pipeline (`data/processed/`). Les valeurs sont mesurées, non estimées.

## Source

| Champ | Valeur |
| --- | --- |
| Jeu de données | **Amazon Reviews'23** (McAuley Lab, UC San Diego) — catégories retenues : `Electronics`, `Clothing_Shoes_and_Jewelry`, `Books` |
| Référence | Hou, Li, He, Yan, Chen & McAuley (2024), *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952 |
| URL | https://amazon-reviews-2023.github.io/ (téléchargement direct des trois fichiers bruts par catégorie : `Clothing_Shoes_and_Jewelry`, `Electronics`, `Books`) |
| Version | Édition 2023 ; le corpus complet couvre mai 1996 → sept. 2023. **Ce projet applique un filtre temporel de prétraitement à 2014-01-01 → 2018-12-30** (les valeurs ci-dessous portent sur la fenêtre filtrée, pas sur le corpus entier). |
| Licence / conditions | **Aucune licence explicite** n'est publiée, ni sur la page source, ni sur la page HuggingFace du jeu de données (`McAuley-Lab/Amazon-Reviews-2023`) — vérifié le 2026-08-27. Diffusé par le McAuley Lab (UC San Diego) pour un usage **recherche** ; en l'absence de licence formelle, confirmer les conditions auprès des auteurs (contact : yphou@ucsd.edu) avant toute redistribution. Ce dépôt ne redistribue **pas** les données brutes (voir `.gitignore`). |
| Date de récupération | avril 2026 (téléchargement direct des trois fichiers bruts depuis la page source) |
| Empreintes SHA-256 (artefacts traités, commit `060c8bb`) | `amazon_clean.parquet` : `13ed78d4…a38e99` · `train.parquet` : `c88c5a4b…bc4a54d` · `test.parquet` : `011831f7…1aebd3b` · `split_cutoffs.json` : `2deab5e5…6495293` |
| Dépôt / commit | ce dépôt, commit `060c8bb744099a1fe00ce539b70aead82fe49c94` |

Les empreintes complètes sont reproductibles via
`python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('data/processed/amazon_clean.parquet').read_bytes()).hexdigest())"`.

## Contenu

Schéma de `data/processed/amazon_clean.parquet` (7 colonnes) :

| Champ | Type | Description |
| --- | --- | --- |
| `user_id` | string | identifiant utilisateur pseudonymisé (fourni tel quel par la source) |
| `item_id` | string | identifiant produit (ASIN) |
| `category` | string | catégorie Amazon d'origine du produit |
| `preference_score` | float ∈ [1, 5] | note explicite attribuée par l'utilisateur |
| `timestamp` | datetime | date de l'interaction |
| `user_idx` | int64 | index dense utilisateur (dérivé) |
| `item_idx` | int64 | index dense produit (dérivé) |

## Volumétrie et sparsité (mesurée)

| Grandeur | Valeur |
| --- | --- |
| Interactions | **3 971 952** |
| Utilisateurs | **130 987** |
| Produits | **765 033** |
| Densité de la matrice | **3,96 × 10⁻⁵** |
| Interactions / utilisateur | moyenne 30,3 · médiane 26 |
| Étendue temporelle | 2014-01-01 → 2018-12-30 |
| Répartition par catégorie | Electronics 1 556 253 · Clothing_Shoes_and_Jewelry 1 316 061 · Books 1 099 638 |

La densité de ~4 × 10⁻⁵ (un utilisateur voit en moyenne 4 produits sur 100 000)
explique directement les valeurs absolues faibles de NDCG@10 : c'est une propriété du
jeu de données, documentée ici pour être citable en soutenance, et non un défaut de la
méthode. L'analyse porte sur les écarts **relatifs** à la baseline pré-dérive et sur la
latence de détection.

## Découpage temporel (global, chronologique — sans fuite)

Coupures issues de `data/processed/split_cutoffs.json` :
`T1 = 2017-06-09 13:25:32`, `T2 = 2017-12-10 09:50:49`.

| Split | Condition | Lignes | Utilisateurs | Produits | Fenêtre |
| --- | --- | --: | --: | --: | --- |
| Entraînement | `t < T1` | 2 780 366 | 127 344 | 589 045 | 2014-01-01 → 2017-06-09 |
| Validation | `T1 ≤ t < T2` | — | — | — | 2017-06-09 → 2017-12-10 |
| Test | `t ≥ T2` | 794 391 | 98 852 | 286 622 | 2017-12-10 → 2018-12-30 |

Le découpage est **global et chronologique** : aucune interaction future ne peut
influencer une prédiction passée. Les seuils de détection sont réglés sur la validation
uniquement ; le test n'est jamais utilisé pour le réglage.

## Prétraitement

1. **Filtrage k-core** (`k = 5`, standard SASRec / GRU4Rec) : seuls les produits ayant
   ≥ 5 interactions d'entraînement sont conservés — les produits à support insuffisant
   ne peuvent être ni appris ni classés (voir
   [ADR-0006](decisions/ADR-0006-kcore-vocabulary-filtering.md)).
2. **Découpage temporel** entraînement / validation / test aux coupures ci-dessus.
3. **Vocabulaire produit construit sur l'entraînement seul** : les produits de
   validation / test absents de l'entraînement sont mappés vers un index OOV réservé —
   exactement le phénomène « article inédit » attendu à l'échelle réelle.
4. **Injection de dérive contrôlée** à onset connu pour mesurer détection et latence
   (`src/drift_reco/data/injection.py`), en complément de la dérive naturelle non
   annotée.

Chaque étape est implémentée dans `src/drift_reco/data/` et rejouable via le pipeline ;
aucun traitement manuel. Chaque exécution écrit les empreintes SHA-256 des artefacts
produits dans le manifeste de son run.

## Limites connues

- **Biais de sélection** : seuls les utilisateurs actifs (avis rédigés) sont
  représentés ; population non représentative de l'ensemble des acheteurs.
- **Dérive naturelle non annotée** : les points de rupture évalués sont partiellement
  **injectés** (onset connu) ; la dérive réelle est plus faible et plus bruitée.
- **Retour explicite uniquement** (notes 1–5) : pas de signal négatif implicite ni de
  contexte de session.
- **Catégorie ≠ préférence** : un changement de catégorie est un *indicateur* de
  dérive, jamais sa définition (cf. `docs/scope.md`).
