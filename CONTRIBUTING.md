# Contribuer

## Environnement

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

## Règles de code

- Classes plutôt que fonctions flottantes dès qu'il y a un état ou une configuration ; une classe de base abstraite par famille de composants.
- Pydantic v2 pour toute donnée franchissant une frontière de module : configuration, hyperparamètres, résultats, schémas de données.
- DRY : toute logique répétée deux fois est factorisée ; constantes et noms de colonnes centralisés dans `config/constants.py`.
- Typage complet des signatures publiques, docstrings style Google en anglais.
- Fichiers de 400 lignes au maximum, fonctions de 50 lignes au maximum, complexité cyclomatique inférieure ou égale à 10.
- `logging` uniquement, jamais `print()` dans `src/`.
- Toute source d'aléa reçoit une seed issue de la configuration.

## Avant de commiter

```bash
make format lint types test check
```

Le hook `pre-commit` rejoue le contrôle de conformité. Si un contrôle échoue pour une raison assumée, documenter l'exception dans `compliance.toml` avec une justification en une ligne plutôt que de désactiver le contrôle.

## Convention de commit

`<type>(<portée>): <résumé à l'impératif>` avec `type` parmi `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `exp`.

Exemple : `exp(ablation): ajouter la configuration C avec seuil KL dynamique`

Tout commit qui produit un résultat référence le `RUN_ID` correspondant dans son corps.

## Branches

`main` reste toujours vert. Le travail se fait sur `feat/...`, `exp/...` ou `docs/...`, fusionné après passage de la CI.
