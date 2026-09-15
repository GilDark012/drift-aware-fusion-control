#!/usr/bin/env python3
"""Compliance checker for the PFE repository.

Verifies in a single pass that the repository is organised like a public GitHub
project, that the Python code follows the agreed standards (clean code, DRY,
classes, Pydantic) and that every run left a complete, traceable trail of
artifacts. Writes a Markdown report, a JSON report and a log file, and exits with
a non-zero status when a blocking check fails.

Usage:
    python check_compliance.py --repo . --report
    python check_compliance.py --repo . --fix
    python check_compliance.py --repo . --strict      # warnings become errors

Standard library only, so it runs anywhere the project runs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import shutil
import subprocess
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

LOGGER: Final[logging.Logger] = logging.getLogger("compliance")

REQUIRED_FILES: Final[tuple[str, ...]] = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    "JOURNAL.md",
    "Makefile",
    ".pre-commit-config.yaml",
)
REQUIRED_DIRS: Final[tuple[str, ...]] = (
    "src",
    "tests",
    "docs",
    "scripts",
    "artifacts",
    "data",
)
REQUIRED_GITIGNORE_ENTRIES: Final[tuple[str, ...]] = (
    "data/",
    "__pycache__/",
    ".venv",
    "*.egg-info",
)
REQUIRED_RUN_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "metrics.csv",
    "events.csv",
    "run.log",
    "results.md",
    "manifest.json",
)
REQUIRED_DOCS: Final[tuple[str, ...]] = (
    "architecture.md",
    "experiment_protocol.md",
    "data_card.md",
)
INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "started_at",
    "finished_at",
    "task",
    "status",
    "duration_seconds",
    "git_commit",
    "description",
)

MAX_FILE_LINES: Final[int] = 400
MAX_FUNCTION_LINES: Final[int] = 50
MAX_FUNCTION_ARGS: Final[int] = 7
MIN_TEST_COVERAGE_RATIO: Final[float] = 0.6
MIN_DUPLICATE_BODY_NODES: Final[int] = 6


class Severity(StrEnum):
    """How much a failed check matters."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Status(StrEnum):
    """Outcome of a single check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(slots=True)
class CheckResult:
    """Outcome of one check, with enough detail to act on it."""

    check_id: str
    title: str
    status: Status
    severity: Severity = Severity.ERROR
    details: list[str] = field(default_factory=list)
    hint: str = ""

    @property
    def icon(self) -> str:
        """Return the glyph used in the Markdown report."""
        if self.status is Status.PASS:
            return "OK"
        if self.status is Status.SKIP:
            return "--"
        return "!!" if self.severity is Severity.ERROR else "/!\\"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.check_id,
            "title": self.title,
            "status": self.status.value,
            "severity": self.severity.value,
            "details": self.details,
            "hint": self.hint,
        }


@dataclass(slots=True)
class Context:
    """Everything a check needs to know about the repository under inspection."""

    repo: Path
    fix: bool = False
    strict: bool = False
    run_tools: bool = True
    exemptions: dict[str, str] = field(default_factory=dict)

    @property
    def src_root(self) -> Path:
        """Return the ``src`` directory."""
        return self.repo / "src"

    @property
    def packages(self) -> list[Path]:
        """Return every importable package directly under ``src``."""
        if not self.src_root.is_dir():
            return []
        return [
            path
            for path in sorted(self.src_root.iterdir())
            if path.is_dir() and (path / "__init__.py").exists()
        ]

    def python_files(self) -> list[Path]:
        """Return every Python file under ``src``, excluding caches."""
        if not self.src_root.is_dir():
            return []
        return [
            path
            for path in sorted(self.src_root.rglob("*.py"))
            if "__pycache__" not in path.parts
        ]

    def run_dirs(self) -> list[Path]:
        """Return every run directory under ``artifacts/runs``."""
        runs = self.repo / "artifacts" / "runs"
        if not runs.is_dir():
            return []
        return [path for path in sorted(runs.iterdir()) if path.is_dir()]

    def relative(self, path: Path) -> str:
        """Return a path relative to the repository root, for readable reports."""
        try:
            return str(path.relative_to(self.repo))
        except ValueError:
            return str(path)


class Check:
    """Base class for all checks.

    Subclasses declare their identity and implement :meth:`run`. Keeping every
    check behind one interface means the runner, the report and the exemption
    mechanism are written once.
    """

    check_id: str = "unset"
    title: str = "unset"
    severity: Severity = Severity.ERROR
    hint: str = ""

    def run(self, ctx: Context) -> CheckResult:
        """Execute the check against the repository.

        Args:
            ctx: Repository context.

        Returns:
            The result of the check.
        """
        raise NotImplementedError

    # Helpers shared by subclasses, so no subclass rebuilds a result by hand.

    def ok(self, *details: str) -> CheckResult:
        """Return a passing result."""
        return CheckResult(
            self.check_id, self.title, Status.PASS, self.severity, list(details)
        )

    def fail(self, details: Sequence[str]) -> CheckResult:
        """Return a failing result carrying the offending items."""
        return CheckResult(
            self.check_id,
            self.title,
            Status.FAIL,
            self.severity,
            list(details),
            self.hint,
        )

    def skip(self, reason: str) -> CheckResult:
        """Return a skipped result with the reason it could not run."""
        return CheckResult(
            self.check_id, self.title, Status.SKIP, Severity.INFO, [reason]
        )

    def verdict(self, problems: Sequence[str], success_note: str = "") -> CheckResult:
        """Return a pass or a fail depending on whether problems were found."""
        return self.fail(problems) if problems else self.ok(success_note)


# --------------------------------------------------------------------- layout


class RepositoryLayoutCheck(Check):
    """Ensure the mandatory files and directories exist."""

    check_id = "layout.structure"
    title = "Structure du dépôt (fichiers et dossiers obligatoires)"
    hint = (
        "Relancer `init_workspace.py`, ou `check_compliance.py --fix` "
        "pour les dossiers."
    )

    def run(self, ctx: Context) -> CheckResult:
        """Report every missing required path, creating directories when fixing."""
        problems: list[str] = []
        for name in REQUIRED_DIRS:
            path = ctx.repo / name
            if path.is_dir():
                continue
            if ctx.fix:
                path.mkdir(parents=True, exist_ok=True)
                (path / ".gitkeep").touch()
                LOGGER.info("created missing directory %s", name)
            else:
                problems.append(f"dossier manquant : `{name}/`")
        problems += [
            f"fichier manquant : `{name}`"
            for name in REQUIRED_FILES
            if not (ctx.repo / name).is_file()
        ]
        return self.verdict(problems, "tous les chemins requis sont présents")


class GitIgnoreCheck(Check):
    """Ensure heavy or private paths never reach the public repository."""

    check_id = "layout.gitignore"
    title = ".gitignore couvre les données et les artefacts lourds"
    severity = Severity.WARNING
    hint = "Ajouter les entrées manquantes à `.gitignore`."

    def run(self, ctx: Context) -> CheckResult:
        """Check that every mandatory ignore pattern is present."""
        path = ctx.repo / ".gitignore"
        if not path.is_file():
            return self.fail(["`.gitignore` absent"])
        content = path.read_text(encoding="utf-8")
        missing = [
            entry for entry in REQUIRED_GITIGNORE_ENTRIES if entry not in content
        ]
        return self.verdict(
            [f"entrée manquante : `{entry}`" for entry in missing],
            "les motifs essentiels sont ignorés",
        )


class DocumentationCheck(Check):
    """Ensure the documentation a jury expects is present and non-empty."""

    check_id = "docs.core"
    title = "Documentation de projet (docs/ et README)"
    severity = Severity.WARNING
    hint = "Compléter les documents de `docs/` ; un fichier vide ne compte pas."

    def run(self, ctx: Context) -> CheckResult:
        """Flag missing or placeholder documentation files."""
        problems: list[str] = []
        docs = ctx.repo / "docs"
        for name in REQUIRED_DOCS:
            path = docs / name
            if not path.is_file():
                problems.append(f"document manquant : `docs/{name}`")
            elif len(path.read_text(encoding="utf-8").split()) < 40:
                problems.append(f"document trop succinct : `docs/{name}`")
        readme = ctx.repo / "README.md"
        if readme.is_file() and len(readme.read_text(encoding="utf-8").split()) < 80:
            problems.append("`README.md` trop succinct pour un dépôt public")
        return self.verdict(problems, "documentation présente et substantielle")


# ------------------------------------------------------------------ artifacts


class RunArtifactsCheck(Check):
    """Ensure every run directory holds the full set of expected artifacts."""

    check_id = "artifacts.runs"
    title = "Complétude des artefacts de chaque run"
    hint = "Utiliser `LabJournal` comme gestionnaire de contexte pour tout produire."

    def run(self, ctx: Context) -> CheckResult:
        """Report runs missing files, and repair CSV headers when fixing."""
        runs = ctx.run_dirs()
        if not runs:
            return self.skip("aucun run enregistré pour le moment")
        problems: list[str] = []
        for run in runs:
            for name in REQUIRED_RUN_FILES:
                if not (run / name).is_file():
                    problems.append(f"`{ctx.relative(run)}` : `{name}` manquant")
            problems += self._check_manifest(ctx, run)
        return self.verdict(problems, f"{len(runs)} run(s) complet(s) et documenté(s)")

    @staticmethod
    def _check_manifest(ctx: Context, run: Path) -> list[str]:
        """Validate the manifest of a single run."""
        path = run / "manifest.json"
        if not path.is_file():
            return []
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return [f"`{ctx.relative(path)}` : JSON invalide ({error.msg})"]
        problems = [
            f"`{ctx.relative(path)}` : champ `{key}` absent"
            for key in ("run_id", "status", "started_at", "task")
            if key not in manifest
        ]
        if manifest.get("status") == "running":
            problems.append(
                f"`{ctx.relative(run)}` : run jamais clôturé (statut `running`)"
            )
        return problems


class RunIndexCheck(Check):
    """Ensure ``artifacts/index.csv`` mirrors the run directories."""

    check_id = "artifacts.index"
    title = "Registre `artifacts/index.csv` synchronisé"
    severity = Severity.WARNING
    hint = "Lancer avec `--fix` pour réindexer les runs orphelins."

    def run(self, ctx: Context) -> CheckResult:
        """Compare indexed run identifiers with the directories on disk."""
        runs = ctx.run_dirs()
        if not runs:
            return self.skip("aucun run à indexer")
        index_path = ctx.repo / "artifacts" / "index.csv"
        indexed = self._read_index(index_path)
        missing = [run for run in runs if run.name not in indexed]
        if missing and ctx.fix:
            self._reindex(index_path, missing)
            LOGGER.info("reindexed %d run(s)", len(missing))
            missing = []
        problems = [f"run absent du registre : `{run.name}`" for run in missing]
        problems += [
            f"registre : entrée sans dossier `{run_id}`"
            for run_id in indexed
            if not (ctx.repo / "artifacts" / "runs" / run_id).is_dir()
        ]
        return self.verdict(problems, f"{len(runs)} run(s) référencé(s)")

    @staticmethod
    def _read_index(path: Path) -> set[str]:
        """Return the set of run identifiers already present in the index."""
        if not path.is_file():
            return set()
        with path.open(encoding="utf-8", newline="") as handle:
            return {row.get("run_id", "") for row in csv.DictReader(handle)}

    @staticmethod
    def _reindex(path: Path, runs: Iterable[Path]) -> None:
        """Append missing runs to the index, reading each run's manifest."""
        is_new = not path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
            if is_new:
                writer.writeheader()
            for run in runs:
                manifest_path = run / "manifest.json"
                manifest: dict[str, object] = {}
                if manifest_path.is_file():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        manifest = {}
                writer.writerow(
                    {column: str(manifest.get(column, "")) for column in INDEX_COLUMNS}
                    | {"run_id": run.name}
                )


# ----------------------------------------------------------------- code style


class PythonAstCheck(Check):
    """Base class for checks that need the parsed AST of every source file."""

    def iter_modules(self, ctx: Context) -> Iterator[tuple[Path, ast.Module]]:
        """Yield each source file with its parsed module, skipping unparsable ones."""
        for path in ctx.python_files():
            try:
                yield path, ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as error:
                LOGGER.warning("cannot parse %s: %s", path, error)


class DocstringCheck(PythonAstCheck):
    """Ensure public modules, classes and functions are documented."""

    check_id = "code.docstrings"
    title = "Docstrings sur les modules, classes et fonctions publiques"
    severity = Severity.WARNING
    hint = "Ajouter une docstring style Google (résumé, Args, Returns)."

    def run(self, ctx: Context) -> CheckResult:
        """List every public definition lacking a docstring."""
        problems: list[str] = []
        for path, module in self.iter_modules(ctx):
            location = ctx.relative(path)
            if ast.get_docstring(module) is None:
                problems.append(f"`{location}` : docstring de module absente")
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if node.name.startswith("_") or ast.get_docstring(node) is not None:
                    continue
                problems.append(
                    f"`{location}:{node.lineno}` : `{node.name}` sans docstring"
                )
        return self.verdict(problems, "tout le code public est documenté")


class ComplexityCheck(PythonAstCheck):
    """Ensure files and functions stay small enough to be read in one sitting."""

    check_id = "code.complexity"
    title = "Taille des fichiers et des fonctions (clean code)"
    severity = Severity.WARNING
    hint = (
        f"Découper au-delà de {MAX_FILE_LINES} lignes par fichier ou "
        f"{MAX_FUNCTION_LINES} lignes par fonction."
    )

    def run(self, ctx: Context) -> CheckResult:
        """Report oversized files, functions and parameter lists."""
        problems: list[str] = []
        for path, module in self.iter_modules(ctx):
            location = ctx.relative(path)
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_FILE_LINES:
                problems.append(
                    f"`{location}` : {line_count} lignes (> {MAX_FILE_LINES})"
                )
            for node in ast.walk(module):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                problems += self._inspect_function(location, node)
        return self.verdict(problems, "granularité conforme")

    @staticmethod
    def _inspect_function(
        location: str, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Return the issues found on a single function definition."""
        problems: list[str] = []
        length = (node.end_lineno or node.lineno) - node.lineno
        if length > MAX_FUNCTION_LINES:
            problems.append(
                f"`{location}:{node.lineno}` : `{node.name}` fait {length} lignes "
                f"(> {MAX_FUNCTION_LINES})"
            )
        arg_count = len(node.args.args) + len(node.args.kwonlyargs)
        if arg_count > MAX_FUNCTION_ARGS:
            problems.append(
                f"`{location}:{node.lineno}` : `{node.name}` prend {arg_count} "
                f"paramètres (> {MAX_FUNCTION_ARGS}) — regrouper dans un modèle"
            )
        return problems


class TypingCheck(PythonAstCheck):
    """Ensure public functions carry type annotations."""

    check_id = "code.typing"
    title = "Annotations de type sur les signatures publiques"
    severity = Severity.WARNING
    hint = "Annoter paramètres et valeur de retour ; `mypy` doit passer."

    def run(self, ctx: Context) -> CheckResult:
        """Report public functions with unannotated parameters or return values."""
        problems: list[str] = []
        for path, module in self.iter_modules(ctx):
            location = ctx.relative(path)
            for node in ast.walk(module):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                if node.returns is None:
                    problems.append(
                        f"`{location}:{node.lineno}` : `{node.name}` sans retour typé"
                    )
                problems += [
                    f"`{location}:{node.lineno}` : paramètre `{arg.arg}` non annoté"
                    for arg in (*node.args.args, *node.args.kwonlyargs)
                    if arg.annotation is None and arg.arg not in {"self", "cls"}
                ]
        return self.verdict(problems, "signatures publiques entièrement typées")


class PrintStatementCheck(PythonAstCheck):
    """Ensure ``src`` logs instead of printing."""

    check_id = "code.no_print"
    title = "Aucun `print()` dans `src/` (journalisation via logging)"
    hint = "Remplacer par `logger.info(...)` ou `journal.event(...)`."

    def run(self, ctx: Context) -> CheckResult:
        """Locate every call to the built-in ``print`` inside the package."""
        problems: list[str] = []
        for path, module in self.iter_modules(ctx):
            location = ctx.relative(path)
            problems += [
                f"`{location}:{node.lineno}` : appel à `print()`"
                for node in ast.walk(module)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ]
        return self.verdict(problems, "journalisation propre")


class HardcodedPathCheck(PythonAstCheck):
    """Ensure no machine-specific path leaks into a public repository."""

    check_id = "code.no_absolute_paths"
    title = "Aucun chemin absolu en dur dans `src/`"
    hint = "Passer les chemins par la configuration Pydantic, pas en littéral."

    def run(self, ctx: Context) -> CheckResult:
        """Detect string literals that look like absolute filesystem paths."""
        problems: list[str] = []
        prefixes = ("/home/", "/Users/", "/mnt/", "C:\\", "/content/")
        for path, module in self.iter_modules(ctx):
            location = ctx.relative(path)
            problems += [
                f"`{location}:{node.lineno}` : chemin absolu `{node.value}`"
                for node in ast.walk(module)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(prefixes)
            ]
        return self.verdict(problems, "aucun chemin machine-dépendant")


class PydanticUsageCheck(PythonAstCheck):
    """Ensure configuration and data schemas actually rely on Pydantic."""

    check_id = "code.pydantic"
    title = "Modèles Pydantic pour la configuration et les schémas"
    hint = "Déclarer les configs comme `BaseModel` (frozen, extra='forbid')."

    def run(self, ctx: Context) -> CheckResult:
        """Confirm at least one Pydantic model exists in each package config module."""
        packages = ctx.packages
        if not packages:
            return self.skip("aucun package détecté sous `src/`")
        problems: list[str] = []
        for package in packages:
            config_dir = package / "config"
            if not config_dir.is_dir():
                problems.append(f"`{ctx.relative(package)}` : module `config/` absent")
                continue
            if not self._has_model(config_dir):
                problems.append(
                    f"`{ctx.relative(config_dir)}` : aucune classe héritant de "
                    "`BaseModel`/`BaseSettings`"
                )
        return self.verdict(problems, "configuration validée par Pydantic")

    @staticmethod
    def _has_model(config_dir: Path) -> bool:
        """Return True if any module in the directory declares a Pydantic model."""
        bases = {"BaseModel", "BaseSettings"}
        for path in config_dir.rglob("*.py"):
            try:
                module = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(module):
                if not isinstance(node, ast.ClassDef):
                    continue
                names = {
                    base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                    for base in node.bases
                }
                if names & bases:
                    return True
        return False


class DuplicationCheck(PythonAstCheck):
    """Ensure the same logic is not written twice (DRY)."""

    check_id = "code.dry"
    title = "Absence de duplication de logique (DRY)"
    severity = Severity.WARNING
    hint = "Factoriser dans une fonction ou une classe de base partagée."

    def run(self, ctx: Context) -> CheckResult:
        """Group functions by normalised body and report identical implementations."""
        bodies: dict[str, list[str]] = defaultdict(list)
        for path, module in self.iter_modules(ctx):
            location = ctx.relative(path)
            for node in ast.walk(module):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = [stmt for stmt in node.body if not self._is_docstring(stmt)]
                if len(list(ast.walk(ast.Module(body=body, type_ignores=[])))) < (
                    MIN_DUPLICATE_BODY_NODES
                ):
                    continue
                signature = ast.dump(
                    ast.Module(body=body, type_ignores=[]), annotate_fields=False
                )
                bodies[signature].append(f"`{location}:{node.lineno}` `{node.name}`")
        problems = [
            "implémentations identiques : " + ", ".join(sites)
            for sites in bodies.values()
            if len(sites) > 1
        ]
        return self.verdict(problems, "aucune duplication significative détectée")

    @staticmethod
    def _is_docstring(node: ast.stmt) -> bool:
        """Return True if the statement is a bare string expression."""
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


class TestCoverageCheck(Check):
    """Ensure the test suite mirrors the source tree."""

    check_id = "tests.mirror"
    title = "Chaque module de `src/` a un test correspondant"
    severity = Severity.WARNING
    hint = "Créer `tests/test_<module>.py` pour les modules non couverts."

    def run(self, ctx: Context) -> CheckResult:
        """Compare module stems with test file stems."""
        modules = [
            path
            for path in ctx.python_files()
            if path.name not in {"__init__.py", "__main__.py"}
        ]
        if not modules:
            return self.skip("aucun module à couvrir pour l'instant")
        tests_dir = ctx.repo / "tests"
        test_names = {path.stem for path in tests_dir.rglob("test_*.py")}
        uncovered = [
            ctx.relative(path)
            for path in modules
            if f"test_{path.stem}" not in test_names
        ]
        ratio = 1 - len(uncovered) / len(modules)
        if ratio >= MIN_TEST_COVERAGE_RATIO:
            return self.ok(f"{ratio:.0%} des modules ont un test dédié")
        return self.fail(
            [f"module sans test : `{name}`" for name in uncovered]
            + [
                f"ratio de modules testés : {ratio:.0%} "
                f"(< {MIN_TEST_COVERAGE_RATIO:.0%})"
            ]
        )


# ------------------------------------------------------------ external tools


class ExternalToolCheck(Check):
    """Base class for checks delegating to an installed command-line tool."""

    command: tuple[str, ...] = ()
    tool_name: str = ""

    def run(self, ctx: Context) -> CheckResult:
        """Run the tool at the repository root and interpret its exit code."""
        if not ctx.run_tools:
            return self.skip("exécution des outils externes désactivée (`--no-tools`)")
        if shutil.which(self.command[0]) is None:
            return self.skip(
                f"`{self.tool_name}` non installé — `pip install {self.tool_name}`"
            )
        try:
            result = subprocess.run(
                self.command,
                cwd=ctx.repo,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return self.fail([f"`{self.tool_name}` n'a pas pu s'exécuter : {error}"])
        if result.returncode == 0:
            return self.ok(f"`{self.tool_name}` ne remonte aucun problème")
        output = (result.stdout + result.stderr).strip().splitlines()
        return self.fail(output[-25:] or [f"`{self.tool_name}` a échoué"])


class RuffCheck(ExternalToolCheck):
    """Lint and format compliance via ruff."""

    check_id = "tools.ruff"
    title = "Lint et formatage (`ruff`)"
    tool_name = "ruff"
    command = ("ruff", "check", ".")
    hint = "Corriger avec `ruff check --fix .` puis `ruff format .`."


class MypyCheck(ExternalToolCheck):
    """Static type checking via mypy."""

    check_id = "tools.mypy"
    title = "Vérification statique des types (`mypy`)"
    tool_name = "mypy"
    command = ("mypy", "src")
    severity = Severity.WARNING
    hint = "Corriger les types signalés ou préciser les annotations."


class PytestCheck(ExternalToolCheck):
    """Test suite execution via pytest."""

    check_id = "tools.pytest"
    title = "Suite de tests (`pytest`)"
    tool_name = "pytest"
    command = ("pytest", "-q", "--no-header")
    hint = "Réparer les tests en échec avant de commiter."


# ---------------------------------------------------------------- reporting


class ComplianceReport:
    """Aggregate results and render them as Markdown, JSON and log lines."""

    def __init__(self, repo: Path, results: Sequence[CheckResult]) -> None:
        """Store the results and the repository they describe."""
        self.repo = repo
        self.results = list(results)
        self.generated_at = datetime.now()

    @property
    def failures(self) -> list[CheckResult]:
        """Return failing checks whose severity is ``ERROR``."""
        return [
            result
            for result in self.results
            if result.status is Status.FAIL and result.severity is Severity.ERROR
        ]

    @property
    def warnings(self) -> list[CheckResult]:
        """Return failing checks whose severity is ``WARNING``."""
        return [
            result
            for result in self.results
            if result.status is Status.FAIL and result.severity is Severity.WARNING
        ]

    @property
    def passed(self) -> list[CheckResult]:
        """Return passing checks."""
        return [result for result in self.results if result.status is Status.PASS]

    def headline(self) -> str:
        """Return the one-line verdict shown in the console."""
        return (
            f"{len(self.passed)} contrôle(s) OK, "
            f"{len(self.warnings)} avertissement(s), "
            f"{len(self.failures)} erreur(s)"
        )

    def to_markdown(self) -> str:
        """Render the full report as Markdown."""
        lines = [
            "# Rapport de conformité",
            "",
            f"- **Dépôt** : `{self.repo}`",
            f"- **Généré le** : {self.generated_at.isoformat(timespec='seconds')}",
            f"- **Verdict** : {self.headline()}",
            "",
            "## Synthèse",
            "",
            "| | Contrôle | Statut |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {result.icon} | {result.title} | {result.status.value} |"
            for result in self.results
        ]
        lines += ["", "## Détails", ""]
        for result in self.results:
            if result.status is Status.PASS and not result.details:
                continue
            lines += [f"### {result.icon} {result.title} (`{result.check_id}`)", ""]
            lines += [f"- {detail}" for detail in result.details[:40]]
            if len(result.details) > 40:
                lines.append(f"- … et {len(result.details) - 40} autre(s)")
            if result.status is Status.FAIL and result.hint:
                lines += ["", f"> **Correctif** : {result.hint}"]
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Render the report as JSON for CI consumption."""
        return json.dumps(
            {
                "repo": str(self.repo),
                "generated_at": self.generated_at.isoformat(),
                "passed": len(self.passed),
                "warnings": len(self.warnings),
                "errors": len(self.failures),
                "checks": [result.to_dict() for result in self.results],
            },
            indent=2,
            ensure_ascii=False,
        )

    def write(self, directory: Path) -> Path:
        """Write the timestamped report plus the ``latest`` copies.

        Args:
            directory: Destination directory, created if needed.

        Returns:
            Path of the timestamped Markdown report.
        """
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.generated_at.strftime("%Y%m%d-%H%M%S")
        markdown = self.to_markdown()
        report_path = directory / f"{stamp}.md"
        report_path.write_text(markdown, encoding="utf-8")
        (directory / "latest.md").write_text(markdown, encoding="utf-8")
        (directory / "latest.json").write_text(self.to_json(), encoding="utf-8")
        return report_path


# -------------------------------------------------------------------- runner


CHECKS: Final[tuple[type[Check], ...]] = (
    RepositoryLayoutCheck,
    GitIgnoreCheck,
    DocumentationCheck,
    RunArtifactsCheck,
    RunIndexCheck,
    DocstringCheck,
    TypingCheck,
    ComplexityCheck,
    PrintStatementCheck,
    HardcodedPathCheck,
    PydanticUsageCheck,
    DuplicationCheck,
    TestCoverageCheck,
    RuffCheck,
    MypyCheck,
    PytestCheck,
)


def load_exemptions(repo: Path) -> dict[str, str]:
    """Read documented exemptions from ``compliance.toml``.

    The file maps a check id to a one-line justification. Exempted checks are still
    executed and still reported, but they never make the run fail — the point is to
    keep the compromise visible rather than to hide it.
    """
    path = repo / "compliance.toml"
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        LOGGER.warning("compliance.toml illisible : %s", error)
        return {}
    exemptions = data.get("exemptions", {})
    return {str(key): str(value) for key, value in exemptions.items()}


def run_checks(ctx: Context) -> list[CheckResult]:
    """Execute every check in order and apply documented exemptions."""
    results: list[CheckResult] = []
    for check_class in CHECKS:
        check = check_class()
        LOGGER.debug("running check %s", check.check_id)
        try:
            result = check.run(ctx)
        except Exception as error:  # noqa: BLE001 - a broken check must not stop all
            LOGGER.exception("check %s crashed", check.check_id)
            result = CheckResult(
                check.check_id,
                check.title,
                Status.FAIL,
                Severity.WARNING,
                [f"le contrôle a levé une exception : {error!r}"],
            )
        if result.status is Status.FAIL and result.check_id in ctx.exemptions:
            result.severity = Severity.WARNING
            result.details.append(
                f"exemption documentée : {ctx.exemptions[result.check_id]}"
            )
        if ctx.strict and result.status is Status.FAIL:
            result.severity = Severity.ERROR
        results.append(result)
    return results


def configure_logging(log_path: Path | None, verbose: bool) -> None:
    """Send log records to stderr and, when requested, to a log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", default=".", help="racine du dépôt à contrôler")
    parser.add_argument(
        "--report",
        action="store_true",
        help="écrire les rapports dans reports/compliance/",
    )
    parser.add_argument(
        "--fix", action="store_true", help="corriger ce qui est automatisable"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="traiter les avertissements comme des erreurs",
    )
    parser.add_argument(
        "--no-tools", action="store_true", help="ne pas lancer ruff/mypy/pytest"
    )
    parser.add_argument("--verbose", action="store_true", help="journalisation DEBUG")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run the checks, write the reports, return the exit code."""
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    reports_dir = repo / "reports" / "compliance"
    configure_logging(
        reports_dir / "compliance.log" if args.report else None, args.verbose
    )

    if not repo.is_dir():
        LOGGER.error("dépôt introuvable : %s", repo)
        return 2

    ctx = Context(
        repo=repo,
        fix=args.fix,
        strict=args.strict,
        run_tools=not args.no_tools,
        exemptions=load_exemptions(repo),
    )
    report = ComplianceReport(repo, run_checks(ctx))

    if args.report:
        path = report.write(reports_dir)
        LOGGER.info("rapport écrit : %s", path)

    LOGGER.info("verdict : %s", report.headline())
    for result in report.failures + report.warnings:
        first = result.details[0] if result.details else ""
        LOGGER.warning("%s — %s", result.title, first)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
