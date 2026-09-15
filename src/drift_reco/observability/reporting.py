"""Rendering of the human-readable outputs of a run.

``metrics.csv`` and ``manifest.json`` are for machines; ``results.md`` and
``JOURNAL.md`` are for people — a supervisor, a jury, or the author six weeks
later. Keeping the rendering here means the presentation can change without
touching the recording logic.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import INDEX_COLUMNS, RunManifest

__all__ = ["ResultsRenderer", "RunIndex"]

_MARKDOWN_TABLE_HEADER = "| metric | split | step | value | unit | context |"
_MARKDOWN_TABLE_RULE = "| --- | --- | --- | --- | --- | --- |"
_MARKDOWN_TABLE_ROW = "| {metric} | {split} | {step} | {value} | {unit} | {context} |"


class ResultsRenderer:
    """Turn the artifacts of a finished run into ``results.md``."""

    def __init__(self, run_dir: Path) -> None:
        """Store the directory holding the run artifacts."""
        self.run_dir = run_dir

    def render(
        self,
        manifest: RunManifest,
        params: dict[str, Any],
        summary: dict[str, Any] | None,
        notes: Sequence[str],
        figures: Sequence[str],
    ) -> Path:
        """Write ``results.md`` and return its path.

        Args:
            manifest: Provenance record of the run.
            params: Parameters as declared in the run configuration.
            summary: Narrative content provided through ``LabJournal.summary``.
            notes: Timestamped free-form notes.
            figures: File names of the figures saved during the run.

        Returns:
            Path of the written Markdown file.
        """
        summary = summary or {}
        lines = self._header(manifest)
        lines += self._prose(manifest, summary, params)
        lines += ["## Results", "", self.metrics_table(), ""]
        for title, key in (
            ("Findings", "findings"),
            ("Limitations", "limitations"),
            ("Next steps", "next_steps"),
        ):
            lines += self._bullets(title, summary.get(key, []))
        lines += self._figures(figures)
        lines += self._notes(notes)
        lines += self._error(manifest)

        path = self.run_dir / "results.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _header(manifest: RunManifest) -> list[str]:
        """Return the metadata block opening the report."""
        return [
            f"# Run `{manifest.run_id}`",
            "",
            f"- **Task**: {manifest.task}",
            f"- **Status**: {manifest.status}",
            f"- **Started**: {manifest.started_at.isoformat()}",
            f"- **Duration**: {manifest.duration_seconds or 0.0:.2f} s",
            f"- **Git commit**: {manifest.git_commit or 'n/a'}",
            f"- **Seed**: {manifest.seed}",
            f"- **Dataset**: {manifest.dataset or 'n/a'}",
            "",
        ]

    @staticmethod
    def _prose(
        manifest: RunManifest, summary: dict[str, Any], params: dict[str, Any]
    ) -> list[str]:
        """Return the objective, method and configuration sections."""
        return [
            "## Objective",
            "",
            str(summary.get("objective", manifest.description)),
            "",
            "## Method",
            "",
            str(summary.get("method", "See `config.json` for the exact parameters.")),
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(params, indent=2, ensure_ascii=False, default=str),
            "```",
            "",
        ]

    @staticmethod
    def _bullets(title: str, items: Sequence[str]) -> list[str]:
        """Return a bulleted section, or nothing when there is nothing to say."""
        if not items:
            return []
        return [f"## {title}", "", *[f"- {item}" for item in items], ""]

    @staticmethod
    def _figures(figures: Sequence[str]) -> list[str]:
        """Return the figure gallery section."""
        if not figures:
            return []
        gallery = [f"![{name}](figures/{name})" for name in figures]
        return ["## Figures", "", *gallery, ""]

    @staticmethod
    def _notes(notes: Sequence[str]) -> list[str]:
        """Return the notes section as a verbatim block."""
        if not notes:
            return []
        return ["## Notes", "", "```text", *notes, "```", ""]

    @staticmethod
    def _error(manifest: RunManifest) -> list[str]:
        """Return the traceback section for a failed run."""
        if not manifest.error:
            return []
        return ["## Error", "", "```text", manifest.error.strip(), "```", ""]

    def metrics_table(self) -> str:
        """Return ``metrics.csv`` rendered as a Markdown table."""
        path = self.run_dir / "metrics.csv"
        if not path.is_file():
            return "_No metric recorded._"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return "_No metric recorded._"
        body = [_MARKDOWN_TABLE_ROW.format(**row) for row in rows]
        return "\n".join([_MARKDOWN_TABLE_HEADER, _MARKDOWN_TABLE_RULE, *body])


class RunIndex:
    """Maintain the repository-level views over all runs.

    Two views are kept in sync: ``artifacts/index.csv`` for querying and
    ``JOURNAL.md`` for reading.
    """

    def __init__(self, artifacts_root: Path, repo_root: Path) -> None:
        """Store the artifacts directory and the repository root."""
        self.artifacts_root = artifacts_root
        self.repo_root = repo_root

    def append(self, manifest: RunManifest) -> None:
        """Record a finished run in both the CSV index and the journal."""
        self._append_csv(manifest)
        self._append_journal(manifest)

    def _append_csv(self, manifest: RunManifest) -> None:
        """Append one row to ``index.csv``, writing the header on first use."""
        path = self.artifacts_root / "index.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.is_file()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
            if is_new:
                writer.writeheader()
            writer.writerow(manifest.as_index_row())

    def _append_journal(self, manifest: RunManifest) -> None:
        """Append a readable entry to ``JOURNAL.md``, creating it if needed."""
        path = self.repo_root / "JOURNAL.md"
        if not path.is_file():
            path.write_text(
                "# Journal de bord\n\nJournal chronologique des runs, "
                "alimenté automatiquement par `LabJournal`.\n",
                encoding="utf-8",
            )
        entry = (
            f"\n## {manifest.started_at.strftime('%Y-%m-%d %H:%M')} — "
            f"{manifest.task} (`{manifest.status}`)\n\n"
            f"{manifest.description}\n\n"
            f"- Run : [`{manifest.run_id}`]"
            f"(artifacts/runs/{manifest.run_id}/results.md)\n"
            f"- Métriques : {manifest.metric_count} | "
            f"Événements : {manifest.event_count} | "
            f"Durée : {manifest.duration_seconds or 0.0:.1f} s\n"
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
