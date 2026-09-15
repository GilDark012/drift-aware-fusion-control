"""Structured run journaling for the drift-awareness PFE.

Every unit of work — a training run, an ablation, a baseline computation, even an
exploratory analysis — is recorded as a *run*: a timestamped directory holding the
configuration, the metrics, the event trace, the figures and a human-readable
summary. The point is that any number reported in the thesis can be traced back to
the exact configuration and code revision that produced it.

Typical usage:

    config = RunConfig(task="ablation-c", description="2-of-3 vote, dynamic KL")
    with LabJournal(config) as journal:
        journal.event("data", "load_start", detail="amazon_reviews.parquet")
        journal.metric("ndcg@10", 0.0412, split="test", step=0)
        journal.note("Low absolute NDCG is expected on sparse Amazon data.")
        journal.summary(objective="...", method="...", findings=["..."])
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final, Literal, Self

from .models import (
    EVENT_COLUMNS,
    METRIC_COLUMNS,
    EventRecord,
    LogLevel,
    MetricRecord,
    RunConfig,
    RunManifest,
    RunStatus,
    git_commit,
    hash_file,
    slugify,
    utcnow,
)
from .reporting import ResultsRenderer, RunIndex

__all__ = ["LabJournal", "RunConfig", "MetricRecord", "EventRecord", "RunManifest"]

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


class _CsvStream:
    """An open CSV file that flushes after every row.

    Flushing eagerly costs almost nothing at the rate metrics are produced, and it
    means a run killed mid-flight still leaves behind every measurement it took.
    """

    def __init__(self, path: Path, columns: Iterable[str]) -> None:
        """Open the file and write its header row."""
        self._handle: IO[str] = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(columns))
        self._writer.writeheader()
        self._handle.flush()

    def write(self, row: dict[str, str]) -> None:
        """Append one row and flush it to disk."""
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._handle.flush()
        self._handle.close()


class LabJournal:
    """Context manager that records everything a run produces.

    On entry it creates ``<root>/runs/<run_id>/``, opens ``metrics.csv``,
    ``events.csv`` and ``run.log``, and writes ``config.json``. On exit it writes
    ``results.md`` and ``manifest.json``, appends a row to ``<root>/index.csv`` and
    an entry to ``JOURNAL.md``. Exceptions are recorded as a failed run and then
    re-raised, so a crash still leaves a documented trail.
    """

    def __init__(
        self,
        config: RunConfig,
        root: str | Path = "artifacts",
        repo_root: str | Path | None = None,
        console: bool = True,
    ) -> None:
        """Prepare the journal without touching the filesystem yet.

        Args:
            config: Validated run configuration.
            root: Directory holding ``runs/`` and ``index.csv``.
            repo_root: Repository root, used for the Git commit and ``JOURNAL.md``.
            console: Whether log records are also echoed to stderr.
        """
        self.config = config
        self.root = Path(root)
        self.repo_root = Path(repo_root) if repo_root else self.root.parent
        self.console = console
        self.started_at = utcnow()
        self.run_id = (
            f"{self.started_at.strftime('%Y%m%d-%H%M%S')}__{slugify(config.task)}"
        )
        self.run_dir = self.root / "runs" / self.run_id
        self.logger = logging.getLogger(f"lab.{self.run_id}")

        self._metrics: _CsvStream | None = None
        self._events: _CsvStream | None = None
        self._handlers: list[logging.Handler] = []
        self._metric_count = 0
        self._event_count = 0
        self._notes: list[str] = []
        self._figures: list[str] = []
        self._summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------ setup

    def __enter__(self) -> Self:
        """Create the run directory and open every output stream."""
        for sub in ("figures", "data"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self._write_json("config.json", self.config.model_dump(mode="json"))
        self._metrics = _CsvStream(self.run_dir / "metrics.csv", METRIC_COLUMNS)
        self._events = _CsvStream(self.run_dir / "events.csv", EVENT_COLUMNS)
        self._configure_logging()
        self.event("run", "start", detail=self.config.description)
        return self

    def _configure_logging(self) -> None:
        """Attach a file handler, and a console handler when requested."""
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        file_handler = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        self._handlers.append(file_handler)
        if self.console:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setLevel(logging.INFO)
            self._handlers.append(stream_handler)

        for handler in self._handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # ------------------------------------------------------------- recording

    def metric(
        self,
        metric: str,
        value: float,
        *,
        step: int = 0,
        split: str = "all",
        unit: str = "",
        context: str = "",
    ) -> MetricRecord:
        """Record one measurement and append it to ``metrics.csv``.

        Args:
            metric: Metric name, e.g. ``"ndcg@10"`` or ``"detection_latency"``.
            value: Measured value.
            step: Training step, epoch or drift-window index.
            split: Data split the value refers to.
            unit: Optional unit, e.g. ``"windows"`` or ``"seconds"``.
            context: Free text qualifying the measurement.

        Returns:
            The stored record.
        """
        record = MetricRecord(
            metric=metric,
            value=value,
            step=step,
            split=split,
            unit=unit,
            context=context,
        )
        self._stream(self._metrics).write(record.as_row())
        self._metric_count += 1
        self.logger.info("metric %s[%s]=%s step=%s", metric, split, value, step)
        return record

    def event(
        self,
        stage: str,
        event: str,
        *,
        level: LogLevel = "INFO",
        detail: str = "",
    ) -> EventRecord:
        """Record a lifecycle event and append it to ``events.csv``.

        Args:
            stage: Pipeline stage, e.g. ``"data"``, ``"detection"``, ``"adaptation"``.
            event: Event name, e.g. ``"drift_detected"``.
            level: Severity of the event.
            detail: Free text describing the event.

        Returns:
            The stored record.
        """
        record = EventRecord(stage=stage, event=event, level=level, detail=detail)
        self._stream(self._events).write(record.as_row())
        self._event_count += 1
        self.logger.log(getattr(logging, level), "[%s] %s %s", stage, event, detail)
        return record

    def note(self, text: str) -> None:
        """Append a timestamped free-form observation to ``notes.txt``.

        Notes carry what metrics cannot: doubts, surprises, ideas to follow up. They
        are what makes a decision replayable months later.
        """
        stamped = f"[{utcnow().isoformat()}] {text}"
        self._notes.append(stamped)
        with (self.run_dir / "notes.txt").open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")
        self.logger.debug("note: %s", text)

    def figure(self, fig: Any, name: str, extension: str = "png") -> Path:
        """Save a Matplotlib figure under ``figures/`` and record it.

        Args:
            fig: Object exposing ``savefig``.
            name: Base filename, without extension.
            extension: Image format.

        Returns:
            Path of the written file.
        """
        path = self.run_dir / "figures" / f"{slugify(name, 80)}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        self._figures.append(path.name)
        self.event("report", "figure_saved", detail=path.name)
        return path

    def table(self, rows: list[dict[str, Any]], name: str) -> Path:
        """Save a list of records as a CSV table under ``data/``.

        Args:
            rows: Records sharing the same keys.
            name: Base filename, without extension.

        Returns:
            Path of the written file.

        Raises:
            ValueError: If ``rows`` is empty, which would produce a headerless file.
        """
        if not rows:
            raise ValueError("cannot write an empty table; provide at least one row")
        path = self.run_dir / "data" / f"{slugify(name, 80)}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.event("report", "table_saved", detail=path.name)
        return path

    def summary(
        self,
        objective: str,
        method: str,
        findings: list[str],
        limitations: list[str] | None = None,
        next_steps: list[str] | None = None,
    ) -> None:
        """Provide the narrative content of ``results.md``.

        This is what a supervisor or a jury reads first, so it must state what the
        run shows, not only what it measured.
        """
        self._summary = {
            "objective": objective,
            "method": method,
            "findings": findings,
            "limitations": limitations or [],
            "next_steps": next_steps or [],
        }

    # -------------------------------------------------------------- teardown

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """Finalise the run, whether it succeeded or crashed.

        Returns:
            ``False``, so that any exception propagates to the caller.
        """
        manifest = self._build_manifest(exc_type, exc, tb)
        self._close_streams()
        manifest.artifacts = {
            str(path.relative_to(self.run_dir)): hash_file(path)
            for path in sorted(self.run_dir.rglob("*"))
            if path.is_file()
        }
        ResultsRenderer(self.run_dir).render(
            manifest, self.config.params, self._summary, self._notes, self._figures
        )
        self._write_json("manifest.json", manifest.model_dump(mode="json"))
        RunIndex(self.root, self.repo_root).append(manifest)
        return False

    def _build_manifest(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> RunManifest:
        """Close the event trace and assemble the provenance record."""
        status: RunStatus = "failed" if exc_type else "completed"
        error: str | None = None
        if exc is not None:
            error = "".join(traceback.format_exception(exc_type, exc, tb))
            self.logger.error("run failed: %s", exc)
            self.event("run", "failed", level="ERROR", detail=str(exc))
        else:
            self.event("run", "end", detail="completed")

        finished_at = utcnow()
        return RunManifest(
            run_id=self.run_id,
            task=self.config.task,
            description=self.config.description,
            status=status,
            started_at=self.started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - self.started_at).total_seconds(),
            git_commit=git_commit(self.repo_root),
            seed=self.config.seed,
            dataset=self.config.dataset,
            tags=list(self.config.tags),
            metric_count=self._metric_count,
            event_count=self._event_count,
            error=error,
        )

    def _close_streams(self) -> None:
        """Close the CSV files and detach the logging handlers."""
        for stream in (self._metrics, self._events):
            if stream is not None:
                stream.close()
        for handler in self._handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
        self._handlers.clear()

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _stream(stream: _CsvStream | None) -> _CsvStream:
        """Return the stream, or explain that the journal was used incorrectly.

        Raises:
            RuntimeError: If the journal is used outside its ``with`` block.
        """
        if stream is None:
            raise RuntimeError("LabJournal must be used as a context manager")
        return stream

    def _write_json(self, name: str, payload: dict[str, Any]) -> Path:
        """Serialise a mapping to a UTF-8 JSON file in the run directory."""
        path = self.run_dir / name
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path
