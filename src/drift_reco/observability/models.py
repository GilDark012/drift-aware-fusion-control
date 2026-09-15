"""Validated records produced by a run, plus the helpers they rely on.

Splitting the models away from the journal keeps each file small enough to read in
one sitting and makes the record schema importable on its own — analysis scripts
can parse ``metrics.csv`` back into :class:`MetricRecord` without pulling in the
whole journaling machinery.
"""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "METRIC_COLUMNS",
    "EVENT_COLUMNS",
    "INDEX_COLUMNS",
    "RunStatus",
    "LogLevel",
    "RunConfig",
    "MetricRecord",
    "EventRecord",
    "RunManifest",
    "utcnow",
    "slugify",
    "git_commit",
    "hash_file",
]

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "step",
    "split",
    "metric",
    "value",
    "unit",
    "context",
)
EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "level",
    "stage",
    "event",
    "detail",
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

RunStatus = Literal["running", "completed", "failed"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def utcnow() -> datetime:
    """Return the current UTC time as an aware datetime."""
    return datetime.now(UTC)


def slugify(value: str, max_length: int = 48) -> str:
    """Turn free text into a filesystem-safe slug.

    Args:
        value: Text to normalise.
        max_length: Maximum length of the returned slug.

    Returns:
        A lowercase, hyphen-separated slug, never empty.
    """
    slug = _SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return (slug[:max_length].rstrip("-")) or "run"


def git_commit(repo: Path) -> str | None:
    """Return the current Git commit hash, or None outside a repository.

    Args:
        repo: Directory from which ``git`` is invoked.

    Returns:
        The commit hash, or None when Git is unavailable or the directory is not
        a working tree.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file, read in chunks to bound memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunConfig(BaseModel):
    """Immutable description of what a run is about and how it is parameterised.

    Attributes:
        task: Short machine-friendly identifier, e.g. ``"ablation-c"``.
        description: One sentence explaining the intent of the run.
        params: Hyperparameters and settings, serialised verbatim to ``config.json``.
        tags: Free labels used to group runs in reports.
        seed: Random seed applied to every stochastic component.
        dataset: Identifier of the dataset the run operates on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)
    params: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    seed: int = 42
    dataset: str | None = None

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        """Reject task names that would not survive a round trip through a path."""
        if slugify(value) != value.lower():
            raise ValueError(
                "task must contain only lowercase letters, digits and hyphens"
            )
        return value


class MetricRecord(BaseModel):
    """A single measurement written as one row of ``metrics.csv``."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utcnow)
    step: int = 0
    split: str = "all"
    metric: str = Field(min_length=1)
    value: float
    unit: str = ""
    context: str = ""

    def as_row(self) -> dict[str, str]:
        """Return the record as a CSV-ready mapping."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": str(self.step),
            "split": self.split,
            "metric": self.metric,
            "value": repr(self.value),
            "unit": self.unit,
            "context": self.context,
        }


class EventRecord(BaseModel):
    """A lifecycle event written as one row of ``events.csv``."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utcnow)
    level: LogLevel = "INFO"
    stage: str = Field(min_length=1)
    event: str = Field(min_length=1)
    detail: str = ""

    def as_row(self) -> dict[str, str]:
        """Return the record as a CSV-ready mapping."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "stage": self.stage,
            "event": self.event,
            "detail": self.detail,
        }


class RunManifest(BaseModel):
    """Provenance record written to ``manifest.json`` when a run terminates."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task: str
    description: str
    status: RunStatus = "running"
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    git_commit: str | None = None
    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)
    seed: int = 42
    dataset: str | None = None
    tags: list[str] = Field(default_factory=list)
    metric_count: int = 0
    event_count: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None

    def as_index_row(self) -> dict[str, str]:
        """Return the manifest as one row of ``artifacts/index.csv``."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else "",
            "task": self.task,
            "status": self.status,
            "duration_seconds": f"{self.duration_seconds or 0.0:.3f}",
            "git_commit": self.git_commit or "",
            "description": self.description,
        }
