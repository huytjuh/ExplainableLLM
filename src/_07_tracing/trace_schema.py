"""Trace data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TraceEvent:
    name: str
    payload: dict[str, Any]
    timestamp: str=field(default_factory=utc_now)


@dataclass
class TraceRun:
    name: str
    run_id: str=field(default_factory=lambda: str(uuid4()))
    started_at: str=field(default_factory=utc_now)
    events: list[TraceEvent]=field(default_factory=list)

