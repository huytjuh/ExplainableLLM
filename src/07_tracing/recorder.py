"""Trace recorder that writes JSON artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trace_schema import TraceEvent, TraceRun


class TraceRecorder:
    def __init__(self, name: str) -> None:
        self.run = TraceRun(name=name)

    def add_event(self, name: str, **payload: Any) -> TraceEvent:
        event = TraceEvent(name=name, payload=payload)
        self.run.events.append(event)
        return event

    def write(self, directory: str | Path = "artifacts/traces") -> Path:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.run.name}-{self.run.run_id}.json"
        path.write_text(json.dumps(asdict(self.run), indent=2), encoding="utf-8")
        return path
