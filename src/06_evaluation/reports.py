"""Evaluation report artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_evaluation_report(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": len(rows),
        "average_score": _average(row.get("score") for row in rows),
        "rows": rows,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _average(values) -> float:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return round(sum(numeric) / len(numeric), 3) if numeric else 0.0

