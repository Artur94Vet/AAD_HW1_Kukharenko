"""Атомарне JSON-логування траєкторії для post-mortem аналізу."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Збирає кроки в пам'яті та надійно замінює файл через temporary sibling."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.steps: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(
        self,
        node: str,
        action: object,
        observation: object,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        entry = {
            "index": len(self.steps) + 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "node": node,
            "action": action,
            "observation": observation,
            "tool_calls": tool_calls or [],
        }
        with self._lock:
            self.steps.append(entry)
            self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.steps, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.path)
