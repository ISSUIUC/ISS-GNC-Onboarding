"""Dead-simple per-student progress, persisted to one JSON file.

Shape on disk:
    {
      "<student>": {
        "<module_id>": {
          "<exercise_id>": {"passed": true, "score": 1.0, "ts": 1690000000}
        }
      }
    }

Good enough for a club onboarding tool. Swap for SQLite if you outgrow it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class Progress:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    def mark(self, student: str, module_id: str, exercise_id: str, score: float) -> None:
        with self._lock:
            data = self._read()
            data.setdefault(student, {}).setdefault(module_id, {})[exercise_id] = {
                "passed": True,
                "score": round(score, 4),
                "ts": int(time.time()),
            }
            self._write(data)

    def completed(self, student: str, module_id: str) -> set[str]:
        return set(self._read().get(student, {}).get(module_id, {}).keys())

    def module_counts(self, student: str, modules) -> dict[str, dict]:
        """Return {module_id: {done, total, complete, unlocked, blocked_by}}.

        Modules unlock strictly in order: the first is always open, and module N
        stays locked until every graded exercise in modules 1..N-1 is passed.
        `blocked_by` names the earliest module still standing in the way (None
        once the module is unlocked), so the UI can say what to go finish.

        A module with no graded exercises can't be "completed" by definition, so
        it counts as complete and never blocks the chain.
        """
        done_map = self._read().get(student, {})
        out: dict[str, dict] = {}
        blocker: str | None = None  # title of the first incomplete module so far
        for module in modules:
            graded = [ex for ex in module.exercises if ex.graded]
            done_ids = set(done_map.get(module.id, {}).keys())
            done = sum(1 for ex in graded if ex.id in done_ids)
            complete = done == len(graded)
            out[module.id] = {
                "done": done,
                "total": len(graded),
                "complete": complete,
                "unlocked": blocker is None,
                "blocked_by": blocker,
            }
            if blocker is None and not complete:
                blocker = module.title
        return out
