"""Lightweight metrics: append one JSON line per request summarising savings."""
from __future__ import annotations

import json
import threading
import time
from typing import List

from .stages.base import StageResult

_lock = threading.Lock()


class Metrics:
    def __init__(self, path: str):
        self.path = path

    def record(self, session_id: str, stage_results: List[StageResult],
               tokens_in: int, tokens_out: int, streaming: bool):
        total_saved = max(0, tokens_in - tokens_out)
        pct = (total_saved / tokens_in * 100.0) if tokens_in else 0.0
        row = {
            "ts": time.time(),
            "session": session_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "saved": total_saved,
            "saved_pct": round(pct, 1),
            "streaming": streaming,
            "stages": [
                {
                    "name": r.name,
                    "before": r.tokens_before,
                    "after": r.tokens_after,
                    "saved": r.saved,
                    "note": r.note,
                }
                for r in stage_results
            ],
        }
        line = json.dumps(row)
        with _lock:
            try:
                with open(self.path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        return row
