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
            "kind": "estimate",
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
        self._write(row)
        return row

    def record_usage(self, session_id: str, usage, est_tokens_out: int = 0):
        """Log the API's ground-truth usage. This is the number you cite as proof."""
        row = {
            "kind": "ground_truth",
            "ts": time.time(),
            "session": session_id,
            "usage": usage.to_dict(),
            "cost_usd": round(usage.cost(), 6),
            "estimated_compressed_input": est_tokens_out,
        }
        self._write(row)
        return row

    def _write(self, row: dict):
        line = json.dumps(row)
        with _lock:
            try:
                with open(self.path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
