"""Stage interface + helpers shared across stages.

A Stage transforms a request body in place-ish (returns the new body) and reports
how it changed token counts. Stages must be SAFE: they must never corrupt
tool_use / tool_result / image blocks, and must respect "protect last N" windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from ..state import SessionState


@dataclass
class StageResult:
    name: str
    tokens_before: int
    tokens_after: int
    note: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class Stage:
    name: str = "stage"

    def enabled(self) -> bool:
        return True

    def apply(self, request: dict, state: SessionState) -> StageResult:
        raise NotImplementedError


# ---- helpers -------------------------------------------------------------

def iter_text_blocks(
    request: dict, protect_last_n: int = 0
) -> List[Tuple[int, int, dict]]:
    """Yield (message_index, block_index, block) for every TEXT block that is
    eligible for transformation.

    - Only `type == "text"` blocks (and bare-string contents, normalised first).
    - Skips the last `protect_last_n` messages.
    - Never yields tool_use / tool_result / image blocks.
    """
    msgs = request.get("messages", [])
    cutoff = len(msgs) - protect_last_n
    out: List[Tuple[int, int, dict]] = []
    for mi, msg in enumerate(msgs):
        if mi >= cutoff:
            break
        content = msg.get("content")
        if isinstance(content, str):
            # normalise to a block so callers can edit uniformly
            block = {"type": "text", "text": content}
            msg["content"] = [block]
            out.append((mi, 0, block))
            continue
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append((mi, bi, block))
    return out


def text_blocks_in_message(msg: dict) -> List[dict]:
    content = msg.get("content")
    if isinstance(content, str):
        block = {"type": "text", "text": content}
        msg["content"] = [block]
        return [block]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return []
