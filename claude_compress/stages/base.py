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


def extract_tool_result_text(block: dict) -> str:
    """Extract plain text from a tool_result block.

    tool_result content can be a plain string or a list of typed sub-blocks.
    Returns empty string for non-tool_result blocks or blocks with no text.
    """
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return ""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for sub in content:
            if isinstance(sub, dict) and sub.get("type") == "text":
                parts.append(sub.get("text", ""))
        return "\n".join(parts)
    return ""


def iter_all_text(
    request: dict, protect_last_n: int = 0, include_tool_results: bool = True
) -> List[Tuple[int, int, str, str]]:
    """Yield (message_index, block_index, text, source_type) for every block
    that contains readable text, including tool_result blocks.

    source_type is one of: 'text', 'tool_result'

    Used by dedup and checkpoint so they operate on tool-heavy sessions too.
    Never yields tool_use or image blocks (no useful plain text to extract).
    Skips the last protect_last_n messages.
    """
    msgs = request.get("messages", [])
    cutoff = len(msgs) - protect_last_n
    out: List[Tuple[int, int, str, str]] = []
    for mi, msg in enumerate(msgs):
        if mi >= cutoff:
            break
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                out.append((mi, 0, content, "text"))
            continue
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if text.strip():
                        out.append((mi, bi, text, "text"))
                elif btype == "tool_result" and include_tool_results:
                    text = extract_tool_result_text(block)
                    if text.strip():
                        out.append((mi, bi, text, "tool_result"))
    return out


def count_tool_result_tokens(request: dict) -> int:
    """Count tokens that live inside tool_result blocks.

    Used by the tier selector to detect tool-heavy sessions.
    """
    from ..tokens import count_text
    total = 0
    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    total += count_text(extract_tool_result_text(block))
    return total
