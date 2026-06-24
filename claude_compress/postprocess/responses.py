"""Response post-processing.

If the alias stage ran, Claude may emit aliases (@a0, ...) in its output. We
expand them back to the originals so the client sees real strings. Works for both
non-streaming responses and streaming text deltas.

The state-updater piece records the assistant turn's hashes so future delta/cache
decisions have an accurate picture of what the model has already produced.
"""
from __future__ import annotations

import re
from typing import Dict

from ..state import SessionState, content_hash


def _expand(text: str, legend: Dict[str, str]) -> str:
    if not legend or not text:
        return text
    # replace longer aliases first to avoid prefix collisions (@a1 vs @a10)
    for alias in sorted(legend, key=len, reverse=True):
        if alias in text:
            text = text.replace(alias, legend[alias])
    return text


def expand_response_body(body: dict, state: SessionState) -> dict:
    """Expand aliases inside a non-streaming /v1/messages JSON response."""
    legend = state.alias_legend
    if not legend:
        return body
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            block["text"] = _expand(block.get("text", ""), legend)
    return body


def expand_sse_event(event_data: str, state: SessionState) -> str:
    """Expand aliases inside a single SSE `data:` JSON payload (streaming)."""
    legend = state.alias_legend
    if not legend:
        return event_data
    import json

    try:
        obj = json.loads(event_data)
    except Exception:
        return event_data
    if obj.get("type") == "content_block_delta":
        delta = obj.get("delta", {})
        if delta.get("type") == "text_delta" and "text" in delta:
            delta["text"] = _expand(delta["text"], legend)
            return json.dumps(obj)
    return event_data


def record_assistant_turn(body: dict, state: SessionState):
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            h = content_hash(block.get("text", ""))
            state.seen_hashes[h] = state.seen_hashes.get(h, 0) + 1
