"""Token counting.

We can't run Claude's exact tokenizer locally, so this is an approximation
used only for *relative* measurement (how much a stage saved). It is not used
for billing or to make correctness-critical decisions.

Backends, in order of preference:
  1. tiktoken (cl100k_base) if installed -- a decent proxy for English+code.
  2. A char/word heuristic fallback that needs no dependencies.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

try:  # optional
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path
    _ENC = None


def count_text(text: str) -> int:
    """Approximate token count for a single string."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Heuristic: ~4 chars/token for prose, but punctuation/code skews lower.
    # Blend char and whitespace-word estimates for stability.
    char_est = len(text) / 4.0
    word_est = len(text.split()) * 1.3
    return int(round((char_est + word_est) / 2))


def _block_text(block: Any) -> str:
    """Extract the text we can meaningfully count from a content block."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        btype = block.get("type")
        if btype == "text":
            return block.get("text", "")
        if btype == "tool_result":
            # tool_result content can itself be a string or list of blocks
            return content_to_text(block.get("content", ""))
        if btype == "tool_use":
            # the input json contributes tokens too
            import json

            return json.dumps(block.get("input", {}))
    return ""


def content_to_text(content: Any) -> str:
    """Flatten message `content` (str or list of blocks) into countable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_block_text(b) for b in content)
    return ""


def count_request(request: dict) -> int:
    """Approximate total prompt tokens for a Messages API request body."""
    total = 0
    system = request.get("system")
    if isinstance(system, str):
        total += count_text(system)
    elif isinstance(system, list):
        total += sum(count_text(b.get("text", "")) for b in system if isinstance(b, dict))
    for msg in request.get("messages", []):
        total += count_text(content_to_text(msg.get("content", "")))
    # tools schema contributes too
    tools = request.get("tools")
    if tools:
        import json

        total += count_text(json.dumps(tools))
    return total
