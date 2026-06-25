"""Token counting via tiktoken (cl100k_base).

We can't run Claude's exact tokenizer locally, so this is an approximation
used only for *relative* measurement (how much a stage saved). It is not used
for billing or to make correctness-critical decisions. tiktoken is accurate
for English+code and is a required dependency.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_text(text: str) -> int:
    """Token count for a single string via tiktoken cl100k_base."""
    if not text:
        return 0
    return len(_ENC.encode(text))


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
