"""Delta stage = automatic prompt-cache breakpoints (lossless cost reduction).

Honest framing: the Messages API is stateless, so we can't *omit* unchanged
context from Claude. What we CAN do is mark the stable prefix (system, tools,
older turns) with `cache_control: {type: "ephemeral"}` breakpoints so Anthropic
serves it from prompt cache on the next turn. Same tokens reach the model, but
cached input tokens are billed at a large discount.

We place breakpoints at the last content block of:
  - the system prompt (if present),
  - the tools array (handled by marking on the last tool),
  - the boundary just before the most recent user turn (the "delta" line).

This is the genuinely-shippable version of "delta encoding" for a stateless API.
"""
from __future__ import annotations

import json

from ..config import DeltaConfig
from ..state import SessionState
from ..tokens import content_to_text, count_request
from ..state import content_hash
from .base import Stage, StageResult


def _mark(block: dict):
    if "cache_control" not in block:
        block["cache_control"] = {"type": "ephemeral"}

def _has_any_cache_control(request: dict) -> bool:
    """Check if the client already set cache_control anywhere in the request."""
    system = request.get("system")
    if isinstance(system, list):
        if any(b.get("cache_control") for b in system if isinstance(b, dict)):
            return True
    for tool in request.get("tools") or []:
        if isinstance(tool, dict) and tool.get("cache_control"):
            return True
    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            if any(b.get("cache_control") for b in content if isinstance(b, dict)):
                return True
    return False


class DeltaStage(Stage):
    name = "delta_cache_breakpoints"

    def __init__(self, cfg: DeltaConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        if _has_any_cache_control(request):
            return StageResult(self.name, before, before,
                               note="skipped: client already set cache_control blocks")

        budget = self.cfg.max_breakpoints
        placed = 0

        # 1) system prompt -> normalise to block list, mark the last block
        if budget > 0:
            system = request.get("system")
            if isinstance(system, str) and system.strip():
                request["system"] = [{"type": "text", "text": system}]
                system = request["system"]
            if isinstance(system, list) and system:
                _mark(system[-1])
                placed += 1
                budget -= 1

        # 2) tools -> mark the last tool definition
        if budget > 0:
            tools = request.get("tools")
            if isinstance(tools, list) and tools:
                _mark(tools[-1])
                placed += 1
                budget -= 1

        # 3) prefix boundary inside messages: mark the last block of the message
        # right before the final user turn, so the whole history prefix caches.
        if budget > 0:
            msgs = request.get("messages", [])
            if len(msgs) >= 2:
                boundary = len(msgs) - 1  # message before the last
                # walk back to last assistant/user boundary that has blocks
                target = msgs[boundary - 1] if boundary - 1 >= 0 else None
                if target is not None:
                    content = target.get("content")
                    if isinstance(content, str):
                        target["content"] = [{"type": "text", "text": content}]
                        content = target["content"]
                    if isinstance(content, list) and content:
                        _mark(content[-1])
                        placed += 1
                        budget -= 1

        # Hash the full stable prefix that the cache breakpoints protect:
        # system + tools + every message except the final user turn.
        parts: list[str] = []
        sys_ = request.get("system")
        if sys_:
            parts.append(sys_ if isinstance(sys_, str) else json.dumps(sys_))
        tools_ = request.get("tools")
        if tools_:
            parts.append(json.dumps(tools_))
        for m in request.get("messages", [])[:-1]:
            parts.append(content_to_text(m.get("content", "")))
        h = content_hash("\n".join(parts))
        state.seen_hashes[h] = state.seen_hashes.get(h, 0) + 1

        after = count_request(request)  # unchanged; this stage is lossless
        return StageResult(
            self.name,
            before,
            after,
            note=f"inserted {placed} cache breakpoint(s) (cost, not size)",
            detail={"breakpoints": placed},
        )
