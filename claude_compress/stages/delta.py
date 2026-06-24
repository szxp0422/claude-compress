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

from ..config import DeltaConfig
from ..state import SessionState
from ..tokens import content_to_text, count_request
from ..state import content_hash
from .base import Stage, StageResult


def _mark(block: dict):
    block["cache_control"] = {"type": "ephemeral"}


class DeltaStage(Stage):
    name = "delta_cache_breakpoints"

    def __init__(self, cfg: DeltaConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
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

        # record the prefix hash for observability (delta detection)
        prefix_text = content_to_text(request.get("messages", [{}])[0].get("content", "")) \
            if request.get("messages") else ""
        state.seen_hashes[content_hash(prefix_text)] = \
            state.seen_hashes.get(content_hash(prefix_text), 0) + 1

        after = count_request(request)  # unchanged; this stage is lossless
        return StageResult(
            self.name,
            before,
            after,
            note=f"inserted {placed} cache breakpoint(s) (cost, not size)",
            detail={"breakpoints": placed},
        )
