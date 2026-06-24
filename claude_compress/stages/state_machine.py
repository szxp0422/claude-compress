"""State-machine stage (opt-in).

If the client supplies a finite-state-machine spec via request metadata
(`metadata._fsm`), inject a compact transition table plus the current node,
instead of relying on verbose prose workflow instructions carried every turn.

Spec shape (client-provided):
    metadata._fsm = {
        "current": "draft",
        "states": {
            "draft":     {"on_ok": "review", "on_fail": "draft"},
            "review":    {"on_ok": "ship",   "on_fail": "draft"},
            "ship":      {}
        }
    }

This stage doesn't *reduce* an existing prompt automatically; it gives the client
a token-frugal way to express workflow state. It's a building block, hence opt-in.
"""
from __future__ import annotations

from ..config import StateMachineConfig
from ..state import SessionState
from ..tokens import count_request
from .base import Stage, StageResult


class StateMachineStage(Stage):
    name = "state_machine"

    def __init__(self, cfg: StateMachineConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        meta = request.get("metadata") or {}
        fsm = meta.get("_fsm")
        if not isinstance(fsm, dict) or "states" not in fsm:
            return StageResult(self.name, before, before, note="no _fsm in metadata")

        cur = fsm.get("current", "?")
        lines = [f"WORKFLOW (compact FSM). current={cur}"]
        for node, edges in fsm["states"].items():
            if edges:
                trans = " ".join(f"{ev}->{dst}" for ev, dst in edges.items())
                lines.append(f"  {node}: {trans}")
            else:
                lines.append(f"  {node}: (terminal)")
        table = "\n".join(lines)

        block = {"type": "text", "text": table}
        msgs = request.get("messages", [])
        if msgs:
            first = msgs[0]
            content = first.get("content")
            if isinstance(content, str):
                first["content"] = [{"type": "text", "text": content}]
                content = first["content"]
            if isinstance(content, list):
                content.insert(0, block)

        # strip our private hint so it doesn't go upstream
        meta.pop("_fsm", None)
        if meta:
            request["metadata"] = meta
        else:
            request.pop("metadata", None)

        after = count_request(request)
        return StageResult(
            self.name, before, after, note=f"injected FSM table (current={cur})"
        )
