"""Per-conversation state, kept server-side and out of Claude's context.

Claude Code doesn't send a stable conversation id, so we fingerprint a
conversation by hashing its earliest stable messages. That lets us carry state
(checkpoints already made, alias legends, content-hash registry for delta cache
hints) across turns of the same session.

State is in-memory with optional JSON persistence. For a local single-user proxy
that is plenty; swap in Redis if you ever run this multi-tenant.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tokens import content_to_text


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # alias -> original string, persisted so we can expand responses
    alias_legend: Dict[str, str] = field(default_factory=dict)
    # checkpoint summary text we've injected, if any
    checkpoint_text: Optional[str] = None
    # how many leading messages the checkpoint already subsumes
    checkpoint_covers: int = 0
    # registry of content hashes we've seen (for delta / cache decisions)
    seen_hashes: Dict[str, int] = field(default_factory=dict)

    def touch(self):
        self.updated_at = time.time()


def fingerprint(request: dict) -> str:
    """Stable id for a conversation from its first user message + system."""
    msgs = request.get("messages", [])
    anchor = ""
    if msgs:
        anchor = content_to_text(msgs[0].get("content", ""))[:2000]
    system = request.get("system")
    sys_text = system if isinstance(system, str) else json.dumps(system or "")
    h = hashlib.sha256((sys_text[:1000] + "||" + anchor).encode("utf-8")).hexdigest()
    return h[:16]


class StateStore:
    def __init__(self, persist_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}
        self._persist_path = persist_path
        self._load()

    def _load(self):
        if self._persist_path and os.path.exists(self._persist_path):
            try:
                with open(self._persist_path) as f:
                    raw = json.load(f)
                for sid, data in raw.items():
                    self._sessions[sid] = SessionState(**data)
            except Exception:
                pass

    def _save(self):
        if not self._persist_path:
            return
        try:
            with open(self._persist_path, "w") as f:
                json.dump({sid: s.__dict__ for sid, s in self._sessions.items()}, f)
        except Exception:
            pass

    def get(self, request: dict) -> SessionState:
        sid = fingerprint(request)
        with self._lock:
            st = self._sessions.get(sid)
            if st is None:
                st = SessionState(session_id=sid)
                self._sessions[sid] = st
            return st

    def commit(self, state: SessionState):
        with self._lock:
            state.touch()
            self._sessions[state.session_id] = state
            self._save()


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
