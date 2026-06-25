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
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .tokens import content_to_text


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # alias -> original string, persisted so we can expand responses
    alias_legend: Dict[str, str] = field(default_factory=dict)
    # monotonic counter for alias numbering across turns
    next_alias_index: int = 0
    # checkpoint summary text we've injected, if any
    checkpoint_text: Optional[str] = None
    # how many leading messages the checkpoint already subsumes
    checkpoint_covers: int = 0
    # hash of the message blob that produced checkpoint_text; used to skip
    # re-summarising the same prefix on every turn without changes
    checkpoint_hash: Optional[str] = None
    # registry of content hashes we've seen (for delta / cache decisions)
    seen_hashes: Dict[str, int] = field(default_factory=dict)

    def touch(self):
        self.updated_at = time.time()


def fingerprint(request: dict) -> str:
    """Stable id for a conversation from its first few messages + system.

    Uses the first 3 messages (not just the first) to reduce collision risk for
    sessions that share an opening message or system prompt. Returns a 24-char
    hex prefix (96 bits), making accidental collision vanishingly unlikely.
    """
    msgs = request.get("messages", [])
    anchor_parts = []
    for msg in msgs[:3]:
        anchor_parts.append(content_to_text(msg.get("content", ""))[:1000])
    anchor = "||".join(anchor_parts)
    system = request.get("system")
    sys_text = system if isinstance(system, str) else json.dumps(system or "")
    h = hashlib.sha256((sys_text[:2000] + "|||" + anchor).encode("utf-8")).hexdigest()
    return h[:24]


class StateStore:
    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_sessions: int = 500,
        session_ttl_seconds: float = 86400.0,
    ):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}
        self._persist_path = persist_path
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl_seconds
        self._load()

    def _load(self):
        if self._persist_path and os.path.exists(self._persist_path):
            try:
                with open(self._persist_path) as f:
                    raw = json.load(f)
                known = set(SessionState.__dataclass_fields__)
                for sid, data in raw.items():
                    self._sessions[sid] = SessionState(
                        **{k: v for k, v in data.items() if k in known}
                    )
            except Exception:
                pass

    def _save(self):
        if not self._persist_path:
            return
        try:
            dir_ = os.path.dirname(os.path.abspath(self._persist_path))
            with tempfile.NamedTemporaryFile(
                "w", dir=dir_, delete=False, suffix=".tmp"
            ) as f:
                json.dump(
                    {sid: s.__dict__ for sid, s in self._sessions.items()}, f
                )
                tmp = f.name
            os.replace(tmp, self._persist_path)
        except Exception:
            pass

    def _evict(self):
        """Remove expired and excess sessions. Must be called under self._lock."""
        now = time.time()
        cutoff = now - self._session_ttl
        expired = [
            sid for sid, s in self._sessions.items() if s.updated_at < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]
        if len(self._sessions) > self._max_sessions:
            oldest = sorted(
                self._sessions.items(), key=lambda kv: kv[1].updated_at
            )
            for sid, _ in oldest[: len(self._sessions) - self._max_sessions]:
                del self._sessions[sid]

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
            self._evict()
            self._save()


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
