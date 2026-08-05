"""SQLite-backed session index with tiered retention.

Three tiers:
  full    — last FULL_DAYS days: raw session JSON preserved
  summary — FULL_DAYS to SUMMARY_DAYS: checkpoint summary only (~2KB)
  index   — SUMMARY_DAYS to INDEX_DAYS: file paths touched only (~200B)
  (deleted after INDEX_DAYS)

The SQLite database holds metadata and searchable content. Raw files live in
~/.claude-compress/sessions/{full,summary,index}/.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Retention thresholds in days
FULL_DAYS = 7
SUMMARY_DAYS = 30
INDEX_DAYS = 90

_DEFAULT_DIR = Path.home() / ".claude-compress"


@dataclass
class SessionMeta:
    session_id: str
    created_at: float
    updated_at: float
    turn_count: int
    token_count: int
    summary: str          # always present; full text for recent, brief for old
    file_paths: List[str] # files touched during the session
    tier: str             # 'full', 'summary', or 'index'


def _extract_file_paths(rows: list) -> List[str]:
    """Pull file paths from tool_result blocks in recorded session rows."""
    paths = set()
    path_re = re.compile(r'[\w./\-]+\.\w{1,6}')
    for row in rows:
        for msg in row.get("request", {}).get("messages", []):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    text = ""
                    c = block.get("content", "")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = " ".join(
                            b.get("text", "") for b in c
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    for match in path_re.findall(text):
                        if "/" in match or match.count(".") == 1:
                            paths.add(match)
    return sorted(paths)[:50]  # cap at 50 paths per session


def _make_summary(rows: list, max_chars: int = 1500) -> str:
    """Extract a brief summary from session rows."""
    lines = []
    for row in rows:
        msgs = row.get("request", {}).get("messages", [])
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                text = content.strip()
                if text and not text.startswith("<system-reminder>") and len(text) > 10:
                    lines.append(text[:200])
                break
    summary = "\n".join(lines)
    return summary[:max_chars]


class SessionStore:
    """SQLite-backed session index. Thread-safe. No server required."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_DIR
        self.db_path = self.base_dir / "sessions.db"
        self.full_dir = self.base_dir / "sessions" / "full"
        self.summary_dir = self.base_dir / "sessions" / "summary"
        self.index_dir = self.base_dir / "sessions" / "index"
        self._lock = threading.Lock()
        self._setup()

    def _setup(self):
        for d in (self.full_dir, self.summary_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    created_at   REAL NOT NULL,
                    updated_at   REAL NOT NULL,
                    turn_count   INTEGER DEFAULT 0,
                    token_count  INTEGER DEFAULT 0,
                    summary      TEXT DEFAULT '',
                    tier         TEXT DEFAULT 'full'
                );
                CREATE TABLE IF NOT EXISTS session_files (
                    session_id  TEXT NOT NULL,
                    file_path   TEXT NOT NULL,
                    PRIMARY KEY (session_id, file_path),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                    ON sessions(updated_at);
                CREATE INDEX IF NOT EXISTS idx_files_path
                    ON session_files(file_path);
                CREATE INDEX IF NOT EXISTS idx_files_session
                    ON session_files(session_id);
            """)

    def ingest(self, session_id: str, rows: list) -> None:
        """Add or update a session from a list of recorded rows."""
        if not rows:
            return
        rows_sorted = sorted(rows, key=lambda r: r.get("ts", 0))
        created_at = rows_sorted[0].get("ts", time.time())
        updated_at = rows_sorted[-1].get("ts", time.time())
        turn_count = len(rows_sorted)
        token_count = sum(
            len(json.dumps(r.get("request", {}))) // 4
            for r in rows_sorted
        )
        summary = _make_summary(rows_sorted)
        file_paths = _extract_file_paths(rows_sorted)

        full_path = self.full_dir / f"{session_id}.jsonl"
        with open(full_path, "w") as f:
            for row in rows_sorted:
                f.write(json.dumps(row) + "\n")

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions
                        (session_id, created_at, updated_at, turn_count,
                         token_count, summary, tier)
                    VALUES (?, ?, ?, ?, ?, ?, 'full')
                """, (session_id, created_at, updated_at, turn_count,
                      token_count, summary))
                conn.execute(
                    "DELETE FROM session_files WHERE session_id = ?",
                    (session_id,)
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO session_files VALUES (?, ?)",
                    [(session_id, p) for p in file_paths]
                )

    def search_by_file(self, file_path: str, limit: int = 10) -> List[SessionMeta]:
        """Find sessions that touched a given file path (exact or partial match)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT s.session_id, s.created_at, s.updated_at,
                       s.turn_count, s.token_count, s.summary, s.tier
                FROM sessions s
                JOIN session_files f ON s.session_id = f.session_id
                WHERE f.file_path LIKE ?
                ORDER BY s.updated_at DESC
                LIMIT ?
            """, (f"%{file_path}%", limit)).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def search_by_text(self, query: str, limit: int = 10) -> List[SessionMeta]:
        """Find sessions whose summary contains the query string."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT session_id, created_at, updated_at,
                       turn_count, token_count, summary, tier
                FROM sessions
                WHERE summary LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def recent(self, limit: int = 20) -> List[SessionMeta]:
        """Return the most recently updated sessions."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT session_id, created_at, updated_at,
                       turn_count, token_count, summary, tier
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def get_full(self, session_id: str) -> Optional[list]:
        """Load full session rows if available (full tier only)."""
        path = self.full_dir / f"{session_id}.jsonl"
        if not path.exists():
            return None
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def apply_retention(self) -> dict:
        """
        Enforce tiered retention. Call this periodically (e.g. on proxy startup).

        Returns a dict with counts of sessions transitioned between tiers.
        """
        now = time.time()
        full_cutoff = now - FULL_DAYS * 86400
        summary_cutoff = now - SUMMARY_DAYS * 86400
        index_cutoff = now - INDEX_DAYS * 86400
        stats = {"full_to_summary": 0, "summary_to_index": 0, "deleted": 0}

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # full → summary
                rows = conn.execute("""
                    SELECT session_id, summary FROM sessions
                    WHERE tier = 'full' AND updated_at < ?
                """, (full_cutoff,)).fetchall()
                for sid, summary in rows:
                    s_path = self.summary_dir / f"{sid}.json"
                    s_path.write_text(json.dumps({"summary": summary}))
                    f_path = self.full_dir / f"{sid}.jsonl"
                    if f_path.exists():
                        f_path.unlink()
                    conn.execute(
                        "UPDATE sessions SET tier = 'summary' WHERE session_id = ?",
                        (sid,)
                    )
                    stats["full_to_summary"] += 1

                # summary → index
                rows = conn.execute("""
                    SELECT s.session_id,
                           GROUP_CONCAT(f.file_path, '|') AS paths
                    FROM sessions s
                    LEFT JOIN session_files f ON s.session_id = f.session_id
                    WHERE s.tier = 'summary' AND s.updated_at < ?
                    GROUP BY s.session_id
                """, (summary_cutoff,)).fetchall()
                for sid, paths_str in rows:
                    paths = paths_str.split("|") if paths_str else []
                    i_path = self.index_dir / f"{sid}.json"
                    i_path.write_text(json.dumps({"file_paths": paths}))
                    s_path = self.summary_dir / f"{sid}.json"
                    if s_path.exists():
                        s_path.unlink()
                    conn.execute(
                        "UPDATE sessions SET tier = 'index', summary = '' "
                        "WHERE session_id = ?", (sid,)
                    )
                    stats["summary_to_index"] += 1

                # delete index tier beyond INDEX_DAYS
                rows = conn.execute("""
                    SELECT session_id FROM sessions
                    WHERE tier = 'index' AND updated_at < ?
                """, (index_cutoff,)).fetchall()
                for (sid,) in rows:
                    for d in (self.full_dir, self.summary_dir, self.index_dir):
                        for ext in (".jsonl", ".json"):
                            p = d / f"{sid}{ext}"
                            if p.exists():
                                p.unlink()
                    conn.execute(
                        "DELETE FROM session_files WHERE session_id = ?", (sid,)
                    )
                    conn.execute(
                        "DELETE FROM sessions WHERE session_id = ?", (sid,)
                    )
                    stats["deleted"] += 1

        return stats

    def stats(self) -> dict:
        """Return storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            counts = conn.execute("""
                SELECT tier, COUNT(*), SUM(token_count)
                FROM sessions GROUP BY tier
            """).fetchall()
        result = {}
        for tier, count, tokens in counts:
            result[tier] = {"sessions": count, "approx_tokens": tokens or 0}
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        result["db_size_bytes"] = db_size
        return result

    def _row_to_meta(self, row: tuple) -> SessionMeta:
        sid, created, updated, turns, tokens, summary, tier = row
        with sqlite3.connect(self.db_path) as conn:
            paths = [r[0] for r in conn.execute(
                "SELECT file_path FROM session_files WHERE session_id = ?",
                (sid,)
            ).fetchall()]
        return SessionMeta(
            session_id=sid,
            created_at=created,
            updated_at=updated,
            turn_count=turns,
            token_count=tokens,
            summary=summary,
            file_paths=paths,
            tier=tier,
        )
