"""Local dream command: consolidate past sessions into a reusable context prefix.

Reads summaries from the local SQLite session store and produces a compact
context.md file. No data leaves the machine. No server required.

Usage:
    python -m claude_compress dream [--limit N] [--out path] [--query topic]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from .session_store import SessionStore, _DEFAULT_DIR


def _format_age(ts: float) -> str:
    days = int((time.time() - ts) / 86400)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days}d ago"


def build_context(
    store: SessionStore,
    limit: int = 30,
    query: Optional[str] = None,
    file_filter: Optional[str] = None,
) -> str:
    """Build a context string from recent session summaries."""
    if query:
        sessions = store.search_by_text(query, limit=limit)
    elif file_filter:
        sessions = store.search_by_file(file_filter, limit=limit)
    else:
        sessions = store.recent(limit=limit)

    if not sessions:
        return ""

    lines = [
        "<!-- claude-compress context: generated from past sessions -->",
        "## Past session context",
        "",
        "The following is a summary of relevant past work. "
        "Use it as background context.",
        "",
    ]

    for s in sessions:
        if not s.summary.strip():
            continue
        age = _format_age(s.updated_at)
        lines.append(f"### Session ({age}, {s.turn_count} turns)")
        if s.file_paths:
            lines.append(f"**Files touched:** {', '.join(s.file_paths[:8])}")
        lines.append("")
        lines.append(s.summary.strip())
        lines.append("")

    return "\n".join(lines)


def run_dream(
    limit: int = 30,
    out: Optional[str] = None,
    query: Optional[str] = None,
    file_filter: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> str:
    store = SessionStore(base_dir=base_dir)

    # apply retention first so we're not summarising already-expired sessions
    stats = store.apply_retention()
    if any(v > 0 for v in stats.values()):
        print(f"Retention: {stats}")

    context = build_context(store, limit=limit, query=query, file_filter=file_filter)

    out_path = Path(out) if out else (_DEFAULT_DIR / "context.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(context)

    session_stats = store.stats()
    print(f"Written to {out_path}")
    print(f"Store: {session_stats}")
    return context


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate context.md from past sessions")
    ap.add_argument("--limit", type=int, default=30, help="Max sessions to include")
    ap.add_argument("--out", default=None, help="Output path (default: ~/.claude-compress/context.md)")
    ap.add_argument("--query", default=None, help="Filter sessions by topic")
    ap.add_argument("--file", default=None, help="Filter sessions by file path")
    ap.add_argument("--base-dir", default=None, help="Base directory for store")
    args = ap.parse_args()
    run_dream(
        limit=args.limit,
        out=args.out,
        query=args.query,
        file_filter=args.file,
        base_dir=args.base_dir,
    )
