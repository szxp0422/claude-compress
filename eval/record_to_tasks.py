"""Convert recorded proxy sessions into eval task JSONL.

Usage:
    python -m eval.record_to_tasks --input sessions.jsonl --out eval/my_tasks.jsonl
"""
import argparse
import json
import re
from collections import defaultdict

SKIP_PREFIXES = (
    "[SUGGESTION MODE",
    "[Request interrupted",
    "The user stepped away",
    "<system-reminder>",
    "[CONTEXT:",
)

def _strip_cache_control(obj):
    """Recursively remove cache_control from any dict/list structure."""
    if isinstance(obj, list):
        for item in obj:
            _strip_cache_control(item)
    elif isinstance(obj, dict):
        obj.pop("cache_control", None)
        for v in obj.values():
            _strip_cache_control(v)


def convert(input_path: str, out_path: str, min_turns: int = 2):
    sessions = defaultdict(list)
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                sessions[row["session"]].append(row)

    tasks = []
    for session_id, rows in sessions.items():
        rows.sort(key=lambda r: r["ts"])
        if len(rows) < min_turns:
            continue

        turns = []
        for row in rows:
            msgs = row["request"].get("messages", [])
            if not msgs:
                continue
            last_user = next(
                (m for m in reversed(msgs) if m.get("role") == "user"), None
            )
            if not last_user:
                continue

            # flatten content blocks to plain text
            content = last_user.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not content.strip():
                continue
            if any(content.startswith(p) for p in SKIP_PREFIXES):
                continue

            turn = {"user": content}
            m = re.search(r"NEEDLE=(\w+)", content)
            if m:
                turn["check"] = {"type": "contains", "value": m.group(1)}
            turns.append(turn)

        if len(turns) < min_turns:
            continue

        task = {
            "id": "recorded-" + session_id[:8],
            "turns": turns,
        }
        _strip_cache_control(task)
        tasks.append(task)

    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"wrote {len(tasks)} tasks from {len(sessions)} sessions to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-turns", type=int, default=2)
    args = ap.parse_args()
    convert(args.input, args.out, args.min_turns)