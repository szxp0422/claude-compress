"""Split sessions.jsonl into per-session files in a directory.

Usage:
    python split_sessions.py --input sessions.jsonl --output sessions_dir/
"""
import argparse
import json
import os


def split_sessions(input_path: str, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    counts: dict[str, int] = {}
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            session_id = row["session"]
            out_file = os.path.join(output_dir, f"{session_id}.jsonl")
            with open(out_file, "a") as out:
                out.write(line + "\n")
            counts[session_id] = counts.get(session_id, 0) + 1
    return len(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to sessions.jsonl")
    ap.add_argument("--output", required=True, help="Output directory for per-session files")
    args = ap.parse_args()
    n = split_sessions(args.input, args.output)
    print(f"split {n} sessions into {args.output}")
