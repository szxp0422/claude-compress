"""CLI: `python -m claude_compress` starts the proxy.

Usage:
    python -m claude_compress [--config path.json] [--host H] [--port P]
"""
from __future__ import annotations

import argparse

import uvicorn

from .config import load_config
from .server import create_app


def main():
    import sys
    # check for dream subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "dream":
        from .dream import run_dream
        ap = argparse.ArgumentParser(description="Generate context.md from past sessions")
        ap.add_argument("dream")  # consume the subcommand
        ap.add_argument("--limit", type=int, default=30)
        ap.add_argument("--out", default=None)
        ap.add_argument("--query", default=None)
        ap.add_argument("--file", default=None)
        ap.add_argument("--base-dir", default=None)
        args = ap.parse_args()
        run_dream(limit=args.limit, out=args.out, query=args.query,
                  file_filter=args.file, base_dir=args.base_dir)
        return

    ap = argparse.ArgumentParser(description="claude-compress proxy for Claude Code")
    ap.add_argument("--config", default=None, help="path to JSON config")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.host:
        cfg.listen_host = args.host
    if args.port:
        cfg.listen_port = args.port

    app = create_app(cfg)
    print(f"claude-compress listening on http://{cfg.listen_host}:{cfg.listen_port}")
    print(f"upstream: {cfg.upstream_base_url}")
    print("point Claude Code at it with:")
    print(f'  export ANTHROPIC_BASE_URL="http://{cfg.listen_host}:{cfg.listen_port}"')
    uvicorn.run(app, host=cfg.listen_host, port=cfg.listen_port,
                log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
