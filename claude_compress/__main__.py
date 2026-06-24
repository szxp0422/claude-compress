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
