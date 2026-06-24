"""FastAPI proxy that Claude Code points at via ANTHROPIC_BASE_URL.

Flow:
  Claude Code --> POST /v1/messages (this proxy)
              --> input Pipeline (compression stages)
              --> upstream Anthropic API (Claude, unchanged)
              --> response post-processing (alias expand, state update)
              --> back to Claude Code

Supports streaming (SSE) and non-streaming. Credentials are passed straight
through from the incoming request; the proxy never stores or needs its own key.
The checkpoint summariser reuses those same credentials for a cheap side-call.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Config, load_config
from .metrics import Metrics
from .pipeline import Pipeline
from .postprocess.responses import (
    expand_response_body,
    expand_sse_event,
    record_assistant_turn,
)
from .state import StateStore

logger = logging.getLogger("ccomp")

# headers we must forward upstream; hop-by-hop and host are dropped
_FORWARD_HEADERS = {
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "content-type",
}


def _auth_headers(request: Request) -> dict:
    out = {}
    for k, v in request.headers.items():
        if k.lower() in _FORWARD_HEADERS:
            out[k] = v
    return out


def make_summarizer(cfg: Config, headers: dict):
    """Build a summarize_fn bound to the caller's credentials for this request."""

    def summarize(text: str, target_tokens: int, model: str) -> str:
        prompt = (
            "Summarise the following conversation excerpt for use as a compact "
            "context checkpoint. Preserve decisions, file names, identifiers, "
            "open TODOs, and constraints. Drop pleasantries and superseded "
            f"detail. Target ~{target_tokens} tokens. Output only the summary.\n\n"
            f"{text}"
        )
        body = {
            "model": model,
            "max_tokens": min(2048, target_tokens * 2),
            "messages": [{"role": "user", "content": prompt}],
        }
        url = cfg.upstream_base_url.rstrip("/") + "/v1/messages"
        h = dict(headers)
        h["content-type"] = "application/json"
        with httpx.Client(timeout=60) as client:
            r = client.post(url, headers=h, json=body)
            r.raise_for_status()
            data = r.json()
        parts = [
            b.get("text", "")
            for b in data.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(parts).strip()

    return summarize


def create_app(cfg: Optional[Config] = None) -> FastAPI:
    cfg = cfg or load_config()
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    app = FastAPI(title="claude-compress proxy")
    store = StateStore(persist_path=None)
    metrics = Metrics(cfg.metrics_path)
    app.state.cfg = cfg

    @app.get("/healthz")
    async def healthz():
        from . import embeddings

        return {"ok": True, "embedding_mode": embeddings.mode(), "upstream": cfg.upstream_base_url}

    @app.post("/v1/messages")
    async def messages(request: Request):
        raw = await request.body()
        try:
            req_body = json.loads(raw)
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        headers = _auth_headers(request)
        streaming = bool(req_body.get("stream"))

        # --- input pipeline ------------------------------------------------
        state = store.get(req_body)
        pipeline = Pipeline(cfg, summarize_fn=make_summarizer(cfg, headers))
        new_body, results, tok_in, tok_out = pipeline.run(req_body, state)
        store.commit(state)
        metric_row = metrics.record(
            state.session_id, results, tok_in, tok_out, streaming
        )
        logger.info(
            "session=%s saved=%d (%.1f%%) stages=%s",
            state.session_id,
            metric_row["saved"],
            metric_row["saved_pct"],
            ",".join(f"{r.name}:{r.saved}" for r in results if r.saved),
        )

        url = cfg.upstream_base_url.rstrip("/") + "/v1/messages"
        up_headers = dict(headers)
        up_headers["content-type"] = "application/json"

        # --- forward + post-process ---------------------------------------
        if streaming:
            async def event_stream():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST", url, headers=up_headers, json=new_body
                    ) as r:
                        async for line in r.aiter_lines():
                            if line.startswith("data:"):
                                payload = line[len("data:"):].strip()
                                if payload and payload != "[DONE]":
                                    payload = expand_sse_event(payload, state)
                                yield f"data: {payload}\n\n"
                            elif line.startswith("event:"):
                                yield line + "\n"
                            elif line == "":
                                continue
                            else:
                                yield line + "\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, headers=up_headers, json=new_body)
        try:
            resp_body = r.json()
        except Exception:
            return Response(content=r.content, status_code=r.status_code,
                            media_type=r.headers.get("content-type"))

        if r.status_code == 200 and isinstance(resp_body, dict):
            resp_body = expand_response_body(resp_body, state)
            record_assistant_turn(resp_body, state)
            store.commit(state)

        return JSONResponse(resp_body, status_code=r.status_code)

    return app


app = create_app()
