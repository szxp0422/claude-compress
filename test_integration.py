"""Integration test: run the proxy against a MOCK upstream (no real API key).

Verifies:
  - request reaches upstream in compressed form,
  - non-streaming alias expansion works end to end,
  - streaming SSE passes through and expands text deltas.
"""
import json
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from claude_compress.config import Config
from claude_compress.server import create_app

# ---- mock upstream that echoes what it received + emits an alias ----------
mock = FastAPI()
RECEIVED = {}


@mock.post("/v1/messages")
async def upstream(request: Request):
    body = await request.json()
    RECEIVED["body"] = body
    if body.get("stream"):
        async def gen():
            events = [
                ("message_start", {"type": "message_start", "message": {"id": "m"}}),
                ("content_block_start", {"type": "content_block_start", "index": 0,
                                          "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "text_delta",
                                                    "text": "edited @a0 done"}}),
                ("message_stop", {"type": "message_stop"}),
            ]
            for name, data in events:
                yield f"event: {name}\n"
                yield f"data: {json.dumps(data)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    # non-streaming: respond mentioning an alias so we can test expansion
    return JSONResponse({
        "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "I edited @a0 as requested."}],
        "model": body.get("model"), "stop_reason": "end_turn",
    })


def run_server(app, port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def main():
    cfg = Config()
    cfg.upstream_base_url = "http://127.0.0.1:9911"
    cfg.alias.enabled = True
    cfg.alias.min_occurrences = 6
    cfg.alias.min_length = 10
    cfg.checkpoint.enabled = False  # keep this test focused
    proxy = create_app(cfg)

    t1 = threading.Thread(target=run_server, args=(mock, 9911), daemon=True)
    t2 = threading.Thread(target=run_server, args=(proxy, 9912), daemon=True)
    t1.start(); t2.start()
    time.sleep(2.0)

    lp = "lib/really/long/module/path/runner.rb"
    payload = {
        "model": "claude-sonnet-4-6", "max_tokens": 100,
        "messages": [{"role": "user", "content":
            " ".join([f"open {lp}", f"test {lp}", f"lint {lp}",
                    f"deploy {lp}", f"revert {lp}", f"edit {lp}",
                    f"read {lp}", f"check {lp}"])}],
    }
    h = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01",
         "content-type": "application/json"}

    # non-streaming
    r = httpx.post("http://127.0.0.1:9912/v1/messages", json=payload, headers=h, timeout=10)
    print("status:", r.status_code)
    upstream_text = json.dumps(RECEIVED["body"])
    assert "@a0" in upstream_text, "upstream should have received aliased text"
    # the raw path should now appear only once (inside the injected legend),
    # not at each of its three usage sites
    assert upstream_text.count(lp) == 1, \
        f"expected path once (legend), got {upstream_text.count(lp)}"
    out = r.json()
    assert lp in out["content"][0]["text"], "response should be expanded back"
    print("non-streaming expanded response:", out["content"][0]["text"])

    # streaming
    with httpx.stream("POST", "http://127.0.0.1:9912/v1/messages",
                      json={**payload, "stream": True}, headers=h, timeout=10) as s:
        collected = ""
        for line in s.iter_lines():
            if line.startswith("data:"):
                d = line[5:].strip()
                if d and d != "[DONE]":
                    obj = json.loads(d)
                    if obj.get("type") == "content_block_delta":
                        collected += obj["delta"]["text"]
    print("streaming expanded text:", collected)
    assert lp in collected, "streaming delta should be expanded"

    print("\nINTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
