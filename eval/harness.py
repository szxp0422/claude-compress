"""Paired A/B eval harness.

For each task we run the SAME conversation through two arms:
  - baseline   : original request -> model
  - compressed : Pipeline(config).run(request) -> model

Both arms hit the same model; only the input transform differs. We evaluate the
*transform* in-process (not the HTTP proxy layer) so ablations are just config
swaps and there's no server to manage. The production proxy applies the identical
transform, so this faithfully measures what ships.

Multi-turn tasks are REPLAYED turn by turn, re-applying compression each turn, so
stateful effects (checkpoint summarisation, growing cache prefix) accumulate
exactly as they would in a real session. This is what surfaces *longitudinal*
degradation -- the failure mode single-turn tests miss.

A `model_call(request, headers) -> (response_body, Usage)` is injected:
  - real_model_call posts to the upstream API
  - mock_model_call fabricates deterministic output + synthetic usage so the
    harness runs with no network/key (numbers are clearly marked synthetic).
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from claude_compress.config import Config
from claude_compress.pipeline import Pipeline
from claude_compress.state import SessionState
from claude_compress.tokens import count_request, content_to_text
from claude_compress.usage import Usage

from .judge import score_objective, score_pairwise, JudgeCall

ModelCall = Callable[[dict, dict], Tuple[dict, Usage]]


@dataclass
class TurnResult:
    turn_index: int
    baseline_usage: Usage
    compressed_usage: Usage
    baseline_score: Optional[float] = None
    compressed_score: Optional[float] = None
    est_input_baseline: int = 0
    est_input_compressed: int = 0


@dataclass
class TaskResult:
    task_id: str
    turns: List[TurnResult] = field(default_factory=list)

    def final_scores(self) -> Tuple[Optional[float], Optional[float]]:
        scored = [t for t in self.turns if t.baseline_score is not None]
        if not scored:
            return None, None
        last = scored[-1]
        return last.baseline_score, last.compressed_score


def _assistant_text(resp: dict) -> str:
    return "\n".join(
        b.get("text", "")
        for b in resp.get("content", [])
        if isinstance(b, dict) and b.get("type") == "text"
    )


def run_task(
    task: dict,
    cfg: Config,
    model_call: ModelCall,
    judge_call: Optional[JudgeCall] = None,
    headers: Optional[dict] = None,
) -> TaskResult:
    headers = headers or {}
    base_msgs: List[dict] = []
    comp_msgs: List[dict] = []
    comp_state = SessionState(session_id=task.get("id", "t"))
    pipeline = Pipeline(cfg, summarize_fn=None)  # summariser optional in eval

    tr = TaskResult(task_id=task.get("id", "t"))
    system = task.get("system")
    tools = task.get("tools")

    for i, turn in enumerate(task["turns"]):
        user_msg = {"role": "user", "content": turn["user"]}
        base_msgs.append(copy.deepcopy(user_msg))
        comp_msgs.append(copy.deepcopy(user_msg))

        base_req = {"model": task.get("model", "claude-sonnet-4-6"),
                    "max_tokens": task.get("max_tokens", 1024),
                    "messages": copy.deepcopy(base_msgs)}
        comp_req = {"model": task.get("model", "claude-sonnet-4-6"),
                    "max_tokens": task.get("max_tokens", 1024),
                    "messages": copy.deepcopy(comp_msgs)}
        if system:
            base_req["system"] = system
            comp_req["system"] = system
        if tools:
            base_req["tools"] = tools
            comp_req["tools"] = tools

        est_base = count_request(base_req)
        new_comp_req, _results, _ti, est_comp = pipeline.run(comp_req, comp_state)

        base_resp, base_usage = model_call(base_req, headers)
        comp_resp, comp_usage = model_call(new_comp_req, headers)

        # append assistant replies to each arm's running history
        base_msgs.append({"role": "assistant", "content": _assistant_text(base_resp)})
        comp_msgs.append({"role": "assistant", "content": _assistant_text(comp_resp)})

        turn_res = TurnResult(
            turn_index=i,
            baseline_usage=base_usage, compressed_usage=comp_usage,
            est_input_baseline=est_base, est_input_compressed=est_comp,
        )

        # score this turn if it carries a check or we have a judge
        check_task = {**task, **turn}  # turn-level check overrides task-level
        bs = score_objective(check_task, _assistant_text(base_resp))
        cs = score_objective(check_task, _assistant_text(comp_resp))
        if bs is None and cs is None and judge_call and turn.get("judge", False):
            pj = score_pairwise(check_task, _assistant_text(base_resp),
                                _assistant_text(comp_resp), judge_call, seed=i)
            bs, cs = pj["baseline"], pj["compressed"]
        turn_res.baseline_score = bs
        turn_res.compressed_score = cs
        tr.turns.append(turn_res)

    return tr


# ---- model callers --------------------------------------------------------

def make_real_model_call(upstream_base_url: str) -> ModelCall:
    import httpx
    from claude_compress.usage import parse_usage

    def call(request: dict, headers: dict) -> Tuple[dict, Usage]:
        url = upstream_base_url.rstrip("/") + "/v1/messages"
        h = dict(headers)
        h["content-type"] = "application/json"
        with httpx.Client(timeout=120) as client:
            r = client.post(url, headers=h, json=request)
            if r.status_code >= 400:
                print("API error response:", r.text)
            r.raise_for_status()
            body = r.json()
        return body, parse_usage(body)

    return call


def make_mock_model_call(cache_aware: bool = True) -> ModelCall:
    """Synthetic model. Usage is derived from the request so that:
      - smaller compressed prompts report fewer input_tokens (savings show up)
      - cache_control breakpoints move tokens into cache_read (delta stage shows)
    Output text deterministically 'answers' tasks whose check is a 'contains',
    and occasionally drops the answer when the prompt is heavily compressed, so
    the quality axis isn't trivially perfect. SYNTHETIC -- for wiring only.
    """
    seen_prefixes: Dict[str, int] = {}

    def call(request: dict, headers: dict) -> Tuple[dict, Usage]:
        approx = count_request(request)
        # detect cache breakpoints to simulate cache reads on repeat prefixes
        cached = 0
        if cache_aware:
            sig = hashlib.sha1(
                content_to_text(request.get("messages", [{}])[0].get("content", "")).encode()
            ).hexdigest()[:10]
            prev = seen_prefixes.get(sig, 0)
            has_bp = _has_breakpoint(request)
            if has_bp and prev:
                cached = int(approx * 0.6)  # 60% of input served from cache
            seen_prefixes[sig] = prev + 1
        input_tokens = max(1, approx - cached)

        # produce an answer; if the prompt got very small, sometimes omit the
        # 'needle' to create a measurable (synthetic) quality cost
        needle = _find_needle(request)
        drop = approx < 60 and (approx % 7 == 0)
        text = "" if (needle and drop) else (f"Answer: {needle}." if needle else "Done.")

        body = {
            "id": "mock", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": request.get("model"),
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
                "output_tokens": max(1, len(text) // 4),
            },
        }
        return body, Usage(
            input_tokens=input_tokens, cache_read_input_tokens=cached,
            output_tokens=max(1, len(text) // 4),
        )

    return call


def _has_breakpoint(request: dict) -> bool:
    def scan(blocks):
        for b in blocks or []:
            if isinstance(b, dict) and b.get("cache_control"):
                return True
        return False
    sysv = request.get("system")
    if isinstance(sysv, list) and scan(sysv):
        return True
    if scan(request.get("tools")):
        return True
    for m in request.get("messages", []):
        c = m.get("content")
        if isinstance(c, list) and scan(c):
            return True
    return False


def _find_needle(request: dict) -> Optional[str]:
    """Pull a 'NEEDLE=xyz' marker the eval tasks plant, to test if compression
    preserved a specific fact buried in context."""
    import re
    for m in request.get("messages", []):
        t = content_to_text(m.get("content", ""))
        mt = re.search(r"NEEDLE=(\w+)", t)
        if mt:
            return mt.group(1)
    return None
