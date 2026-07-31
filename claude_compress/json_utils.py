"""JSON minification and smart truncation for tool_result content.

Two-pass compression:
  1. Minify  — re-serialise with no whitespace (lossless).
  2. Truncate — collapse oversized arrays to head + tail and cap long strings.
     The omission markers are human-readable strings Claude can parse, so the
     model still knows what was dropped.

No external dependencies — stdlib json only.
"""
from __future__ import annotations

import json
from typing import Any, Optional


def compress_json(
    text: str,
    max_array_items: int,
    max_string_chars: int,
) -> Optional[str]:
    """Parse *text* as JSON and return a minified, optionally-truncated form.

    Returns ``None`` when *text* is not valid JSON (caller leaves it unchanged).
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    data = _truncate(data, max_array_items, max_string_chars, depth=0)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def looks_like_json(text: str) -> bool:
    """Quick structural check — no full parse."""
    s = text.lstrip()
    return s.startswith(("{", "[", '"')) and (s.endswith(("]", "}", '"')) or len(s) > 2)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _truncate(val: Any, max_arr: int, max_str: int, depth: int) -> Any:
    if depth > 12:
        # Bail on pathologically deep structures to avoid stack overflow.
        return val
    if isinstance(val, dict):
        return {k: _truncate(v, max_arr, max_str, depth + 1) for k, v in val.items()}
    if isinstance(val, list):
        if len(val) > max_arr:
            n_head = max(1, max_arr // 2)
            n_tail = max(1, max_arr - n_head)
            omitted = len(val) - max_arr
            head = [_truncate(v, max_arr, max_str, depth + 1) for v in val[:n_head]]
            tail = [_truncate(v, max_arr, max_str, depth + 1) for v in val[-n_tail:]]
            return head + [f"[…{omitted} items omitted…]"] + tail
        return [_truncate(v, max_arr, max_str, depth + 1) for v in val]
    if isinstance(val, str) and len(val) > max_str:
        omitted = len(val) - max_str
        return val[:max_str] + f"…[{omitted} chars omitted]"
    return val
