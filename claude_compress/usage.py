"""Ground-truth token & cost accounting from the API's own `usage` block.

The heuristic counter in tokens.py is fine for *tuning* a stage, but you cannot
*prove savings* with an estimate. The Messages API returns the real numbers:

    usage: {
        input_tokens,                  # uncached input billed at base rate
        cache_creation_input_tokens,   # written to cache, billed at 1.25x base
        cache_read_input_tokens,       # served from cache, billed at 0.10x base
        output_tokens
    }

Total input the model saw = input_tokens + cache_creation + cache_read.
Total *billed* input cost weights those three buckets differently, which is the
whole point of the delta/cache stage: it moves tokens from the 1.0x bucket into
the 0.10x bucket without changing what Claude sees.

PRICING NOTE: per-million-token prices change and depend on the model. The values
below are placeholders you MUST verify against current Anthropic pricing for your
model before trusting any dollar figure. The cache *multipliers* (1.25x / 0.10x)
have been stable and are encoded as ratios.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

# ---- VERIFY THESE against current pricing for your model -------------------
# dollars per 1M tokens
DEFAULT_PRICES = {
    "input_per_mtok": 3.00,    # placeholder — verify
    "output_per_mtok": 15.00,  # placeholder — verify
}
CACHE_WRITE_MULT = 1.25   # cache_creation billed at 1.25x base input
CACHE_READ_MULT = 0.10    # cache_read billed at 0.10x base input


@dataclass
class Usage:
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input(self) -> int:
        """What the model actually processed on the input side."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def cost(self, prices: Optional[dict] = None) -> float:
        p = prices or DEFAULT_PRICES
        inp = p["input_per_mtok"] / 1_000_000
        out = p["output_per_mtok"] / 1_000_000
        return (
            self.input_tokens * inp
            + self.cache_creation_input_tokens * inp * CACHE_WRITE_MULT
            + self.cache_read_input_tokens * inp * CACHE_READ_MULT
            + self.output_tokens * out
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_input"] = self.total_input
        return d


def parse_usage(obj: dict) -> Usage:
    """Extract a Usage from a non-streaming response body's `usage` field."""
    u = (obj or {}).get("usage", {}) or {}
    return Usage(
        input_tokens=u.get("input_tokens", 0) or 0,
        cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=u.get("cache_read_input_tokens", 0) or 0,
        output_tokens=u.get("output_tokens", 0) or 0,
    )


class StreamingUsageAccumulator:
    """Usage in streaming arrives split across events:
       - message_start.message.usage : input + cache fields (+ partial output)
       - message_delta.usage         : final output_tokens
    Feed it each parsed SSE data object; read .usage at the end.
    """

    def __init__(self):
        self.usage = Usage()

    def feed(self, obj: dict):
        t = obj.get("type")
        if t == "message_start":
            u = parse_usage(obj.get("message", {}))
            self.usage.input_tokens = u.input_tokens
            self.usage.cache_creation_input_tokens = u.cache_creation_input_tokens
            self.usage.cache_read_input_tokens = u.cache_read_input_tokens
            self.usage.output_tokens = u.output_tokens
        elif t == "message_delta":
            u = obj.get("usage", {}) or {}
            if "output_tokens" in u:
                self.usage.output_tokens = u["output_tokens"]
