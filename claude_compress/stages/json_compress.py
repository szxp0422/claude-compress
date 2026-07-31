"""JSON compression stage.

Targets tool_result blocks whose content is valid JSON. Two passes:
  1. Minify — re-serialise with no whitespace (always lossless).
  2. Truncate — collapse arrays longer than max_array_items to head + tail,
     and cap string values longer than max_string_chars.

Both passes are applied together in a single json.loads / json.dumps cycle.
The stage is a no-op on non-JSON content (parse failure → skip silently).
"""
from __future__ import annotations

from ..config import JsonConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from ..json_utils import compress_json
from .base import (
    Stage,
    StageResult,
    get_tool_result_text,
    iter_tool_result_blocks,
    set_tool_result_text,
)


class JsonCompressStage(Stage):
    name = "json_compress"

    def __init__(self, cfg: JsonConfig) -> None:
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        blocks = iter_tool_result_blocks(request, self.cfg.protect_last_n_messages)

        n_compressed = 0
        tokens_saved = 0

        for _mi, _bi, block in blocks:
            text = get_tool_result_text(block)
            if not text or count_text(text) < self.cfg.min_compress_tokens:
                continue

            compressed = compress_json(
                text,
                self.cfg.max_array_items,
                self.cfg.max_string_chars,
            )
            if compressed is None or len(compressed) >= len(text):
                continue  # not JSON, or already minimal

            saved = count_text(text) - count_text(compressed)
            if saved <= 0:
                continue

            set_tool_result_text(block, compressed)
            tokens_saved += saved
            n_compressed += 1

        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=f"compressed {n_compressed} JSON block(s), saved ~{tokens_saved} tokens",
            detail={"blocks_compressed": n_compressed, "tokens_saved": tokens_saved},
        )
