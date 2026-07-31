"""Log and stack-trace compression stage.

Targets tool_result blocks that look like shell/log output or exception traces.
Two operations applied in sequence:
  1. Repeated-line collapse — consecutive identical lines become one entry with
     a "[above line repeated ×N]" annotation.
  2. Stack-frame truncation — long frame runs are reduced to head + tail with
     an "[N frames omitted]" gap in the middle.

Both operations are self-gating: collapse is a no-op when there are no repeated
lines; truncation fires only when it detects actual stack-frame patterns. Running
the stage on non-log content is safe (output equals input).
"""
from __future__ import annotations

from ..config import LogConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from ..log_utils import compress_log, looks_like_log
from .base import (
    Stage,
    StageResult,
    get_tool_result_text,
    iter_tool_result_blocks,
    set_tool_result_text,
)


class LogCompressStage(Stage):
    name = "log_compress"

    def __init__(self, cfg: LogConfig) -> None:
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
            if not looks_like_log(text):
                continue

            compressed = compress_log(text, self.cfg.max_stack_frames)
            if compressed == text:
                continue

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
            note=f"compressed {n_compressed} log/trace block(s), saved ~{tokens_saved} tokens",
            detail={"blocks_compressed": n_compressed, "tokens_saved": tokens_saved},
        )
