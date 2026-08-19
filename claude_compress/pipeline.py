"""Pipeline: run the input stages in a safe, fixed order.

Order matters:
  1. image_compress  -- reduce image visual tokens (opt-in; Pillow required)
  2. json_compress   -- minify + truncate JSON in tool_results (on by default)
  3. log_compress    -- collapse repeated lines + truncate stack traces (on by default)
  4. html_compress   -- strip HTML boilerplate, extract text (opt-in)
  5. state_machine   -- inject compact workflow (adds a few tokens, opt-in)
  6. checkpoint      -- fold old turns so later stages work on less text
  7. dedup           -- remove near-duplicate survivors
  8. eigencontext    -- prune tagged reference material (lossy, opt-in)
  9. alias           -- substitute repeated long strings (risky, opt-in)
  10. delta          -- LAST: insert cache breakpoints on the final shape
                        (must run last so breakpoints sit on stable content)

Dynamic tiering:
  TINY     (< 2000 tokens):  pass through (nothing to compress)
  SHORT    (< 4000 tokens):  cache breakpoints only (no overhead)
  TOOL-HEAVY (>30% tool results): skip alias/eigencontext (they don't understand
                                   tool_result structure)
  NORMAL:  all enabled stages
"""
from __future__ import annotations

import copy
from typing import List, Optional

from .config import Config
from .state import SessionState
from .stages.base import StageResult, count_tool_result_tokens
from .stages.alias import AliasStage
from .stages.checkpoint import CheckpointStage, SummarizeFn
from .stages.dedup import DedupStage
from .stages.delta import DeltaStage
from .stages.eigencontext import EigencontextStage
from .stages.html_compress import HtmlCompressStage
from .stages.image_compress import ImageCompressStage
from .stages.video_compress import VideoCompressStage
from .stages.json_compress import JsonCompressStage
from .stages.log_compress import LogCompressStage
from .stages.state_machine import StateMachineStage
from .tokens import count_request


class Pipeline:
    def __init__(self, cfg: Config, summarize_fn: Optional[SummarizeFn] = None):
        self.cfg = cfg
        self.stages = [
            # Format-specific stages run first so checkpoint and dedup see
            # already-compressed tool results.
            VideoCompressStage(cfg.video),
            ImageCompressStage(cfg.image),
            JsonCompressStage(cfg.json),
            LogCompressStage(cfg.log),
            HtmlCompressStage(cfg.html),
            StateMachineStage(cfg.state_machine),
            CheckpointStage(cfg.checkpoint, summarize_fn=summarize_fn),
            DedupStage(cfg.dedup),
            EigencontextStage(cfg.eigencontext),
            AliasStage(cfg.alias),
            DeltaStage(cfg.delta),
        ]

    def _select_stages(self, request: dict) -> List:
        """Return the stages to run for this specific request based on tier."""
        total = count_request(request)

        if total < self.cfg.tier.tiny_threshold:
            # too small to benefit from any compression
            return [s for s in self.stages if s.name == "delta_cache_breakpoints" and s.enabled()]

        if total < self.cfg.tier.short_threshold:
            # small session — only lossless cache stage, avoid any overhead
            return [s for s in self.stages if s.name == "delta_cache_breakpoints" and s.enabled()]

        tool_tokens = count_tool_result_tokens(request)
        tool_ratio = tool_tokens / total if total else 0.0

        if tool_ratio > 0.30:
            # tool-heavy: skip alias and eigencontext — they don't understand
            # tool_result structure and add overhead without benefit.
            # Format stages (image, json, log, html) are especially valuable here.
            safe_for_tools = {
                "video_compress",
                "image_compress",
                "json_compress",
                "log_compress",
                "html_compress",
                "semantic_dedup",
                "checkpoint_compression",
                "delta_cache_breakpoints",
            }
            return [s for s in self.stages if s.name in safe_for_tools and s.enabled()]

        # normal prose/code session: run all enabled stages
        return [s for s in self.stages if s.enabled()]

    def run(self, request: dict, state: SessionState):
        """Returns (new_request, [StageResult], tokens_in, tokens_out)."""
        tokens_in = count_request(request)
        work = copy.deepcopy(request)
        results: List[StageResult] = []
        for stage in self._select_stages(request):
            try:
                res = stage.apply(work, state)
                results.append(res)
            except Exception as e:  # a broken stage must never break the request
                results.append(
                    StageResult(
                        stage.name,
                        count_request(work),
                        count_request(work),
                        note=f"ERROR skipped: {type(e).__name__}: {e}",
                    )
                )
        # delta stage is lossless and last; report size before it as tokens_out
        # so "saved" reflects real size reduction, not cache (which saves cost)
        size_stages = [r for r in results if r.name != "delta_cache_breakpoints"]
        tokens_out = size_stages[-1].tokens_after if size_stages else tokens_in
        return work, results, tokens_in, tokens_out
