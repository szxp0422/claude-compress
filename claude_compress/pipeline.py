"""Pipeline: run the input stages in a safe, fixed order.

Order matters:
  1. state_machine   -- inject compact workflow (adds a few tokens, opt-in)
  2. checkpoint      -- fold old turns FIRST so later stages work on less text
  3. dedup           -- remove near-duplicate survivors
  4. eigencontext    -- prune tagged reference material (lossy, opt-in)
  5. alias           -- substitute repeated long strings (risky, opt-in)
  6. delta           -- LAST: insert cache breakpoints on the final shape
                        (must run last so breakpoints sit on stable content)
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional

from .config import Config
from .state import SessionState
from .stages.base import StageResult
from .stages.alias import AliasStage
from .stages.checkpoint import CheckpointStage, SummarizeFn
from .stages.dedup import DedupStage
from .stages.delta import DeltaStage
from .stages.eigencontext import EigencontextStage
from .stages.state_machine import StateMachineStage
from .tokens import count_request


class Pipeline:
    def __init__(self, cfg: Config, summarize_fn: Optional[SummarizeFn] = None):
        self.cfg = cfg
        self.stages = [
            StateMachineStage(cfg.state_machine),
            CheckpointStage(cfg.checkpoint, summarize_fn=summarize_fn),
            DedupStage(cfg.dedup),
            EigencontextStage(cfg.eigencontext),
            AliasStage(cfg.alias),
            DeltaStage(cfg.delta),
        ]

    def run(self, request: dict, state: SessionState):
        """Returns (new_request, [StageResult], tokens_in, tokens_out)."""
        tokens_in = count_request(request)
        work = copy.deepcopy(request)
        results: List[StageResult] = []
        for stage in self.stages:
            if not stage.enabled():
                continue
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
