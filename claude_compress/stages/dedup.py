"""Semantic dedup stage.

Removes text blocks in the conversation *history* whose meaning is already
represented by an earlier block (cosine >= threshold). This is the safest
high-value compression: boilerplate, re-pasted files, repeated tool framing.

Safety:
  - Never touches the last N messages (live working set).
  - Never touches tool/image blocks.
  - Keeps the FIRST occurrence, drops later near-duplicates.
  - Leaves a tiny stub so the turn structure / role alternation is preserved.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .. import embeddings
from ..config import DedupConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from .base import Stage, StageResult, iter_text_blocks


class DedupStage(Stage):
    name = "semantic_dedup"

    def __init__(self, cfg: DedupConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        blocks = iter_text_blocks(request, self.cfg.protect_last_n_messages)
        candidates = [
            (mi, bi, blk)
            for (mi, bi, blk) in blocks
            if count_text(blk.get("text", "")) >= self.cfg.min_block_tokens
        ]
        if len(candidates) < 2:
            return StageResult(self.name, before, before, note="nothing to dedup")

        texts = [blk["text"] for (_, _, blk) in candidates]
        vecs = embeddings.embed(texts)

        kept: List[int] = []
        dropped = 0
        for i in range(len(candidates)):
            is_dup = False
            for j in kept:
                if embeddings.cosine(vecs[i], vecs[j]) >= self.cfg.threshold:
                    is_dup = True
                    break
            if is_dup:
                _, _, blk = candidates[i]
                blk["text"] = "[redundant context removed by ccomp]"
                dropped += 1
            else:
                kept.append(i)

        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=f"dropped {dropped} duplicate block(s); embed={embeddings.mode()}",
            detail={"dropped": dropped, "candidates": len(candidates)},
        )
