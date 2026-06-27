"""Semantic dedup stage.

Removes text blocks in the conversation *history* whose meaning is already
represented by an earlier block (cosine >= threshold). Now handles both
`text` blocks and `tool_result` blocks so repeated file reads in Claude Code
sessions are caught and removed.

Safety:
  - Never touches the last N messages (live working set).
  - Never modifies tool_use or image blocks.
  - Keeps the FIRST occurrence, drops later near-duplicates.
  - Leaves a tiny stub so turn structure / role alternation is preserved.
  - Never treats checkpoint summary messages as anchors.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import embeddings
from ..config import DedupConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from .base import Stage, StageResult, iter_all_text

_CHECKPOINT_MARKER = "=== CONVERSATION CHECKPOINT"
_STUB_TEXT = "[redundant context removed by ccomp]"
_STUB_TOOL = "[duplicate tool result removed by ccomp]"

# Approximate usable context budget; pressure ramps up beyond 50% of this.
_CONTEXT_LIMIT = 180_000


def _otsu_threshold(vals: np.ndarray, bins: int = 100) -> float:
    """Find the similarity cutoff that best separates dissimilar from near-duplicate pairs.

    Uses Otsu's method on the histogram of pairwise cosine similarities: finds the
    threshold that maximises between-class variance, i.e. the natural valley between
    the "clearly different" and "near-duplicate" clusters.
    """
    if len(vals) < 3:
        return 0.93
    counts, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(counts.sum())
    if total == 0:
        return 0.93
    total_mean = float(np.sum(counts * centers)) / total
    best_var, best_t = -1.0, 0.93
    w0 = 0.0
    sum0 = 0.0
    for i in range(len(counts)):
        w0 += counts[i] / total
        sum0 += counts[i] * centers[i] / total
        w1 = 1.0 - w0
        if w0 <= 0.0 or w1 <= 0.0:
            continue
        mu0 = sum0 / w0
        mu1 = (total_mean - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = centers[i]
    return float(best_t)


class DedupStage(Stage):
    name = "semantic_dedup"

    def __init__(self, cfg: DedupConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        all_blocks = iter_all_text(
            request,
            self.cfg.protect_last_n_messages,
            include_tool_results=True,
        )
        candidates = [
            (mi, bi, text, src)
            for (mi, bi, text, src) in all_blocks
            if count_text(text) >= self.cfg.min_block_tokens
            and _CHECKPOINT_MARKER not in text
        ]
        if len(candidates) < 2:
            return StageResult(self.name, before, before, note="nothing to dedup")

        texts = [text for (_, _, text, _) in candidates]

        vecs = embeddings.embed(texts)
        sims = vecs @ vecs.T  # (n, n) cosine similarities

        # Dynamic threshold: two signals combined.
        #
        # 1. Otsu's method on the pairwise similarity distribution — finds the
        #    natural valley between "clearly different" and "near-duplicate" clusters
        #    in *this* session rather than using a fixed global cutoff.
        #
        # 2. Token pressure — as the session approaches the context limit, pull the
        #    threshold down further so dedup becomes more aggressive exactly when
        #    compression is most needed.  No effect below 50% utilisation; maximum
        #    reduction of 0.13 at 100%.
        n = len(candidates)
        if n >= 3:
            upper = sims[np.triu_indices(n, k=1)]
            threshold = _otsu_threshold(upper)
        else:
            threshold = self.cfg.threshold

        pressure = max(0.0, (before / _CONTEXT_LIMIT - 0.5) * 2.0)
        threshold -= 0.13 * pressure

        # Hash embeddings are lexical, not semantic; cap more conservatively.
        if embeddings.mode() == "hash":
            threshold = min(threshold, 0.90)

        threshold = float(np.clip(threshold, 0.80, 0.97))

        msgs = request.get("messages", [])
        kept: List[int] = []
        dropped_text = 0
        dropped_tool = 0

        for i in range(len(candidates)):
            mi, bi, text, src = candidates[i]
            is_dup = any(sims[i, j] >= threshold for j in kept)
            if is_dup:
                block = msgs[mi]["content"][bi]
                if src == "text":
                    block["text"] = _STUB_TEXT
                    dropped_text += 1
                elif src == "tool_result":
                    # replace the content of the tool_result block
                    if isinstance(block.get("content"), str):
                        block["content"] = _STUB_TOOL
                    elif isinstance(block.get("content"), list):
                        # keep the first text sub-block, stub it
                        for sub in block["content"]:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                sub["text"] = _STUB_TOOL
                                break
                    dropped_tool += 1
            else:
                kept.append(i)

        dropped = dropped_text + dropped_tool
        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=(
                f"dropped {dropped} duplicate block(s) "
                f"({dropped_text} text, {dropped_tool} tool_result); "
                f"embed={embeddings.mode()} threshold={threshold:.3f} pressure={pressure:.2f}"
            ),
            detail={
                "dropped": dropped,
                "dropped_text": dropped_text,
                "dropped_tool": dropped_tool,
                "candidates": len(candidates),
            },
        )
