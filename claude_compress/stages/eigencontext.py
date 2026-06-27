"""Eigencontext stage (LOSSY, default OFF).

Idea: treat tagged reference material as a set of sentences and keep the smallest
subset that "covers" the information mass, using a greedy max-coverage selection
over sentence embeddings (a facility-location / submodular approximation, which
is the practical stand-in for the PCA-style "minimum basis" framing).

Only operates on blocks explicitly tagged with `cfg.marker` so it can never
silently prune live instructions or code. Even so, it's off by default because
dropping context from a coding agent is risky.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from .. import embeddings
from ..config import EigencontextConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from .base import Stage, StageResult, iter_text_blocks

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts


def _greedy_cover(vecs: np.ndarray, coverage: float) -> List[int]:
    """Greedy facility-location: repeatedly add the sentence that most increases
    total coverage (max similarity of every sentence to the chosen set)."""
    n = vecs.shape[0]
    if n == 0:
        return []
    sim = vecs @ vecs.T  # (n, n), all in [-1, 1] for normalised vecs
    np.clip(sim, 0.0, 1.0, out=sim)
    best_cov = np.zeros(n)
    chosen: List[int] = []
    total_target = coverage * n
    while len(chosen) < n:
        # pick sentence maximising marginal gain
        gains = np.maximum(sim, best_cov[None, :]).sum(axis=1) - best_cov.sum()
        for c in chosen:
            gains[c] = -1
        nxt = int(np.argmax(gains))
        if gains[nxt] <= 0:
            break
        chosen.append(nxt)
        best_cov = np.maximum(best_cov, sim[nxt])
        if best_cov.sum() >= total_target:
            break
    return sorted(chosen)


class EigencontextStage(Stage):
    name = "eigencontext"

    def __init__(self, cfg: EigencontextConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        blocks = iter_text_blocks(request, self.cfg.protect_last_n_messages)
        touched = 0
        for _, _, blk in blocks:
            text = blk.get("text", "")
            if not text.startswith(self.cfg.marker):
                continue
            body = text[len(self.cfg.marker):]
            sents = _split_sentences(body)
            if len(sents) < 4:
                continue
            vecs = embeddings.embed(sents)
            keep_idx = _greedy_cover(vecs, self.cfg.coverage)
            if len(keep_idx) >= len(sents):
                continue
            blk["text"] = self.cfg.marker + " ".join(sents[i] for i in keep_idx)
            touched += 1

        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=f"reduced {touched} reference block(s); embed={embeddings.mode()}",
            detail={"blocks": touched},
        )
