"""Alias substitution stage (RISKY, default OFF).

Finds long strings that repeat across the conversation history and replaces them
with short aliases (e.g. @A1), injecting a one-time legend so Claude can resolve
them. The alias_legend is saved in session state so the response post-processor
can expand any aliases Claude echoes back.

Caveats that make this off-by-default:
  - Short aliases don't always tokenise to fewer tokens than the original.
  - The model can get confused or leak aliases into user-visible output.
  - Only worth it for very repetitive, very long identifiers/paths.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from ..config import AliasConfig
from ..state import SessionState
from ..tokens import count_request
from .base import Stage, StageResult, iter_text_blocks


class AliasStage(Stage):
    name = "alias_substitution"

    def __init__(self, cfg: AliasConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def _candidates(self, texts: List[str]) -> Dict[str, int]:
        # candidate = long-ish token sequences (paths, qualified names, urls)
        pat = re.compile(r"[\w./:\-]{%d,}" % self.cfg.min_length)
        counts: Counter = Counter()
        for t in texts:
            for m in pat.findall(t):
                counts[m] += 1
        return {
            s: c
            for s, c in counts.items()
            if c >= self.cfg.min_occurrences
        }

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        blocks = iter_text_blocks(request, protect_last_n=0)
        texts = [b.get("text", "") for (_, _, b) in blocks]
        cands = self._candidates(texts)
        if not cands:
            return StageResult(self.name, before, before, note="no alias candidates")

        # rank by (len * count) savings potential, take top-k
        ranked = sorted(cands.items(), key=lambda kv: len(kv[0]) * kv[1], reverse=True)
        ranked = ranked[: self.cfg.max_aliases]

        legend: Dict[str, str] = {}
        for i, (s, _) in enumerate(ranked):
            alias = f"@a{i}"
            legend[alias] = s

        # apply substitutions
        for _, _, blk in blocks:
            txt = blk.get("text", "")
            for alias, original in legend.items():
                if original in txt:
                    txt = txt.replace(original, alias)
            blk["text"] = txt

        # persist for response expansion, and inject the legend once at the front
        state.alias_legend.update(legend)
        legend_lines = "\n".join(f"{a} = {o}" for a, o in legend.items())
        legend_block = {
            "type": "text",
            "text": "[ALIAS LEGEND — these short tokens stand in for the strings "
            "on the right; treat them as identical]\n" + legend_lines,
        }
        msgs = request.get("messages", [])
        if msgs:
            first = msgs[0]
            content = first.get("content")
            if isinstance(content, str):
                first["content"] = [{"type": "text", "text": content}]
                content = first["content"]
            if isinstance(content, list):
                content.insert(0, legend_block)

        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=f"aliased {len(legend)} string(s)",
            detail={"aliases": len(legend)},
        )
