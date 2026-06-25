"""Alias substitution stage (RISKY, default OFF).

Finds long strings that repeat across the conversation history and replaces them
with short aliases (e.g. @a0), injecting a one-time legend so Claude can resolve
them. The alias_legend is saved in session state so the response post-processor
can expand any aliases Claude echoes back.

Caveats that make this off-by-default:
  - Only profitable when a string repeats enough times to pay back the legend
    overhead. We compute actual token savings before committing.
  - Short aliases don't always tokenise to fewer tokens than the original
    (especially with BPE tokenisers that merge common subwords).
  - The model can get confused or leak aliases into user-visible output.
  - Only worth it for very repetitive, very long identifiers/paths.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from ..config import AliasConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from .base import Stage, StageResult, iter_text_blocks

# Static legend header cost (paid once regardless of how many aliases)
_LEGEND_HEADER = (
    "[ALIAS LEGEND — these short tokens stand in for the strings "
    "on the right; treat them as identical]\n"
)


def _alias_name(i: int) -> str:
    return f"@a{i}"


def _entry_cost(alias: str, original: str) -> int:
    """Tokens for one legend line: '@a0 = original\n'"""
    return count_text(f"{alias} = {original}\n")


def _per_use_saving(alias: str, original: str) -> int:
    """Tokens saved per substituted occurrence (can be negative with BPE)."""
    return count_text(original) - count_text(alias)


class AliasStage(Stage):
    name = "alias_substitution"

    def __init__(self, cfg: AliasConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def _candidates(self, texts: List[str]) -> Dict[str, int]:
        pat = re.compile(r"[\w./:\-]{%d,}" % self.cfg.min_length)
        counts: Counter = Counter()
        for t in texts:
            for m in pat.findall(t):
                counts[m] += 1
        return {s: c for s, c in counts.items() if c >= self.cfg.min_occurrences}

    def _profitable_aliases(
        self, cands: Dict[str, int], header_cost: int, start_idx: int = 0
    ) -> List[Tuple[str, str, int]]:
        """Return [(alias, original, net_saving)] for strings where aliasing
        actually saves tokens after paying the legend overhead.

        The header cost is shared across all selected aliases, so we do a
        greedy selection: sort by per-entry net saving (savings - entry_cost)
        and accumulate until adding the next alias would consume more than it
        saves.
        start_idx offsets alias numbering so aliases never get reused across turns.
        """
        # rank by (per_use_saving * count - entry_cost), best first
        scored = []
        for i, (s, count) in enumerate(
            sorted(cands.items(), key=lambda kv: len(kv[0]) * kv[1], reverse=True)
        ):
            alias = _alias_name(start_idx + i)
            pu = _per_use_saving(alias, s)
            entry = _entry_cost(alias, s)
            gross = pu * count
            net_before_header = gross - entry
            scored.append((alias, s, count, gross, entry, net_before_header))

        selected: List[Tuple[str, str, int]] = []
        cumulative_gross = 0
        cumulative_entry = 0
        for alias, original, count, gross, entry, _ in scored[: self.cfg.max_aliases]:
            cumulative_gross += gross
            cumulative_entry += entry
            total_net = cumulative_gross - cumulative_entry - header_cost
            if total_net > 0:
                selected.append((alias, original, total_net))
            else:
                # adding this alias doesn't recover its own overhead; stop
                break
        return selected

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)
        blocks = list(iter_text_blocks(request, protect_last_n=self.cfg.protect_last_n_messages))
        texts = [b.get("text", "") for (_, _, b) in blocks]
        cands = self._candidates(texts)

        # strip strings already aliased in a previous turn
        already_aliased = set(state.alias_legend.values())
        cands = {s: c for s, c in cands.items() if s not in already_aliased}

        if not cands:
            return StageResult(self.name, before, before, note="no alias candidates")

        header_cost = count_text(_LEGEND_HEADER)
        profitable = self._profitable_aliases(cands, header_cost, start_idx=state.next_alias_index)
        if not profitable:
            return StageResult(
                self.name, before, before,
                note=f"found {len(cands)} candidate(s) but none profitable after "
                     f"legend overhead ({header_cost} tok header)"
            )

        new_legend: Dict[str, str] = {alias: original for alias, original, _ in profitable}

        # full legend = prior aliases + new ones; apply all substitutions so
        # existing aliases already in the text are preserved
        full_legend = {**state.alias_legend, **new_legend}

        # apply substitutions longest-first to prevent a shorter alias from
        # corrupting a longer one that shares a prefix (e.g. /api before /api/users)
        sorted_legend = sorted(full_legend.items(), key=lambda kv: len(kv[1]), reverse=True)
        for _, _, blk in blocks:
            txt = blk.get("text", "")
            for alias, original in sorted_legend:
                if original in txt:
                    txt = txt.replace(original, alias)
            blk["text"] = txt

        # build and prepend the legend block (only new entries this turn)
        legend_lines = "\n".join(f"{a} = {o}" for a, o in new_legend.items())
        legend_block = {
            "type": "text",
            "text": _LEGEND_HEADER + legend_lines,
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

        state.alias_legend.update(new_legend)
        state.next_alias_index += len(new_legend)
        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=f"aliased {len(new_legend)} new string(s), net ~{before - after} tok",
            detail={"aliases": len(new_legend)},
        )
