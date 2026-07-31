"""HTML extraction stage.

Targets tool_result blocks whose content is HTML (web-scraping output is the
primary source). Strips scripts, styles, navigation, and boilerplate; converts
semantic structure to markdown-adjacent text using stdlib html.parser.

The conversion loses inline hyperlink destinations and fine-grained formatting,
but preserves all information content (headings, lists, body text, code blocks).
For most web-content retrieval tasks, text extraction is strictly better than
raw HTML: shorter, no irrelevant tokens, easier for Claude to reason about.

Disabled by default for HTML that is structured data meant to be parsed rather
than read — disable this stage if the HTML structure itself is the subject of
the conversation.
"""
from __future__ import annotations

from ..config import HtmlConfig
from ..state import SessionState
from ..tokens import count_request, count_text
from ..html_utils import html_to_text, looks_like_html
from .base import (
    Stage,
    StageResult,
    get_tool_result_text,
    iter_tool_result_blocks,
    set_tool_result_text,
)


class HtmlCompressStage(Stage):
    name = "html_compress"

    def __init__(self, cfg: HtmlConfig) -> None:
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
            if not looks_like_html(text):
                continue

            try:
                extracted = html_to_text(text)
            except Exception:
                continue  # never break the request

            if not extracted or len(extracted) >= len(text):
                continue

            saved = count_text(text) - count_text(extracted)
            if saved <= 0:
                continue

            set_tool_result_text(block, extracted)
            tokens_saved += saved
            n_compressed += 1

        after = count_request(request)
        return StageResult(
            self.name,
            before,
            after,
            note=f"extracted text from {n_compressed} HTML block(s), saved ~{tokens_saved} tokens",
            detail={"blocks_compressed": n_compressed, "tokens_saved": tokens_saved},
        )
