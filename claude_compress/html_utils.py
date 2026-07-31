"""HTML → readable text extraction for tool_result content.

Strips scripts, styles, navigation, and boilerplate; converts semantic
structure (headings, lists, code blocks) to markdown-adjacent text.
Uses only stdlib html.parser — no external dependencies.

Limitations vs. a full HTML→markdown library:
  - Inline links are rendered as plain text (href is discarded).
  - Table structure is approximated with pipe characters.
  - Deeply nested layouts may collapse to a flat paragraph.
These are acceptable for the use case: reducing web-scraping output so
Claude can extract information from it without paying for scripts + nav tokens.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List

# Tags whose entire subtree (including children) should be dropped.
_DROP_TAGS = frozenset({
    "script", "style", "nav", "header", "footer", "aside",
    "noscript", "iframe", "template", "svg", "path", "symbol",
    "button", "form", "input", "select", "option", "textarea",
    "meta", "link", "head",
})

# Tags that map to a markdown heading prefix.
_HEADING = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}

# Tags that should introduce a newline in the output.
_BLOCK = frozenset({
    "p", "div", "section", "article", "main", "blockquote",
    "dd", "dt", "figcaption", "address", "dl",
})

# Semantic HTML tags used to identify HTML content (see looks_like_html).
_SEMANTIC_RE = re.compile(
    r"<(?:html|head|body|div|span|p|a\b|h[1-6]|ul|ol|li|table|tr|td|th"
    r"|nav|header|footer|article|section|main|pre|code|script|style)[^>]*>",
    re.IGNORECASE,
)


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip: int = 0   # nesting depth inside a _DROP_TAGS subtree
        self._pre: int = 0    # nesting depth inside <pre>

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _DROP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return

        if tag in _HEADING:
            self._parts.append("\n\n" + _HEADING[tag])
        elif tag == "pre":
            self._pre += 1
            self._parts.append("\n```\n")
        elif tag == "code" and not self._pre:
            self._parts.append("`")
        elif tag in ("li",):
            self._parts.append("\n- ")
        elif tag in ("tr",):
            self._parts.append("\n")
        elif tag in ("th", "td"):
            self._parts.append(" | ")
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "hr":
            self._parts.append("\n---\n")
        elif tag in _BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return

        if tag in _HEADING:
            self._parts.append("\n")
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._parts.append("\n```\n")
        elif tag == "code" and not self._pre:
            self._parts.append("`")
        elif tag in _BLOCK or tag in ("ul", "ol", "table"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._pre:
            self._parts.append(data)
        else:
            # Collapse whitespace in normal flow text.
            cleaned = " ".join(data.split())
            if cleaned:
                self._parts.append(cleaned + " ")

    def result(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    """Convert *html* to clean markdown-adjacent text.

    Returns the original string unchanged if parsing raises an exception.
    """
    ext = _Extractor()
    try:
        ext.feed(html)
        return ext.result()
    except Exception:
        return html


def looks_like_html(text: str) -> bool:
    """Return True when *text* is almost certainly HTML content.

    Requires either a known opening declaration or a high density of
    semantic HTML tags in the first 2 KB.
    """
    head = text.lstrip()[:20].lower()
    if head.startswith(("<!doctype", "<html", "<head", "<body")):
        return True
    sample = text[:2000]
    return len(_SEMANTIC_RE.findall(sample)) >= 5
