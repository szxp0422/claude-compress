"""Layer 2 shared transforms: comment stripping, whitespace normalization."""
from __future__ import annotations

import re
from math import gcd
from functools import reduce
from typing import Set

from .parser import get_leaf_nodes


# ---------------------------------------------------------------------------
# Shared: comment stripping
# ---------------------------------------------------------------------------

def strip_comments(root, source_bytes: bytes, comment_types: Set[str]) -> bytes:
    """Remove all nodes whose type is in comment_types, return cleaned source."""
    spans = []

    def collect(node):
        if node.type in comment_types:
            spans.append((node.start_byte, node.end_byte))
        else:
            for child in node.children:
                collect(child)

    collect(root)
    if not spans:
        return source_bytes

    result = bytearray(source_bytes)
    for start, end in sorted(spans, reverse=True):
        del result[start:end]
    return bytes(result)


def strip_trailing_whitespace(source: str) -> str:
    """Remove trailing whitespace from every line (after comment removal leaves gaps)."""
    return '\n'.join(line.rstrip() for line in source.split('\n'))


# ---------------------------------------------------------------------------
# Group B (whitespace-significant): blank-line collapse + indent reduction
# ---------------------------------------------------------------------------

def collapse_blank_lines(source: str) -> str:
    """Reduce runs of 3+ newlines to exactly 2 (one blank line)."""
    return re.sub(r'\n{3,}', '\n\n', source)


def _indent_unit(source: str) -> int:
    """Return the base indentation unit (GCD of all non-zero indent widths)."""
    widths = set()
    for line in source.split('\n'):
        if line.strip():
            w = len(line) - len(line.lstrip(' '))
            if w > 0:
                widths.add(w)
    if not widths:
        return 1
    return reduce(gcd, sorted(widths))


def reduce_indent(source: str) -> str:
    """Replace each N-space indent unit with 1 space, minimising token cost."""
    unit = _indent_unit(source)
    if unit <= 1:
        return source
    lines = []
    for line in source.split('\n'):
        if line.strip():
            content = line.lstrip(' ')
            level = (len(line) - len(content)) // unit
            lines.append(' ' * level + content)
        else:
            lines.append(line)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Group A (delimiter-based): reconstruct source from token leaves
# ---------------------------------------------------------------------------

_WORD_CHARS = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$')


def _needs_space(left: str, right: str) -> bool:
    """True if a space must be inserted between adjacent tokens to preserve meaning."""
    if not left or not right:
        return False
    lc, rc = left[-1], right[0]
    # Both word chars → merging would create a new identifier/keyword
    if lc in _WORD_CHARS and rc in _WORD_CHARS:
        return True
    # Operator pairs that must NOT be merged (would create a different/longer op)
    if lc == '+' and rc == '+':   # ++ (increment)
        return True
    if lc == '-' and rc == '-':   # -- (decrement)
        return True
    if lc == '/' and rc in ('*', '/'):  # /* block comment or // line comment
        return True
    if lc == '>' and rc == '>':   # >> (right-shift or generic close)
        return True
    if lc == '<' and rc == '<':   # << (left-shift)
        return True
    if lc == '?' and rc == '.':   # ?. (optional chain)
        return True
    if lc == '?' and rc == '?':   # ?? (nullish coalescing)
        return True
    if lc == '*' and rc == '*':   # ** (exponentiation)
        return True
    return False


def _has_preproc_ancestor(node) -> bool:
    """Return True if node is a descendant of a preproc_* named node."""
    n = node.parent
    while n is not None:
        if n.type.startswith("preproc_"):
            return True
        n = n.parent
    return False


def reconstruct_group_a(
    root,
    source_bytes: bytes,
    newline_before_types: frozenset = frozenset(),
) -> bytes:
    """
    Rebuild the source for a Group-A language by concatenating leaf tokens
    with the minimal whitespace required to keep meaning intact.

    newline_before_types: leaf node types that must begin on their own line
    (used for C/C++ preprocessor directive keywords like '#include', '#define').
    When non-empty, newlines are also inserted at the boundary EXITING a
    preproc node back into regular code (i.e., after the macro body).
    """
    parts: list[str] = []
    prev_text: str | None = None
    prev_in_preproc: bool = False

    for node in get_leaf_nodes(root):
        raw = source_bytes[node.start_byte:node.end_byte]
        text = raw.decode('utf-8')

        # Skip pure-whitespace leaf nodes (newlines, spaces between tokens)
        if not text.strip():
            continue

        curr_in_preproc = _has_preproc_ancestor(node) if newline_before_types else False

        if prev_text is not None:
            if newline_before_types and node.type in newline_before_types:
                # Start of a preprocessor directive: must be on its own line
                parts.append('\n')
            elif newline_before_types and prev_in_preproc and not curr_in_preproc:
                # Leaving a preprocessor node: the directive must end here
                parts.append('\n')
            elif _needs_space(prev_text, text):
                parts.append(' ')
        parts.append(text)
        prev_text = text
        prev_in_preproc = curr_in_preproc

    return ''.join(parts).encode('utf-8')
