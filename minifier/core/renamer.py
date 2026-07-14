"""Layer 2: generic identifier renaming engine.

The language module provides a ScopeNode tree; this module assigns short names
and applies substitutions to the source bytes.

Name pool: single lowercase letters (a-z), then two-letter pairs (aa-zz).
Each of these is a single BPE token in nearly every tokenizer vocabulary —
no per-candidate token check needed.

Collision rule (from spec): never reuse a short name across two variables in
overlapping scopes (parent ↔ child).  Sibling scopes may reuse names.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class ScopeNode:
    """One lexical scope with renameable local variables."""
    # name → list of (start_byte, end_byte) occurrences in source
    locals: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    children: List["ScopeNode"] = field(default_factory=list)
    # If True, skip renaming everything in this scope (exec/eval detected, etc.)
    unsafe: bool = False
    # Shorthand property occurrences that need expansion: name → [(start, end)]
    # These are NOT in `locals`; they get a different substitution: "key: new_name"
    shorthands: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


def _name_pool():
    """Yield short identifier candidates: a … z, aa … zz (702 total)."""
    for c in string.ascii_lowercase:
        yield c
    for c1 in string.ascii_lowercase:
        for c2 in string.ascii_lowercase:
            yield c1 + c2


def _assign(scope: ScopeNode, forbidden: Set[str]) -> Dict[str, str]:
    """
    Recursively assign short names to scope and all descendants.
    Returns a flat {(scope_id, original_name): short_name} mapping
    encoded as {scope_id_str: {orig: short}} — but we return a flat
    substitution list instead.

    Internal use: returns assigned = {orig_name: short_name} for THIS scope.
    Side-effect: populates scope._name_map (attached at runtime).
    """
    if scope.unsafe or not scope.locals:
        scope._name_map = {}  # type: ignore[attr-defined]
        child_forbidden = set(forbidden)
        for child in scope.children:
            _assign(child, child_forbidden)
        return {}

    pool = (n for n in _name_pool() if n not in forbidden)
    assigned: Dict[str, str] = {}
    for orig in sorted(scope.locals):  # deterministic order
        short = next(pool)
        assigned[orig] = short

    scope._name_map = assigned  # type: ignore[attr-defined]

    # Children inherit this scope's names as forbidden (parent ↔ child overlap)
    child_forbidden = forbidden | set(assigned.values())
    for child in scope.children:
        _assign(child, child_forbidden)

    return assigned


def build_substitutions(
    scope: ScopeNode,
) -> Tuple[List[Tuple[int, int, str]], Dict[str, str]]:
    """
    Assign short names and collect all (start_byte, end_byte, new_text) triples.
    Shorthand property occurrences are expanded to "original_key: new_name" form.
    Also returns the flat old→new map for the .map.json file.
    """
    _assign(scope, forbidden=set())

    subs: List[Tuple[int, int, str]] = []
    flat_map: Dict[str, str] = {}

    def _collect(s: ScopeNode):
        name_map = getattr(s, "_name_map", {})
        for orig, short in name_map.items():
            flat_map[orig] = short
            for start, end in s.locals.get(orig, []):
                subs.append((start, end, short))
            # Shorthand properties: expand `foo` → `foo: short`
            for start, end in s.shorthands.get(orig, []):
                subs.append((start, end, f"{orig}: {short}"))
        for child in s.children:
            _collect(child)

    _collect(scope)
    return subs, flat_map


def apply_substitutions(
    source_bytes: bytes,
    subs: List[Tuple[int, int, str]],
) -> bytes:
    """Apply substitutions in reverse byte order to preserve offsets."""
    result = bytearray(source_bytes)
    for start, end, new_name in sorted(subs, key=lambda x: x[0], reverse=True):
        result[start:end] = new_name.encode("utf-8")
    return bytes(result)
