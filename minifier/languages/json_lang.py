"""JSON language module — Group A.

JSON is pure data: keys are string literals (not identifiers), values are
literals.  The only transform applied is whitespace stripping via the
Group A reconstruct pass (`reconstruct_group_a`), which produces compact
single-line JSON.

No identifier renaming is performed.
"""
from __future__ import annotations

from .base import LanguageModule
from ..core.renamer import ScopeNode


class JsonModule(LanguageModule):
    GROUP = "A"
    COMMENT_TYPES: set = set()  # standard JSON has no comments

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        sn = ScopeNode()
        sn.unsafe = True  # nothing to rename in JSON
        return sn
