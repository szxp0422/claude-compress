"""YAML language module — Group B (whitespace-significant), no indent reduction.

YAML indentation encodes document structure; reducing it would change the
meaning of the document.  The only safe transforms are:
  1. Strip `# comments` (tree-sitter YAML parses them as `comment` nodes).
  2. Collapse consecutive blank lines to at most one.

No identifier renaming is performed (YAML keys are data, not code identifiers).
"""
from __future__ import annotations

from .base import LanguageModule
from ..core.renamer import ScopeNode


class YamlModule(LanguageModule):
    GROUP = "B"
    COMMENT_TYPES = {"comment"}
    REDUCE_INDENT = False  # YAML indentation is structural — never reduce

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        sn = ScopeNode()
        sn.unsafe = True  # nothing to rename in YAML
        return sn
