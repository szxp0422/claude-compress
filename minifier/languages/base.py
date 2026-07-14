"""Abstract interface every language module must satisfy."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

from ..core.renamer import ScopeNode


class LanguageModule(ABC):
    # 'A' = delimiter-based (free whitespace strip)
    # 'B' = whitespace-significant (indent-preserve)
    GROUP: str

    # tree-sitter node types that represent comments
    COMMENT_TYPES: Set[str]

    # Whether to reduce indentation (Group B only); set False for YAML
    REDUCE_INDENT: bool = True

    @abstractmethod
    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        """
        Walk the parsed tree and return a ScopeNode tree describing which
        identifier occurrences are safe to rename and how they nest.
        """
