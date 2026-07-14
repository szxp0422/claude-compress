"""TypeScript language module — Group A (thin JS superset).

The JS walker naturally handles TypeScript because:
  - Type annotations use `type_annotation`, `predefined_type`, and `type_identifier`
    nodes — none of which are `identifier`, so they are silently skipped.
  - Interface/type-alias bodies contain `property_identifier` and `method_signature`
    nodes — again, not `identifier`, so silently skipped.
  - TS parameter syntax (`required_parameter` with field "pattern") is already
    handled by `_extract_binding` via the fallback `node.named_children[0]`.

The only thing that differs from the JS module is the grammar (Language object).
"""
from __future__ import annotations

import tree_sitter_typescript as _tsts
from tree_sitter import Language

from .javascript_lang import JavaScriptModule

_TS_LANGUAGE = Language(_tsts.language_typescript())


class TypeScriptModule(JavaScriptModule):
    """TypeScript: JS walker + TypeScript grammar. No overrides needed."""
