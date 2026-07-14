"""C and C++ language modules — Group A.

Renaming strategy (per spec):
  - File-scope names are NOT renamed (external linkage by default).
  - Function parameters and local variables ARE renamed.
  - Preprocessor nodes (preproc_*) are skipped entirely — macro bodies are
    raw-text blobs and cannot be safely renamed.
  - C++ class/struct/union bodies are marked unsafe (member names are
    accessible externally); methods inside are still renamed at local scope.
  - `field_identifier` and `type_identifier` nodes are never renamed
    (they are struct field names and type names, not variable bindings).

Flat function-scope model:
  All declarations within a function (regardless of nesting depth) are
  collected into one flat scope.  Block-scope shadowing (`int x` in two
  different `{ }` blocks) is safe to rename this way because C's positional
  block scoping is preserved: if both `x` declarations are renamed to `a`,
  the inner `int a` still shadows the outer `int a` in the output.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .base import LanguageModule
from ..core.renamer import ScopeNode

# Names that must never be renamed regardless of context
_ALWAYS_EXCLUDE: Set[str] = {"main", "argc", "argv"}

# Preprocessor node-type prefixes / exact types to skip entirely
_PREPROC_PREFIX = "preproc_"
_PREPROC_EXACT = {
    "preproc_def", "preproc_function_def", "preproc_ifdef", "preproc_ifndef",
    "preproc_else", "preproc_elif", "preproc_if", "preproc_include",
    "preproc_call", "preproc_params", "preproc_arg",
    "#include", "#define", "#ifdef", "#ifndef", "#if", "#else", "#endif", "#elif",
}

# Nodes that are types/field names — never rename
_SKIP_ID_TYPES = {"type_identifier", "field_identifier", "namespace_identifier"}

# C++ class-like bodies to treat as unsafe scopes
_CPP_CLASS_TYPES = {"class_specifier", "struct_specifier", "union_specifier"}


def _text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _is_preproc(node_type: str) -> bool:
    return node_type.startswith(_PREPROC_PREFIX) or node_type in _PREPROC_EXACT


def _unwrap_declarator(node) -> Optional[object]:
    """Return the innermost identifier node buried in a C declarator chain.

    Handles: pointer_declarator, array_declarator, function_declarator,
    reference_declarator (C++), parenthesized_declarator.
    """
    if node is None:
        return None
    t = node.type
    if t == "identifier":
        return node
    if t in (
        "pointer_declarator", "array_declarator", "function_declarator",
        "parenthesized_declarator", "reference_declarator",
        "abstract_pointer_declarator", "abstract_array_declarator",
        "abstract_reference_declarator",
    ):
        inner = node.child_by_field_name("declarator")
        if inner:
            return _unwrap_declarator(inner)
    return None


# ---------------------------------------------------------------------------
# Flat per-function scope
# ---------------------------------------------------------------------------

class _FuncScope:
    """Collects all locals in a function into one flat namespace."""

    def __init__(self):
        self.local_names: Set[str] = set()
        self.occurrences: Dict[str, List[Tuple[int, int]]] = {}

    def declare(self, name: str, start: int, end: int):
        if name in _ALWAYS_EXCLUDE:
            return
        self.local_names.add(name)
        self.occurrences.setdefault(name, []).append((start, end))

    def record(self, name: str, start: int, end: int):
        if name in self.local_names:
            self.occurrences.setdefault(name, []).append((start, end))

    def to_scope_node(self) -> ScopeNode:
        sn = ScopeNode()
        for name in self.local_names:
            if name in self.occurrences:
                sn.locals[name] = self.occurrences[name][:]
        return sn


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

class _Walker:
    def __init__(self, source_bytes: bytes, is_cpp: bool):
        self.sb = source_bytes
        self.is_cpp = is_cpp

    def walk(self, root) -> ScopeNode:
        file_sn = ScopeNode()
        file_sn.unsafe = True  # file-scope: never rename
        self._visit_toplevel(root, file_sn)
        return file_sn

    # ------------------------------------------------------------------
    # Top-level traversal
    # ------------------------------------------------------------------

    def _visit_toplevel(self, node, file_sn: ScopeNode):
        for child in node.children:
            t = child.type
            if _is_preproc(t):
                continue
            if t == "function_definition":
                sn = self._walk_function(child)
                if sn is not None:
                    file_sn.children.append(sn)
            elif self.is_cpp and t in _CPP_CLASS_TYPES:
                body = child.child_by_field_name("body")
                if body:
                    cls_sn = ScopeNode()
                    cls_sn.unsafe = True
                    file_sn.children.append(cls_sn)
                    self._visit_class_body(body, cls_sn)
            elif self.is_cpp and t == "namespace_definition":
                body = child.child_by_field_name("body")
                if body:
                    self._visit_toplevel(body, file_sn)
            elif self.is_cpp and t == "template_declaration":
                # Recurse to find the inner function/class
                self._visit_toplevel(child, file_sn)
            # All other top-level declarations (global vars, typedefs): skip

    def _visit_class_body(self, body, cls_sn: ScopeNode):
        for child in body.children:
            if child.type == "function_definition":
                sn = self._walk_function(child)
                if sn is not None:
                    cls_sn.children.append(sn)
            elif self.is_cpp and child.type in _CPP_CLASS_TYPES:
                # Nested class: recurse
                nested_body = child.child_by_field_name("body")
                if nested_body:
                    nested_sn = ScopeNode()
                    nested_sn.unsafe = True
                    cls_sn.children.append(nested_sn)
                    self._visit_class_body(nested_body, nested_sn)

    # ------------------------------------------------------------------
    # Function scope
    # ------------------------------------------------------------------

    def _walk_function(self, fn_node) -> Optional[ScopeNode]:
        scope = _FuncScope()
        decl = fn_node.child_by_field_name("declarator")
        if decl:
            self._collect_params(decl, scope)
        body = fn_node.child_by_field_name("body")
        if body:
            self._walk_stmt(body, scope)
        return scope.to_scope_node()

    def _collect_params(self, decl_node, scope: _FuncScope):
        """Extract parameter names from a function_declarator."""
        params = decl_node.child_by_field_name("parameters")
        if not params:
            return
        for param in params.named_children:
            if param.type == "parameter_declaration":
                pdecl = param.child_by_field_name("declarator")
                id_node = _unwrap_declarator(pdecl)
                if id_node and id_node.type == "identifier":
                    name = _text(id_node, self.sb)
                    scope.declare(name, id_node.start_byte, id_node.end_byte)
            elif param.type == "variadic_parameter":
                pass  # `...` — no name

    # ------------------------------------------------------------------
    # Statement / expression walker (flat into function scope)
    # ------------------------------------------------------------------

    def _walk_stmt(self, node, scope: _FuncScope):
        t = node.type

        # Skip preprocessor entirely
        if _is_preproc(t):
            return

        # Type names and field names are never variables
        if t in _SKIP_ID_TYPES:
            return

        # C++ lambda: skip (complex capture semantics)
        if self.is_cpp and t == "lambda_expression":
            return

        # C++ nested class/struct inside function (rare): skip
        if self.is_cpp and t in _CPP_CLASS_TYPES:
            return

        # Local variable declaration
        if t == "declaration":
            self._handle_declaration(node, scope)
            return

        # Identifier reference
        if t == "identifier":
            scope.record(_text(node, self.sb), node.start_byte, node.end_byte)
            return

        # Recurse into all children
        for child in node.children:
            self._walk_stmt(child, scope)

    def _handle_declaration(self, decl_node, scope: _FuncScope):
        """Process a local variable declaration, adding names and visiting values."""
        for child in decl_node.named_children:
            ct = child.type
            if ct == "init_declarator":
                d = child.child_by_field_name("declarator")
                id_node = _unwrap_declarator(d)
                if id_node and id_node.type == "identifier":
                    scope.declare(
                        _text(id_node, self.sb),
                        id_node.start_byte, id_node.end_byte,
                    )
                val = child.child_by_field_name("value")
                if val:
                    self._walk_stmt(val, scope)
            elif ct == "identifier":
                # `int x;` — bare identifier without initializer
                scope.declare(
                    _text(child, self.sb),
                    child.start_byte, child.end_byte,
                )
            elif ct not in (
                "primitive_type", "type_identifier", "sized_type_specifier",
                "storage_class_specifier", "type_qualifier", "struct_specifier",
                "union_specifier", "enum_specifier", "const", "volatile",
                "type_specifier",
            ):
                # Array/pointer declarator without initializer: `int arr[10];`
                id_node = _unwrap_declarator(child)
                if id_node and id_node.type == "identifier":
                    scope.declare(
                        _text(id_node, self.sb),
                        id_node.start_byte, id_node.end_byte,
                    )


# ---------------------------------------------------------------------------
# Public LanguageModule classes
# ---------------------------------------------------------------------------

# Preprocessor keyword leaf types that must begin on their own line.
_PREPROC_LEAF_TYPES = frozenset({
    "#include", "#define", "#ifdef", "#ifndef", "#if", "#elif", "#else",
    "#endif", "#undef", "#pragma", "#error", "#warning", "#line",
})


class CModule(LanguageModule):
    GROUP = "A"
    COMMENT_TYPES = {"comment"}
    PREPROC_LEAF_TYPES = _PREPROC_LEAF_TYPES

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        return _Walker(source_bytes, is_cpp=False).walk(tree.root_node)


class CppModule(LanguageModule):
    GROUP = "A"
    COMMENT_TYPES = {"comment"}
    PREPROC_LEAF_TYPES = _PREPROC_LEAF_TYPES

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        return _Walker(source_bytes, is_cpp=True).walk(tree.root_node)
