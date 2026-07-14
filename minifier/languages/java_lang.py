"""Java language module — Group A.

Renaming strategy:
  - Class member names (field names, method names, class names) are NOT
    renamed — they are accessible externally and via reflection.
  - Method/constructor parameters and local variables ARE renamed.
  - Field access identifiers (after `.`) are skipped using a prev-sibling check.
  - Method names in invocations (also after `.`) are similarly skipped.
  - Lambda parameters are renamed (they are function-local).
  - Exception catch parameters are renamed.
  - Enhanced-for loop variables are renamed.

Comment node types: `line_comment` and `block_comment` (not `comment`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .base import LanguageModule
from ..core.renamer import ScopeNode


def _text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _is_member_name(node) -> bool:
    """True if this identifier follows a '.' (i.e., is a field/method name)."""
    prev = node.prev_sibling
    return prev is not None and prev.type == "."


# ---------------------------------------------------------------------------
# Per-method scope (flat)
# ---------------------------------------------------------------------------

class _MethodScope:
    def __init__(self):
        self.local_names: Set[str] = set()
        self.occurrences: Dict[str, List[Tuple[int, int]]] = {}

    def declare(self, name: str, start: int, end: int):
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
    def __init__(self, source_bytes: bytes):
        self.sb = source_bytes

    def walk(self, root) -> ScopeNode:
        file_sn = ScopeNode()
        file_sn.unsafe = True  # file (program) scope is unsafe
        self._visit_toplevel(root, file_sn)
        return file_sn

    # ------------------------------------------------------------------

    def _visit_toplevel(self, node, parent_sn: ScopeNode):
        for child in node.children:
            t = child.type
            if t in ("class_declaration", "interface_declaration",
                     "annotation_type_declaration", "enum_declaration",
                     "record_declaration"):
                cls_sn = ScopeNode()
                cls_sn.unsafe = True
                parent_sn.children.append(cls_sn)
                body = child.child_by_field_name("body")
                if body:
                    self._visit_class_body(body, cls_sn)

    def _visit_class_body(self, body, cls_sn: ScopeNode):
        for child in body.children:
            t = child.type
            if t in ("method_declaration", "constructor_declaration",
                     "compact_constructor_declaration"):
                method_sn = self._walk_method(child)
                if method_sn is not None:
                    cls_sn.children.append(method_sn)
            elif t in ("class_declaration", "interface_declaration",
                       "enum_declaration", "record_declaration"):
                # Inner class: recurse
                nested_sn = ScopeNode()
                nested_sn.unsafe = True
                cls_sn.children.append(nested_sn)
                nested_body = child.child_by_field_name("body")
                if nested_body:
                    self._visit_class_body(nested_body, nested_sn)

    # ------------------------------------------------------------------

    def _walk_method(self, method_node) -> Optional[ScopeNode]:
        scope = _MethodScope()

        # Parameters
        params = method_node.child_by_field_name("parameters")
        if params:
            for param in params.named_children:
                if param.type == "formal_parameter":
                    id_node = param.child_by_field_name("name")
                    if id_node and id_node.type == "identifier":
                        scope.declare(
                            _text(id_node, self.sb),
                            id_node.start_byte, id_node.end_byte,
                        )
                elif param.type == "spread_parameter":
                    # varargs: `String... args`
                    for c in param.named_children:
                        if c.type == "identifier":
                            scope.declare(
                                _text(c, self.sb), c.start_byte, c.end_byte
                            )

        # Body
        body = method_node.child_by_field_name("body")
        if body:
            self._walk_body(body, scope)

        return scope.to_scope_node()

    # ------------------------------------------------------------------

    def _walk_body(self, node, scope: _MethodScope):
        t = node.type

        # Local variable declaration: `int r = x + y;`
        if t == "local_variable_declaration":
            self._handle_local_var(node, scope)
            return

        # Enhanced-for: `for (String item : list)`
        if t == "enhanced_for_statement":
            self._handle_enhanced_for(node, scope)
            return

        # Catch parameter: `catch (Exception ex)`
        if t == "catch_formal_parameter":
            id_node = node.child_by_field_name("name")
            if id_node and id_node.type == "identifier":
                scope.declare(
                    _text(id_node, self.sb), id_node.start_byte, id_node.end_byte
                )
            return

        # Lambda: treat as a nested function with its own (safe) locals
        if t == "lambda_expression":
            self._handle_lambda(node, scope)
            return

        # Identifier reference
        if t == "identifier":
            if not _is_member_name(node):
                scope.record(
                    _text(node, self.sb), node.start_byte, node.end_byte
                )
            return

        # Type identifiers (class names used as types): skip
        if t in ("type_identifier", "void_type", "integral_type",
                 "floating_point_type", "boolean_type"):
            return

        # Default: recurse
        for child in node.children:
            self._walk_body(child, scope)

    def _handle_local_var(self, lvd_node, scope: _MethodScope):
        for child in lvd_node.named_children:
            if child.type == "variable_declarator":
                id_node = child.child_by_field_name("name")
                if id_node and id_node.type == "identifier":
                    scope.declare(
                        _text(id_node, self.sb),
                        id_node.start_byte, id_node.end_byte,
                    )
                val = child.child_by_field_name("value")
                if val:
                    self._walk_body(val, scope)

    def _handle_enhanced_for(self, node, scope: _MethodScope):
        # Declare the loop variable
        name_node = node.child_by_field_name("name")
        if name_node and name_node.type == "identifier":
            scope.declare(
                _text(name_node, self.sb),
                name_node.start_byte, name_node.end_byte,
            )
        # Visit the iterable expression
        val_node = node.child_by_field_name("value")
        if val_node:
            self._walk_body(val_node, scope)
        # Visit the body
        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, scope)

    def _handle_lambda(self, node, scope: _MethodScope):
        # Lambda parameters are new locals visible in the lambda body.
        # We fold them into the enclosing method scope (flat model is safe
        # since lambda locals cannot shadow outer method locals at the
        # byte-position level — the lambda body's declarations stand alone).
        params_node = node.child_by_field_name("parameters")
        if params_node is not None:
            pt = params_node.type
            if pt == "identifier":
                # Single un-typed parameter: `x -> x + 1`
                scope.declare(
                    _text(params_node, self.sb),
                    params_node.start_byte, params_node.end_byte,
                )
            elif pt == "formal_parameters":
                for param in params_node.named_children:
                    if param.type in ("formal_parameter", "inferred_parameters"):
                        for c in param.children:
                            if c.type == "identifier":
                                scope.declare(
                                    _text(c, self.sb), c.start_byte, c.end_byte
                                )
            elif pt == "inferred_parameters":
                # (x, y) -> x + y  (no types)
                for c in params_node.named_children:
                    if c.type == "identifier":
                        scope.declare(
                            _text(c, self.sb), c.start_byte, c.end_byte
                        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, scope)


# ---------------------------------------------------------------------------
# Public LanguageModule
# ---------------------------------------------------------------------------

class JavaModule(LanguageModule):
    GROUP = "A"
    COMMENT_TYPES = {"line_comment", "block_comment"}

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        return _Walker(source_bytes).walk(tree.root_node)
