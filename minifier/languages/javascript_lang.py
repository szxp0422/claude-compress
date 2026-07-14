"""JavaScript language module — Group A (delimiter-based).

Scope model
-----------
  - `var` declarations are function-scoped (hoisted).
  - `let` / `const` declarations are block-scoped.
  - Function parameters are scoped to the function body.
  - Arrow functions, function expressions, and function declarations all
    create a new function scope.
  - Class bodies create a scope for the class name but class methods
    are normal function scopes.

Renaming exclusions (per spec)
-------------------------------
  - Exported identifiers: anything following `export` keyword, or assigned
    to `module.exports` / `exports.*`.
  - `arguments` (magic inside non-arrow functions).
  - Names used as object keys in shorthand properties ({ foo } = …) where
    the key and value are the same — these are kept because the key is a
    string-referenced name.
  - Property identifiers after `.` — these are property names, not variable
    references; tree-sitter already uses `property_identifier` for them.
  - Dynamic bracket access: obj[varName] — we skip renaming variables whose
    names appear inside computed member expressions as string literals.

Known limitations (documented, not silently broken)
-----------------------------------------------------
  - `eval()` / `with` disables renaming for the surrounding function scope.
  - Template literal tag functions and Proxy traps use string-based dispatch;
    we err on the side of NOT renaming in those patterns.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .base import LanguageModule
from ..core.renamer import ScopeNode

GROUP = "A"
COMMENT_TYPES: Set[str] = {"comment"}

# Scope types in tree-sitter JS
_FUNCTION_SCOPE_TYPES = {
    "function_declaration",
    "function_expression",
    "arrow_function",
    "generator_function",
    "generator_function_declaration",
    "method_definition",
}
_BLOCK_SCOPE_TYPES = {
    "statement_block",
    "for_statement",
    "for_in_statement",
    "for_of_statement",
    "if_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "try_statement",
    "catch_clause",
    "with_statement",
}
_CLASS_TYPES = {"class_declaration", "class_expression"}

# These identifiers are never renamed regardless of scope
_BUILTIN_NAMES: Set[str] = {
    "arguments", "undefined", "null", "true", "false",
    "NaN", "Infinity", "globalThis", "window", "document",
    "module", "exports", "require", "__dirname", "__filename",
    "Promise", "Symbol", "Proxy", "Reflect",
    "console", "process",
}


def _text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


# ---------------------------------------------------------------------------
# Scope builder
# ---------------------------------------------------------------------------

class _Scope:
    """Mutable scope used during the tree walk; converted to ScopeNode at end."""

    def __init__(self, parent: "_Scope | None" = None, kind: str = "block"):
        self.parent = parent
        self.kind = kind  # 'function', 'block', 'module', 'class'
        self.children: List["_Scope"] = []
        # var_names: hoisted to nearest function scope
        self.var_names: Set[str] = set()
        # let_const_names: local to this block
        self.let_const_names: Set[str] = set()
        # param_names: function parameters
        self.param_names: Set[str] = set()
        # names that must NOT be renamed (exported, etc.)
        self.excluded: Set[str] = set()
        # unsafe = has eval/with, disable ALL renaming in this scope
        self.unsafe: bool = False

        # occurrences: name → [(start, end)] — regular identifier references
        self.occurrences: Dict[str, List[Tuple[int, int]]] = {}
        # shorthand_occurrences: name → [(start, end)] — shorthand props in object literals
        # These need expansion to "original: renamed" form, not simple substitution.
        self.shorthand_occurrences: Dict[str, List[Tuple[int, int]]] = {}

    def nearest_function(self) -> "_Scope":
        s = self
        while s.kind not in ("function", "module") and s.parent is not None:
            s = s.parent
        return s

    def declare_var(self, name: str, source_bytes: bytes = b""):
        self.nearest_function().var_names.add(name)

    def is_locally_defined(self, name: str) -> bool:
        """True if name is defined in THIS scope (not ancestors)."""
        return (
            name in self.param_names
            or name in self.let_const_names
            or name in self.var_names
        )

    def is_renameable(self, name: str) -> bool:
        if name in _BUILTIN_NAMES:
            return False
        if name.startswith("__"):
            return False
        return True

    def to_scope_node(self) -> ScopeNode:
        sn = ScopeNode()
        sn.unsafe = self.unsafe
        # Gather all locally-defined names and their occurrences
        local_names = self.param_names | self.let_const_names | self.var_names
        safe_names = {
            n for n in local_names
            if self.is_renameable(n) and n not in self.excluded
        }
        if not self.unsafe:
            for name in safe_names:
                if name in self.occurrences:
                    sn.locals[name] = list(self.occurrences[name])
                if name in self.shorthand_occurrences:
                    sn.shorthands[name] = list(self.shorthand_occurrences[name])
        for child in self.children:
            sn.children.append(child.to_scope_node())
        return sn


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

class _Walker:
    def __init__(self, source_bytes: bytes):
        self.sb = source_bytes
        self.module_scope = _Scope(kind="module")
        # Collect export names to exclude from renaming
        self._export_names: Set[str] = set()
        # First pass: discover all exports
        # (done in main walk by detecting export nodes)

    def walk(self, root) -> _Scope:
        self._visit(root, self.module_scope)
        return self.module_scope

    # ------------------------------------------------------------------

    def _visit(self, node, scope: _Scope):
        t = node.type

        # --- ES6 export ---
        if t == "export_statement":
            self._handle_export(node, scope)
            return

        # --- CommonJS: module.exports = {...} / exports.x = ... ---
        if t in ("assignment_expression", "augmented_assignment_expression"):
            self._maybe_handle_cjs_export(node, scope)
            # Don't return — still recurse to process RHS declarations/functions

        # --- Function scopes ---
        if t in _FUNCTION_SCOPE_TYPES:
            self._enter_function(node, scope)
            return

        # --- Class ---
        if t in _CLASS_TYPES:
            self._enter_class(node, scope)
            return

        # --- Variable declarations ---
        if t == "variable_declaration":
            kind_node = node.children[0] if node.children else None
            kind = _text(kind_node, self.sb) if kind_node else "var"
            self._handle_var_decl(node, scope, kind)
            return

        if t == "lexical_declaration":
            kind_node = node.children[0] if node.children else None
            kind = _text(kind_node, self.sb) if kind_node else "let"
            self._handle_var_decl(node, scope, kind)
            return

        # --- Catch clause (introduces a binding) ---
        if t == "catch_clause":
            self._handle_catch(node, scope)
            return

        # --- With / eval → mark unsafe ---
        if t == "with_statement":
            scope.nearest_function().unsafe = True
            for child in node.children:
                self._visit(child, scope)
            return

        if t == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node and fn_node.type == "identifier":
                if _text(fn_node, self.sb) in ("eval", "Function"):
                    scope.nearest_function().unsafe = True
            for child in node.children:
                self._visit(child, scope)
            return

        # --- Identifier reference (use) ---
        if t == "identifier":
            self._handle_identifier(node, scope)
            return

        # --- Shorthand property in object LITERAL: { foo } ---
        # The key and value share one node; renaming needs expansion to {foo: newname}.
        if t == "shorthand_property_identifier":
            parent = node.parent
            if parent is not None and parent.type == "object":
                # Object literal shorthand — must expand on rename
                self._handle_shorthand_property(node, scope)
            else:
                # Destructuring pattern shorthand — treat as binding/reference
                self._handle_identifier(node, scope)
            return

        # --- Property identifier after `.`: never rename ---
        if t == "property_identifier":
            return

        # --- Computed member: obj[name] — conservative: skip renaming `name` ---
        if t == "subscript_expression":
            obj_node = node.child_by_field_name("object")
            idx_node = node.child_by_field_name("index")
            if obj_node:
                self._visit(obj_node, scope)
            # Don't descend into index — to be safe with string-keyed dispatch
            return

        # Default: recurse
        for child in node.children:
            self._visit(child, scope)

    # ------------------------------------------------------------------

    def _enter_function(self, node, parent_scope: _Scope):
        fn_scope = _Scope(parent=parent_scope, kind="function")
        parent_scope.children.append(fn_scope)

        # Function name (for declarations / named expressions) lives in PARENT scope
        name_node = node.child_by_field_name("name")
        if name_node and name_node.type == "identifier":
            name = _text(name_node, self.sb)
            parent_scope.var_names.add(name)
            # Record the occurrence in the parent scope
            parent_scope.occurrences.setdefault(name, []).append(
                (name_node.start_byte, name_node.end_byte)
            )

        # Parameters
        params_node = node.child_by_field_name("parameters")
        if params_node:
            self._collect_params(params_node, fn_scope)

        # Body
        body_node = node.child_by_field_name("body")
        if body_node:
            self._visit(body_node, fn_scope)

    def _enter_class(self, node, parent_scope: _Scope):
        cls_scope = _Scope(parent=parent_scope, kind="class")
        cls_scope.unsafe = True  # class attribute names accessible externally
        parent_scope.children.append(cls_scope)

        # Class name in parent scope
        name_node = node.child_by_field_name("name")
        if name_node and name_node.type == "identifier":
            name = _text(name_node, self.sb)
            parent_scope.let_const_names.add(name)
            parent_scope.occurrences.setdefault(name, []).append(
                (name_node.start_byte, name_node.end_byte)
            )

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                self._visit(child, cls_scope)

    def _collect_params(self, params_node, fn_scope: _Scope):
        """Extract identifier names from a formal_parameters node."""
        for child in params_node.named_children:
            self._extract_binding(child, fn_scope, into="param")

    def _extract_binding(self, node, scope: _Scope, into: str = "let"):
        """Recursively extract binding names from destructuring patterns."""
        t = node.type
        if t == "identifier":
            name = _text(node, self.sb)
            if into == "param":
                scope.param_names.add(name)
            elif into == "var":
                scope.nearest_function().var_names.add(name)
            else:
                scope.let_const_names.add(name)
            scope.occurrences.setdefault(name, []).append(
                (node.start_byte, node.end_byte)
            )
        elif t in ("array_pattern", "object_pattern"):
            for child in node.named_children:
                self._extract_binding(child, scope, into)
        elif t == "pair_pattern":
            # { key: value } — only the value side is the binding
            val = node.child_by_field_name("value")
            if val:
                self._extract_binding(val, scope, into)
        elif t == "rest_pattern":
            inner = node.named_children[0] if node.named_children else None
            if inner:
                self._extract_binding(inner, scope, into)
        elif t in ("assignment_pattern", "required_parameter"):
            # default: param = default_value
            left = node.child_by_field_name("left") or (
                node.named_children[0] if node.named_children else None
            )
            if left:
                self._extract_binding(left, scope, into)
        elif t == "shorthand_property_identifier_pattern":
            name = _text(node, self.sb)
            if into == "param":
                scope.param_names.add(name)
            else:
                scope.let_const_names.add(name)
            scope.occurrences.setdefault(name, []).append(
                (node.start_byte, node.end_byte)
            )

    def _handle_var_decl(self, node, scope: _Scope, kind: str):
        """Process a variable_declaration or lexical_declaration node."""
        into = "var" if kind == "var" else "let"
        for child in node.named_children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    self._extract_binding(name_node, scope, into)
                # Visit the initialiser
                val_node = child.child_by_field_name("value")
                if val_node:
                    self._visit(val_node, scope)

    def _handle_catch(self, node, scope: _Scope):
        """catch (err) — err is a local of the catch block."""
        param_node = node.child_by_field_name("parameter")
        if param_node:
            self._extract_binding(param_node, scope, into="let")
        body_node = node.child_by_field_name("body")
        if body_node:
            self._visit(body_node, scope)

    def _handle_shorthand_property(self, node, scope: _Scope):
        """Handle { foo } in an object literal — records as expansion-needed occurrence."""
        name = _text(node, self.sb)
        defining = self._find_defining_scope(name, scope)
        if defining is not None:
            defining.shorthand_occurrences.setdefault(name, []).append(
                (node.start_byte, node.end_byte)
            )

    def _handle_identifier(self, node, scope: _Scope):
        name = _text(node, self.sb)
        # Find the scope that defines this name (walk up)
        defining = self._find_defining_scope(name, scope)
        if defining is not None:
            defining.occurrences.setdefault(name, []).append(
                (node.start_byte, node.end_byte)
            )
        # If not found in any scope, it's a global/free variable — skip

    def _find_defining_scope(self, name: str, scope: _Scope) -> "_Scope | None":
        s: "_Scope | None" = scope
        while s is not None:
            if s.is_locally_defined(name):
                return s
            s = s.parent
        return None

    def _handle_export(self, node, scope: _Scope):
        """Mark exported identifiers as excluded from renaming (ES6 exports)."""
        def mark(n):
            if n.type in ("identifier", "shorthand_property_identifier"):
                name = _text(n, self.sb)
                scope.excluded.add(name)
                self.module_scope.excluded.add(name)
            for child in n.children:
                mark(child)
        mark(node)
        for child in node.children:
            if child.type not in ("export", "default", "from", "source"):
                self._visit(child, scope)

    def _maybe_handle_cjs_export(self, node, scope: _Scope):
        """Detect CommonJS module.exports / exports.x patterns and exclude names."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return

        # module.exports = ...  or  exports.X = ...
        is_module_exports = (
            left.type == "member_expression"
            and _text(left.child_by_field_name("object") or left, self.sb) in ("module", "exports")
        )
        if not is_module_exports:
            return

        # Mark every identifier / shorthand key in the RHS as excluded
        def mark(n):
            if n.type in ("identifier", "shorthand_property_identifier",
                          "property_identifier"):
                name = _text(n, self.sb)
                scope.excluded.add(name)
                self.module_scope.excluded.add(name)
            for child in n.children:
                mark(child)

        mark(right)


# ---------------------------------------------------------------------------
# Public LanguageModule
# ---------------------------------------------------------------------------

class JavaScriptModule(LanguageModule):
    GROUP = "A"
    COMMENT_TYPES = COMMENT_TYPES

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        walker = _Walker(source_bytes)
        root_scope = walker.walk(tree.root_node)
        return root_scope.to_scope_node()
