"""Python language module — Group B (whitespace-significant).

Scope resolution strategy (as specified):
  - Use stdlib `ast` + `symtable` for authoritative scope membership.
  - Use tree-sitter only for comment stripping, indent reduction, and
    finding byte-level positions of identifiers.
  - The two are reconciled via (line, col) coordinates, which are identical
    between the ast module and tree-sitter.

Renaming exclusions (see spec):
  - Module-level names: skipped entirely (may be imported by other modules).
  - Class-body-level names: skipped (accessible as attributes via reflection).
  - Dunder names (__foo__ or __foo).
  - Names appearing in `global` or `nonlocal` statements (symtable handles
    this: they won't be marked is_local() in the function's symbol table).
  - Any name in a scope that contains exec() or eval() calls.
  - `self` and `cls` (convention; also used by serialization/reflection tools).
"""
from __future__ import annotations

import ast
import symtable
from typing import Dict, List, Set, Tuple

from .base import LanguageModule
from ..core.renamer import ScopeNode

GROUP = "B"
COMMENT_TYPES: Set[str] = {"comment"}

# Names that must never be renamed regardless of scope
_ALWAYS_EXCLUDE: Set[str] = {"self", "cls", "__all__"}


def _is_renameable(name: str) -> bool:
    if name in _ALWAYS_EXCLUDE:
        return False
    # Dunder names (any __xxx__ or __xxx)
    if name.startswith("__"):
        return False
    return True


def _has_exec_eval(fn_node: ast.AST) -> bool:
    """Return True if any exec() or eval() call appears in fn_node's subtree."""
    for child in ast.walk(fn_node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id in ("exec", "eval"):
                return True
            # Also catch builtins.exec / builtins.eval
            if isinstance(func, ast.Attribute) and func.attr in ("exec", "eval"):
                return True
    return False


def _char_to_byte_offset(source: str, lineno: int, col: int) -> int:
    """Convert 1-based lineno + 0-based char col to byte offset in UTF-8 source."""
    lines = source.split("\n")
    byte_off = sum(len(ln.encode("utf-8")) + 1 for ln in lines[: lineno - 1])
    byte_off += len(lines[lineno - 1][:col].encode("utf-8"))
    return byte_off


# ---------------------------------------------------------------------------
# Safe-local collection from symtable
# ---------------------------------------------------------------------------

def _safe_locals(sym_table: symtable.SymbolTable, fn_node: ast.AST) -> Set[str]:
    """Names that are truly local to sym_table and safe to rename."""
    if _has_exec_eval(fn_node):
        return set()
    result: Set[str] = set()
    for sym in sym_table.get_symbols():
        name = sym.get_name()
        if sym.is_local() and _is_renameable(name):
            result.add(name)
    return result


# ---------------------------------------------------------------------------
# Two-pass collector
# ---------------------------------------------------------------------------

class _Collector:
    """
    Walks the ast tree and symtable tree in parallel to build a ScopeNode tree.

    The symtable children are returned in source order by get_children(), which
    matches the order we encounter nested function/class defs in the ast walk.
    We consume them with an iterator so sibling scopes match up correctly.
    """

    def __init__(self, source: str):
        self.source = source
        self.ast_tree = ast.parse(source)
        self.sym_root = symtable.symtable(source, "<string>", "exec")

    def _nonlocal_positions(self, node: ast.Nonlocal):
        """
        Yield (name, start_byte, end_byte) for each name in a nonlocal statement.
        ast.Nonlocal gives (lineno, col_offset) of the `nonlocal` keyword.
        We scan the source line to find each name's character position.
        """
        lines = self.source.split("\n")
        line = lines[node.lineno - 1]
        # Skip past `nonlocal ` keyword
        search_start = node.col_offset + len("nonlocal ")
        for name in node.names:
            idx = line.find(name, search_start)
            if idx == -1:
                continue
            start = _char_to_byte_offset(self.source, node.lineno, idx)
            end = start + len(name.encode("utf-8"))
            yield name, start, end
            search_start = idx + len(name)

    def collect(self) -> ScopeNode:
        root_scope = ScopeNode()  # module scope — no renaming
        root_scope.unsafe = True  # module-level names: skip
        child_iter = iter(self.sym_root.get_children())
        self._walk_body(self.ast_tree.body, self.sym_root, root_scope, child_iter)
        return root_scope

    # ------------------------------------------------------------------

    def _walk_body(
        self,
        stmts: list,
        sym_table: symtable.SymbolTable,
        parent_scope: ScopeNode,
        child_sym_iter,
    ):
        """Process a list of statements in the context of sym_table."""
        for stmt in stmts:
            self._walk_stmt(stmt, sym_table, parent_scope, child_sym_iter)

    def _walk_stmt(self, node, sym_table, parent_scope, child_sym_iter):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._enter_function(node, sym_table, parent_scope, child_sym_iter)
        elif isinstance(node, ast.ClassDef):
            self._enter_class(node, sym_table, parent_scope, child_sym_iter)
        elif isinstance(node, (ast.If, ast.While, ast.For, ast.With,
                                ast.Try, ast.ExceptHandler)):
            # Compound statements — same scope, recurse into sub-bodies
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (list,)):
                    pass
                elif isinstance(child, ast.AST):
                    self._walk_stmt(child, sym_table, parent_scope, child_sym_iter)
        elif hasattr(node, "body"):
            # Generic compound with a body list
            body = getattr(node, "body", [])
            if isinstance(body, list):
                self._walk_body(body, sym_table, parent_scope, child_sym_iter)

    def _enter_function(self, fn_node, parent_sym, parent_scope, child_sym_iter):
        try:
            fn_sym = next(child_sym_iter)
        except StopIteration:
            return

        fn_scope = ScopeNode()
        parent_scope.children.append(fn_scope)

        safe = _safe_locals(fn_sym, fn_node)
        if not safe:
            fn_scope.unsafe = True

        if safe:
            # Collect occurrences inside this function's own body
            # Pass 1: find all identifier positions in this function's body
            self._collect_fn_occurrences(fn_node, fn_sym, safe, fn_scope)

        # Recurse into nested functions / classes within this function
        nested_iter = iter(fn_sym.get_children())
        for stmt in fn_node.body:
            self._walk_stmt(stmt, fn_sym, fn_scope, nested_iter)

    def _enter_class(self, cls_node, parent_sym, parent_scope, child_sym_iter):
        try:
            cls_sym = next(child_sym_iter)
        except StopIteration:
            return

        cls_scope = ScopeNode()
        cls_scope.unsafe = True  # class-level names: skip renaming
        parent_scope.children.append(cls_scope)

        # Recurse into methods / nested classes
        nested_iter = iter(cls_sym.get_children())
        for stmt in cls_node.body:
            self._walk_stmt(stmt, cls_sym, cls_scope, nested_iter)

    def _collect_fn_occurrences(
        self,
        fn_node,
        fn_sym: symtable.SymbolTable,
        safe_locals: Set[str],
        fn_scope: ScopeNode,
    ):
        """
        Walk fn_node's args + body, recording (start_byte, end_byte) for every
        occurrence of a safe_locals name.

        Critically: we DO descend into nested functions to find free-variable
        references to this scope's locals.  We track which names are shadowed
        by each nested scope so we stop renaming them there.
        """
        # Parameters
        args_obj = fn_node.args
        all_args = (
            args_obj.posonlyargs
            + args_obj.args
            + args_obj.kwonlyargs
            + ([args_obj.vararg] if args_obj.vararg else [])
            + ([args_obj.kwarg] if args_obj.kwarg else [])
        )
        for arg in all_args:
            if arg.arg in safe_locals:
                start = _char_to_byte_offset(self.source, arg.lineno, arg.col_offset)
                end = start + len(arg.arg.encode("utf-8"))
                fn_scope.locals.setdefault(arg.arg, []).append((start, end))

        # child_sym_iter tracks the ordered child symtables so we can look up
        # what each nested function/class/comprehension locally defines.
        child_sym_iter = iter(fn_sym.get_children())

        def _get_nested_locals(sym: symtable.SymbolTable) -> Set[str]:
            return {s.get_name() for s in sym.get_symbols() if s.is_local()}

        def _walk(node, shadowed: frozenset, nested_iter):
            """
            Walk node looking for occurrences of (safe_locals - shadowed).
            nested_iter: the child symbol-table iterator for the CURRENT scope level.
            """
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = node.name
                # The function NAME is a local of the OUTER scope
                if fn_name in safe_locals and fn_name not in shadowed:
                    kw = 10 if isinstance(node, ast.AsyncFunctionDef) else 4
                    start = _char_to_byte_offset(
                        self.source, node.lineno, node.col_offset + kw
                    )
                    end = start + len(fn_name.encode("utf-8"))
                    fn_scope.locals.setdefault(fn_name, []).append((start, end))

                # Consume nested sym, compute what it shadows
                try:
                    ns = next(nested_iter)
                    nested_locals = _get_nested_locals(ns)
                    sub_iter = iter(ns.get_children())
                except StopIteration:
                    nested_locals = set()
                    sub_iter = iter([])

                new_shadowed = shadowed | nested_locals
                # Recurse into nested body (to catch free-var references)
                for stmt in node.body:
                    _walk(stmt, new_shadowed, sub_iter)
                return

            if isinstance(node, ast.Lambda):
                try:
                    ns = next(nested_iter)
                    nested_locals = _get_nested_locals(ns)
                    sub_iter = iter(ns.get_children())
                except StopIteration:
                    nested_locals = set()
                    sub_iter = iter([])
                new_shadowed = shadowed | nested_locals
                _walk(node.body, new_shadowed, sub_iter)
                return

            if isinstance(node, ast.ClassDef):
                # Class bodies don't participate in closure scoping —
                # names inside are class attributes, not free variables.
                try:
                    ns = next(nested_iter)
                    # Still need to consume class's children so iterator stays in sync
                    # but don't look for outer-scope references inside class body.
                except StopIteration:
                    pass
                return

            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                  ast.GeneratorExp)):
                # Comprehensions are their own scope; consume the child sym table.
                try:
                    ns = next(nested_iter)
                    nested_locals = _get_nested_locals(ns)
                    sub_iter = iter(ns.get_children())
                except StopIteration:
                    nested_locals = set()
                    sub_iter = iter([])
                new_shadowed = shadowed | nested_locals
                # Do recurse to find free-var references inside comprehensions
                for child in ast.iter_child_nodes(node):
                    _walk(child, new_shadowed, sub_iter)
                return

            if isinstance(node, ast.Nonlocal):
                # `nonlocal x, y` declares names as free — but those names ARE in
                # the outer scope's safe_locals, so the declaration must be renamed.
                for name, start, end in self._nonlocal_positions(node):
                    if name in safe_locals and name not in shadowed:
                        fn_scope.locals.setdefault(name, []).append((start, end))
                return

            if isinstance(node, ast.Name):
                if node.id in safe_locals and node.id not in shadowed:
                    start = _char_to_byte_offset(
                        self.source, node.lineno, node.col_offset
                    )
                    end = start + len(node.id.encode("utf-8"))
                    fn_scope.locals.setdefault(node.id, []).append((start, end))
                return

            if isinstance(node, ast.arg):
                return  # Handled in parameter loop above

            for child in ast.iter_child_nodes(node):
                _walk(child, shadowed, nested_iter)

        for stmt in fn_node.body:
            _walk(stmt, frozenset(), child_sym_iter)


# ---------------------------------------------------------------------------
# Public LanguageModule
# ---------------------------------------------------------------------------

class PythonModule(LanguageModule):
    GROUP = "B"
    COMMENT_TYPES = COMMENT_TYPES

    def build_scope_tree(self, tree, source_bytes: bytes) -> ScopeNode:
        source = source_bytes.decode("utf-8")
        collector = _Collector(source)
        return collector.collect()
