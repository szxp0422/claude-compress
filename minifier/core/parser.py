"""Layer 1: tree-sitter parsing — uniform AST API across languages."""
from __future__ import annotations

from typing import Dict
from tree_sitter import Language, Parser, Tree

import tree_sitter_python as _tspython
import tree_sitter_javascript as _tsjs
import tree_sitter_typescript as _tsts
import tree_sitter_c as _tsc
import tree_sitter_cpp as _tscpp
import tree_sitter_java as _tsjava
import tree_sitter_json as _tsjson
import tree_sitter_yaml as _tsyaml

_LANGUAGE_OBJECTS: Dict[str, Language] = {
    "python": Language(_tspython.language()),
    "javascript": Language(_tsjs.language()),
    "js": Language(_tsjs.language()),
    "typescript": Language(_tsts.language_typescript()),
    "ts": Language(_tsts.language_typescript()),
    "c": Language(_tsc.language()),
    "cpp": Language(_tscpp.language()),
    "c++": Language(_tscpp.language()),
    "java": Language(_tsjava.language()),
    "json": Language(_tsjson.language()),
    "yaml": Language(_tsyaml.language()),
    "yml": Language(_tsyaml.language()),
}


def get_language(lang: str) -> Language:
    key = lang.lower()
    if key not in _LANGUAGE_OBJECTS:
        raise ValueError(
            f"Unsupported language: {lang!r}. Supported: {sorted(_LANGUAGE_OBJECTS)}"
        )
    return _LANGUAGE_OBJECTS[key]


def parse(source_bytes: bytes, lang: str) -> Tree:
    language = get_language(lang)
    parser = Parser(language)
    return parser.parse(source_bytes)


def get_leaf_nodes(node):
    """Yield all terminal (leaf) nodes in document order."""
    if node.child_count == 0:
        yield node
    else:
        for child in node.children:
            yield from get_leaf_nodes(child)
