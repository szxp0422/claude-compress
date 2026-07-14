"""Top-level minification API.

Usage:
    from minifier.minify import minify, MinifyResult

    result = minify(source_code, lang="python")
    print(result.output)
    print(result.stats)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .core.parser import parse
from .core.transforms import (
    strip_comments,
    strip_trailing_whitespace,
    collapse_blank_lines,
    reduce_indent,
    reconstruct_group_a,
)
from .core.renamer import build_substitutions, apply_substitutions
from .languages.python_lang import PythonModule
from .languages.javascript_lang import JavaScriptModule
from .languages.typescript_lang import TypeScriptModule
from .languages.c_lang import CModule, CppModule
from .languages.java_lang import JavaModule
from .languages.json_lang import JsonModule
from .languages.yaml_lang import YamlModule

_MODULES = {
    "python": PythonModule(),
    "javascript": JavaScriptModule(),
    "js": JavaScriptModule(),
    "typescript": TypeScriptModule(),
    "ts": TypeScriptModule(),
    "c": CModule(),
    "cpp": CppModule(),
    "c++": CppModule(),
    "java": JavaModule(),
    "json": JsonModule(),
    "yaml": YamlModule(),
    "yml": YamlModule(),
}


@dataclass
class MinifyStats:
    input_bytes: int = 0
    output_bytes: int = 0
    identifiers_renamed: int = 0

    @property
    def byte_reduction_pct(self) -> float:
        if self.input_bytes == 0:
            return 0.0
        return 100.0 * (1 - self.output_bytes / self.input_bytes)


@dataclass
class MinifyResult:
    output: str
    name_map: Dict[str, str] = field(default_factory=dict)
    stats: MinifyStats = field(default_factory=MinifyStats)


def minify(
    source: str,
    lang: str,
    *,
    rename: bool = True,
    map_path: Optional[str | Path] = None,
) -> MinifyResult:
    """
    Minify source code for token efficiency.

    Args:
        source:    The source code as a string.
        lang:      Language identifier ('python', 'javascript', 'typescript',
                   'c', 'cpp', 'java', 'json', 'yaml', etc.).
        rename:    If True, apply identifier renaming (default True).
                   Set False when the code will be actively reasoned about.
        map_path:  Optional path to write the old→new name map JSON file.

    Returns:
        MinifyResult with .output (str), .name_map (dict), .stats.
    """
    lang_key = lang.lower()
    if lang_key not in _MODULES:
        raise ValueError(f"Unsupported language: {lang!r}")

    mod = _MODULES[lang_key]
    source_bytes = source.encode("utf-8")
    stats = MinifyStats(input_bytes=len(source_bytes))

    # --- Step 1: parse ---
    tree = parse(source_bytes, lang_key)

    # --- Step 2: strip comments ---
    clean_bytes = strip_comments(tree.root_node, source_bytes, mod.COMMENT_TYPES)

    # Re-parse after comment removal so transforms work on the cleaned source
    tree2 = parse(clean_bytes, lang_key)

    # --- Step 3: group-specific whitespace transforms ---
    if mod.GROUP == "A":
        preproc_types = getattr(mod, "PREPROC_LEAF_TYPES", frozenset())
        clean_bytes = reconstruct_group_a(tree2.root_node, clean_bytes, preproc_types)
    else:
        # Group B: collapse blank lines + optionally reduce indentation
        clean_str = clean_bytes.decode("utf-8")
        clean_str = strip_trailing_whitespace(clean_str)
        clean_str = collapse_blank_lines(clean_str)
        if getattr(mod, "REDUCE_INDENT", True):
            clean_str = reduce_indent(clean_str)
        clean_bytes = clean_str.encode("utf-8")

    # --- Step 4: identifier renaming (optional) ---
    name_map: Dict[str, str] = {}
    if rename:
        # Build scope tree from the ORIGINAL parse (before whitespace changes)
        # so byte offsets are still valid.  For Group A we reconstruct first
        # then rename, but we need to re-derive positions on the clean source.
        clean_tree = parse(clean_bytes, lang_key)
        scope_root = mod.build_scope_tree(clean_tree, clean_bytes)
        subs, name_map = build_substitutions(scope_root)
        if subs:
            clean_bytes = apply_substitutions(clean_bytes, subs)
            stats.identifiers_renamed = len(name_map)

    # --- Step 5: final trim ---
    output = clean_bytes.decode("utf-8").strip()
    if mod.GROUP == "B":
        output = output + "\n"  # preserve trailing newline for Python/YAML

    stats.output_bytes = len(output.encode("utf-8"))

    if map_path and name_map:
        Path(map_path).write_text(json.dumps(name_map, indent=2))

    return MinifyResult(output=output, name_map=name_map, stats=stats)
