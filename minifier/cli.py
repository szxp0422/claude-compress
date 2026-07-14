"""CLI entry point: python -m minifier <file> [options]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .minify import minify


def main():
    parser = argparse.ArgumentParser(
        description="Token-optimised source code minifier (Claude tokenizer target)."
    )
    parser.add_argument("file", help="Source file to minify")
    parser.add_argument(
        "--lang",
        help="Language override (python, javascript). Auto-detected from extension if omitted.",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help="Skip identifier renaming (preserves readable names).",
    )
    parser.add_argument(
        "--map",
        metavar="PATH",
        help="Write old→new name map JSON to this path.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write minified output to this path (default: stdout).",
    )
    args = parser.parse_args()

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")

    lang = args.lang
    if not lang:
        ext = path.suffix.lower().lstrip(".")
        _EXT_MAP = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "c": "c",
            "h": "c",
            "cpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "hpp": "cpp",
            "java": "java",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
        }
        lang = _EXT_MAP.get(ext)
        if not lang:
            print(f"Cannot infer language from extension {ext!r}. Use --lang.", file=sys.stderr)
            sys.exit(1)

    result = minify(source, lang=lang, rename=not args.no_rename, map_path=args.map)

    if args.out:
        Path(args.out).write_text(result.output, encoding="utf-8")
        print(
            f"Minified: {result.stats.input_bytes} → {result.stats.output_bytes} bytes "
            f"({result.stats.byte_reduction_pct:.1f}% reduction), "
            f"{result.stats.identifiers_renamed} identifiers renamed.",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(result.output)


if __name__ == "__main__":
    main()
