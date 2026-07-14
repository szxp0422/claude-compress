"""JavaScript minifier unit tests."""
import subprocess
import sys
from pathlib import Path
from shutil import which

import pytest

from minifier.minify import minify
from .conftest import JS_GOLDEN


NODE = which("node")

_skip_no_node = pytest.mark.skipif(NODE is None, reason="node not found")


def _run_node(source: str) -> str:
    """Execute JS source via node and capture stdout."""
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", source]
        if "import " in source or "export " in source
        else [NODE, "-e", source],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Node execution failed:\n{result.stderr}\n\nSource:\n{source[:500]}")
    return result.stdout


# ---------------------------------------------------------------------------
# Tests: comment stripping
# ---------------------------------------------------------------------------

class TestCommentStripping:
    def test_line_comment_removed(self):
        src = "const x = 1; // inline comment\nconst y = 2;\n"
        result = minify(src, "javascript", rename=False)
        assert "//" not in result.output

    def test_block_comment_removed(self):
        src = "/* block */\nconst x = 1;\n"
        result = minify(src, "javascript", rename=False)
        assert "/*" not in result.output
        assert "*/" not in result.output


# ---------------------------------------------------------------------------
# Tests: whitespace collapse (Group A)
# ---------------------------------------------------------------------------

class TestWhitespaceCollapse:
    def test_newlines_removed(self):
        src = "function f(x) {\n    return x + 1;\n}\n"
        result = minify(src, "javascript", rename=False)
        assert "\n" not in result.output

    def test_spaces_around_braces_removed(self):
        src = "function f(x) { return x; }\n"
        result = minify(src, "javascript", rename=False)
        # Should not have space before {
        assert "f(x){" in result.output or "f(x) {" in result.output

    def test_keyword_space_preserved(self):
        src = "function f(x) { return x; }\n"
        result = minify(src, "javascript", rename=False)
        # 'return' followed by identifier needs a space
        assert "return " in result.output or "returnx" not in result.output

    def test_increment_operator_safe(self):
        # x++ + y  must not become x+++y (ambiguous)
        src = "function f(x, y) { return x++ + y; }\n"
        result = minify(src, "javascript", rename=False)
        # Should contain ++ and + as separate tokens
        assert "+++" not in result.output

    def test_decrement_operator_safe(self):
        src = "function f(x) { return x--; }\n"
        result = minify(src, "javascript", rename=False)
        assert "--" in result.output
        # x-- should not have become part of a longer -- sequence
        assert "---" not in result.output


# ---------------------------------------------------------------------------
# Tests: identifier renaming
# ---------------------------------------------------------------------------

class TestIdentifierRenaming:
    def test_local_vars_renamed(self):
        src = "function f(longParamName) { const longVarName = longParamName * 2; return longVarName; }\n"
        result = minify(src, "javascript", rename=True)
        assert "longParamName" not in result.output
        assert "longVarName" not in result.output

    def test_exported_names_not_renamed(self):
        src = "function exported() { return 1; }\nmodule.exports = { exported };\n"
        result = minify(src, "javascript", rename=True)
        # The name 'exported' must survive
        assert "exported" in result.output

    def test_no_collision_nested_scopes(self):
        src = (
            "function outer(outerParam) {\n"
            "    const outerLocal = outerParam * 2;\n"
            "    function inner(innerParam) {\n"
            "        const innerLocal = innerParam + outerLocal;\n"
            "        return innerLocal;\n"
            "    }\n"
            "    return inner;\n"
            "}\n"
        )
        result = minify(src, "javascript", rename=True)
        assert len(result.name_map) > 0

    @pytest.mark.skipif(NODE is None, reason="node not found")
    def test_renamed_output_executes_correctly(self):
        src = (
            "function add(firstNum, secondNum) {\n"
            "    const sumResult = firstNum + secondNum;\n"
            "    return sumResult;\n"
            "}\n"
            "console.log(add(3, 4));\n"
        )
        result = minify(src, "javascript", rename=True)
        assert "firstNum" not in result.output
        output = _run_node(result.output)
        assert output.strip() == "7"


# ---------------------------------------------------------------------------
# Tests: output correctness via node
# ---------------------------------------------------------------------------

class TestOutputCorrectness:
    @pytest.mark.skipif(NODE is None, reason="node not found")
    @pytest.mark.parametrize("name", ["basic", "closures", "exports"])
    def test_golden_files_execute_identically(self, name: str):
        path = JS_GOLDEN / f"{name}_input.js"
        if not path.exists():
            pytest.skip(f"Golden file not found: {path}")
        source = path.read_text()
        original_out = _run_node(source)
        result = minify(source, "javascript", rename=True)
        minified_out = _run_node(result.output)
        assert original_out == minified_out, (
            f"Output mismatch for {name}:\n"
            f"Original:  {original_out!r}\n"
            f"Minified:  {minified_out!r}\n"
            f"Source:\n{result.output}"
        )

    @pytest.mark.parametrize("name", ["basic", "closures", "exports"])
    def test_golden_files_are_smaller(self, name: str):
        path = JS_GOLDEN / f"{name}_input.js"
        if not path.exists():
            pytest.skip(f"Golden file not found: {path}")
        source = path.read_text()
        result = minify(source, "javascript", rename=True)
        assert result.stats.output_bytes < result.stats.input_bytes, (
            "Output should be smaller than input"
        )
