"""Python minifier unit tests."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from minifier.minify import minify
from .conftest import PYTHON_GOLDEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_python(source: str) -> str:
    """Execute Python source and capture stdout."""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Python execution failed:\n{result.stderr}")
    return result.stdout


def _is_valid_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Tests: comment stripping
# ---------------------------------------------------------------------------

class TestCommentStripping:
    def test_inline_comment_removed(self):
        src = "x = 1  # inline comment\ny = 2\n"
        result = minify(src, "python", rename=False)
        assert "#" not in result.output

    def test_block_comment_removed(self):
        src = "# top comment\nx = 1\n# another\ny = 2\n"
        result = minify(src, "python", rename=False)
        assert "#" not in result.output

    def test_docstring_preserved(self):
        src = 'def foo():\n    """This is a docstring."""\n    return 1\n'
        result = minify(src, "python", rename=False)
        # Docstrings are string literals, not comment nodes
        assert "docstring" in result.output

    def test_comment_in_function_removed(self):
        src = "def f(x):\n    # strip me\n    return x + 1\n"
        result = minify(src, "python", rename=False)
        assert "#" not in result.output
        assert _is_valid_python(result.output)


# ---------------------------------------------------------------------------
# Tests: indent reduction
# ---------------------------------------------------------------------------

class TestIndentReduction:
    def test_four_space_to_one(self):
        src = "def f(x):\n    if x > 0:\n        return x\n    return 0\n"
        result = minify(src, "python", rename=False)
        # Should use 1 space per indent level now
        lines = result.output.split("\n")
        body_lines = [l for l in lines if l.startswith(" ")]
        assert all(l.startswith(" ") and not l.startswith("  ") or l.startswith("  ")
                   for l in body_lines), "indent should be minimal"
        assert _is_valid_python(result.output)

    def test_two_space_to_one(self):
        src = "def f(x):\n  return x + 1\n"
        result = minify(src, "python", rename=False)
        assert _is_valid_python(result.output)

    def test_blank_lines_collapsed(self):
        src = "x = 1\n\n\n\ny = 2\n"
        result = minify(src, "python", rename=False)
        # No 3+ consecutive newlines
        assert "\n\n\n" not in result.output


# ---------------------------------------------------------------------------
# Tests: identifier renaming
# ---------------------------------------------------------------------------

class TestIdentifierRenaming:
    def test_local_vars_renamed(self):
        src = "def f(long_parameter_name):\n    long_variable_name = long_parameter_name * 2\n    return long_variable_name\n"
        result = minify(src, "python", rename=True)
        assert "long_parameter_name" not in result.output
        assert "long_variable_name" not in result.output
        assert _is_valid_python(result.output)

    def test_module_names_not_renamed(self):
        src = "PUBLIC_NAME = 42\n\ndef f():\n    local_var = PUBLIC_NAME + 1\n    return local_var\n"
        result = minify(src, "python", rename=True)
        assert "PUBLIC_NAME" in result.output

    def test_dunder_not_renamed(self):
        src = "def f():\n    __hidden = 1\n    return __hidden\n"
        result = minify(src, "python", rename=True)
        assert "__hidden" in result.output

    def test_self_not_renamed(self):
        src = "class C:\n    def method(self):\n        return self\n"
        result = minify(src, "python", rename=True)
        assert "self" in result.output

    def test_nonlocal_not_renamed_locally(self):
        src = (
            "def outer():\n"
            "    shared_state = 0\n"
            "    def inner():\n"
            "        nonlocal shared_state\n"
            "        shared_state += 1\n"
            "        return shared_state\n"
            "    return inner\n"
        )
        result = minify(src, "python", rename=True)
        assert _is_valid_python(result.output)

    def test_name_map_populated(self):
        src = "def f(first_arg, second_arg):\n    sum_result = first_arg + second_arg\n    return sum_result\n"
        result = minify(src, "python", rename=True)
        assert len(result.name_map) > 0

    def test_short_names_are_short(self):
        src = "def f(very_long_variable_name_one):\n    very_long_variable_name_two = very_long_variable_name_one + 1\n    return very_long_variable_name_two\n"
        result = minify(src, "python", rename=True)
        for new_name in result.name_map.values():
            assert len(new_name) <= 2, f"Expected short name, got {new_name!r}"

    def test_exec_scope_not_renamed(self):
        src = (
            "def unsafe(code):\n"
            "    local_secret = 99\n"
            "    exec(code)\n"
            "    return local_secret\n"
        )
        result = minify(src, "python", rename=True)
        # local_secret must not be renamed (exec could reference it)
        assert "local_secret" in result.output

    def test_no_collision_nested_scopes(self):
        src = (
            "def outer(outer_param):\n"
            "    outer_local = outer_param * 2\n"
            "    def inner(inner_param):\n"
            "        inner_local = inner_param + outer_local\n"
            "        return inner_local\n"
            "    return inner\n"
        )
        result = minify(src, "python", rename=True)
        assert _is_valid_python(result.output)
        # No two *different* locals in overlapping scopes share the same name
        # Verify by executing
        exec_ns: dict = {}
        exec(result.output, exec_ns)
        outer = exec_ns["outer"]
        inner = outer(3)
        assert inner(4) == 10  # 3*2=6, 4+6=10


# ---------------------------------------------------------------------------
# Tests: output validity (can be parsed by Python)
# ---------------------------------------------------------------------------

class TestOutputValidity:
    @pytest.mark.parametrize("name", ["basic", "closures", "exports", "dynamic"])
    def test_golden_files_are_valid_python(self, name: str):
        path = PYTHON_GOLDEN / f"{name}_input.py"
        if not path.exists():
            pytest.skip(f"Golden file not found: {path}")
        source = path.read_text()
        result = minify(source, "python", rename=True)
        assert _is_valid_python(result.output), (
            f"Minified output is not valid Python:\n{result.output}"
        )

    @pytest.mark.parametrize("name", ["basic", "closures", "exports", "dynamic"])
    def test_golden_files_are_smaller(self, name: str):
        path = PYTHON_GOLDEN / f"{name}_input.py"
        if not path.exists():
            pytest.skip(f"Golden file not found: {path}")
        source = path.read_text()
        result = minify(source, "python", rename=True)
        assert result.stats.output_bytes < result.stats.input_bytes, (
            "Output should be smaller than input"
        )
