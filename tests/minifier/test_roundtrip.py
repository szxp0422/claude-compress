"""Round-trip behavioral equivalence tests.

For every golden-file input, run original and minified through the language's
runtime and assert stdout is identical.  This is the correctness gate (spec).
"""
import subprocess
import sys
from pathlib import Path
from shutil import which

import pytest

from minifier.minify import minify
from .conftest import PYTHON_GOLDEN, JS_GOLDEN

NODE = which("node")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_python(source: str) -> str:
    r = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Python execution failed:\n{r.stderr}")
    return r.stdout


def _run_node(source: str) -> str:
    r = subprocess.run(
        [NODE, "-e", source],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Node execution failed:\n{r.stderr}\n---\n{source[:800]}"
        )
    return r.stdout


# ---------------------------------------------------------------------------
# Python round-trip
# ---------------------------------------------------------------------------

PYTHON_GOLDEN_FILES = sorted(PYTHON_GOLDEN.glob("*_input.py"))


@pytest.mark.parametrize("path", PYTHON_GOLDEN_FILES, ids=[p.stem for p in PYTHON_GOLDEN_FILES])
def test_python_roundtrip(path: Path):
    source = path.read_text()
    original_out = _run_python(source)

    result = minify(source, "python", rename=True)
    try:
        minified_out = _run_python(result.output)
    except RuntimeError as e:
        pytest.fail(f"Minified Python failed to execute:\n{e}\n\n--- Minified ---\n{result.output}")

    assert original_out == minified_out, (
        f"Behavioural mismatch for {path.name}:\n"
        f"  original  stdout: {original_out!r}\n"
        f"  minified  stdout: {minified_out!r}\n"
        f"--- Minified source ---\n{result.output}"
    )

    # Also verify size reduction
    assert result.stats.output_bytes < result.stats.input_bytes, (
        f"Minified output ({result.stats.output_bytes} B) is not smaller than "
        f"input ({result.stats.input_bytes} B) for {path.name}"
    )


# ---------------------------------------------------------------------------
# JavaScript round-trip
# ---------------------------------------------------------------------------

JS_GOLDEN_FILES = sorted(JS_GOLDEN.glob("*_input.js"))


@pytest.mark.skipif(NODE is None, reason="node.js not found in PATH")
@pytest.mark.parametrize("path", JS_GOLDEN_FILES, ids=[p.stem for p in JS_GOLDEN_FILES])
def test_javascript_roundtrip(path: Path):
    source = path.read_text()
    original_out = _run_node(source)

    result = minify(source, "javascript", rename=True)
    try:
        minified_out = _run_node(result.output)
    except RuntimeError as e:
        pytest.fail(
            f"Minified JavaScript failed to execute:\n{e}\n\n--- Minified ---\n{result.output}"
        )

    assert original_out == minified_out, (
        f"Behavioural mismatch for {path.name}:\n"
        f"  original  stdout: {original_out!r}\n"
        f"  minified  stdout: {minified_out!r}\n"
        f"--- Minified source ---\n{result.output}"
    )

    assert result.stats.output_bytes < result.stats.input_bytes, (
        f"Minified output ({result.stats.output_bytes} B) is not smaller than "
        f"input ({result.stats.input_bytes} B) for {path.name}"
    )
