"""Tests for TypeScript, C, C++, Java, JSON, and YAML language modules."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from shutil import which

import pytest

from minifier.minify import minify

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).parent / "golden"
TS_GOLDEN = GOLDEN / "typescript"
C_GOLDEN = GOLDEN / "c"
CPP_GOLDEN = GOLDEN / "cpp"
JAVA_GOLDEN = GOLDEN / "java"
JSON_GOLDEN = GOLDEN / "json"
YAML_GOLDEN = GOLDEN / "yaml"

NODE = which("node")
GCC = which("gcc") or which("clang")
GPP = which("g++") or which("clang++")
JAVAC = which("javac")
JAVA_BIN = which("java")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, input_text=None, timeout=30):
    r = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r


# ===========================================================================
# TypeScript
# ===========================================================================

class TestTypeScript:
    def test_comment_stripping(self):
        src = "// single line\nconst x: number = 1;\n/* block */\nconst y = 2;\n"
        result = minify(src, "typescript", rename=False)
        assert "//" not in result.output
        assert "/*" not in result.output
        assert "const" in result.output

    def test_type_annotations_not_renamed(self):
        src = "function f(x: number, y: number): number { const r = x + y; return r; }\n"
        result = minify(src, "typescript")
        # 'number' is a predefined_type, must not be renamed
        assert "number" in result.output

    def test_interface_names_not_renamed(self):
        src = "interface MyInterface { getValue(): string; }\n"
        result = minify(src, "typescript", rename=False)
        assert "MyInterface" in result.output

    def test_local_vars_renamed(self):
        src = (
            "function compute(inputX: number, inputY: number): number {\n"
            "  const localResult: number = inputX + inputY;\n"
            "  return localResult;\n"
            "}\n"
        )
        result = minify(src, "typescript")
        assert "inputX" not in result.output
        assert "inputY" not in result.output
        assert "localResult" not in result.output

    def test_size_reduction(self):
        src = (TS_GOLDEN / "basic_input.ts").read_text()
        result = minify(src, "typescript")
        assert result.stats.output_bytes < result.stats.input_bytes

    @pytest.mark.skipif(NODE is None, reason="node not found")
    def test_roundtrip_node(self):
        src = (TS_GOLDEN / "basic_input.ts").read_text()
        # TypeScript requires transpilation; run original through node with type stripping
        # via node --input-type=module (TS can't run directly). Skip full round-trip;
        # just verify minified is syntactically valid by re-parsing.
        result = minify(src, "typescript")
        from minifier.core.parser import parse
        tree = parse(result.output.encode(), "typescript")
        assert tree.root_node.has_error is False, (
            f"Minified TypeScript has parse error:\n{result.output[:500]}"
        )

    def test_no_parse_errors(self):
        src = (TS_GOLDEN / "basic_input.ts").read_text()
        result = minify(src, "typescript")
        from minifier.core.parser import parse
        tree = parse(result.output.encode(), "typescript")
        assert not tree.root_node.has_error


# ===========================================================================
# C
# ===========================================================================

class TestC:
    def test_comment_stripping(self):
        src = "/* header */\nint x = 1; // inline\nint y = 2;\n"
        result = minify(src, "c", rename=False)
        assert "/*" not in result.output
        assert "//" not in result.output
        assert "int" in result.output

    def test_whitespace_collapse(self):
        src = "int   f  (  int   x  )  {  return   x  ;  }\n"
        result = minify(src, "c", rename=False)
        assert "  " not in result.output

    def test_local_vars_renamed(self):
        src = (
            "int compute(int firstParam, int secondParam) {\n"
            "  int localResult = firstParam + secondParam;\n"
            "  return localResult;\n"
            "}\n"
        )
        result = minify(src, "c")
        assert "firstParam" not in result.output
        assert "secondParam" not in result.output
        assert "localResult" not in result.output

    def test_preproc_preserved(self):
        src = "#define SQUARE(x) ((x)*(x))\nint f(int v) { return SQUARE(v); }\n"
        result = minify(src, "c")
        assert "SQUARE" in result.output
        assert "#define" in result.output

    def test_global_not_renamed(self):
        src = "int global_var = 0;\nint f(int x) { global_var = x; return global_var; }\n"
        result = minify(src, "c")
        assert "global_var" in result.output

    def test_size_reduction(self):
        src = (C_GOLDEN / "basic_input.c").read_text()
        result = minify(src, "c")
        assert result.stats.output_bytes < result.stats.input_bytes

    def test_no_parse_errors(self):
        src = (C_GOLDEN / "basic_input.c").read_text()
        result = minify(src, "c")
        from minifier.core.parser import parse
        tree = parse(result.output.encode(), "c")
        assert not tree.root_node.has_error

    @pytest.mark.skipif(GCC is None, reason="gcc/clang not found")
    def test_roundtrip_compile_c(self, tmp_path):
        src = (C_GOLDEN / "basic_input.c").read_text()
        result = minify(src, "c")

        orig_file = tmp_path / "orig.c"
        mini_file = tmp_path / "mini.c"
        orig_exe  = tmp_path / "orig_out"
        mini_exe  = tmp_path / "mini_out"

        orig_file.write_text(src)
        mini_file.write_text(result.output)

        r1 = _run([GCC, str(orig_file), "-o", str(orig_exe)])
        assert r1.returncode == 0, f"Original C compile failed:\n{r1.stderr}"

        r2 = _run([GCC, str(mini_file), "-o", str(mini_exe)])
        assert r2.returncode == 0, (
            f"Minified C compile failed:\n{r2.stderr}\n--- Minified ---\n{result.output}"
        )

        orig_out = _run([str(orig_exe)]).stdout
        mini_out = _run([str(mini_exe)]).stdout
        assert orig_out == mini_out, (
            f"Output mismatch:\n  original: {orig_out!r}\n  minified: {mini_out!r}"
        )


# ===========================================================================
# C++
# ===========================================================================

class TestCpp:
    def test_comment_stripping(self):
        src = "// line comment\nint x = 1; /* block */ int y = 2;\n"
        result = minify(src, "cpp", rename=False)
        assert "//" not in result.output
        assert "/*" not in result.output

    def test_local_vars_renamed(self):
        src = (
            "int add(int firstVal, int secondVal) {\n"
            "  int localSum = firstVal + secondVal;\n"
            "  return localSum;\n"
            "}\n"
        )
        result = minify(src, "cpp")
        assert "firstVal" not in result.output
        assert "localSum" not in result.output

    def test_class_method_locals_renamed(self):
        src = (
            "class Calc {\npublic:\n"
            "  int multiply(int factorA, int factorB) {\n"
            "    int productResult = factorA * factorB;\n"
            "    return productResult;\n"
            "  }\n"
            "};\n"
        )
        result = minify(src, "cpp")
        assert "factorA" not in result.output
        assert "productResult" not in result.output

    def test_size_reduction(self):
        src = (CPP_GOLDEN / "basic_input.cpp").read_text()
        result = minify(src, "cpp")
        assert result.stats.output_bytes < result.stats.input_bytes

    def test_no_parse_errors(self):
        src = (CPP_GOLDEN / "basic_input.cpp").read_text()
        result = minify(src, "cpp")
        from minifier.core.parser import parse
        tree = parse(result.output.encode(), "cpp")
        assert not tree.root_node.has_error

    @pytest.mark.skipif(GPP is None, reason="g++/clang++ not found")
    def test_roundtrip_compile_cpp(self, tmp_path):
        src = (CPP_GOLDEN / "basic_input.cpp").read_text()
        result = minify(src, "cpp")

        orig_file = tmp_path / "orig.cpp"
        mini_file = tmp_path / "mini.cpp"
        orig_exe  = tmp_path / "orig_out"
        mini_exe  = tmp_path / "mini_out"

        orig_file.write_text(src)
        mini_file.write_text(result.output)

        r1 = _run([GPP, "-std=c++17", str(orig_file), "-o", str(orig_exe)])
        assert r1.returncode == 0, f"Original C++ compile failed:\n{r1.stderr}"

        r2 = _run([GPP, "-std=c++17", str(mini_file), "-o", str(mini_exe)])
        assert r2.returncode == 0, (
            f"Minified C++ compile failed:\n{r2.stderr}\n--- Minified ---\n{result.output}"
        )

        orig_out = _run([str(orig_exe)]).stdout
        mini_out = _run([str(mini_exe)]).stdout
        assert orig_out == mini_out


# ===========================================================================
# Java
# ===========================================================================

class TestJava:
    def test_comment_stripping(self):
        src = "// line comment\npublic class F { /* block */ int x = 1; }\n"
        result = minify(src, "java", rename=False)
        assert "//" not in result.output
        assert "/*" not in result.output

    def test_local_vars_renamed(self):
        src = (
            "public class T {\n"
            "  public int add(int firstNum, int secondNum) {\n"
            "    int localResult = firstNum + secondNum;\n"
            "    return localResult;\n"
            "  }\n"
            "}\n"
        )
        result = minify(src, "java")
        assert "firstNum" not in result.output
        assert "localResult" not in result.output

    def test_field_names_not_renamed(self):
        src = (
            "public class T {\n"
            "  private int instanceField = 0;\n"
            "  public void set(int newValue) { this.instanceField = newValue; }\n"
            "}\n"
        )
        result = minify(src, "java")
        assert "instanceField" in result.output

    def test_method_names_not_renamed(self):
        src = (
            "public class T {\n"
            "  public int compute(int x) { return x * 2; }\n"
            "  public void run() { int r = compute(5); }\n"
            "}\n"
        )
        result = minify(src, "java")
        assert "compute" in result.output

    def test_for_loop_vars_renamed(self):
        src = (
            "public class T {\n"
            "  public int sum(int[] arr) {\n"
            "    int totalSum = 0;\n"
            "    for (int loopIdx = 0; loopIdx < arr.length; loopIdx++) {\n"
            "      totalSum += arr[loopIdx];\n"
            "    }\n"
            "    return totalSum;\n"
            "  }\n"
            "}\n"
        )
        result = minify(src, "java")
        assert "loopIdx" not in result.output
        assert "totalSum" not in result.output

    def test_size_reduction(self):
        src = (JAVA_GOLDEN / "BasicInput.java").read_text()
        result = minify(src, "java")
        assert result.stats.output_bytes < result.stats.input_bytes

    def test_no_parse_errors(self):
        src = (JAVA_GOLDEN / "BasicInput.java").read_text()
        result = minify(src, "java")
        from minifier.core.parser import parse
        tree = parse(result.output.encode(), "java")
        assert not tree.root_node.has_error

    @pytest.mark.skipif(
        JAVAC is None or JAVA_BIN is None, reason="javac/java not found"
    )
    def test_roundtrip_compile_java(self, tmp_path):
        src = (JAVA_GOLDEN / "BasicInput.java").read_text()
        result = minify(src, "java")

        orig_dir = tmp_path / "orig"
        mini_dir = tmp_path / "mini"
        orig_dir.mkdir()
        mini_dir.mkdir()

        (orig_dir / "BasicInput.java").write_text(src)
        (mini_dir / "BasicInput.java").write_text(result.output)

        r1 = _run([JAVAC, str(orig_dir / "BasicInput.java"), "-d", str(orig_dir)])
        assert r1.returncode == 0, f"Original Java compile failed:\n{r1.stderr}"

        r2 = _run([JAVAC, str(mini_dir / "BasicInput.java"), "-d", str(mini_dir)])
        assert r2.returncode == 0, (
            f"Minified Java compile failed:\n{r2.stderr}\n--- Minified ---\n{result.output}"
        )

        orig_out = _run([JAVA_BIN, "-cp", str(orig_dir), "BasicInput"]).stdout
        mini_out = _run([JAVA_BIN, "-cp", str(mini_dir), "BasicInput"]).stdout
        assert orig_out == mini_out


# ===========================================================================
# JSON
# ===========================================================================

class TestJson:
    def test_whitespace_stripped(self):
        src = '{\n  "key": "value",\n  "num": 42\n}\n'
        result = minify(src, "json")
        assert result.output == '{"key":"value","num":42}'

    def test_valid_json_after_minify(self):
        src = (JSON_GOLDEN / "basic_input.json").read_text()
        result = minify(src, "json")
        parsed = json.loads(result.output)
        original = json.loads(src)
        assert parsed == original

    def test_nested_json_preserved(self):
        src = '{"a": {"b": {"c": [1, 2, 3]}}}'
        result = minify(src, "json")
        assert json.loads(result.output) == json.loads(src)

    def test_no_newlines(self):
        src = (JSON_GOLDEN / "basic_input.json").read_text()
        result = minify(src, "json")
        # Group A output should be on a single line
        assert "\n" not in result.output

    def test_size_reduction(self):
        src = (JSON_GOLDEN / "basic_input.json").read_text()
        result = minify(src, "json")
        assert result.stats.output_bytes < result.stats.input_bytes

    def test_null_true_false_preserved(self):
        src = '{"a": null, "b": true, "c": false}'
        result = minify(src, "json")
        data = json.loads(result.output)
        assert data["a"] is None
        assert data["b"] is True
        assert data["c"] is False


# ===========================================================================
# YAML
# ===========================================================================

class TestYaml:
    def test_comment_stripping(self):
        src = "# top comment\nname: alice\n# another comment\nage: 30\n"
        result = minify(src, "yaml")
        assert "#" not in result.output
        assert "name: alice" in result.output
        assert "age: 30" in result.output

    def test_indentation_preserved(self):
        src = "parent:\n  child:\n    grandchild: value\n"
        result = minify(src, "yaml")
        # Indentation must be preserved (YAML structure)
        assert "  child:" in result.output
        assert "    grandchild:" in result.output

    def test_blank_lines_collapsed(self):
        src = "a: 1\n\n\n\nb: 2\n"
        result = minify(src, "yaml")
        assert "\n\n\n" not in result.output

    def test_data_preserved(self):
        """Verify YAML data is unchanged after comment stripping."""
        try:
            import yaml as pyyaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        src = (YAML_GOLDEN / "basic_input.yaml").read_text()
        result = minify(src, "yaml")

        original_data = pyyaml.safe_load(src)
        minified_data = pyyaml.safe_load(result.output)
        assert original_data == minified_data

    def test_size_reduction(self):
        src = (YAML_GOLDEN / "basic_input.yaml").read_text()
        result = minify(src, "yaml")
        assert result.stats.output_bytes < result.stats.input_bytes

    def test_no_rename_in_yaml(self):
        src = "name: alice\nage: 30\n"
        result = minify(src, "yaml")
        assert result.name_map == {}
