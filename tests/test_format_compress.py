"""Tests for JSON, log, and HTML compression stages and utilities."""
from __future__ import annotations

import json

import pytest

from claude_compress.config import JsonConfig, LogConfig, HtmlConfig
from claude_compress.json_utils import compress_json, looks_like_json, _truncate
from claude_compress.log_utils import (
    collapse_repeated_lines,
    truncate_stack_frames,
    compress_log,
    looks_like_log,
)
from claude_compress.html_utils import html_to_text, looks_like_html
from claude_compress.stages.json_compress import JsonCompressStage
from claude_compress.stages.log_compress import LogCompressStage
from claude_compress.stages.html_compress import HtmlCompressStage
from claude_compress.state import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_result(text: str, tool_use_id: str = "call_1") -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


def _request(*tool_texts: str, protect: int = 0) -> dict:
    """Build a minimal request with tool_result blocks in the first message."""
    content = [_tool_result(t, f"call_{i}") for i, t in enumerate(tool_texts)]
    msgs = [{"role": "user", "content": content}]
    # Add a final user turn so protect_last_n_messages works correctly.
    if protect:
        for _ in range(protect):
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
            msgs.append({"role": "user", "content": [{"type": "text", "text": "next"}]})
    return {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": msgs}


def _get_tool_text(request: dict, msg_idx: int = 0, block_idx: int = 0) -> str:
    return request["messages"][msg_idx]["content"][block_idx]["content"]


def _state() -> SessionState:
    return SessionState(session_id="test")


# ===========================================================================
# JSON utilities
# ===========================================================================

class TestLooksLikeJson:
    def test_object(self):
        assert looks_like_json('{"a": 1}')

    def test_array(self):
        assert looks_like_json('[1, 2, 3]')

    def test_string(self):
        assert looks_like_json('"hello"')

    def test_plain_text(self):
        assert not looks_like_json("just some text")


class TestCompressJson:
    def test_minifies_whitespace(self):
        text = '{\n  "a": 1,\n  "b": 2\n}'
        result = compress_json(text, max_array_items=20, max_string_chars=500)
        assert result == '{"a":1,"b":2}'

    def test_not_json_returns_none(self):
        assert compress_json("not json", 20, 500) is None

    def test_truncates_long_array(self):
        data = list(range(100))
        text = json.dumps(data)
        result = compress_json(text, max_array_items=10, max_string_chars=500)
        parsed = json.loads(result)
        # head(5) + omission marker + tail(5) = 11 elements
        assert len(parsed) == 11
        assert parsed[5] == "[…90 items omitted…]"
        assert parsed[0] == 0
        assert parsed[-1] == 99

    def test_truncates_long_string_values(self):
        text = json.dumps({"key": "x" * 1000})
        result = compress_json(text, max_array_items=20, max_string_chars=50)
        parsed = json.loads(result)
        assert len(parsed["key"]) < 100
        assert "omitted" in parsed["key"]

    def test_preserves_short_content(self):
        text = json.dumps({"name": "Alice", "age": 30})
        result = compress_json(text, max_array_items=20, max_string_chars=500)
        parsed = json.loads(result)
        assert parsed == {"name": "Alice", "age": 30}

    def test_recursive_nested(self):
        data = {"items": list(range(50)), "meta": {"values": list(range(50))}}
        text = json.dumps(data)
        result = compress_json(text, max_array_items=6, max_string_chars=500)
        parsed = json.loads(result)
        assert len(parsed["items"]) < 50
        assert len(parsed["meta"]["values"]) < 50

    def test_depth_limit_prevents_recursion(self):
        # Build a deeply nested structure
        val: dict = {}
        cur = val
        for _ in range(20):
            cur["x"] = {}
            cur = cur["x"]
        cur["leaf"] = "end"
        text = json.dumps(val)
        result = compress_json(text, max_array_items=20, max_string_chars=500)
        assert result is not None  # should not raise


# ===========================================================================
# Log utilities
# ===========================================================================

class TestCollapseRepeatedLines:
    def test_collapses_consecutive_repeats(self):
        text = "line1\nline2\nline2\nline2\nline3"
        result = collapse_repeated_lines(text)
        assert "line2\n[above line repeated ×2]" in result
        assert result.count("line2") == 1

    def test_no_repeats_unchanged(self):
        text = "a\nb\nc"
        assert collapse_repeated_lines(text) == text

    def test_non_consecutive_repeats_kept(self):
        text = "a\nb\na"
        result = collapse_repeated_lines(text)
        assert result.count("a") == 2  # not consecutive, both kept


class TestTruncateStackFrames:
    _PYTHON_TRACE = "\n".join([
        "Traceback (most recent call last):",
        *[f'  File "app.py", line {i}, in func_{i}' for i in range(30)],
        "ValueError: something went wrong",
    ])

    _JAVA_TRACE = "\n".join([
        "Exception in thread \"main\" java.lang.RuntimeException: msg",
        *[f"\tat com.example.Class{i}.method{i}(Class{i}.java:{i})" for i in range(30)],
    ])

    def test_truncates_long_python_trace(self):
        result = truncate_stack_frames(self._PYTHON_TRACE, max_frames=10)
        assert "frames omitted" in result
        frame_lines = [l for l in result.split("\n") if 'File "app.py"' in l]
        assert len(frame_lines) == 10

    def test_truncates_long_java_trace(self):
        result = truncate_stack_frames(self._JAVA_TRACE, max_frames=8)
        assert "frames omitted" in result

    def test_short_trace_unchanged(self):
        trace = "\n".join([
            "Error:",
            '  File "x.py", line 1, in f',
            '  File "x.py", line 2, in g',
            "ValueError",
        ])
        result = truncate_stack_frames(trace, max_frames=10)
        assert "omitted" not in result

    def test_non_frame_lines_preserved(self):
        result = truncate_stack_frames(self._PYTHON_TRACE, max_frames=6)
        assert "Traceback (most recent call last):" in result
        assert "ValueError: something went wrong" in result


class TestLooksLikeLog:
    def test_detects_python_stacktrace(self):
        trace = 'Traceback:\n  File "a.py", line 1, in f\n  File "b.py", line 2, in g\n  File "c.py", line 3\nError'
        assert looks_like_log(trace)

    def test_detects_repeated_lines(self):
        text = "WARN: connection refused\nWARN: connection refused\nWARN: connection refused\nDone"
        assert looks_like_log(text)

    def test_normal_text_not_log(self):
        assert not looks_like_log("This is a regular paragraph.\nIt has two lines.")

    def test_short_text_not_log(self):
        assert not looks_like_log("error\nerror\nerror")  # < 4 lines


# ===========================================================================
# HTML utilities
# ===========================================================================

_SIMPLE_PAGE = """<!DOCTYPE html>
<html>
<head><title>Test</title><style>body{color:red}</style></head>
<body>
<nav><a href="/">Home</a></nav>
<main>
  <h1>Hello World</h1>
  <p>This is a paragraph.</p>
  <ul>
    <li>Item one</li>
    <li>Item two</li>
  </ul>
  <pre><code>x = 1 + 2</code></pre>
</main>
<footer>Footer text</footer>
<script>alert('hi')</script>
</body>
</html>"""

_ARTICLE = """<html><body>
<article>
  <h2>Article Title</h2>
  <p>First paragraph with <strong>bold</strong> text.</p>
  <p>Second paragraph.</p>
</article>
</body></html>"""


class TestLooksLikeHtml:
    def test_doctype(self):
        assert looks_like_html("<!DOCTYPE html><html></html>")

    def test_html_tag(self):
        assert looks_like_html("<html><body>content</body></html>")

    def test_semantic_tag_density(self):
        html = "<div><p>text</p><ul><li>a</li><li>b</li></ul><nav>nav</nav></div>"
        assert looks_like_html(html)

    def test_plain_text(self):
        assert not looks_like_html("This is just plain text without any tags.")

    def test_code_with_angle_brackets(self):
        code = "if (a < b) { return x > y; }"
        assert not looks_like_html(code)


class TestHtmlToText:
    def test_strips_script_and_style(self):
        result = html_to_text(_SIMPLE_PAGE)
        assert "alert" not in result
        assert "color:red" not in result

    def test_strips_nav_and_footer(self):
        result = html_to_text(_SIMPLE_PAGE)
        assert "Home" not in result
        assert "Footer text" not in result

    def test_preserves_main_content(self):
        result = html_to_text(_SIMPLE_PAGE)
        assert "Hello World" in result
        assert "This is a paragraph" in result
        assert "Item one" in result
        assert "Item two" in result

    def test_preserves_code_block(self):
        result = html_to_text(_SIMPLE_PAGE)
        assert "x = 1 + 2" in result

    def test_headings_as_markdown(self):
        result = html_to_text(_ARTICLE)
        assert "## Article Title" in result

    def test_result_shorter_than_input(self):
        result = html_to_text(_SIMPLE_PAGE)
        assert len(result) < len(_SIMPLE_PAGE)

    def test_malformed_html_returns_original(self):
        broken = "<div>unclosed"
        result = html_to_text(broken)
        # Should not raise; either returns something or original
        assert isinstance(result, str)


# ===========================================================================
# Stage integration
# ===========================================================================

class TestJsonCompressStage:
    def _stage(self, **kwargs) -> JsonCompressStage:
        # Default protect to 0 so single-message test requests aren't all protected.
        kwargs.setdefault("protect_last_n_messages", 0)
        cfg = JsonConfig(min_compress_tokens=0, **kwargs)
        return JsonCompressStage(cfg)

    def test_compresses_json_tool_result(self):
        payload = json.dumps([{"id": i, "value": "x" * 10} for i in range(50)])
        req = _request(payload)
        stage = self._stage(max_array_items=10, max_string_chars=500)
        result = stage.apply(req, _state())
        compressed = _get_tool_text(req)
        assert len(compressed) < len(payload)
        assert result.detail["blocks_compressed"] == 1

    def test_skips_non_json(self):
        req = _request("plain text, not JSON")
        stage = self._stage()
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 0

    def test_respects_protect_last_n(self):
        payload = json.dumps({"a": list(range(100))})
        # Tool_result in the LAST message so it falls inside the protection zone.
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "prior context"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [_tool_result(payload)]},
        ]
        req = {"model": "m", "max_tokens": 1, "messages": msgs}
        stage = self._stage(max_array_items=5, protect_last_n_messages=2)
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 0

    def test_list_content_format(self):
        # Use a payload large enough that minification saves at least one token.
        payload = json.dumps([{"id": i, "label": f"item_{i}", "active": True} for i in range(30)])
        block = {
            "type": "tool_result",
            "tool_use_id": "x",
            "content": [{"type": "text", "text": payload}],
        }
        msgs = [{"role": "user", "content": [block]}]
        req = {"model": "m", "max_tokens": 1, "messages": msgs}
        cfg = JsonConfig(min_compress_tokens=0, protect_last_n_messages=0)
        stage = JsonCompressStage(cfg)
        stage.apply(req, _state())
        result_text = block["content"][0]["text"]
        # Minified form has no spaces after : or ,
        assert ": " not in result_text
        assert ", " not in result_text


class TestLogCompressStage:
    def _stage(self, **kwargs) -> LogCompressStage:
        kwargs.setdefault("protect_last_n_messages", 0)
        cfg = LogConfig(min_compress_tokens=0, **kwargs)
        return LogCompressStage(cfg)

    def _python_trace(self, n_frames: int = 30) -> str:
        lines = ["Traceback (most recent call last):"]
        lines += [f'  File "app.py", line {i}, in func_{i}' for i in range(n_frames)]
        lines += ["ValueError: boom"]
        return "\n".join(lines)

    def test_truncates_long_stack_trace(self):
        trace = self._python_trace(30)
        req = _request(trace)
        stage = self._stage(max_stack_frames=10)
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 1
        compressed = _get_tool_text(req)
        assert "frames omitted" in compressed

    def test_collapses_repeated_lines(self):
        log = "\n".join(["WARN: retry"] * 20 + ["Done"])
        req = _request(log)
        stage = self._stage()
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 1
        compressed = _get_tool_text(req)
        assert "repeated" in compressed
        assert compressed.count("WARN: retry") == 1

    def test_skips_normal_text(self):
        req = _request("This is a normal response with no stack traces.")
        stage = self._stage()
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 0

    def test_respects_protect_last_n(self):
        trace = self._python_trace(30)
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "prior"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [_tool_result(trace)]},
        ]
        req = {"model": "m", "max_tokens": 1, "messages": msgs}
        stage = self._stage(max_stack_frames=4, protect_last_n_messages=2)
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 0


class TestHtmlCompressStage:
    def _stage(self, **kwargs) -> HtmlCompressStage:
        kwargs.setdefault("protect_last_n_messages", 0)
        cfg = HtmlConfig(enabled=True, min_compress_tokens=0, **kwargs)
        return HtmlCompressStage(cfg)

    def test_extracts_text_from_html(self):
        req = _request(_SIMPLE_PAGE)
        stage = self._stage()
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 1
        compressed = _get_tool_text(req)
        assert "Hello World" in compressed
        assert "alert" not in compressed
        assert len(compressed) < len(_SIMPLE_PAGE)

    def test_skips_plain_text(self):
        req = _request("Just a normal response from a tool.")
        stage = self._stage()
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 0

    def test_disabled_by_default(self):
        cfg = HtmlConfig()  # default: enabled=False
        stage = HtmlCompressStage(cfg)
        assert not stage.enabled()

    def test_respects_protect_last_n(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "prior"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [_tool_result(_SIMPLE_PAGE)]},
        ]
        req = {"model": "m", "max_tokens": 1, "messages": msgs}
        stage = self._stage(protect_last_n_messages=2)
        result = stage.apply(req, _state())
        assert result.detail["blocks_compressed"] == 0
