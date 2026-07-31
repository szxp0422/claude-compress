"""Log output and stack-trace compression for tool_result content.

Two operations, both applied unconditionally (they are self-gating — no-ops when
the content doesn't match):

  1. Repeated-line collapse — consecutive identical lines become one line plus
     a "[above line repeated ×N]" annotation.  Lossless: the annotation says
     exactly what was dropped.

  2. Stack-frame truncation — long runs of stack frames are reduced to
     head + tail with an "[N frames omitted]" gap.  Keeps the call site and
     the crash site, which together answer >90% of debugging questions.

No external dependencies.
"""
from __future__ import annotations

import re
from typing import List

# Patterns that identify a single stack frame line across common runtimes.
_FRAME_RES = [
    re.compile(r'^\s+File "[^"]+", line \d+'),            # Python
    re.compile(r'^\s+at [\w$.<>]+\([\w.]*(?::\d+)?\)'),   # Java / Kotlin / Scala
    re.compile(r'^\s+at [\w$.< >]+\s*\(?[^)]*:\d+:\d+\)?'),  # Node.js / V8
    re.compile(r'^\s+#\d+\s+(?:0x[0-9a-fA-F]+ in |[\w<>]+)'), # GDB / LLDB
    re.compile(r'^\s+\d+\s+[\w.]+\s+0x[0-9a-fA-F]+'),    # macOS symbolicated
]


def _is_frame(line: str) -> bool:
    return any(p.match(line) for p in _FRAME_RES)


def collapse_repeated_lines(text: str) -> str:
    """Replace runs of identical consecutive lines with a count annotation."""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j] == lines[i]:
            j += 1
        out.append(lines[i])
        count = j - i
        if count > 1:
            out.append(f"[above line repeated ×{count - 1}]")
        i = j
    return "\n".join(out)


def truncate_stack_frames(text: str, max_frames: int) -> str:
    """Find runs of stack frames; if a run exceeds max_frames, keep head + tail."""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        if not _is_frame(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        # Collect the full frame run.
        start = i
        while i < len(lines) and _is_frame(lines[i]):
            i += 1
        run = lines[start:i]
        if len(run) <= max_frames:
            out.extend(run)
        else:
            n_head = max_frames // 2
            n_tail = max_frames - n_head
            omitted = len(run) - max_frames
            out.extend(run[:n_head])
            out.append(f"    ... [{omitted} frames omitted] ...")
            out.extend(run[-n_tail:])
    return "\n".join(out)


def compress_log(text: str, max_frames: int) -> str:
    """Apply repeated-line collapse then stack-frame truncation."""
    text = collapse_repeated_lines(text)
    text = truncate_stack_frames(text, max_frames)
    return text


def looks_like_log(text: str) -> bool:
    """Heuristic: return True when text is worth running through compress_log.

    Triggers on any stack trace or on content with repeated consecutive lines.
    Scans only the first 100 lines to bound cost on huge blobs.
    """
    lines = text.split("\n", 100)[:100]
    if len(lines) < 4:
        return False
    if sum(1 for l in lines if _is_frame(l)) >= 3:
        return True
    for i in range(len(lines) - 1):
        if lines[i] == lines[i + 1] and lines[i].strip():
            return True
    return False
