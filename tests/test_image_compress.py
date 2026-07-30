"""Tests for image compression stage and utilities.

Tests that don't require Pillow cover the token counting and header parsing
paths. Tests that need Pillow are skipped when it is not installed.
"""
from __future__ import annotations

import base64
import io
import math
import struct

import pytest

from claude_compress.image_utils import (
    count_image_tokens,
    count_image_block_tokens,
    dims_from_header,
    _dims_from_png_header,
    _dims_from_jpeg_header,
    _dims_from_gif_header,
    _dims_from_webp_header,
    STUB_PNG_B64,
    pillow_available,
)
from claude_compress.tokens import count_request
from claude_compress.config import Config, ImageConfig
from claude_compress.stages.image_compress import ImageCompressStage, _iter_image_blocks
from claude_compress.state import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int, height: int) -> bytes:
    """Synthesise a minimal valid PNG with given dimensions."""
    import zlib

    def write_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Minimal IDAT: one scanline per row of zeros
    raw_rows = b"".join(b"\x00" + b"\x00" * width * 3 for _ in range(height))
    idat_data = zlib.compress(raw_rows)

    return (
        b"\x89PNG\r\n\x1a\n"
        + write_chunk(b"IHDR", ihdr_data)
        + write_chunk(b"IDAT", idat_data)
        + write_chunk(b"IEND", b"")
    )


def _make_image_block(width: int, height: int, media_type: str = "image/png") -> dict:
    raw = _make_png_bytes(width, height)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(raw).decode(),
        },
    }


def _request_with_images(image_blocks: list, n_old_turns: int = 2) -> dict:
    msgs = []
    for i in range(n_old_turns):
        content = [{"type": "text", "text": f"user turn {i}"}]
        if i == 0:
            content.extend(image_blocks)
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"answer {i}"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": "final question"}]})
    return {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": msgs}


# ---------------------------------------------------------------------------
# Token counting (no Pillow)
# ---------------------------------------------------------------------------

def test_count_image_tokens_formula():
    # ceil(28/28) * ceil(28/28) = 1
    assert count_image_tokens(28, 28) == 1
    # ceil(29/28) * ceil(28/28) = 2 * 1 = 2
    assert count_image_tokens(29, 28) == 2
    assert count_image_tokens(56, 56) == 4
    assert count_image_tokens(100, 100) == math.ceil(100 / 28) ** 2


def test_png_header_parsing():
    raw = _make_png_bytes(320, 240)
    dims = _dims_from_png_header(raw)
    assert dims == (320, 240)


def test_png_header_wrong_magic():
    assert _dims_from_png_header(b"notapng" + b"\x00" * 20) is None


def test_count_image_block_tokens_base64_png():
    block = _make_image_block(280, 280)
    # ceil(280/28) = 10; 10*10 = 100
    assert count_image_block_tokens(block) == 100


def test_count_image_block_tokens_url_fallback():
    block = {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
    # Should return fixed fallback, not crash
    result = count_image_block_tokens(block)
    assert result > 0


def test_count_image_block_tokens_non_image():
    assert count_image_block_tokens({"type": "text", "text": "hello"}) == 0


def test_count_request_includes_image_tokens():
    block = _make_image_block(280, 280)
    request = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, block]}
        ],
    }
    total = count_request(request)
    img_tokens = count_image_block_tokens(block)
    assert total >= img_tokens  # total includes text + image


def test_count_request_images_in_tool_result():
    img_block = _make_image_block(56, 56)  # 4 tokens
    tool_result = {
        "type": "tool_result",
        "tool_use_id": "x",
        "content": [{"type": "text", "text": "result"}, img_block],
    }
    request = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [tool_result]}],
    }
    total = count_request(request)
    assert total >= 4  # at minimum the image tokens


# ---------------------------------------------------------------------------
# Stage gate: protect_last_n_messages
# ---------------------------------------------------------------------------

def test_iter_image_blocks_respects_protect_last_n():
    block = _make_image_block(280, 280)
    request = _request_with_images([block])
    # With protect_last_n=4, the image in turn 0 is within the last 4 msgs
    # of a 5-message conversation → should be protected
    msgs = request["messages"]
    # 5 messages total: protect_last_n=4 → cutoff at index 1
    found = _iter_image_blocks(request, protect_last_n=4)
    # image is at msg index 0, which is < cutoff(1), so it IS found
    assert len(found) == 1

    # With protect_last_n=6 → cutoff at -1 (nothing processed)
    found2 = _iter_image_blocks(request, protect_last_n=6)
    assert len(found2) == 0


# ---------------------------------------------------------------------------
# Stage: no Pillow → graceful skip
# ---------------------------------------------------------------------------

def test_stage_skips_without_pillow(monkeypatch):
    monkeypatch.setattr(
        "claude_compress.stages.image_compress.pillow_available", lambda: False
    )
    cfg = ImageConfig(enabled=True, max_tokens_per_image=100)
    stage = ImageCompressStage(cfg)
    block = _make_image_block(280, 280)
    request = _request_with_images([block])
    state = SessionState(session_id="test")
    result = stage.apply(request, state)
    assert "Pillow" in result.note
    assert result.saved == 0


# ---------------------------------------------------------------------------
# Stage: Pillow-dependent tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not pillow_available(), reason="Pillow not installed")
def test_stage_compresses_large_image():
    # 560x560 → count_image_tokens = ceil(560/28)^2 = 20^2 = 400 tokens
    block = _make_image_block(560, 560)
    assert count_image_block_tokens(block) == 400

    cfg = ImageConfig(enabled=True, max_tokens_per_image=100, protect_last_n_messages=0)
    stage = ImageCompressStage(cfg)
    request = _request_with_images([block], n_old_turns=1)
    # put image in first message so it's not protected even with protect=0
    state = SessionState(session_id="test")
    result = stage.apply(request, state)

    # After compression the image should be at or below budget
    compressed_block = request["messages"][0]["content"][1]
    tok_after = count_image_block_tokens(compressed_block)
    assert tok_after <= 100, f"Expected ≤100 tokens, got {tok_after}"
    assert result.detail["images_compressed"] >= 1


@pytest.mark.skipif(not pillow_available(), reason="Pillow not installed")
def test_stage_skips_small_image():
    # 28x28 → 1 token → well under any budget
    block = _make_image_block(28, 28)
    cfg = ImageConfig(enabled=True, max_tokens_per_image=1024, protect_last_n_messages=0)
    stage = ImageCompressStage(cfg)
    request = _request_with_images([block], n_old_turns=1)
    state = SessionState(session_id="test")
    result = stage.apply(request, state)
    assert result.detail.get("images_compressed", 0) == 0


@pytest.mark.skipif(not pillow_available(), reason="Pillow not installed")
def test_crop_whitespace_removes_margins():
    from PIL import Image as PILImage
    import numpy as np
    from claude_compress.image_utils import crop_whitespace

    # White canvas with a 10x10 black square in the centre
    img = PILImage.new("RGB", (100, 100), color=(255, 255, 255))
    arr = np.array(img)
    arr[45:55, 45:55] = 0
    img = PILImage.fromarray(arr)

    cropped = crop_whitespace(img)
    # Should be smaller than original (margins removed)
    assert cropped.size[0] < 100 or cropped.size[1] < 100


@pytest.mark.skipif(not pillow_available(), reason="Pillow not installed")
def test_classify_image_photo_vs_diagram():
    from PIL import Image as PILImage
    import numpy as np
    from claude_compress.image_utils import classify_image

    # Random-pixel image simulates many unique colors → photo
    rng = np.random.default_rng(42)
    photo_arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    photo = PILImage.fromarray(photo_arr)
    assert classify_image(photo) == "photo"

    # Solid-colour + few edge pixels → diagram_ui
    ui_arr = np.full((64, 64, 3), 240, dtype=np.uint8)
    ui_arr[10, :] = 0  # one thin dark line
    ui = PILImage.fromarray(ui_arr)
    result = classify_image(ui)
    assert result in ("diagram_ui", "document_text")  # either is reasonable for this


@pytest.mark.skipif(not pillow_available(), reason="Pillow not installed")
def test_downscale_to_token_budget():
    from PIL import Image as PILImage
    from claude_compress.image_utils import downscale_to_token_budget

    img = PILImage.new("RGB", (560, 560))  # 400 visual tokens
    compressed = downscale_to_token_budget(img, 100)
    assert count_image_tokens(*compressed.size) <= 100


@pytest.mark.skipif(not pillow_available(), reason="Pillow not installed")
def test_seam_carve_reduces_width():
    from PIL import Image as PILImage
    import numpy as np
    from claude_compress.image_utils import seam_carve

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr)

    result = seam_carve(img, 50)
    assert result.size[0] == 50
    assert result.size[1] == 40  # height unchanged


# ---------------------------------------------------------------------------
# GIF + WebP header parsing
# ---------------------------------------------------------------------------

def _make_gif_bytes(width: int, height: int) -> bytes:
    """Minimal GIF89a with a logical screen descriptor and nothing else."""
    header = b"GIF89a"
    lsd = struct.pack("<HH", width, height) + b"\x00\x00\x00"  # packed, bgcolor, aspect
    trailer = b";"
    return header + lsd + trailer


def _make_webp_vp8x_bytes(width: int, height: int) -> bytes:
    """Minimal VP8X WebP with the extended chunk carrying canvas dimensions."""
    # VP8X chunk data: flags (4B) + canvas_width-1 (3B LE) + canvas_height-1 (3B LE)
    vp8x_data = b"\x00\x00\x00\x00"  # flags (no alpha, anim, exif, xmp, icc)
    vp8x_data += (width - 1).to_bytes(3, "little")
    vp8x_data += (height - 1).to_bytes(3, "little")
    chunk_size = struct.pack("<I", len(vp8x_data))
    webp_body = b"VP8X" + chunk_size + vp8x_data
    riff_size = struct.pack("<I", 4 + len(webp_body))
    return b"RIFF" + riff_size + b"WEBP" + webp_body


def test_gif_header_parsing():
    raw = _make_gif_bytes(320, 240)
    assert _dims_from_gif_header(raw) == (320, 240)


def test_gif_header_wrong_magic():
    assert _dims_from_gif_header(b"GIF86a" + b"\x00" * 10) is None


def test_webp_vp8x_header_parsing():
    raw = _make_webp_vp8x_bytes(800, 600)
    assert _dims_from_webp_header(raw) == (800, 600)


def test_webp_header_wrong_magic():
    assert _dims_from_webp_header(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20) is None


def test_dims_from_header_sniffs_gif():
    raw = _make_gif_bytes(100, 50)
    # media_type absent — should sniff by magic bytes
    dims = dims_from_header(raw, "")
    assert dims == (100, 50)


def test_dims_from_header_sniffs_webp():
    raw = _make_webp_vp8x_bytes(400, 300)
    dims = dims_from_header(raw, "")
    assert dims == (400, 300)


def test_gif_block_token_count():
    raw = _make_gif_bytes(280, 280)
    block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/gif",
            "data": base64.b64encode(raw).decode(),
        },
    }
    # ceil(280/28) * ceil(280/28) = 10 * 10 = 100
    assert count_image_block_tokens(block) == 100


def test_webp_block_token_count():
    raw = _make_webp_vp8x_bytes(280, 280)
    block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/webp",
            "data": base64.b64encode(raw).decode(),
        },
    }
    assert count_image_block_tokens(block) == 100


# ---------------------------------------------------------------------------
# Stub PNG constant
# ---------------------------------------------------------------------------

def test_stub_png_is_valid_1x1():
    raw = base64.b64decode(STUB_PNG_B64)
    dims = _dims_from_png_header(raw)
    assert dims == (1, 1)


def test_stub_png_costs_one_token():
    assert count_image_tokens(1, 1) == 1


# ---------------------------------------------------------------------------
# Image deduplication
# ---------------------------------------------------------------------------

def test_stage_deduplicates_identical_images():
    block_a = _make_image_block(280, 280)  # 100 tokens
    block_b = _make_image_block(280, 280)  # same bytes → duplicate

    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "turn 0"}, block_a]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "turn 2"}, block_b]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "final"}]},
    ]
    request = {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": msgs}

    from claude_compress.config import ImageConfig
    from claude_compress.stages.image_compress import ImageCompressStage
    from claude_compress.state import SessionState

    cfg = ImageConfig(
        enabled=True,
        dedup_exact=True,
        max_tokens_per_image=9999,  # disable size compression; only test dedup
        protect_last_n_messages=2,
    )
    stage = ImageCompressStage(cfg)
    state = SessionState(session_id="test")
    result = stage.apply(request, state)

    # block_b should be stubbed; block_a should be untouched
    assert result.detail["images_deduped"] == 1
    stubbed_src = msgs[2]["content"][1]["source"]
    assert stubbed_src["data"] == STUB_PNG_B64
    # Original first occurrence unchanged
    assert msgs[0]["content"][1]["source"]["data"] != STUB_PNG_B64


def test_stage_dedup_respects_different_images():
    block_a = _make_image_block(280, 280)
    block_b = _make_image_block(56, 56)  # different size → different bytes

    msgs = [
        {"role": "user", "content": [block_a]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [block_b]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "final"}]},
    ]
    request = {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": msgs}

    from claude_compress.config import ImageConfig
    from claude_compress.stages.image_compress import ImageCompressStage
    from claude_compress.state import SessionState

    cfg = ImageConfig(enabled=True, dedup_exact=True, max_tokens_per_image=9999, protect_last_n_messages=2)
    stage = ImageCompressStage(cfg)
    result = stage.apply(request, state=SessionState(session_id="t"))
    assert result.detail["images_deduped"] == 0


# ---------------------------------------------------------------------------
# Age-based progressive compression
# ---------------------------------------------------------------------------

def test_age_based_budget_compresses_old_images_more():
    # 560×560 = 400 tokens, much larger than old_age_max_tokens=64
    block_old = _make_image_block(560, 560)
    block_new = _make_image_block(560, 560)

    msgs = [
        {"role": "user", "content": [block_old]},   # index 0 (old)
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [block_new]},    # index 2 (new-ish)
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "final"}]},
    ]
    request = {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": msgs}

    from claude_compress.config import ImageConfig
    from claude_compress.stages.image_compress import ImageCompressStage
    from claude_compress.state import SessionState

    cfg = ImageConfig(
        enabled=True,
        dedup_exact=False,  # disable dedup so both images get processed
        max_tokens_per_image=200,        # normal budget: 400→≤200
        old_age_threshold_messages=3,    # msgs 0..1 are "old" (age ≥ 4 from end)
        old_age_max_tokens=64,           # old budget: 400→≤64
        protect_last_n_messages=2,       # protect last 2 messages
    )
    stage = ImageCompressStage(cfg)
    state = SessionState(session_id="test")
    result = stage.apply(request, state)

    tok_old = count_image_block_tokens(msgs[0]["content"][0])
    tok_new = count_image_block_tokens(msgs[2]["content"][0])

    assert tok_old <= 64, f"Old image should be compressed to ≤64 tokens, got {tok_old}"
    assert tok_new <= 200, f"New image should be compressed to ≤200 tokens, got {tok_new}"
    # Old image should be smaller than new image (hit the stricter budget)
    assert tok_old <= tok_new
