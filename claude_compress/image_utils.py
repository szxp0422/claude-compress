"""Image processing utilities for VLM token compression.

Requires Pillow (pip install Pillow). If unavailable the image stage skips
gracefully, identical to the optional sentence-transformers pattern in embeddings.

Token formula for Claude: ceil(width / 28) * ceil(height / 28) visual tokens.
One 28x28-pixel patch = one token. Shrinking only saves tokens when it crosses
a real tile-count boundary.
"""
from __future__ import annotations

import base64
import io
import math
import struct
import zlib
from typing import Optional, Tuple

import numpy as np

try:
    from PIL import Image as _PILImage

    _PILLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PILImage = None  # type: ignore
    _PILLOW_AVAILABLE = False


def pillow_available() -> bool:
    return _PILLOW_AVAILABLE


# ---------------------------------------------------------------------------
# Token counting (no Pillow needed — reads image headers directly)
# ---------------------------------------------------------------------------

def count_image_tokens(width: int, height: int) -> int:
    """Claude visual token cost: ceil(w/28) * ceil(h/28)."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def _dims_from_png_header(raw: bytes) -> Optional[Tuple[int, int]]:
    """Read (width, height) from PNG file bytes without decoding the image."""
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    # IHDR chunk: 4B length, 4B "IHDR", 4B width, 4B height
    w, h = struct.unpack(">II", raw[16:24])
    return w, h


def _dims_from_jpeg_header(raw: bytes) -> Optional[Tuple[int, int]]:
    """Scan JPEG markers to find SOF0/SOF1/SOF2 with frame dimensions."""
    if len(raw) < 2 or raw[0] != 0xFF or raw[1] != 0xD8:
        return None
    i = 2
    while i < len(raw) - 3:
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):  # SOF0, SOF1, SOF2
            if i + 9 <= len(raw):
                h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
                return w, h
        seg_len = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        i += 2 + seg_len
    return None


def _dims_from_gif_header(raw: bytes) -> Optional[Tuple[int, int]]:
    """Read (width, height) from GIF87a / GIF89a logical screen descriptor."""
    if len(raw) < 10 or raw[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    w, h = struct.unpack_from("<HH", raw, 6)
    return w, h


def _dims_from_webp_header(raw: bytes) -> Optional[Tuple[int, int]]:
    """Read (width, height) from a WebP VP8X (extended) chunk.

    VP8X is the most common format for WebP files with alpha or metadata.
    For plain VP8 (lossy) and VP8L (lossless) the bitstream format is more
    involved; those fall back to the fixed estimate.
    """
    if len(raw) < 30 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        return None
    chunk = raw[12:16]
    if chunk == b"VP8X":
        # VP8X canvas width-1 (3B LE) at offset 24, height-1 (3B LE) at offset 27
        w = int.from_bytes(raw[24:27], "little") + 1
        h = int.from_bytes(raw[27:30], "little") + 1
        return w, h
    return None


def dims_from_header(raw: bytes, media_type: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) by peeking at image file header bytes."""
    if "png" in media_type:
        return _dims_from_png_header(raw)
    if "jpeg" in media_type or "jpg" in media_type:
        return _dims_from_jpeg_header(raw)
    if "gif" in media_type:
        return _dims_from_gif_header(raw)
    if "webp" in media_type:
        return _dims_from_webp_header(raw)
    # Sniff by magic bytes when media_type is missing or generic
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return _dims_from_gif_header(raw)
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return _dims_from_webp_header(raw)
    return None


def _build_stub_png() -> bytes:
    """Synthesise a minimal valid 1×1 mid-gray PNG (no Pillow needed)."""
    def _chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)  # 1×1 8-bit grayscale
    idat = zlib.compress(b"\x00\x80")  # filter=None, pixel=128 (mid-gray)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


# Pre-computed once at import time — used to stub duplicate images.
STUB_PNG_B64: str = base64.b64encode(_build_stub_png()).decode()
STUB_MEDIA_TYPE: str = "image/png"


def count_image_block_tokens(block: dict) -> int:
    """Estimate visual token cost for a single image content block.

    Returns 0 for non-image blocks. Returns a fixed estimate for URL images
    (we can't fetch them) and for formats whose header we can't parse.
    """
    if not isinstance(block, dict) or block.get("type") != "image":
        return 0
    src = block.get("source", {})
    if src.get("type") == "base64":
        raw = base64.b64decode(src.get("data", ""))
        dims = dims_from_header(raw, src.get("media_type", ""))
        if dims:
            return count_image_tokens(*dims)
        # Unknown format — rough estimate: 1000 tokens (~896x896 equivalent)
        return 1000
    # URL source — can't determine without fetching; use fixed estimate
    return 1000


# ---------------------------------------------------------------------------
# Decode / encode (requires Pillow)
# ---------------------------------------------------------------------------

def decode_image(b64_data: str) -> "_PILImage.Image":
    """Decode a base64 image string to a PIL Image."""
    raw = base64.b64decode(b64_data)
    return _PILImage.open(io.BytesIO(raw))


def encode_image(img: "_PILImage.Image", media_type: str) -> Tuple[str, str]:
    """Encode a PIL Image back to (base64_string, media_type).

    JPEG for photos (smaller file), PNG for everything else (lossless quality).
    """
    fmt = "JPEG" if "jpeg" in media_type or "jpg" in media_type else "PNG"
    if fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    save_kw = {"quality": 85, "optimize": True} if fmt == "JPEG" else {"optimize": True}
    img.save(buf, format=fmt, **save_kw)
    return base64.b64encode(buf.getvalue()).decode(), f"image/{fmt.lower()}"


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------

def classify_image(img: "_PILImage.Image") -> str:
    """Classify image content type using cheap numpy heuristics.

    Returns 'photo', 'document_text', or 'diagram_ui'.

    Three signals:
    1. Color palette entropy  — few distinct colors → UI / diagram
    2. Edge density (Sobel)   — high + row-periodic → document text
    3. Row-mean variance       — periodic row peaks → text baselines
    """
    thumb = img.copy()
    thumb.thumbnail((256, 256), _PILImage.LANCZOS)
    gray = np.asarray(thumb.convert("L"), dtype=np.float64) / 255.0
    h, w = gray.shape

    # 1. Color palette: quantise RGB to 5-bit per channel, count distinct tuples
    rgb = np.asarray(thumb.convert("RGB"), dtype=np.uint8) >> 3  # 32 levels/channel
    n_distinct = len({(int(r), int(g), int(b)) for r, g, b in rgb.reshape(-1, 3)})
    color_entropy = n_distinct / (h * w)  # 0 = monochrome, 1 = all pixels unique

    # 2. Edge density via finite-difference Sobel approximation
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    edge_density = float(np.mean((dx + dy) > 0.05))

    # 3. Row-wise mean intensity variance (periodic peaks = text baselines)
    row_variance = float(np.var(gray.mean(axis=1)))

    # Decision: photos have high color entropy; text has dense periodic edges;
    # diagrams/UI have low color entropy and moderate structure.
    if color_entropy > 0.15:
        return "photo"
    if edge_density > 0.12 and row_variance > 0.005:
        return "document_text"
    return "diagram_ui"


# ---------------------------------------------------------------------------
# Compression operations
# ---------------------------------------------------------------------------

def _target_dims_for_budget(
    width: int, height: int, token_budget: int
) -> Tuple[int, int]:
    """Largest (w, h) preserving aspect ratio that fits within token_budget tokens."""
    if count_image_tokens(width, height) <= token_budget:
        return width, height
    # Binary search on scale factor
    lo, hi = 0.0, 1.0
    best_w, best_h = 1, 1
    for _ in range(40):
        mid = (lo + hi) / 2
        nw = max(1, int(width * mid))
        nh = max(1, int(height * mid))
        if count_image_tokens(nw, nh) <= token_budget:
            best_w, best_h = nw, nh
            lo = mid
        else:
            hi = mid
    return best_w, best_h


def crop_whitespace(img: "_PILImage.Image") -> "_PILImage.Image":
    """Remove uniform whitespace / letterboxing margins (near-free, zero distortion)."""
    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    row_std = np.std(gray, axis=1)
    col_std = np.std(gray, axis=0)

    if not row_std.any() or not col_std.any():
        return img

    top = int(np.argmax(row_std > 0.02))
    bottom = int(len(row_std) - np.argmax((row_std > 0.02)[::-1]))
    left = int(np.argmax(col_std > 0.02))
    right = int(len(col_std) - np.argmax((col_std > 0.02)[::-1]))

    # Sanity: don't produce a degenerate sliver (< 5% of original dimension)
    if (bottom - top) < max(1, img.height * 0.05) or (right - left) < max(1, img.width * 0.05):
        return img

    return img.crop((left, top, right, bottom))


def downscale_to_token_budget(
    img: "_PILImage.Image", token_budget: int
) -> "_PILImage.Image":
    """Resize image to stay within token_budget visual tokens (maintains aspect ratio)."""
    w, h = img.size
    tw, th = _target_dims_for_budget(w, h, token_budget)
    if tw >= w and th >= h:
        return img
    return img.resize((tw, th), _PILImage.LANCZOS)


# ---------------------------------------------------------------------------
# Seam carving (backward energy, pure numpy — last resort for photos only)
# ---------------------------------------------------------------------------

def _compute_energy(gray: np.ndarray) -> np.ndarray:
    """Backward-energy map via finite-difference Sobel approximation."""
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    return (dx + dy).astype(np.float64)


def _find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Minimum-cost vertical seam via dynamic programming. O(W*H)."""
    h, w = energy.shape
    M = energy.copy()
    # choice[y, x] ∈ {-1, 0, 1}: parent column offset
    choice = np.zeros((h, w), dtype=np.int8)

    for y in range(1, h):
        row = M[y - 1]
        left = np.empty(w)
        left[0] = np.inf
        left[1:] = row[:-1]
        right = np.empty(w)
        right[-1] = np.inf
        right[:-1] = row[1:]

        best = np.minimum(np.minimum(left, row), right)
        M[y] += best
        # record which parent was cheapest
        choice[y] = np.where(best == left, -1, np.where(best == row, 0, 1))

    seam = np.empty(h, dtype=int)
    seam[-1] = int(np.argmin(M[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = int(np.clip(seam[y + 1] + choice[y + 1, seam[y + 1]], 0, w - 1))
    return seam


def _remove_vertical_seam(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    out_shape = (h, w - 1, arr.shape[2]) if arr.ndim == 3 else (h, w - 1)
    out = np.empty(out_shape, dtype=arr.dtype)
    for y in range(h):
        x = seam[y]
        out[y, :x] = arr[y, :x]
        out[y, x:] = arr[y, x + 1 :]
    return out


def seam_carve(img: "_PILImage.Image", target_width: int) -> "_PILImage.Image":
    """Content-aware resize to target_width by removing vertical seams.

    ONLY appropriate for photographic content. Geometric warping from seam
    removal corrupts text, tables, diagrams, and UI — use downscaling there.

    This implements backward energy (standard Sobel). For production use,
    forward energy (Rubinstein et al. 2008) reduces warping artifacts further
    but requires per-row cost tables that complicate the DP recurrence.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    current_width = arr.shape[1]
    if target_width >= current_width:
        return img

    n_seams = current_width - target_width
    for _ in range(n_seams):
        gray = arr.mean(axis=2) / 255.0
        energy = _compute_energy(gray)
        seam = _find_vertical_seam(energy)
        arr = _remove_vertical_seam(arr, seam)

    mode = img.mode
    result = _PILImage.fromarray(arr)
    if mode != "RGB":
        result = result.convert(mode)
    return result


# ---------------------------------------------------------------------------
# OCR extraction (requires pytesseract + Tesseract binary, or EasyOCR)
# ---------------------------------------------------------------------------

try:
    import pytesseract as _pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _pytesseract = None  # type: ignore
    _TESSERACT_AVAILABLE = False

try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
    _easyocr_reader = None  # lazy-init to avoid loading model at import time
except ImportError:
    _easyocr = None  # type: ignore
    _EASYOCR_AVAILABLE = False
    _easyocr_reader = None


def tesseract_available() -> bool:
    return _TESSERACT_AVAILABLE


def easyocr_available() -> bool:
    return _EASYOCR_AVAILABLE


def ocr_with_tesseract(img: "_PILImage.Image") -> str:
    """Extract text using pytesseract (wraps local Tesseract binary).

    Best for: terminal output, stack traces, code editors, log files.
    Poor for: handwriting, low-contrast text, rotated text.

    Install: pip install pytesseract
             brew install tesseract  (macOS)
             apt install tesseract-ocr  (Linux)
    """
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed")
    # PSM 6 = assume uniform block of text — works well for terminal/code screenshots
    config = "--psm 6 -c preserve_interword_spaces=1"
    text = _pytesseract.image_to_string(img, config=config)
    return text.strip()


def ocr_with_easyocr(img: "_PILImage.Image") -> str:
    """Extract text using EasyOCR (local neural OCR, higher accuracy than Tesseract).

    Best for: mixed fonts, non-standard layouts, lower-contrast screenshots.
    Slower than Tesseract; downloads ~200MB model on first use.

    Install: pip install easyocr
    """
    global _easyocr_reader
    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr not installed")
    if _easyocr_reader is None:
        _easyocr_reader = _easyocr.Reader(["en"], gpu=False, verbose=False)
    arr = np.asarray(img.convert("RGB"))
    results = _easyocr_reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join(results).strip()


def extract_text_from_image(
    img: "_PILImage.Image",
    backend: str = "tesseract",
) -> str:
    """Extract text from a document_text image using the specified OCR backend.

    backend options:
      'tesseract' — local Tesseract (fast, good for clean screenshots)
      'easyocr'   — local neural OCR (slower, better on complex layouts)

    Returns empty string on failure rather than raising — callers should
    fall back to downscaling when this returns ''.
    """
    try:
        if backend == "easyocr":
            return ocr_with_easyocr(img)
        return ocr_with_tesseract(img)
    except Exception:
        return ""


def is_ocr_result_valid(text: str, min_chars: int = 30) -> bool:
    """Sanity-check OCR output before replacing an image block with it.

    Rejects:
    - Too short (likely a misclassified image or failed OCR)
    - Mostly non-ASCII garbage (corrupted extraction)
    - Fewer than 3 whitespace-separated tokens (not real text)
    """
    if len(text) < min_chars:
        return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.80:
        return False
    if len(text.split()) < 3:
        return False
    return True
