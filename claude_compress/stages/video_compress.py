"""Video compression stage.

Converts video content blocks in the request into text transcriptions before
forwarding to Claude. Claude does not natively accept video, so this stage
extracts OCR text from the scrolling content region of each frame and injects
it as a text block.

Supported source formats inside a content block of type "video":
  {"type": "video", "source": {"type": "path",   "path": "/abs/path/to/video.mp4"}}
  {"type": "video", "source": {"type": "base64",  "media_type": "video/mp4", "data": "<b64>"}}

The stage writes base64 video to a temp file, processes it, and cleans up.

Requires opencv-python and pytesseract (or easyocr):
  pip install opencv-python pytesseract
  brew install tesseract   # macOS
"""
from __future__ import annotations

import base64
import tempfile
import os
from typing import List, Tuple

from ..config import VideoConfig
from ..state import SessionState
from ..tokens import count_request
from .base import Stage, StageResult

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

_OCR_AVAILABLE = True
try:
    import pytesseract as _tess_check  # noqa: F401
except ImportError:
    try:
        import easyocr as _easy_check  # noqa: F401
    except ImportError:
        _OCR_AVAILABLE = False


def _iter_video_blocks(request: dict, protect_last_n: int) -> List[Tuple[int, int, dict]]:
    """Yield (message_index, block_index, block) for every video content block."""
    msgs = request.get("messages", [])
    cutoff = len(msgs) - protect_last_n
    out = []
    for mi, msg in enumerate(msgs):
        if mi >= cutoff:
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "video":
                out.append((mi, bi, block))
    return out


def _resolve_video_path(block: dict) -> str | None:
    """Return a filesystem path to the video, writing base64 data to a temp file."""
    src = block.get("source", {})
    src_type = src.get("type")

    if src_type == "path":
        path = src.get("path", "")
        return path if os.path.isfile(path) else None

    if src_type == "base64":
        data = src.get("data", "")
        if not data:
            return None
        raw = base64.b64decode(data)
        suffix = ".mp4"
        media_type = src.get("media_type", "video/mp4")
        if "webm" in media_type:
            suffix = ".webm"
        elif "avi" in media_type:
            suffix = ".avi"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name

    return None


def _transcribe_video(
    video_path: str,
    cfg: VideoConfig,
    is_temp: bool = False,
) -> str:
    """
    Extract OCR text from the video and return a single transcript string.
    Cleans up temp files when is_temp=True.
    """
    try:
        # Import here so the stage can self-disable when deps are missing
        from detect_roi import detect_roi_from_video, ROITracker, TextAnchorTracker

        cap = _cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""

        # Build tracker
        roi = tuple(cfg.roi) if cfg.roi else None
        tracker = None

        if cfg.anchor_text:
            search_region = tuple(cfg.search_region) if cfg.search_region else None
            tracker = TextAnchorTracker(
                anchor_text=cfg.anchor_text,
                offset_y=cfg.offset_y,
                offset_x=cfg.offset_x,
                content_height=cfg.content_height,
                content_width=cfg.content_width,
                search_region=search_region,
                redetect_every=cfg.redetect_every,
                fuzzy_threshold=cfg.fuzzy_threshold,
            )
        elif roi and cfg.stabilize:
            tracker = ROITracker(roi, max_drift_px=cfg.max_drift_px)
        elif roi is None:
            detected = detect_roi_from_video(video_path)
            if detected:
                roi = detected

        # Lazy import of frame processor
        from video_transcribe_roi import frames_to_audio_via_ocr

        segments = []
        for _frame_idx, text in frames_to_audio_via_ocr(
            cap,
            tracker=tracker,
            roi=roi,
            ocr_every_n_frames=cfg.ocr_every_n_frames,
            change_threshold=cfg.change_threshold,
            backend=cfg.ocr_backend,
            max_frames=cfg.max_frames,
        ):
            segments.append(text)

        cap.release()
        return "\n\n".join(segments)

    finally:
        if is_temp and os.path.exists(video_path):
            os.unlink(video_path)


class VideoCompressStage(Stage):
    name = "video_compress"

    def __init__(self, cfg: VideoConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)

        if not _CV2_AVAILABLE:
            return StageResult(
                self.name, before, before,
                note="skipped: opencv-python not installed (pip install opencv-python)"
            )
        if not _OCR_AVAILABLE:
            return StageResult(
                self.name, before, before,
                note="skipped: no OCR backend found (pip install pytesseract)"
            )

        video_blocks = _iter_video_blocks(request, self.cfg.protect_last_n_messages)
        if not video_blocks:
            return StageResult(self.name, before, before, note="no video blocks")

        n_converted = 0
        n_failed = 0

        for _mi, _bi, block in video_blocks:
            src = block.get("source", {})
            is_temp = src.get("type") == "base64"
            path = _resolve_video_path(block)
            if path is None:
                n_failed += 1
                continue

            transcript = _transcribe_video(path, self.cfg, is_temp=is_temp)
            if not transcript.strip():
                n_failed += 1
                continue

            block.clear()
            block["type"] = "text"
            block["text"] = f"[video transcription]\n{transcript}"
            n_converted += 1

        after = count_request(request)
        note_parts = [f"converted {n_converted}/{len(video_blocks)} video(s) to text"]
        if n_failed:
            note_parts.append(f"{n_failed} failed/empty")
        return StageResult(
            self.name, before, after,
            note=", ".join(note_parts),
            detail={
                "videos_found": len(video_blocks),
                "videos_converted": n_converted,
                "videos_failed": n_failed,
            },
        )
