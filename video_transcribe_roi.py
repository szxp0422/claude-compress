"""
video_transcribe_roi.py

Transcribe a screen-recording to text by OCR-ing the scrolling content region
frame-by-frame.

Tracker modes
-------------
  Fixed ROI   (--roi X Y W H)            — fastest, use when camera is stable
  Optical-flow (--roi ... --stabilize)   — corrects small drift in the UI chrome
  Auto-ROI    (--auto-roi)               — samples frames to guess content area
  Anchor text (--anchor TEXT --offset-y) — re-locates anchor via OCR each N frames

Output modes
------------
  Plain text  (default)                  — deduped segments to stdout or --output
  Scene-split (--output-dir DIR)         — separate JSONL per scene + index.json;
                                           uses pHash + tab OCR for scene identity

Usage examples
--------------
    python video_transcribe_roi.py recording.mp4 --roi 0 30 1920 1050
    python video_transcribe_roi.py recording.mp4 --anchor "VS Code" --offset-y 28
    python video_transcribe_roi.py recording.mp4 --auto-roi --output transcript.txt
    python video_transcribe_roi.py recording.mp4 --anchor "VS Code" --output-dir out/
    python video_transcribe_roi.py recording.mp4 \\
        --anchor "VS Code" --tab-region 0 40 1200 28 --check-tab-every 3
"""
from __future__ import annotations

import argparse
import difflib
import sys
from typing import Optional

import cv2

from detect_roi import detect_roi_from_video, ROITracker, TextAnchorTracker


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _ocr_frame(frame, backend: str = "tesseract") -> str:
    """Run OCR on a single frame crop. Returns raw text string."""
    if backend == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(frame, detail=0)
        return "\n".join(results)

    import pytesseract
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return pytesseract.image_to_string(pil, config="--psm 6")


def _text_changed(prev: str, curr: str, threshold: float = 0.15) -> bool:
    """Return True if the two strings differ enough to be worth emitting."""
    if not prev:
        return bool(curr.strip())
    ratio = difflib.SequenceMatcher(None, prev, curr).ratio()
    return (1.0 - ratio) >= threshold


def _safe_crop(frame, roi):
    """Clamp roi to frame bounds and return crop, or None if degenerate."""
    x, y, w, h = roi
    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = min(w, fw - x)
    h = min(h, fh - y)
    if w <= 0 or h <= 0:
        return None
    return frame[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# Core frame-processing loop (plain-text / stdout mode)
# ---------------------------------------------------------------------------

def frames_to_audio_via_ocr(
    cap: cv2.VideoCapture,
    tracker,
    roi: Optional[tuple],
    ocr_every_n_frames: int = 5,
    change_threshold: float = 0.15,
    backend: str = "tesseract",
    max_frames: Optional[int] = None,
):
    """
    Iterate over video frames and yield text segments when the content changes.

    Yields (frame_idx, text) pairs for each unique screen state encountered.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Opened video capture object positioned at the desired start frame.
    tracker : ROITracker | TextAnchorTracker | None
        Active tracker, or None to use the fixed roi directly.
    roi : (x, y, w, h) or None
        Fallback ROI when tracker is None. Must be set if tracker is None.
    ocr_every_n_frames : int
        Only run OCR every N frames (tracker.update is still called every frame).
    change_threshold : float
        Minimum text difference ratio to emit a new segment (0.0–1.0).
    backend : str
        'tesseract' or 'easyocr'.
    max_frames : int or None
        Stop after this many frames. None = process to end.
    """
    prev_text = ""
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        if tracker:
            adjusted_roi = tracker.update(frame)
            if adjusted_roi is None:
                frame_idx += 1
                continue  # anchor not found yet
        else:
            adjusted_roi = roi

        if frame_idx % ocr_every_n_frames == 0 and adjusted_roi is not None:
            crop = _safe_crop(frame, adjusted_roi)
            if crop is not None:
                try:
                    text = _ocr_frame(crop, backend=backend).strip()
                    if text and _text_changed(prev_text, text, change_threshold):
                        yield frame_idx, text
                        prev_text = text
                except Exception:
                    pass

        frame_idx += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Transcribe a screen-recording using OCR on the content ROI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("video", help="Path to input video file")

    # ROI source (mutually exclusive)
    roi_group = ap.add_mutually_exclusive_group()
    roi_group.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Fixed content region (pixels)"
    )
    roi_group.add_argument(
        "--auto-roi", action="store_true",
        help="Auto-detect content region by comparing sampled frames"
    )

    # Optical-flow stabilisation
    ap.add_argument(
        "--stabilize", action="store_true",
        help="Correct small UI drift with optical flow (only with --roi)"
    )
    ap.add_argument(
        "--max-drift", type=int, default=10,
        help="Max pixels per frame to allow drift correction (default 10)"
    )

    # Text-anchor tracking
    ap.add_argument(
        "--anchor", default=None,
        help="Text string to find in static UI (e.g. 'VS Code'). "
             "If set, uses text-anchor tracking instead of fixed ROI."
    )
    ap.add_argument(
        "--offset-y", type=int, default=28,
        help="Pixels below anchor text where content begins (default 28)"
    )
    ap.add_argument(
        "--offset-x", type=int, default=0,
        help="Pixels right of anchor left edge where content begins (default 0)"
    )
    ap.add_argument(
        "--search-region", nargs=4, type=int,
        metavar=("X", "Y", "W", "H"),
        help="Limit anchor search to this sub-region (speeds up OCR)"
    )
    ap.add_argument(
        "--redetect-every", type=int, default=15,
        help="Re-run anchor OCR every N frames (default 15)"
    )
    ap.add_argument(
        "--fuzzy", type=float, default=0.7,
        help="Anchor match threshold 0.0-1.0 (default 0.7)"
    )
    ap.add_argument("--content-height", type=int, default=None,
                    help="Fixed pixel height of content region (anchor mode)")
    ap.add_argument("--content-width", type=int, default=None,
                    help="Fixed pixel width of content region (anchor mode)")

    # OCR settings
    ap.add_argument(
        "--ocr-every", type=int, default=5,
        help="Run OCR every N frames (default 5)"
    )
    ap.add_argument(
        "--change-threshold", type=float, default=0.15,
        help="Min text difference ratio to emit a new segment (default 0.15)"
    )
    ap.add_argument(
        "--backend", choices=["tesseract", "easyocr"], default="tesseract",
        help="OCR backend (default: tesseract)"
    )
    ap.add_argument(
        "--start-frame", type=int, default=0,
        help="First frame to process (default 0)"
    )
    ap.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames (default: whole video)"
    )

    # Plain-text output
    ap.add_argument(
        "--output", default=None,
        help="Write plain-text transcription to this file (default: stdout). "
             "Ignored when --output-dir is set."
    )
    ap.add_argument(
        "--timestamps", action="store_true",
        help="Prefix each segment with its frame number (plain-text mode)"
    )

    # Scene-managed output
    ap.add_argument(
        "--output-dir", default=None,
        help="Write per-scene JSONL files to this directory (enables scene tracking)."
    )
    ap.add_argument(
        "--match-threshold", type=int, default=10,
        help="pHash Hamming distance threshold for scene matching (default 10)"
    )
    ap.add_argument(
        "--phash-every", type=int, default=30,
        help="Check for pHash scene changes every N frames (default 30)"
    )

    # Tab-strip detection (Change 2b)
    ap.add_argument(
        "--tab-region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help=(
            "Pixel region of the tab strip (x y w h). "
            "If omitted, searches the top 6%% of the frame automatically. "
            "Providing this speeds up tab detection significantly."
        ),
    )
    ap.add_argument(
        "--check-tab-every", type=int, default=5,
        help="Check for file changes every N sampled frames (default 5).",
    )

    return ap


def main():
    ap = _build_arg_parser()
    args = ap.parse_args()

    # --- Resolve ROI ---
    roi = tuple(args.roi) if args.roi else None

    if args.auto_roi and roi is None:
        print("Auto-detecting content ROI...", file=sys.stderr)
        roi = detect_roi_from_video(args.video)
        if roi:
            print(f"  detected ROI: {roi}", file=sys.stderr)
        else:
            print("  WARNING: auto-detect failed — processing full frame", file=sys.stderr)

    # --- Build tracker ---
    if args.anchor:
        search_region = tuple(args.search_region) if args.search_region else None
        tracker = TextAnchorTracker(
            anchor_text=args.anchor,
            offset_y=args.offset_y,
            offset_x=args.offset_x,
            content_height=args.content_height if args.content_height else (roi[3] if roi else None),
            content_width=args.content_width if args.content_width else (roi[2] if roi else None),
            search_region=search_region,
            redetect_every=args.redetect_every,
            fuzzy_threshold=args.fuzzy,
        )
        print(f"Using text-anchor tracking: '{args.anchor}'", file=sys.stderr)
    else:
        tracker = ROITracker(roi, max_drift_px=args.max_drift) if (args.stabilize and roi) else None

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.video}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"Video: {total} frames @ {fps:.1f} fps  ({total / fps:.1f}s)",
        file=sys.stderr,
    )

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # ================================================================
    # Scene-managed mode (--output-dir)
    # ================================================================
    if args.output_dir:
        from pathlib import Path
        from scene_manager import SceneManager, phash, hamming_distance

        manager = SceneManager(
            output_dir=Path(args.output_dir),
            match_threshold=args.match_threshold,
        )

        # Change 2c: wire new arguments into the manager
        if args.tab_region:
            manager.tab_strip_region = tuple(args.tab_region)
        manager.check_tab_every = args.check_tab_every

        frame_step = args.ocr_every
        frame_idx = 0
        prev_text = ""

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break

                timestamp = (args.start_frame + frame_idx) / fps

                # --- Level-1 scene detection via pHash ---
                if frame_idx % args.phash_every == 0:
                    if manager.current_scene is None:
                        manager.on_scene_change(frame, timestamp, frame_idx)
                    else:
                        fp = phash(frame)
                        if (
                            manager.current_scene.fingerprint is None
                            or hamming_distance(fp, manager.current_scene.fingerprint)
                            > manager.match_threshold
                        ):
                            manager.on_scene_change(frame, timestamp, frame_idx)

                # --- ROI tracking ---
                if tracker:
                    adjusted_roi = tracker.update(frame)
                    if adjusted_roi is None:
                        frame_idx += 1
                        continue  # anchor not found yet
                else:
                    adjusted_roi = roi

                # Change 2a: Periodically check for file changes within the same
                # editor window. This catches switching files in VS Code without
                # a pHash scene change.
                if frame_idx % (frame_step * manager.check_tab_every) == 0:
                    manager.check_file_change(frame, timestamp)

                # --- OCR on sampled frames ---
                if frame_idx % frame_step == 0 and manager.current_scene:
                    if adjusted_roi is not None:
                        crop = _safe_crop(frame, adjusted_roi)
                        if crop is not None:
                            try:
                                text = _ocr_frame(crop, backend=args.backend).strip()
                                if text and _text_changed(prev_text, text, args.change_threshold):
                                    manager.write_ocr_line(text, timestamp)
                                    prev_text = text
                            except Exception:
                                pass

                frame_idx += 1
        finally:
            cap.release()

        n_scenes = len(manager.scenes)
        print(
            f"Done. {n_scenes} scene(s) written to {args.output_dir}",
            file=sys.stderr,
        )
        return

    # ================================================================
    # Plain-text mode (stdout or --output)
    # ================================================================
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    try:
        for frame_idx, text in frames_to_audio_via_ocr(
            cap,
            tracker=tracker,
            roi=roi,
            ocr_every_n_frames=args.ocr_every,
            change_threshold=args.change_threshold,
            backend=args.backend,
            max_frames=args.max_frames,
        ):
            abs_frame = args.start_frame + frame_idx
            if args.timestamps:
                ts = abs_frame / fps
                out.write(f"[frame {abs_frame}  {ts:.2f}s]\n")
            out.write(text)
            out.write("\n\n")
            out.flush()
    finally:
        cap.release()
        if args.output:
            out.close()
            print(f"Transcription written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
