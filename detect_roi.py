"""
detect_roi.py

Region-of-interest detection and tracking for screen-recording videos.

Provides:
  detect_roi_from_video  — auto-detect the content region from a video file
  ROITracker             — hold a fixed ROI with optional optical-flow drift correction
  TextAnchorTracker      — locate a text string in each frame to derive the ROI
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ROITracker
# ---------------------------------------------------------------------------

class ROITracker:
    """
    Tracks a fixed ROI across frames, compensating for small UI drift.

    For perfectly stable recordings use max_drift_px=0 (no tracking overhead).
    For wobbly recordings where the window can shift a few pixels between frames,
    set max_drift_px to the expected maximum per-frame shift.

    Usage:
        tracker = ROITracker(roi=(100, 30, 1200, 900), max_drift_px=5)
        for frame in frames:
            x, y, w, h = tracker.update(frame)
            crop = frame[y:y+h, x:x+w]
    """

    def __init__(self, roi: tuple, max_drift_px: int = 10):
        """
        Parameters
        ----------
        roi : (x, y, w, h)
        max_drift_px : int
            Maximum pixels per frame to allow drift correction.
            0 = rigid ROI, no optical flow.
        """
        self.roi = roi
        self.max_drift_px = max_drift_px
        self._current = list(roi)
        self._prev_gray = None

    def update(self, frame) -> tuple:
        """Process one frame. Returns current (x, y, w, h)."""
        if self.max_drift_px == 0:
            return tuple(self._current)

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3
            else frame
        )

        if self._prev_gray is not None:
            x, y, w, h = self._current
            fh, fw = gray.shape
            # Sample four corner points in the UI chrome just outside the ROI
            pts = np.float32([
                [max(0, x - 4),     max(0, y - 4)],
                [min(fw - 1, x + w + 4), max(0, y - 4)],
                [max(0, x - 4),     min(fh - 1, y + h + 4)],
                [min(fw - 1, x + w + 4), min(fh - 1, y + h + 4)],
            ]).reshape(-1, 1, 2)

            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None
            )
            if new_pts is not None and status is not None:
                good = status.ravel() == 1
                if good.any():
                    deltas = (new_pts - pts)[good, 0, :]
                    dx = float(np.median(deltas[:, 0]))
                    dy = float(np.median(deltas[:, 1]))
                    dx = max(-self.max_drift_px, min(self.max_drift_px, dx))
                    dy = max(-self.max_drift_px, min(self.max_drift_px, dy))
                    self._current[0] = max(0, int(self._current[0] + dx))
                    self._current[1] = max(0, int(self._current[1] + dy))

        self._prev_gray = gray
        return tuple(self._current)


# ---------------------------------------------------------------------------
# detect_roi_from_video
# ---------------------------------------------------------------------------

def detect_roi_from_video(video_path: str, sample_frames: int = 10):
    """
    Auto-detect the scrolling content ROI by comparing sampled frames.

    Pixels that change frequently → scrolling content region.
    Pixels that never change    → static UI chrome (excluded).

    Returns (x, y, w, h) or None if detection fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    step = max(1, total // (sample_frames + 1))
    grays = []
    for i in range(1, sample_frames + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(grays) < 2:
        return None

    diff_acc = np.zeros_like(grays[0], dtype=np.float32)
    for i in range(1, len(grays)):
        diff_acc += cv2.absdiff(grays[i], grays[i - 1]).astype(np.float32)

    _, mask = cv2.threshold(
        diff_acc, max(1.0, diff_acc.mean() * 0.5), 255, cv2.THRESH_BINARY
    )
    mask = mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    fh, fw = grays[0].shape
    x1 = max(0, int(xs.min()) - 5)
    y1 = max(0, int(ys.min()) - 5)
    x2 = min(fw, int(xs.max()) + 5)
    y2 = min(fh, int(ys.max()) + 5)

    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# TextAnchorTracker
# ---------------------------------------------------------------------------

class TextAnchorTracker:
    """
    Tracks a scrolling ROI by locating a known anchor string in each frame.

    The anchor is a piece of text you know appears in a static part of the UI
    — window title, tab name, menu item, fixed header. The content region is
    derived from the anchor's position using calibrated offsets.

    Usage:
        tracker = TextAnchorTracker(
            anchor_text="VS Code",
            offset_y=30,
            content_height=600,
            content_width=900,
        )
        for frame in frames:
            roi = tracker.update(frame)
            if roi:
                x, y, w, h = roi
                crop = frame[y:y+h, x:x+w]
    """

    def __init__(
        self,
        anchor_text: str,
        offset_y: int,
        offset_x: int = 0,
        content_height=None,
        content_width=None,
        search_region=None,
        redetect_every: int = 15,
        fuzzy_threshold: float = 0.7,
    ):
        """
        Parameters
        ----------
        anchor_text : str
            Text to find in each frame. Case-insensitive. Keep it short
            and unique — a tab title works better than a common word.
        offset_y : int
            Pixels below the bottom edge of the anchor text where the
            scrolling content begins.
        offset_x : int
            Pixels right of the anchor text's left edge where content begins.
            Use 0 if content starts at the same x position as the anchor.
        content_height : int or None
            Fixed pixel height of the content region. None = to frame bottom.
        content_width : int or None
            Fixed pixel width of the content region. None = to frame right edge.
        search_region : (x, y, w, h) or None
            Limit Tesseract search to a sub-region of the frame. Use when
            you know roughly where the anchor lives — cuts OCR time by 90%.
            Example: (0, 0, 400, 60) searches only the top-left strip.
        redetect_every : int
            Re-run OCR to find the anchor every N frames. Between detections
            the last known position is held. Lower = more accurate, slower.
        fuzzy_threshold : float
            Minimum character match ratio to accept as the anchor. 1.0 = exact
            only, 0.6 = allows more OCR noise. 0.7 is a safe default.
        """
        self.anchor_text = anchor_text.lower().strip()
        self.offset_y = offset_y
        self.offset_x = offset_x
        self.content_height = content_height
        self.content_width = content_width
        self.search_region = search_region
        self.redetect_every = redetect_every
        self.fuzzy_threshold = fuzzy_threshold

        self._last_roi = None
        self._last_anchor_box = None
        self._frame_count = 0
        self._miss_count = 0

    def _fuzzy_match(self, candidate: str) -> bool:
        candidate = candidate.lower().strip()
        anchor = self.anchor_text
        if anchor in candidate:
            return True
        if not anchor:
            return False
        matches = sum(1 for c in anchor if c in candidate)
        return (matches / len(anchor)) >= self.fuzzy_threshold

    def _find_anchor_in_frame(self, frame):
        import pytesseract
        from PIL import Image

        if self.search_region:
            sx, sy, sw, sh = self.search_region
            search_crop = frame[sy:sy + sh, sx:sx + sw]
            x_offset, y_offset = sx, sy
        else:
            search_crop = frame
            x_offset, y_offset = 0, 0

        pil_img = Image.fromarray(cv2.cvtColor(search_crop, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(
            pil_img,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        words = []
        for i in range(len(data["text"])):
            conf = int(data["conf"][i])
            word = data["text"][i].strip()
            if conf > 30 and word:
                wx = data["left"][i] + x_offset
                wy = data["top"][i] + y_offset
                ww = data["width"][i]
                wh = data["height"][i]
                words.append((word, wx, wy, ww, wh))

        anchor_words = self.anchor_text.split()
        n_anchor = len(anchor_words)

        for i in range(len(words) - n_anchor + 1):
            group = words[i:i + n_anchor]
            combined = " ".join(w[0] for w in group)
            if self._fuzzy_match(combined):
                x1 = min(w[1] for w in group)
                y1 = min(w[2] for w in group)
                x2 = max(w[1] + w[3] for w in group)
                y2 = max(w[2] + w[4] for w in group)
                return (x1, y1, x2 - x1, y2 - y1)

        return None

    def _roi_from_anchor(self, anchor_box, frame_shape):
        ax, ay, aw, ah = anchor_box
        h_frame, w_frame = frame_shape[:2]

        content_x = max(0, ax + self.offset_x)
        content_y = max(0, ay + ah + self.offset_y)
        content_w = self.content_width or (w_frame - content_x)
        content_h = self.content_height or (h_frame - content_y)

        content_w = min(content_w, w_frame - content_x)
        content_h = min(content_h, h_frame - content_y)

        return (content_x, content_y, content_w, content_h)

    def update(self, frame):
        """
        Process one frame. Returns current content ROI as (x, y, w, h),
        or None if the anchor has never been found.

        Call for every frame to maintain accurate tracking even on frames
        you do not OCR for content.
        """
        self._frame_count += 1
        should_detect = (
            self._frame_count % self.redetect_every == 0
            or self._last_anchor_box is None
        )

        if should_detect:
            anchor_box = self._find_anchor_in_frame(frame)
            if anchor_box:
                self._last_anchor_box = anchor_box
                self._last_roi = self._roi_from_anchor(anchor_box, frame.shape)
                self._miss_count = 0
            else:
                self._miss_count += 1
                # After 5 consecutive misses, fall back to searching the full frame
                if self._miss_count > 5:
                    self.search_region = None

        return self._last_roi

    def calibrate(self, frame) -> dict:
        """
        Run on a single sample frame to verify your anchor string and offsets
        before processing the full video. Returns a dict with found/not found,
        anchor bounding box, and derived ROI. Draw these on screen with the
        calibrate_anchor.py helper.
        """
        anchor_box = self._find_anchor_in_frame(frame)
        if anchor_box is None:
            return {
                "found": False,
                "anchor_text": self.anchor_text,
                "tip": "Try a shorter anchor string or lower fuzzy_threshold",
            }
        roi = self._roi_from_anchor(anchor_box, frame.shape)
        return {
            "found": True,
            "anchor_box": anchor_box,
            "derived_roi": roi,
            "anchor_text": self.anchor_text,
        }
