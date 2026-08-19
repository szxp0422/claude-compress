"""
scene_manager.py

Tracks distinct visual scenes in a screen-recording using perceptual hashing
(pHash) plus active-tab OCR for two-level scene identity.

Level 1 — pHash: each unique editor window layout / application gets its own
  window-level scene.

Level 2 — Tab OCR: within the same window, switching files produces a new
  compound scene key (window_scene_id::filename) so output is routed to
  separate files.

Output structure:
    output_dir/
        scene_index.json          ← all scene metadata
        scene_001_label.jsonl     ← OCR lines for scene 001
        scene_002_label.jsonl     ← OCR lines for scene 002
        ...

Each .jsonl line: {"t": <timestamp_sec>, "text": "<ocr_line>"}
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------

def phash(frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    DCT perceptual hash of a video frame.

    Returns a 1-D boolean numpy array of length hash_size**2.
    Two frames with Hamming distance ≤ ~10 are visually similar.
    """
    gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    side = hash_size * 4
    resized = _cv2.resize(gray, (side, side), interpolation=_cv2.INTER_AREA).astype(np.float32)
    dct = _cv2.dct(resized)
    top = dct[:hash_size, :hash_size].flatten()
    # Compare each component against the mean of non-DC components to normalize
    vals = top[1:]
    mean_val = vals.mean() if len(vals) > 0 else 0.0
    return top > mean_val


def hamming_distance(fp1: np.ndarray, fp2: np.ndarray) -> int:
    """Number of bit positions that differ between two phash arrays."""
    return int(np.sum(fp1 != fp2))


def hash_to_hex(fp: np.ndarray) -> str:
    """Serialize a phash bool array to a compact hex string."""
    bits = np.packbits(fp.astype(np.uint8))
    return bits.tobytes().hex()


def hex_to_hash(hex_str: str, hash_size: int = 8) -> np.ndarray:
    """Deserialize a hex string back to a phash bool array."""
    n_bits = hash_size * hash_size
    raw = bytes.fromhex(hex_str)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n_bits]
    return bits.astype(bool)


# ---------------------------------------------------------------------------
# Scene label detection (window title bar)
# ---------------------------------------------------------------------------

def detect_scene_label(frame: np.ndarray) -> Optional[str]:
    """
    Try to read a short label from the window title bar via OCR.

    Returns a cleaned string (≤30 chars) or None when OCR is unavailable
    or the title bar is empty.
    """
    if not _PIL_AVAILABLE:
        return None
    try:
        import pytesseract
        h, w = frame.shape[:2]
        strip_h = max(8, int(h * 0.025))
        title_strip = frame[:strip_h, :]
        pil = _PIL_Image.fromarray(_cv2.cvtColor(title_strip, _cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil, config="--psm 7").strip()
        text = re.sub(r"[^\w\s\-\.]", "", text).strip()[:30]
        return text if text else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tab-strip active file detection (Level-2 identity)
# ---------------------------------------------------------------------------

def extract_active_tab(
    frame: np.ndarray,
    tab_strip_region: Optional[Tuple[int, int, int, int]] = None,
    tab_strip_height_ratio: float = 0.06,
) -> Optional[str]:
    """
    OCR the tab strip to find the active filename.

    The active tab in most editors is visually distinct — lighter background,
    close button visible. Finds the brightest horizontal band in the tab strip,
    OCRs that region, and extracts a filename pattern.

    Parameters
    ----------
    tab_strip_region : (x, y, w, h) or None
        If you know exactly where the tab strip is, pass it directly for speed.
        If None, searches the top tab_strip_height_ratio of the frame.
    tab_strip_height_ratio : float
        Fraction of frame height to search for tabs when no region is given.
        0.06 = top 6% — covers the tab bar in most editors.

    Returns the active filename string (e.g. "auth.py"), or None if not found.

    Reliability notes:
    - Works well when the active tab is visually distinct (VS Code, JetBrains,
      most tabbed editors).
    - Degrades when tabs are too narrow to show the full filename.
    - Returns None in Zen mode / distraction-free layouts with no tab bar.
    - When None is returned, the caller should keep the current scene.
    """
    if not _PIL_AVAILABLE:
        return None
    try:
        import pytesseract
        h_frame, w_frame = frame.shape[:2]

        if tab_strip_region:
            x, y, w, h = tab_strip_region
            strip = frame[y:y + h, x:x + w]
        else:
            strip_h = max(8, int(h_frame * tab_strip_height_ratio))
            # Skip the very top (window titlebar), search just below it
            y_start = max(0, int(h_frame * 0.03))
            strip = frame[y_start:y_start + strip_h, :]

        # Find the brightest column band — that is the active tab
        gray_strip = _cv2.cvtColor(strip, _cv2.COLOR_BGR2GRAY).astype(float)
        col_brightness = gray_strip.mean(axis=0)

        # Smooth to avoid noise from individual bright pixels
        kernel = np.ones(20) / 20
        smoothed = np.convolve(col_brightness, kernel, mode="same")

        # Find the brightness peak and its left/right extent
        peak_x = int(np.argmax(smoothed))
        peak_val = smoothed[peak_x]
        threshold = peak_val * 0.92

        left = peak_x
        while left > 0 and smoothed[left] >= threshold:
            left -= 1
        right = peak_x
        while right < len(smoothed) - 1 and smoothed[right] >= threshold:
            right += 1

        tab_width = right - left
        if tab_width < 20:
            return None  # too narrow to contain a readable filename

        tab_crop = strip[:, max(0, left):min(strip.shape[1], right)]

        # Scale up for better OCR accuracy on small tab text
        scale = max(1, int(40 / max(tab_crop.shape[0], 1)))
        if scale > 1:
            tab_crop = _cv2.resize(
                tab_crop,
                (tab_crop.shape[1] * scale, tab_crop.shape[0] * scale),
                interpolation=_cv2.INTER_CUBIC,
            )

        pil = _PIL_Image.fromarray(_cv2.cvtColor(tab_crop, _cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(
            pil,
            config=(
                "--psm 7 -c tessedit_char_whitelist="
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789._-/ "
            ),
        ).strip()

        # Extract a filename pattern: word.extension
        match = re.search(r"\b[\w\-]+\.[\w]{1,6}\b", text)
        if match:
            return match.group(0).lower()

        return text[:40] if text else None

    except Exception:
        return None


def build_compound_key(window_scene_id: str, active_file: Optional[str]) -> str:
    """
    Build a compound scene key combining window identity and active file.

    Examples:
        "scene_001::auth.py"
        "scene_001::models.py"
        "scene_002::unknown"   (terminal or window with no readable tab)
    """
    file_part = active_file if active_file else "unknown"
    return f"{window_scene_id}::{file_part}"


# ---------------------------------------------------------------------------
# Scene dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    scene_id: str
    label: str
    fingerprint_hex: str
    file_path: Path
    first_seen: float
    last_seen: float
    fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    visit_count: int = 0
    ocr_line_count: int = 0


# ---------------------------------------------------------------------------
# SceneManager
# ---------------------------------------------------------------------------

class SceneManager:
    """
    Manages scene routing for a video transcription session.

    Detects scene changes via pHash (Level 1) and active-tab OCR (Level 2),
    and writes OCR output to per-scene JSONL files.

    Basic usage:
        manager = SceneManager(output_dir=Path("my_video_scenes"))
        for frame in frames:
            fp = phash(frame)
            if manager.current_scene is None or hamming_distance(fp, manager.current_scene.fingerprint) > manager.match_threshold:
                manager.on_scene_change(frame, timestamp)
            manager.check_file_change(frame, timestamp)   # every N frames
            manager.write_ocr_line(ocr_text, timestamp)
    """

    def __init__(self, output_dir: Path, match_threshold: int = 10):
        """
        Parameters
        ----------
        output_dir : Path
            Directory where scene files and the index are written.
        match_threshold : int
            Maximum Hamming distance (in bits) to consider two pHashes the
            same scene. Lower = stricter matching (more scenes). 10 is a
            good default for screen recordings with stable chrome.
        """
        self.output_dir = Path(output_dir)
        self.match_threshold = match_threshold
        self.scenes: List[Scene] = []
        self.current_scene: Optional[Scene] = None
        self._scene_counter: int = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

        # compound_key (window_id::filename) → Scene
        self._compound_scenes: Dict[str, Scene] = {}
        # optional: set to (x, y, w, h) if you know the tab bar location
        self.tab_strip_region: Optional[tuple] = None
        # check for file changes this often (in sampled-frame units)
        self.check_tab_every: int = 5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_matching_scene(self, fp: np.ndarray) -> Optional[Scene]:
        """Return the closest scene within match_threshold, or None."""
        best: Optional[Scene] = None
        best_dist = self.match_threshold + 1
        for scene in self.scenes:
            if scene.fingerprint is not None:
                d = hamming_distance(fp, scene.fingerprint)
                if d < best_dist:
                    best_dist = d
                    best = scene
        return best

    def _new_scene_id(self) -> str:
        self._scene_counter += 1
        return f"scene_{self._scene_counter:03d}"

    def _make_filename(self, scene_id: str, label: str) -> Path:
        safe = re.sub(r"[^\w]", "_", label)[:40]
        return self.output_dir / f"{scene_id}_{safe}.jsonl"

    def _load_index(self) -> None:
        index_path = self.output_dir / "scene_index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            self._scene_counter = data.get("scene_counter", 0)
            for s in data.get("scenes", []):
                fp = hex_to_hash(s["fingerprint_hex"]) if s.get("fingerprint_hex") else None
                scene = Scene(
                    scene_id=s["scene_id"],
                    label=s.get("label", ""),
                    fingerprint_hex=s.get("fingerprint_hex", ""),
                    file_path=self.output_dir / s["file_name"],
                    first_seen=s.get("first_seen", 0.0),
                    last_seen=s.get("last_seen", 0.0),
                    fingerprint=fp,
                    visit_count=s.get("visit_count", 0),
                    ocr_line_count=s.get("ocr_line_count", 0),
                )
                self.scenes.append(scene)
        except Exception:
            pass  # corrupted index — start fresh

    def _save_index(self) -> None:
        index_path = self.output_dir / "scene_index.json"
        data = {
            "scene_counter": self._scene_counter,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "label": s.label,
                    "fingerprint_hex": s.fingerprint_hex,
                    "file_name": s.file_path.name,
                    "first_seen": s.first_seen,
                    "last_seen": s.last_seen,
                    "visit_count": s.visit_count,
                    "ocr_line_count": s.ocr_line_count,
                }
                for s in self.scenes
            ],
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_ocr_line(self, text: str, timestamp: float) -> None:
        """Append one OCR text line to the current scene's JSONL file."""
        if self.current_scene is None:
            return
        line = json.dumps({"t": round(timestamp, 3), "text": text})
        with open(self.current_scene.file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.current_scene.ocr_line_count += 1

    def on_scene_change(
        self, frame: np.ndarray, timestamp: float, frame_idx: int = 0
    ) -> Scene:
        """
        Two-level matching:
          1. pHash to find the window-level scene.
          2. Tab OCR to find the active file within that window.

        A compound key (window_scene_id::filename) is used so that two
        different files open in the same editor window get separate scene
        files, and returning to a previously seen combination appends to
        the existing file.
        """
        fp = phash(frame)
        window_match = self._find_matching_scene(fp)
        now = time.time()

        if window_match:
            window_scene_id = window_match.scene_id
        else:
            window_scene_id = self._new_scene_id()

        # Level 2: read the active tab filename
        active_file = extract_active_tab(frame, self.tab_strip_region)
        compound_key = build_compound_key(window_scene_id, active_file)

        if compound_key in self._compound_scenes:
            # Returning to a known window + file combination
            scene = self._compound_scenes[compound_key]
            scene.last_seen = now
            scene.visit_count += 1
            scene.fingerprint = fp
            scene.fingerprint_hex = hash_to_hex(fp)
            self.current_scene = scene
            print(
                f"[{timestamp:.1f}s] Returned to {scene.scene_id} "
                f"({scene.label}) visit #{scene.visit_count}"
            )
        else:
            # New window + file combination
            label_parts = []
            if window_match:
                base = window_match.label.split("::")[0]
                label_parts.append(base)
            else:
                label_parts.append(detect_scene_label(frame) or "window")
            if active_file:
                label_parts.append(re.sub(r"[^\w]", "_", active_file))
            label = "_".join(label_parts)[:50]

            new_scene_id = self._new_scene_id()
            file_path = self._make_filename(new_scene_id, label)
            scene = Scene(
                scene_id=new_scene_id,
                label=label,
                fingerprint_hex=hash_to_hex(fp),
                file_path=file_path,
                first_seen=now,
                last_seen=now,
                fingerprint=fp,
                visit_count=1,
            )
            self.scenes.append(scene)
            self._compound_scenes[compound_key] = scene
            self.current_scene = scene
            print(
                f"[{timestamp:.1f}s] New scene {new_scene_id} ({label}) "
                f"[file: {active_file or 'unknown'}]"
            )

        self._save_index()
        return self.current_scene

    def check_file_change(
        self, frame: np.ndarray, timestamp: float
    ) -> bool:
        """
        Detect file changes within the same editor window without a pHash
        scene change. Call this periodically (every check_tab_every sampled
        frames) on normal frames.

        When the user switches files in VS Code without the window layout
        changing, pHash will not fire a scene change. This method catches
        that by re-reading the active tab and comparing to the current scene.

        Returns True if a file change was detected and the scene was updated.
        """
        if self.current_scene is None:
            return False

        active_file = extract_active_tab(frame, self.tab_strip_region)
        if active_file is None:
            return False

        current_label = self.current_scene.label.lower()
        active_normalised = re.sub(r"[^\w]", "_", active_file.lower())

        if active_normalised in current_label:
            return False  # same file, no change

        print(
            f"[{timestamp:.1f}s] File change within window: "
            f"{self.current_scene.label} → {active_file}"
        )
        self.on_scene_change(frame, timestamp)
        return True
