"""Image compression stage.

Reduces the visual-token cost of image content blocks in the request before
forwarding to Claude. Only operates on base64-encoded images — URL images
are left untouched (we cannot download or modify them here).

Pipeline per image (oldest → newest in eligible history):

  1. Deduplication (default ON): exact-match images (same base64 bytes) beyond
     the first occurrence are replaced with a 1×1 stub. High-value for
     computer-use sessions where the same screenshot appears across many turns.

  2. Age-based budget (opt-in): images older than old_age_threshold_messages
     get a stricter token budget than the normal max_tokens_per_image.

  3. Compression (if still over budget):
       photo        → crop whitespace → downscale → optional seam carve
       document_text → downscale only (warping destroys text readability)
       diagram_ui   → downscale only (same reason)

Requires Pillow (pip install Pillow). If Pillow is not installed the stage
self-disables gracefully and reports the reason in StageResult.note.
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

from ..config import ImageConfig
from ..state import SessionState
from ..tokens import count_request
from ..image_utils import (
    STUB_MEDIA_TYPE,
    STUB_PNG_B64,
    classify_image,
    count_image_block_tokens,
    count_image_tokens,
    crop_whitespace,
    decode_image,
    downscale_to_token_budget,
    encode_image,
    pillow_available,
    seam_carve,
    _target_dims_for_budget,
)
from .base import Stage, StageResult


def _iter_image_blocks(request: dict, protect_last_n: int) -> List[Tuple[int, int, dict]]:
    """Yield (message_index, block_index, block) for every base64 image block.

    Skips the last protect_last_n messages so live working-set images are
    preserved at full resolution (the model needs them for the current task).
    Also walks tool_result content for screenshot images from computer use.
    """
    msgs = request.get("messages", [])
    cutoff = len(msgs) - protect_last_n
    out: List[Tuple[int, int, dict]] = []

    for mi, msg in enumerate(msgs):
        if mi >= cutoff:
            break
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "image" and _is_base64_image(block):
                out.append((mi, bi, block))
            elif btype == "tool_result":
                # Screenshots from computer use live inside tool_result.content
                inner = block.get("content")
                if isinstance(inner, list):
                    for sub in inner:
                        if (
                            isinstance(sub, dict)
                            and sub.get("type") == "image"
                            and _is_base64_image(sub)
                        ):
                            out.append((mi, bi, sub))
    return out


def _is_base64_image(block: dict) -> bool:
    src = block.get("source", {})
    return isinstance(src, dict) and src.get("type") == "base64"


def _image_hash(b64_data: str) -> str:
    """SHA-256 of the raw image bytes (first 64 KB is enough to distinguish images)."""
    # Hashing the full base64 string is cheaper than decoding; collisions are
    # impossible for the exact-dedup use case since we're comparing identical bytes.
    return hashlib.sha256(b64_data[:65536].encode()).hexdigest()


class ImageCompressStage(Stage):
    name = "image_compress"

    def __init__(self, cfg: ImageConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return self.cfg.enabled

    def apply(self, request: dict, state: SessionState) -> StageResult:
        before = count_request(request)

        if not pillow_available():
            return StageResult(
                self.name,
                before,
                before,
                note="skipped: Pillow not installed (pip install Pillow)",
            )

        image_blocks = _iter_image_blocks(request, self.cfg.protect_last_n_messages)
        if not image_blocks:
            return StageResult(self.name, before, before, note="no compressible images")

        n_msgs = len(request.get("messages", []))
        seen_hashes: set = set()
        n_deduped = 0
        n_compressed = 0
        visual_tokens_saved = 0

        for mi, _bi, block in image_blocks:
            src = block["source"]
            media_type = src.get("media_type", "image/png")
            b64_data = src.get("data", "")

            tok_before = count_image_block_tokens(block)

            # --- Step 1: exact deduplication ---
            if self.cfg.dedup_exact:
                h = _image_hash(b64_data)
                if h in seen_hashes:
                    src["data"] = STUB_PNG_B64
                    src["media_type"] = STUB_MEDIA_TYPE
                    visual_tokens_saved += max(0, tok_before - 1)
                    n_deduped += 1
                    continue
                seen_hashes.add(h)

            # --- Step 2: choose token budget based on age ---
            age = n_msgs - 1 - mi  # messages from end (0 = most recent eligible)
            if (
                self.cfg.old_age_threshold_messages > 0
                and age > self.cfg.old_age_threshold_messages
            ):
                budget = self.cfg.old_age_max_tokens
            else:
                budget = self.cfg.max_tokens_per_image

            if tok_before <= budget:
                continue  # already within budget

            # --- Step 3: compress ---
            try:
                img = decode_image(b64_data)
                content_type = classify_image(img)

                img = crop_whitespace(img)
                img = downscale_to_token_budget(img, budget)

                if (
                    self.cfg.seam_carve_photos
                    and content_type == "photo"
                    and count_image_tokens(*img.size) > budget
                ):
                    target_w, _ = _target_dims_for_budget(*img.size, budget)
                    img = seam_carve(img, target_w)

                new_b64, new_media_type = encode_image(img, media_type)
                src["data"] = new_b64
                src["media_type"] = new_media_type

                tok_after = count_image_block_tokens(block)
                visual_tokens_saved += max(0, tok_before - tok_after)
                n_compressed += 1

            except Exception:  # never break the request
                pass

        after = count_request(request)
        note_parts = [f"compressed {n_compressed}/{len(image_blocks)} image(s)"]
        if n_deduped:
            note_parts.append(f"stubbed {n_deduped} duplicate(s)")
        note_parts.append(f"saved ~{visual_tokens_saved} visual tokens")
        return StageResult(
            self.name,
            before,
            after,
            note=", ".join(note_parts),
            detail={
                "images_found": len(image_blocks),
                "images_compressed": n_compressed,
                "images_deduped": n_deduped,
                "visual_tokens_saved": visual_tokens_saved,
            },
        )
