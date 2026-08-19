"""Configuration for the compression pipeline.

Defaults are deliberately CONSERVATIVE. The stages that can silently degrade a
coding session (eigencontext, alias substitution) ship disabled. The stages that
are safe and high-value (dedup of redundant blocks, checkpoint summarisation of
old turns, prompt-cache breakpoints) ship enabled.

Override via environment (CCOMP_*) or a JSON file passed to load_config().
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

_logger = logging.getLogger(__name__)


@dataclass
class JsonConfig:
    # Minify + truncate JSON in tool_result blocks. Enabled by default: minification
    # is lossless and truncation only fires when arrays/strings exceed the limits below.
    enabled: bool = True
    # Only process blocks whose token count meets this minimum (avoid overhead on tiny results).
    min_compress_tokens: int = 80
    # Maximum array items to keep; longer arrays are reduced to head + tail.
    max_array_items: int = 20
    # Maximum characters for any individual string value; longer strings are truncated.
    max_string_chars: int = 500
    # Never modify tool_result blocks in the last N messages (live working set).
    protect_last_n_messages: int = 4


@dataclass
class LogConfig:
    # Collapse repeated lines and truncate long stack traces in tool_result blocks.
    # Enabled by default: both operations are safe and self-gating.
    enabled: bool = True
    # Only process blocks whose token count meets this minimum.
    min_compress_tokens: int = 80
    # Maximum stack frames to keep in a single trace (head + tail).
    max_stack_frames: int = 20
    # Never modify tool_result blocks in the last N messages.
    protect_last_n_messages: int = 4


@dataclass
class HtmlConfig:
    # Strip scripts/styles/nav and convert HTML to text in tool_result blocks.
    # Disabled by default: enable for web-scraping workloads. Disable if the HTML
    # structure itself is the subject of the conversation.
    enabled: bool = False
    # Only process blocks whose token count meets this minimum.
    min_compress_tokens: int = 200
    # Never modify tool_result blocks in the last N messages.
    protect_last_n_messages: int = 4


@dataclass
class ImageConfig:
    # LOSSY (resizes images). Disabled by default — enable when images inflate
    # context and full resolution isn't needed for the current task.
    enabled: bool = False
    # Compress any image that costs more than this many visual tokens.
    # Claude formula: ceil(w/28) * ceil(h/28). A 1920x1080 image ≈ 2691 tokens.
    max_tokens_per_image: int = 1024
    # Never compress images in the last N messages (live working set).
    protect_last_n_messages: int = 4
    # Apply seam carving to PHOTOS still over budget after downscaling.
    # Off by default: adds CPU cost and only safe for photographic content.
    seam_carve_photos: bool = False
    # Stub exact-duplicate images (same base64 data seen more than once in
    # non-protected history). High value for computer-use sessions where the
    # same screenshot recurs across many turns. Lossless after the first copy.
    dedup_exact: bool = True
    # Messages from the end of conversation beyond which an image is considered
    # "old" and gets a stricter budget. 0 = disabled (flat budget everywhere).
    old_age_threshold_messages: int = 0
    # Token budget applied to images older than old_age_threshold_messages.
    # Ignored when old_age_threshold_messages == 0.
    old_age_max_tokens: int = 256

    # OCR extraction for document_text images — replaces image block with text block.
    # Much cheaper than visual tokens for terminal output, code, stack traces.
    # Requires pytesseract (default) or easyocr — see image_utils.py for install.
    # Falls back to downscaling if OCR fails or returns too little text.
    ocr_enabled: bool = False
    # 'tesseract' (fast, local) or 'easyocr' (slower, more accurate)
    ocr_backend: str = "tesseract"
    # minimum character count for OCR output to be accepted (filters noise)
    ocr_min_chars: int = 30

    # Zone segmentation: split mixed images into typed regions and handle each
    # independently (text zones → OCR, image zones → compress). Disabled by
    # default — useful for large screenshots with mixed content (IDE, browser).
    zone_segment: bool = False
    # Minimum image area (pixels) to attempt zone segmentation. Small images
    # are unlikely to contain multiple zones worth splitting.
    zone_min_area: int = 80000   # roughly 320x250px
    # RLSA horizontal gap threshold: larger = more aggressive word merging
    zone_h_threshold: int = 20
    # RLSA vertical gap threshold: larger = more aggressive line merging
    zone_v_threshold: int = 30
    # Minimum area of a zone blob to be kept (filters noise)
    zone_min_zone_area: int = 2000


@dataclass
class DedupConfig:
    enabled: bool = True
    # cosine similarity above which two history blocks are considered dupes
    threshold: float = 0.93
    # never dedup within the last N messages (the live working set)
    protect_last_n_messages: int = 4
    # only consider blocks at least this many tokens (tiny blocks aren't worth it)
    min_block_tokens: int = 40


@dataclass
class CheckpointConfig:
    enabled: bool = True
    # once the conversation exceeds this many *approx* tokens, summarise the
    # oldest turns down to a compact checkpoint
    trigger_tokens: int = 12000
    # always keep this many most-recent messages verbatim
    keep_recent_messages: int = 8
    # target token budget for the generated summary checkpoint
    summary_target_tokens: int = 600
    # only compress if old-bucket tokens exceed summary_target_tokens by at least
    # this multiple; prevents summarising 700 tokens down to 600 (net loss)
    min_compression_ratio: float = 2.0
    # model used for the cheap summarisation side-call
    summarizer_model: str = "claude-haiku-4-5-20251001"


@dataclass
class DeltaConfig:
    # "delta" here = automatic prompt-cache breakpoint insertion on the stable
    # prefix (system + tools + old turns). This reduces COST, not prompt size,
    # and is completely lossless. Safe to leave on.
    enabled: bool = True
    # max cache breakpoints to insert (Anthropic allows up to 4)
    max_breakpoints: int = 4


@dataclass
class EigencontextConfig:
    # LOSSY. Selects a minimum-coverage subset of sentences from blocks that are
    # explicitly tagged as reference material. Off by default: dropping context
    # from a coding agent is dangerous.
    enabled: bool = False
    coverage: float = 0.92  # stop once this fraction of info mass is covered
    # only operate on blocks whose text starts with this marker
    marker: str = "<<REF>>"
    protect_last_n_messages: int = 4


@dataclass
class AliasConfig:
    # LOSSY-ish / risky. Replaces long repeated strings with short aliases and
    # injects a legend. Can confuse the model. Off by default.
    enabled: bool = False
    # minimum occurrences before considering a string for aliasing.
    # must be high enough for substitution savings to exceed legend overhead
    # (header ~24 tok + ~11 tok/entry). For a typical path (~8 tok), break-even
    # is ~6 occurrences. Default 8 gives a comfortable margin.
    min_occurrences: int = 8
    min_length: int = 24
    max_aliases: int = 24
    # never alias-substitute within the last N messages (the live working set),
    # matching the same safety boundary used by dedup
    protect_last_n_messages: int = 4


@dataclass
class VideoConfig:
    # Convert video content blocks to OCR text transcriptions.
    # Disabled by default — enable for recordings/screencasts passed as video blocks.
    # Requires: pip install opencv-python pytesseract && brew install tesseract
    enabled: bool = False
    # Never process video blocks in the last N messages (live working set).
    protect_last_n_messages: int = 4

    # --- ROI / tracker settings ---
    # Fixed content region (x, y, w, h). None = auto-detect or full frame.
    roi: list = None  # type: ignore[assignment]
    # Use optical-flow drift correction when roi is fixed.
    stabilize: bool = False
    max_drift_px: int = 10

    # Text-anchor tracking (re-locates a UI string via OCR to derive the ROI).
    # If set, takes priority over roi/stabilize.
    anchor_text: str = ""
    offset_y: int = 28
    offset_x: int = 0
    content_height: int = 0   # 0 = to frame bottom
    content_width: int = 0    # 0 = to frame right edge
    search_region: list = None  # type: ignore[assignment]
    redetect_every: int = 15
    fuzzy_threshold: float = 0.7

    # --- OCR / frame sampling ---
    ocr_backend: str = "tesseract"   # 'tesseract' or 'easyocr'
    ocr_every_n_frames: int = 5      # run OCR every N frames
    change_threshold: float = 0.15   # min text-difference ratio to emit a segment
    max_frames: int = 0              # 0 = process entire video


@dataclass
class StateMachineConfig:
    # Opt-in. If the client passes a `_fsm` hint in metadata, inject a compact
    # transition table instead of prose workflow instructions.
    enabled: bool = False


@dataclass
class TierConfig:
    # Token thresholds for the dynamic compression tier selector.
    # Sessions below tiny_threshold are passed through unchanged.
    tiny_threshold: int = 2000
    # Sessions below short_threshold get cache breakpoints only (no lossy ops).
    short_threshold: int = 4000


@dataclass
class Config:
    upstream_base_url: str = "https://api.anthropic.com"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8787
    metrics_path: str = "./ccomp_metrics.jsonl"
    log_level: str = "INFO"
    # session store limits — evict old/excess sessions to bound memory use
    max_sessions: int = 500
    session_ttl_seconds: float = 86400.0

    dedup: DedupConfig = field(default_factory=DedupConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    eigencontext: EigencontextConfig = field(default_factory=EigencontextConfig)
    alias: AliasConfig = field(default_factory=AliasConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    tier: TierConfig = field(default_factory=TierConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    json: JsonConfig = field(default_factory=JsonConfig)
    log: LogConfig = field(default_factory=LogConfig)
    html: HtmlConfig = field(default_factory=HtmlConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(dc_cls, data: dict):
    fields = {f for f in dc_cls.__dataclass_fields__}
    unknown = set(data.keys()) - fields
    if unknown:
        _logger.warning(
            "config: unknown keys in %s section: %s (ignored)",
            dc_cls.__name__,
            sorted(unknown),
        )
    return dc_cls(**{k: v for k, v in data.items() if k in fields})


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        for key in ("upstream_base_url", "listen_host", "listen_port",
                    "metrics_path", "log_level", "max_sessions", "session_ttl_seconds"):
            if key in raw:
                setattr(cfg, key, raw[key])
        if "dedup" in raw:
            cfg.dedup = _coerce(DedupConfig, raw["dedup"])
        if "checkpoint" in raw:
            cfg.checkpoint = _coerce(CheckpointConfig, raw["checkpoint"])
        if "delta" in raw:
            cfg.delta = _coerce(DeltaConfig, raw["delta"])
        if "eigencontext" in raw:
            cfg.eigencontext = _coerce(EigencontextConfig, raw["eigencontext"])
        if "alias" in raw:
            cfg.alias = _coerce(AliasConfig, raw["alias"])
        if "state_machine" in raw:
            cfg.state_machine = _coerce(StateMachineConfig, raw["state_machine"])
        if "tier" in raw:
            cfg.tier = _coerce(TierConfig, raw["tier"])
        if "image" in raw:
            cfg.image = _coerce(ImageConfig, raw["image"])
        if "video" in raw:
            cfg.video = _coerce(VideoConfig, raw["video"])
        if "json" in raw:
            cfg.json = _coerce(JsonConfig, raw["json"])
        if "log" in raw:
            cfg.log = _coerce(LogConfig, raw["log"])
        if "html" in raw:
            cfg.html = _coerce(HtmlConfig, raw["html"])

    # env overrides for the few operational knobs
    cfg.upstream_base_url = os.getenv("CCOMP_UPSTREAM", cfg.upstream_base_url)
    cfg.listen_host = os.getenv("CCOMP_HOST", cfg.listen_host)
    cfg.listen_port = int(os.getenv("CCOMP_PORT", cfg.listen_port))
    cfg.metrics_path = os.getenv("CCOMP_METRICS", cfg.metrics_path)
    cfg.log_level = os.getenv("CCOMP_LOG_LEVEL", cfg.log_level)
    cfg.max_sessions = int(os.getenv("CCOMP_MAX_SESSIONS", cfg.max_sessions))
    cfg.session_ttl_seconds = float(
        os.getenv("CCOMP_SESSION_TTL", cfg.session_ttl_seconds)
    )
    return cfg
