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
