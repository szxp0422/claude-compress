"""Configuration for the compression pipeline.

Defaults are deliberately CONSERVATIVE. The stages that can silently degrade a
coding session (eigencontext, alias substitution) ship disabled. The stages that
are safe and high-value (dedup of redundant blocks, checkpoint summarisation of
old turns, prompt-cache breakpoints) ship enabled.

Override via environment (CCOMP_*) or a JSON file passed to load_config().
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


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


@dataclass
class StateMachineConfig:
    # Opt-in. If the client passes a `_fsm` hint in metadata, inject a compact
    # transition table instead of prose workflow instructions.
    enabled: bool = False


@dataclass
class Config:
    upstream_base_url: str = "https://api.anthropic.com"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8787
    metrics_path: str = "./ccomp_metrics.jsonl"
    log_level: str = "INFO"

    dedup: DedupConfig = field(default_factory=DedupConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    eigencontext: EigencontextConfig = field(default_factory=EigencontextConfig)
    alias: AliasConfig = field(default_factory=AliasConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(dc_cls, data: dict):
    fields = {f for f in dc_cls.__dataclass_fields__}
    return dc_cls(**{k: v for k, v in data.items() if k in fields})


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        for key in ("upstream_base_url", "listen_host", "listen_port",
                    "metrics_path", "log_level"):
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

    # env overrides for the few operational knobs
    cfg.upstream_base_url = os.getenv("CCOMP_UPSTREAM", cfg.upstream_base_url)
    cfg.listen_host = os.getenv("CCOMP_HOST", cfg.listen_host)
    cfg.listen_port = int(os.getenv("CCOMP_PORT", cfg.listen_port))
    cfg.metrics_path = os.getenv("CCOMP_METRICS", cfg.metrics_path)
    cfg.log_level = os.getenv("CCOMP_LOG_LEVEL", cfg.log_level)
    return cfg
