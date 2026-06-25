"""Token shrinkage assertions.

Verifies that each compression stage achieves a minimum reduction ratio on
purpose-built inputs. These are *effectiveness* tests, not safety tests --
safety invariants live in test_pipeline.py.

Run with: python test_shrinkage.py
"""
import sys
from dataclasses import dataclass
from typing import Optional

from claude_compress.config import (
    AliasConfig, CheckpointConfig, DedupConfig, EigencontextConfig,
)
from claude_compress.stages.alias import AliasStage
from claude_compress.stages.checkpoint import CheckpointStage
from claude_compress.stages.dedup import DedupStage
from claude_compress.stages.eigencontext import EigencontextStage
from claude_compress.state import SessionState
from claude_compress.tokens import count_request, count_text


@dataclass
class ShrinkResult:
    stage: str
    tokens_before: int
    tokens_after: int
    saved: int
    ratio: float  # fraction removed, e.g. 0.30 means 30% smaller
    passed: bool
    reason: str


def _req(messages: list, system: str = "") -> dict:
    r: dict = {"model": "claude-sonnet-4-6", "max_tokens": 1024, "messages": messages}
    if system:
        r["system"] = system
    return r


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def check(label: str, before: int, after: int, min_ratio: float) -> ShrinkResult:
    saved = before - after
    ratio = saved / before if before else 0.0
    passed = ratio >= min_ratio
    reason = f"{ratio*100:.1f}% removed (need >= {min_ratio*100:.0f}%)"
    return ShrinkResult(label, before, after, saved, ratio, passed, reason)


# ---------------------------------------------------------------------------
# Stage: dedup
# ---------------------------------------------------------------------------
def test_dedup_shrinkage(min_ratio: float = 0.35) -> ShrinkResult:
    """Near-identical messages should be collapsed to ~1 copy."""
    sentence = (
        "The deployment pipeline uses Kamal to push Docker images to the VPS "
        "and runs health checks before switching traffic. "
    )
    msgs = []
    for i in range(6):
        msgs.append(_msg("user", f"Question {i}. " + sentence * 3))
        msgs.append(_msg("assistant", f"Answer {i}. " + sentence * 2))
    req = _req(msgs)
    before = count_request(req)

    state = SessionState(session_id="dedup-shrink")
    cfg = DedupConfig(enabled=True, threshold=0.82, protect_last_n_messages=2)
    result = DedupStage(cfg).apply(req, state)
    return check("dedup", before, result.tokens_after, min_ratio)


# ---------------------------------------------------------------------------
# Stage: eigencontext
# ---------------------------------------------------------------------------
def test_eigencontext_shrinkage(min_ratio: float = 0.18) -> ShrinkResult:
    """REF-tagged block with redundant sentences should be pruned."""
    redundant = (
        "<<REF>>"
        "The cache layer must never store user secrets. " * 5
        + "Rate limits are 100 rpm per key. "
        + "The cache layer must not store secrets ever. " * 4
        + "Background jobs retry up to three times with backoff. "
        + "Secrets live in Vault, not in the cache. " * 3
    )
    msgs = [
        _msg("user", redundant),
        _msg("assistant", "Understood, I will follow these constraints."),
        _msg("user", "Now implement the rate limiter."),
    ]
    req = _req(msgs)
    before = count_request(req)

    state = SessionState(session_id="eigen-shrink")
    cfg = EigencontextConfig(enabled=True, coverage=0.70, protect_last_n_messages=2)
    result = EigencontextStage(cfg).apply(req, state)
    return check("eigencontext", before, result.tokens_after, min_ratio)


# ---------------------------------------------------------------------------
# Stage: alias
# ---------------------------------------------------------------------------
def test_alias_shrinkage(min_ratio: float = 0.15) -> ShrinkResult:
    """Long repeated path should be aliased, saving at least 15%."""
    lp = "app/services/customer_success/orchestration/billing/platform_runner.rb"
    usages = " ".join([
        f"open {lp}", f"test {lp}", f"lint {lp}", f"deploy {lp}",
        f"revert {lp}", f"edit {lp}", f"read {lp}", f"check {lp}",
        f"diff {lp}", f"blame {lp}",
    ])
    msgs = [
        _msg("user", usages),
        _msg("assistant", f"I'll start with {lp}."),
    ]
    req = _req(msgs)
    before = count_request(req)

    state = SessionState(session_id="alias-shrink")
    cfg = AliasConfig(enabled=True, min_occurrences=4, protect_last_n_messages=0)
    result = AliasStage(cfg).apply(req, state)
    return check("alias", before, result.tokens_after, min_ratio)


# ---------------------------------------------------------------------------
# Stage: checkpoint
# ---------------------------------------------------------------------------
def test_checkpoint_shrinkage(min_ratio: float = 0.40) -> ShrinkResult:
    """Old turns beyond keep_recent should be folded into a summary stub."""
    boiler = (
        "We are building a Rails 7 API backed by Postgres 15 and Sidekiq 7. "
        "Deploys use Kamal on a Hetzner VPS. Tests are RSpec with FactoryBot. "
    )
    msgs = []
    for i in range(10):
        msgs.append(_msg("user", f"Step {i}: {boiler * 4}"))
        msgs.append(_msg("assistant", f"Done step {i}. " + boiler * 2))
    msgs.append(_msg("user", "What is the next step?"))
    req = _req(msgs, system="You are a senior Rails engineer.")
    before = count_request(req)

    state = SessionState(session_id="ckpt-shrink")
    cfg = CheckpointConfig(
        enabled=True,
        trigger_tokens=500,
        keep_recent_messages=4,
        summary_target_tokens=300,
    )
    # Use extractive fallback (no summarize_fn) -- measures worst-case savings.
    result = CheckpointStage(cfg, summarize_fn=None).apply(req, state)
    return check("checkpoint", before, result.tokens_after, min_ratio)


# ---------------------------------------------------------------------------
# End-to-end: full pipeline
# ---------------------------------------------------------------------------
def test_pipeline_shrinkage(min_ratio: float = 0.40) -> ShrinkResult:
    """Full pipeline on a realistic bloated conversation."""
    from claude_compress.config import Config
    from claude_compress.pipeline import Pipeline

    sentence = (
        "The deployment pipeline uses Kamal to push Docker images to the VPS. "
    )
    ref = (
        "<<REF>>The cache layer must never store user secrets. " * 5
        + "Rate limits are 100 rpm. Background jobs retry 3 times. "
        + "The cache layer must not store secrets ever. " * 3
    )
    lp = "app/services/customer_success/orchestration/billing/platform_runner.rb"
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"Turn {i}. " + sentence * 5))
        msgs.append(_msg("assistant", f"Answer {i}. " + sentence * 3))
    msgs.append(_msg("user", ref))
    msgs.append(_msg("assistant", "Understood."))
    msgs.append(_msg("user", f"Edit {lp} and re-check {lp} and then test {lp}."))
    msgs.append(_msg("assistant", "What change?"))
    msgs.append(_msg("user", "Add retry with exponential backoff."))

    req = _req(msgs, system="You are a senior Rails engineer.")
    before = count_request(req)

    cfg = Config()
    cfg.eigencontext.enabled = True
    cfg.alias.enabled = True
    cfg.checkpoint.trigger_tokens = 800
    cfg.checkpoint.keep_recent_messages = 4

    state = SessionState(session_id="e2e-shrink")
    new_req, _, tok_in, tok_out = Pipeline(cfg, summarize_fn=None).run(req, state)
    after = tok_out
    return check("pipeline (e2e)", before, after, min_ratio)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = [
    test_dedup_shrinkage,
    test_eigencontext_shrinkage,
    test_alias_shrinkage,
    test_checkpoint_shrinkage,
    test_pipeline_shrinkage,
]


def main():
    print(f"{'stage':<24} {'before':>7} {'after':>7} {'saved':>6}  {'result'}")
    print("-" * 70)
    failures = []
    for fn in TESTS:
        r = fn()
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.stage:<24} {r.tokens_before:>7} {r.tokens_after:>7} {r.saved:>6}  {status}  {r.reason}")
        if not r.passed:
            failures.append(r)
    print("-" * 70)
    if failures:
        print(f"\n{len(failures)} shrinkage test(s) FAILED:")
        for f in failures:
            print(f"  {f.stage}: {f.reason}")
        sys.exit(1)
    else:
        print("\nAll shrinkage tests passed.")


if __name__ == "__main__":
    main()
