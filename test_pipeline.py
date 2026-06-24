"""Offline smoke test: build a long, redundant conversation and run the pipeline
with all stages enabled (including the normally-off ones) to verify safety and
measure savings. No network required -- uses local fallbacks for embeddings and
summarisation.
"""
import json

from claude_compress.config import Config
from claude_compress.pipeline import Pipeline
from claude_compress.state import SessionState
from claude_compress.tokens import count_request


def build_conversation():
    boiler = (
        "Here is the full project README. The project is a Rails app using "
        "Postgres and Sidekiq. We deploy via Kamal to a single VPS. The test "
        "suite uses RSpec and runs in CI on every push to main. " * 6
    )
    msgs = []
    # several old turns, with the same boilerplate re-pasted (redundant)
    for i in range(8):
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"Turn {i} question.\n{boiler}"},
        ]})
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"Turn {i} answer with some reasoning. " * 20},
        ]})
    # a reference block tagged for eigencontext, with repetitive sentences
    ref = "<<REF>>" + ("The cache layer must never store secrets. " * 4) + \
        "Rate limits are 100 rpm. The cache layer must not store secrets. " + \
        "Background jobs retry three times. Secrets live in the vault. " * 3
    msgs.append({"role": "user", "content": [{"type": "text", "text": ref}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "Understood."}]})
    # repeated long path for alias stage
    longpath = "app/services/customer_success/orchestration/platform_runner.rb"
    msgs.append({"role": "user", "content": [{"type": "text", "text":
        f"Edit {longpath} then run {longpath} and re-check {longpath} plus {longpath}."}]})
    # live final turn (must be protected)
    msgs.append({"role": "assistant", "content": [{"type": "text",
        "text": "What change do you want in the runner?"}]})  # noqa
    msgs.append({"role": "user", "content": [{"type": "text",
        "text": "Add retry with exponential backoff to the runner."}]})
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "system": "You are a senior Rails engineer. Be terse.",
        "tools": [{"name": "edit_file", "description": "edit a file",
                   "input_schema": {"type": "object", "properties": {}}}],
        "messages": msgs,
    }


def main():
    cfg = Config()
    # turn ON the normally-off stages to exercise them
    cfg.eigencontext.enabled = True
    cfg.alias.enabled = True
    cfg.checkpoint.trigger_tokens = 1000  # force checkpoint to fire
    cfg.checkpoint.keep_recent_messages = 6

    req = build_conversation()
    before = count_request(req)

    state = SessionState(session_id="test")
    pipe = Pipeline(cfg, summarize_fn=None)  # None -> local extractive fallback
    new_req, results, tok_in, tok_out = pipe.run(req, state)

    print(f"tokens in  : {tok_in}")
    print(f"tokens out : {tok_out}  ({100*(tok_in-tok_out)/tok_in:.1f}% smaller)")
    print("-" * 60)
    for r in results:
        print(f"{r.name:24s} {r.tokens_before:6d} -> {r.tokens_after:6d}"
              f"  saved {r.saved:5d}  | {r.note}")
    print("-" * 60)

    # SAFETY CHECKS
    msgs = new_req["messages"]
    # 1. last user message preserved verbatim
    assert msgs[-1]["content"][0]["text"] == \
        "Add retry with exponential backoff to the runner.", "live turn corrupted!"
    # 2. tool schema intact
    assert new_req["tools"][0]["name"] == "edit_file", "tool corrupted!"
    # 3. role alternation valid (no two same-role in a row)
    roles = [m["role"] for m in msgs]
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"role alternation broken: {roles}"
    # 4. alias stage round-trips when run in isolation on repetitive input
    from claude_compress.stages.alias import AliasStage
    from claude_compress.config import AliasConfig
    lp = "app/services/customer_success/orchestration/platform_runner.rb"
    alias_req = {"messages": [{"role": "user", "content": [{"type": "text",
        "text": f"open {lp}; test {lp}; lint {lp}; deploy {lp}; revert {lp}"}]}]}
    astate = SessionState(session_id="alias")
    ares = AliasStage(AliasConfig(enabled=True)).apply(alias_req, astate)
    assert astate.alias_legend, "expected aliases in isolation"
    assert ares.saved > 0, "alias stage should save tokens here"
    # round-trip: expanding an aliased response restores the original path
    from claude_compress.postprocess.responses import _expand
    alias = next(iter(astate.alias_legend))
    assert _expand(f"done with {alias}", astate.alias_legend) == f"done with {lp}"
    print("alias legend:", json.dumps(astate.alias_legend, indent=0)[:200])
    print(f"alias stage saved {ares.saved} tokens in isolation")
    print("roles:", roles)
    print("\nALL SAFETY CHECKS PASSED")


if __name__ == "__main__":
    main()
