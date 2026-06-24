# claude-compress

A token-compression **middleware proxy** for Claude Code (and anything that
speaks the Anthropic Messages API). It sits between the client and
`api.anthropic.com`, compresses the outgoing prompt, forwards it, and expands the
response on the way back. **Claude itself is unchanged** — this is pure
middleware.

```
Claude Code ──▶ claude-compress proxy ──▶ Anthropic API (Claude, unchanged)
                     │                              │
            input pipeline (compress)      response post-process (expand)
```

## What it actually does (and the honest caveats)

The Messages API is **stateless**, so middleware can't "hide" context Claude
needs. Each stage is therefore either *lossless*, *safe-lossy* (only touches
clearly-redundant or explicitly-tagged content), or *opt-in risky*. Defaults are
conservative.

| Stage | Default | Lossy? | What it does |
|---|---|---|---|
| `delta_cache_breakpoints` | **on** | no | Inserts `cache_control` breakpoints on the stable prefix (system, tools, old turns). Cuts **cost** (cached input is discounted), not prompt size. This is the only honest form of "delta encoding" for a stateless API. |
| `semantic_dedup` | **on** | safe | Drops history text blocks whose meaning duplicates an earlier block (cosine ≥ threshold). Never touches the last N messages or tool/image blocks. |
| `checkpoint_compression` | **on** | safe | Once the convo passes a token threshold, folds the oldest turns into one compact summary via a cheap Haiku side-call. Highest-value stage for long sessions. |
| `eigencontext` | off | **lossy** | Greedy max-coverage sentence selection over `<<REF>>`-tagged reference blocks only. Off because dropping context from a coding agent is risky. |
| `alias_substitution` | off | **risky** | Replaces long repeated strings with short aliases + a legend; expands them back in the response. Can confuse the model; marginal token wins. |
| `state_machine` | off | n/a | If the client passes a `metadata._fsm` spec, injects a compact transition table instead of prose workflow text. |

Stages that **need model internals** (true runtime attention pruning, latent-space
prompting) are intentionally **not** here — they can't be built as an add-on.

## Install

```bash
pip install -r requirements.txt
# recommended for real semantic dedup/eigencontext:
pip install sentence-transformers tiktoken
```

Without `sentence-transformers`, embeddings fall back to a deterministic
lexical-hash vector (catches near-duplicates, not deep semantics). Without
`tiktoken`, token counts use a char/word heuristic (fine for relative savings).

## Run

```bash
python -m claude_compress --config config.example.json
# or just: python -m claude_compress   (uses safe defaults)
```

## Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
# your normal ANTHROPIC_API_KEY is forwarded through untouched
claude
```

The proxy reads your API key from the request headers and forwards it; it never
stores credentials and never needs its own key. The checkpoint summariser reuses
the same key for its side-call.

## Observability

Every request appends a line to `ccomp_metrics.jsonl`:

```json
{"session":"ab12…","tokens_in":3875,"tokens_out":776,"saved":3099,"saved_pct":80.0,
 "stages":[{"name":"checkpoint_compression","saved":3025,"note":"folded 15 msgs …"}]}
```

`GET /healthz` reports the active embedding mode and upstream.

## Tests

```bash
python test_pipeline.py      # offline: savings + safety invariants (no network)
python test_integration.py   # proxy against a mock upstream (no real key)
```

`test_pipeline.py` asserts the invariants that matter: the live (most recent)
user turn is never altered, tool schemas stay intact, user/assistant alternation
is preserved, and aliases round-trip.

## Safety model

- The most recent `protect_last_n_messages` are never compressed.
- `tool_use` / `tool_result` / `image` blocks are never modified.
- Any stage that throws is skipped, not fatal — a broken stage can never corrupt
  a request.
- `checkpoint` preserves role alternation when injecting its summary message.

## Tuning

Start with defaults. If you want more aggressive savings on long sessions, lower
`checkpoint.trigger_tokens` and `checkpoint.keep_recent_messages`. Only enable
`eigencontext`/`alias` if you've measured that they don't degrade your task —
watch the metrics file and your own output quality.
