# claude-compress

A token-compression **middleware proxy** for Claude Code (and any client that speaks the Anthropic Messages API). It sits between the client and `api.anthropic.com`, compresses the outgoing prompt, forwards it, and expands the response on the way back. Claude itself is unchanged — this is pure middleware.

```
Claude Code ──▶ claude-compress proxy ──▶ Anthropic API (Claude, unchanged)
                     │                              │
            input pipeline (compress)      response post-process (expand)
```

## How it works

The Messages API is stateless, so middleware cannot hide context from Claude. Each compression stage is therefore either *lossless*, *safe-lossy* (only touches clearly-redundant or explicitly-tagged content), or *opt-in risky*. Defaults are conservative.

| Stage | Default | Lossy? | What it does |
|---|---|---|---|
| `delta_cache_breakpoints` | **on** | no | Inserts `cache_control` breakpoints on the stable prefix (system, tools, old turns). Cuts **cost** (cached input is billed at a discount), not prompt size. |
| `semantic_dedup` | **on** | safe | Drops history text blocks whose meaning duplicates an earlier block (cosine similarity ≥ threshold). Never touches the last N messages or tool/image blocks. |
| `checkpoint_compression` | **on** | safe | Once a conversation passes a token threshold, folds the oldest turns into one compact summary via a cheap Haiku side-call. Highest-value stage for long sessions. |
| `eigencontext` | off | **lossy** | Greedy max-coverage sentence selection over `<<REF>>`-tagged reference blocks only. Disabled by default: dropping context from a coding agent is risky. |
| `alias_substitution` | off | **risky** | Replaces long repeated strings with short aliases and a legend; expands them back in the response. Can confuse the model; marginal token wins on most sessions. |
| `state_machine` | off | n/a | If the client passes a `metadata._fsm` spec, injects a compact transition table instead of prose workflow text. |

Stages that would require model internals (runtime attention pruning, latent-space prompting) are intentionally absent — they cannot be built as an API-layer add-on.

## Requirements

- Python 3.9+
- `fastapi`, `uvicorn`, `httpx`, `numpy`, `tiktoken` (all in `requirements.txt`)
- `sentence-transformers` — optional but strongly recommended for real semantic deduplication and eigencontext. Without it, embeddings fall back to a deterministic lexical-hash vector (catches near-duplicates but not deep semantic similarity).

## Install

```bash
pip install -r requirements.txt

# optional but recommended for real semantic behaviour:
pip install sentence-transformers
```

## Run

```bash
python -m claude_compress --config config.example.json
# or: python -m claude_compress   (uses safe defaults)
```

## Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
# ANTHROPIC_API_KEY is forwarded through untouched
claude
```

The proxy reads the API key from request headers and forwards it; it never stores credentials and never needs its own key. The checkpoint summariser reuses the same forwarded key for its side-call.

## Configuration

Configuration is loaded from a JSON file (`--config path`) with optional overrides via environment variables (`CCOMP_*`). See `config.example.json` for all options with annotations.

Key knobs:

| Key | Default | Notes |
|---|---|---|
| `listen_host` / `listen_port` | `127.0.0.1:8787` | |
| `upstream_base_url` | `https://api.anthropic.com` | Override for proxied upstreams |
| `checkpoint.trigger_tokens` | `12000` | Lower for more aggressive summarisation |
| `checkpoint.keep_recent_messages` | `8` | Verbatim turns always preserved |
| `dedup.threshold` | `0.93` | Cosine similarity floor for duplicate detection |

Environment overrides: `CCOMP_UPSTREAM`, `CCOMP_HOST`, `CCOMP_PORT`, `CCOMP_METRICS`, `CCOMP_LOG_LEVEL`.

## Observability

Every request appends a JSONL line to `ccomp_metrics.jsonl`:

```json
{"session":"ab12…","tokens_in":3875,"tokens_out":776,"saved":3099,"saved_pct":80.0,
 "stages":[{"name":"checkpoint_compression","saved":3025,"note":"folded 15 msgs …"}]}
```

`GET /healthz` returns the active embedding mode, upstream URL, and session count.

## Tests

```bash
python test_pipeline.py      # offline: savings + safety invariants (no network)
python test_integration.py   # proxy against a mock upstream (no real API key needed)
```

`test_pipeline.py` asserts the invariants that matter: the live (most recent) user turn is never altered, tool schemas stay intact, user/assistant role alternation is preserved, and aliases round-trip correctly.

## Safety model

- The most recent `protect_last_n_messages` turns are never compressed by any stage.
- `tool_use`, `tool_result`, and `image` blocks are never modified.
- Any stage that throws is skipped, not fatal — a broken stage cannot corrupt a request.
- Checkpoint compression preserves role alternation when injecting its summary message.

## Tuning

Start with defaults. For more aggressive savings on long sessions, lower `checkpoint.trigger_tokens` and `checkpoint.keep_recent_messages`. Enable `eigencontext` or `alias_substitution` only after measuring that they don't degrade output quality on your workload — the metrics file and the eval harness (`eval/`) are the right tools for that.
