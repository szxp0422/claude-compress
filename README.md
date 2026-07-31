# claude-compress

A token-compression **middleware proxy** for Claude Code (and any client that speaks the Anthropic Messages API). It sits between the client and `api.anthropic.com`, compresses the outgoing prompt, forwards it, and expands the response on the way back. Claude itself is unchanged — this is pure middleware.

```
Claude Code ──▶ claude-compress proxy ──▶ Anthropic API (Claude, unchanged)
                     │                              │
            input pipeline (compress)      response post-process (expand)
```

## Results

Evaluated over 7 multi-turn tasks (44 turns, 10 judged turns) using Claude Sonnet 4.6 as judge:

- **17% fewer input tokens** (95% CI 7.9–27.3%)
- **37% lower cost per turn** ($0.0357 → $0.0226)
- **Compressed wins or ties on 7 of 10 judged turns** (5 wins / 2 ties / 3 losses)
- Savings compound with session length: near-zero at t0–t8, 65–80% from t15 onward

The formal non-inferiority test requires ~18 scored turns to pass at δ=0.03 given the current win rate. Results and methodology are in [`eval/README.md`](eval/README.md).

## How it works

The Messages API is stateless, so middleware cannot hide context from Claude. Each compression stage is therefore either *lossless*, *safe-lossy* (only touches clearly-redundant or explicitly-tagged content), or *opt-in risky*. Defaults are conservative.

| Stage | Default | Lossy? | What it does |
|---|---|---|---|
| `delta_cache_breakpoints` | **on** | no | Inserts `cache_control` breakpoints on the stable prefix (system, tools, old turns). Cuts **cost** (cached input is billed at a discount), not prompt size. Steps aside if the client already manages its own cache breakpoints. |
| `semantic_dedup` | **on** | safe | Drops history text blocks whose meaning duplicates an earlier block. Threshold is computed per-session via Otsu's method (finds the natural valley in the pairwise similarity distribution) and pulled down further under token pressure. Never touches the last N messages or tool/image blocks. |
| `checkpoint_compression` | **on** | safe | Once a conversation passes a token threshold, folds the oldest turns into one compact summary via a cheap Haiku side-call. Highest-value stage for long sessions — accounts for roughly half of total size reduction. |
| `json_compress` | **on** | safe | Minifies and truncates JSON in `tool_result` blocks. Minification (removing whitespace) is lossless. Truncation reduces arrays longer than `max_array_items` to a head + tail with a count annotation, and caps long string values at `max_string_chars`. Skips blocks below `min_compress_tokens` and protects the most recent `protect_last_n_messages` turns. High-value for agentic sessions that call JSON APIs or search tools. |
| `log_compress` | **on** | safe | Compresses shell output, log streams, and exception traces in `tool_result` blocks. Two passes: (1) collapses consecutive identical lines to `[above line repeated ×N]`; (2) truncates long stack-frame runs to head + tail with a `[N frames omitted]` gap. Both passes are self-gating — no-op on non-log content. Detects Python, Java/Kotlin, Node.js/V8, GDB/LLDB, and macOS crash frames. |
| `html_compress` | off | safe | Strips HTML boilerplate from `tool_result` blocks and converts semantic content to markdown-adjacent text using stdlib `html.parser`. Drops `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>`, `<form>`, and other non-content tags. Converts headings to `# …` markdown, preserves `<pre>` blocks as code fences, converts lists to `- item` lines. Disabled by default: enable for web-scraping workloads; leave off if the HTML structure itself is the subject of the session. |
| `image_compress` | off | **lossy** | Reduces the visual-token cost of image blocks in conversation history. Works in three passes: (1) **exact dedup** — replaces repeated screenshots with a 1×1 stub (saves 100% of duplicate tokens, especially valuable in computer-use sessions); (2) **age-based budget** — applies a stricter token limit to very old images; (3) **resize** — classifies each image as photo / document_text / diagram_ui via cheap numpy heuristics, crops whitespace margins, and downscales to a per-image token budget. Seam carving for photos is available but off by default. Requires Pillow. Token counting covers PNG, JPEG, GIF, and WebP (VP8X). |
| `eigencontext` | off | **lossy** | Greedy max-coverage sentence selection over `<<REF>>`-tagged reference blocks only. Disabled by default: dropping context from a coding agent is risky. |
| `alias_substitution` | off | **risky** | Replaces long repeated strings with short aliases and a legend; expands them back in the response. Only profitable when a string repeats 8+ times; marginal wins on most sessions. |
| `state_machine` | off | n/a | If the client passes a `metadata._fsm` spec, injects a compact transition table instead of prose workflow text. |

Stages that would require model internals (runtime attention pruning, latent-space prompting) are intentionally absent — they cannot be built as an API-layer add-on.

## Theoretical basis

### Why compression is possible

Shannon's source coding theorem guarantees that any source with redundancy can be represented more compactly. Long Claude Code sessions have high redundancy: repeated file paths and boilerplate (dedup), early turns superseded by later edits (checkpoint), and stable system prompts re-sent every turn (cache breakpoints).

Formally, let $C_t$ be the context at turn $t$ and $Q_t$ be the query. The model computes $P(A_t \mid C_t, Q_t)$. The claim is that a compression function $f$ exists such that:

$$P(A_t \mid f(C_t), Q_t) \approx P(A_t \mid C_t, Q_t)$$

where $|f(C_t)| \ll |C_t|$ and the approximation error is bounded by the non-inferiority margin $\delta$. The empirical eval measures whether this holds on real sessions.

### Cache economics

Anthropic bills cache-read tokens at 0.10× the base input rate. If a fraction $r$ of input tokens are served from cache:

$$\text{cost} = p_i \cdot n \cdot (1 - 0.9r) + p_o \cdot m$$

A session with 60% cache hits pays 46 cents on the dollar for input. The delta stage is pure economics — it moves the stable prefix into the discounted bucket without changing what Claude sees. Size reduction and cache discount compound: a 17% smaller prompt with a 60% cache hit rate produces a ~37% cost reduction, matching the observed results.

### Checkpoint ROI

The checkpoint stage fires a Haiku side-call to summarise old turns. It only commits the summary when:

$$\frac{|C_{old}|}{|S|} \geq \rho_{min} \quad (\rho_{min} = 2.0 \text{ by default})$$

This guarantees at least a 2:1 compression ratio on the summarised portion, ensuring the stage pays for the side-call cost across subsequent turns.

### Semantic dedup bound

Dedup solves an approximate set cover problem. The greedy algorithm retains at most $O(\log n)$ times the optimal minimum set.

The similarity threshold is computed per-session rather than set globally. Otsu's method finds the threshold $\tau^*$ that maximises between-class variance on the histogram of all pairwise cosine similarities — the natural valley between the "clearly different" and "near-duplicate" clusters in the current session. A pressure term then pulls the threshold down as the context window fills:

$$\tau = \text{clip}(\tau^* - 0.13 \cdot \max(0, 2(u - 0.5)),\ 0.80,\ 0.97)$$

where $u = |C_t| / C_{limit}$ is current utilisation. At $u \leq 0.5$ there is no adjustment; at $u = 1.0$ the threshold drops by 0.13. With hash-based embeddings (lexical fallback) the threshold is additionally capped at 0.90 to avoid false dedup.

Any removed block has cosine similarity ≥ $\tau$ to a retained block, so information loss per removed block is bounded by $1 - \tau^2$. At the minimum allowed threshold $\tau = 0.80$ this is $\leq 36\%$ of that block's directional information content; at the typical Otsu-computed value near 0.93 it is $\leq 13\%$.

### Why the test is conservative

The non-inferiority test checks whether the upper CI bound of the quality loss is below $\delta = 0.03$. With $n$ scored turns, standard error $\sigma \approx 0.5$, and observed win rate implying $\mu_{loss} \approx -0.2$:

$$n \geq \left(\frac{1.96 \times 0.5}{0.03 - (-0.2)}\right)^2 \approx 18$$

Roughly 18 scored turns are needed to formally pass. The current dataset has 10, which is why the FAIL label appears despite compressed winning 5–7 of 10 judged turns in every configuration. The math and the direction of results are consistent — the gap is sample size only.

## Source Code Minifier

A standalone, lossless source code minifier targeting **Claude's tokenizer** rather than raw byte count. It strips comments and redundant whitespace and renames local identifiers to single-character names, all while preserving exact runtime behaviour.

### Supported languages

| Language | Group | Comment strip | Whitespace | Rename locals | Typical byte reduction |
|---|---|---|---|---|---|
| Python | B | ✓ | indent reduce | ✓ (`ast`+`symtable`) | 40–55% |
| JavaScript | A | ✓ | reconstruct | ✓ (scope-aware) | 55–68% |
| TypeScript | A | ✓ | reconstruct | ✓ (JS walker, type nodes skipped) | ~48% |
| C | A | ✓ | reconstruct | ✓ (function-local only; macros untouched) | ~57% |
| C++ | A | ✓ | reconstruct | ✓ (methods + free functions; members untouched) | ~59% |
| Java | A | ✓ | reconstruct | ✓ (method-local only; fields/methods untouched) | ~59% |
| JSON | A | — | reconstruct | — (keys are data, not identifiers) | ~24% |
| YAML | B | ✓ | blank-line collapse | — (indentation is structural) | ~25% |

**Group A** languages are delimiter-based: the minifier reconstructs the source from AST leaf tokens with minimum necessary spacing.  
**Group B** languages are whitespace-significant: only comments and excess blank lines are removed; indentation is reduced in place (except YAML where it is structural and preserved as-is).

### CLI

```bash
# auto-detects language from extension (.py, .js, .ts, .c, .cpp, .java, .json, .yaml / .yml)
python -m minifier path/to/file.py

# write to a file and save the old→new name map
python -m minifier path/to/file.ts --out out.ts --map names.json

# preserve readable names (whitespace only)
python -m minifier path/to/file.java --no-rename
```

### Python API

```python
from minifier.minify import minify

result = minify(source_code, lang="typescript")
print(result.output)          # minified source
print(result.name_map)        # {"longName": "a", "anotherName": "b", ...}
print(result.stats.byte_reduction_pct)  # e.g. 48.4

# supported lang keys: python, javascript/js, typescript/ts,
#                      c, cpp/c++, java, json, yaml/yml
```

### Safety guarantees

- **Lossless for code**: a minified Python/JS/TS/C/C++/Java file produces identical stdout to the original (verified by the test suite via subprocess round-trips with `gcc`, `g++`, `javac`, and `node`).
- **Data-preserving for config**: minified JSON round-trips through `json.loads`; minified YAML round-trips through `pyyaml.safe_load`.
- Local identifiers are renamed only when all occurrences are visible within the file. Exported names (ES6 `export`, CommonJS `module.exports`), class fields, method names, and macro bodies are never renamed.

### Measuring token savings

Use `eval/token_tester.py` to get exact Claude token counts (via the `/v1/messages/count_tokens` API) on original vs. minified snippets. See [`eval/README.md`](eval/README.md) for the full workflow.

## Requirements

- Python 3.9+
- `fastapi`, `uvicorn`, `httpx`, `numpy`, `tiktoken` (all in `requirements.txt`)
- `sentence-transformers` — optional but strongly recommended for real semantic deduplication and eigencontext. Without it, embeddings fall back to a deterministic lexical-hash vector (catches near-duplicates but not deep semantic similarity).
- `Pillow` — optional, required for the `image_compress` stage. Without it that stage self-disables gracefully.

## Install

```bash
pip install -r requirements.txt

# optional but recommended for real semantic behaviour:
pip install sentence-transformers

# optional, required for image compression:
pip install Pillow
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

The proxy reads the API key from request headers and forwards it; it never stores credentials and never needs its own key. The checkpoint summariser reuses the same forwarded key for its Haiku side-call.

## Record real sessions for evaluation

```bash
# Start the proxy with recording enabled
CCOMP_RECORD=sessions.jsonl python -m claude_compress --config config.example.json

# Use Claude Code normally in another terminal
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
claude

# Convert recorded sessions to eval tasks
python -m eval.record_to_tasks --input sessions.jsonl --out eval/real_tasks.jsonl

# Run the eval
python -m eval.run_eval --tasks eval/real_tasks.jsonl \
    --api-key $ANTHROPIC_API_KEY \
    --judge-model claude-sonnet-4-6 \
    --margin 0.03 --ablate --out report.md
```

## Configuration

Configuration is loaded from a JSON file (`--config path`) with optional overrides via environment variables (`CCOMP_*`). See `config.example.json` for all options.

Key knobs:

| Key | Default | Notes |
|---|---|---|
| `listen_host` / `listen_port` | `127.0.0.1:8787` | |
| `upstream_base_url` | `https://api.anthropic.com` | Override for proxied upstreams |
| `checkpoint.trigger_tokens` | `40000` | Lower for more aggressive summarisation |
| `checkpoint.keep_recent_messages` | `8` | Verbatim turns always preserved |
| `checkpoint.min_compression_ratio` | `2.0` | ROI gate: skip if summary isn't 2× smaller |
| `dedup.threshold` | `0.93` | Fallback similarity floor used only when fewer than 3 blocks exist; otherwise Otsu's method picks the threshold automatically |
| `json.enabled` | `true` | Enable JSON minification + truncation in tool_result blocks |
| `json.min_compress_tokens` | `80` | Skip blocks below this token count |
| `json.max_array_items` | `20` | Maximum array elements before head+tail truncation |
| `json.max_string_chars` | `500` | Maximum string length before truncation |
| `json.protect_last_n_messages` | `4` | Leave tool_results in the most recent N messages untouched |
| `log.enabled` | `true` | Enable log/stack-trace compression in tool_result blocks |
| `log.min_compress_tokens` | `80` | Skip blocks below this token count |
| `log.max_stack_frames` | `20` | Maximum stack frames to keep in a single trace (head + tail) |
| `log.protect_last_n_messages` | `4` | Leave tool_results in the most recent N messages untouched |
| `html.enabled` | `false` | Enable HTML-to-text extraction in tool_result blocks (for web-scraping workloads) |
| `html.min_compress_tokens` | `200` | Skip blocks below this token count |
| `html.protect_last_n_messages` | `4` | Leave tool_results in the most recent N messages untouched |
| `image.enabled` | `false` | Enable image compression (requires Pillow) |
| `image.max_tokens_per_image` | `1024` | Compress images that exceed this visual-token count. A 1920×1080 image costs ~2691 tokens; 1024 ≈ 896×896. |
| `image.protect_last_n_messages` | `4` | Leave images in the most recent N messages at full resolution |
| `image.seam_carve_photos` | `false` | Apply seam carving after downscaling for photos still over budget (CPU-intensive; only safe for photographic content) |
| `image.dedup_exact` | `true` | Replace exact-duplicate images beyond their first occurrence with a 1×1 stub. Saves 100% of tokens for repeated screenshots. |
| `image.old_age_threshold_messages` | `0` | Apply a stricter budget to images older than this many messages from the end. `0` disables age-based compression. |
| `image.old_age_max_tokens` | `256` | Token budget used for images beyond `old_age_threshold_messages`. Ignored when threshold is 0. |

Environment overrides: `CCOMP_UPSTREAM`, `CCOMP_HOST`, `CCOMP_PORT`, `CCOMP_METRICS`, `CCOMP_LOG_LEVEL`.

## Observability

Every request appends a JSONL line to `ccomp_metrics.jsonl`:

```json
{"session":"ab12…","tokens_in":3875,"tokens_out":776,"saved":3099,"saved_pct":80.0,
 "stages":[{"name":"checkpoint_compression","saved":3025,"note":"folded 15 msgs …"}]}
```

Ground-truth usage from the API (real token counts, cache hits, cost) is logged separately as `"kind": "ground_truth"` rows.

`GET /healthz` returns the active embedding mode, upstream URL, and session count.

## Tests

```bash
# Unit + stage tests (no network, no API key)
python -m pytest tests/

# Offline smoke test: savings + safety invariants across all stages
python test_pipeline.py

# End-to-end proxy test against a mock upstream (no real API key needed)
python test_integration.py
```

`test_pipeline.py` asserts the invariants that matter: the live (most recent) user turn is never altered, tool schemas stay intact, user/assistant role alternation is preserved, and aliases only fire when they are token-profitable.

`tests/test_image_compress.py` covers image token counting (PNG/JPEG/GIF/WebP header parsing), the stub PNG constant, exact-duplicate deduplication, age-based budget selection, whitespace cropping, downscaling, classification heuristics, and seam carving. Tests that require Pillow are automatically skipped when it is not installed.

`tests/test_format_compress.py` covers the JSON, log, and HTML compression utilities and their corresponding stages: minification and array/string truncation, stack-frame pattern detection across Python/Java/Node.js/GDB/macOS trace formats, repeated-line collapse, HTML tag stripping and markdown conversion, the `protect_last_n_messages` boundary, and the `enabled()` flag (HTML disabled by default).

## Safety model

- The most recent `protect_last_n_messages` turns are never compressed by any stage.
- `tool_use` blocks are never modified by text stages.
- Only base64 images are rewritten by `image_compress`; URL images are left untouched.
- Any stage that throws is skipped, not fatal — a broken stage cannot corrupt a request.
- Checkpoint compression preserves role alternation when injecting its summary message.
- The alias stage computes net token savings before committing; it skips if the legend overhead exceeds the substitution savings.
- The delta stage steps aside entirely if the client (e.g. Claude Code) already manages its own `cache_control` blocks.

## Tuning

Start with defaults. For more aggressive savings on long sessions, lower `checkpoint.trigger_tokens` and `checkpoint.keep_recent_messages`. Enable `eigencontext` or `alias_substitution` only after measuring that they do not degrade output quality on your workload — use the eval harness in `eval/` and watch the per-stage savings in `ccomp_metrics.jsonl`.

For **tool-heavy or agentic sessions** (web search, API calls, shell commands), the `json_compress` and `log_compress` stages are on by default and require no tuning. If your agent returns very large JSON payloads, lower `json.max_array_items` or `json.max_string_chars`. If your agent produces dense log streams with many repeated lines or very deep stack traces, lower `log.max_stack_frames`. Enable `html_compress` for sessions where tools return raw web pages — it typically cuts HTML tool_result size by 60–80%.

For **image-heavy or computer-use sessions**, enable `image.enabled` and set `image.max_tokens_per_image` to match your task's visual detail requirements (lower = more aggressive). `image.dedup_exact` is on by default and is the highest-value lever when the same screenshot recurs across turns. If images are still expensive after many turns, set `image.old_age_threshold_messages` to a turn count (e.g. `16`) and reduce `image.old_age_max_tokens` to push very old screenshots to near-thumbnail size.

## Contributing

The project needs more eval data. If you use Claude Code regularly, running the proxy with `CCOMP_RECORD=sessions.jsonl` and contributing your anonymised task files helps tighten the confidence intervals. See [`eval/README.md`](eval/README.md) for the task format and how to add judge rubrics.

The theoretical case for compression is sound; the empirical gap is purely sample size. The highest-value contributions are:

- **Real multi-turn sessions** recorded via `CCOMP_RECORD` — any length, any task type
- **Image-heavy or computer-use sessions** — the image_compress stage has no benchmark data yet; sessions with repeated screenshots are especially useful
- **Objective `check` conditions** for coding tasks (test-suite pass/fail beats LLM-as-judge for code quality)
- **Additional judge rubrics** for open-ended or agentic tasks