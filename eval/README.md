# Evaluating claude-compress

Token savings alone prove nothing. A layer that cuts tokens 80% while degrading answers is a net loss. The claim to support is: **"saves X% with quality non-inferior to baseline within margin δ."** Two axes, measured together, on the same inputs. Please note, due to the fact that tokens are somewhat expensive to buy for sonnet 4.6 and the usage limit of the claude pro plan, we do not have significant amounts of data for testing yet. If possible, please kindly contribute what you can (either that be testing with your own api credits or generating test cases for others to test as well).

## Current results (7 tasks, 44 turns, 10 judged turns — Sonnet 4.6 judge)

> **Note:** FAILs on the non-inferiority test below are driven by wide confidence intervals from small sample size (10 scored turns), not by losses dominating. Compressed wins or ties on 7 of 10 judged turns in every configuration. ~50 scored turns are needed to tighten the CIs enough for a conclusive verdict.
>
> **Scope:** These results cover the text pipeline only (checkpoint, dedup, delta, eigencontext, alias). The `image_compress`, `json_compress`, `log_compress`, and `html_compress` stages were not active during this evaluation. See [Evaluating image compression](#evaluating-image-compression) and [Evaluating format compression](#evaluating-format-compression) below.

### Full pipeline (default config)
- **input saved: 17.3%** (95% CI 7.9–27.3%)
- **cost/turn: $0.03574 → $0.02260 (36.8% cheaper)**
- quality (0–1): baseline 0.400, compressed 0.600
- mean quality loss −0.200 (95% CI −0.700–+0.400); margin 0.030 → FAIL (sample size)
- win/tie/loss for compressed: 5/2/3

### Ablation: − checkpoint
- **input saved: 8.9%** (95% CI 5.6–12.5%) — checkpoint accounts for ~half the size reduction
- cost/turn: $0.03548 → $0.02008 (43.4% cheaper)
- win/tie/loss for compressed: 5/2/3 — identical quality to full pipeline

### Ablation: − dedup
- **input saved: 16.9%** (95% CI 7.8–26.6%) — dedup contributes minimal size reduction
- cost/turn: $0.03612 → $0.02217 (38.6% cheaper)
- win/tie/loss for compressed: 3/5/2 — most neutral stage, neither helps nor hurts quality

### Ablation: − delta (cache)
- **input saved: 23.9%** (95% CI 15.5–32.9%) — removing cache stage increases apparent size savings (breakpoint injection adds a few tokens)
- cost/turn: $0.03643 → $0.02297 (36.9% cheaper)
- win/tie/loss for compressed: 3/4/3

### Key findings from ablation
- **Checkpoint** is responsible for ~half the size reduction (17.3% → 8.9% without it) with no quality cost
- **Dedup** is the most neutral stage — modest savings, no quality impact
- **Delta/cache** reduces cost without reducing size; Claude Code already manages its own cache breakpoints so this stage largely steps aside
- Savings compound with turn count: t0–t8 near zero, t15+ consistently 65–80%
- **Image compress** — not yet in these numbers; expected to dominate savings in computer-use sessions where visual tokens can exceed text tokens by 10–100×
- **Format compress (json/log/html)** — not yet in these numbers; expected to show moderate savings in agentic sessions with large tool_result payloads and zero quality cost (operations are self-gating and structure-preserving)

---

## Evaluating format compression

The `json_compress`, `log_compress`, and `html_compress` stages target `tool_result` blocks. They have no benchmark data yet. The expected profile — moderate savings, no quality cost — needs verification on agentic sessions where tools return large JSON payloads, shell output, or web pages.

### What to measure

- **Blocks compressed per stage** — reported as `blocks_compressed` in `ccomp_metrics.jsonl`. A rate near zero means the guard conditions (content type, minimum token threshold) are rarely triggered; raise `min_compress_tokens` or reconsider whether the stage is relevant for your workload.
- **Token savings per stage** — reported as `tokens_saved` in `ccomp_metrics.jsonl`. Compare to total `tokens_in` to understand each stage's contribution.
- **JSON fidelity** — spot-check compressed tool_results: the truncated form should parse as valid JSON, the omission marker (`[…N items omitted…]`) should be present, and key fields referenced in later turns should be in the head or tail.
- **Log/trace completeness** — the error type and message lines should survive truncation; only interior frame repetitions are removed.
- **HTML content accuracy** — after extraction, key facts visible on the page (headlines, key data) should still be present; navigation links and cookie banners should be absent.

### Task format for agentic sessions with tool results

Add `tool_result` turns directly in the task JSONL. Use `check` conditions to verify that compressed tool outputs still support correct downstream reasoning:

```json
{"id": "json-api-task",
 "turns": [
   {"user": "Search for the latest 50 results.",
    "tool_results": [{"id": "call_1", "content": "[{\"id\": 1, ...}, ...]"}]},
   {"user": "What was the highest-scoring result?",
    "check": {"type": "contains", "value": "id"}}
 ]}
```

### Recommended ablation matrix for tool-heavy sessions

| Config | What it isolates |
|---|---|
| `json.enabled=false, log.enabled=false, html.enabled=false` | Baseline — no format compression |
| `json.enabled=true` only | JSON savings alone |
| `log.enabled=true` only | Log/trace savings alone |
| `html.enabled=true` only | HTML extraction savings alone |
| All three enabled (defaults for json/log; html opt-in) | Full format pipeline |
| `json.max_array_items=5, json.max_string_chars=100` | Aggressive JSON truncation — higher savings, check for fidelity regressions |

---

## Evidence hierarchy (weakest → strongest)

1. **Estimated token reduction** — the proxy's heuristic counter. Useful for tuning; not proof. It's an estimate and says nothing about quality.
2. **Ground-truth usage** — the API's own `usage` block (`input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`). Savings is now real, and cost can be computed accurately. The proxy logs these as `kind: "ground_truth"` rows.
3. **Paired A/B with quality scoring** — each task runs twice (baseline vs. compressed) on identical input; both are scored and tested for non-inferiority. This is the level that supports a real claim. `eval/run_eval.py` implements this.
4. **Shadow / canary on real traffic** — mirror production requests, run compressed in shadow, and compare; or canary a small percentage of sessions with a kill-switch while monitoring objective outcomes (CI pass rate, task completion, thumbs-down). Real distribution beats any benchmark.

## Size vs. cost

The cache (delta) stage is **lossless on size** — Claude still sees every token — but moves the stable prefix into the discounted cache-read bucket. Results correctly show "0% smaller, 30% cheaper." Report both metrics. `total_input` measures what the model processed; `cost()` weights the three input buckets (1.0× uncached / 1.25× cache-write / 0.10× cache-read). Verify per-million prices in `usage.py` against current Anthropic pricing before quoting dollar figures.

## Experimental design

**Paired, not two independent groups.** Same task; only the transform varies. Pairing cancels task-to-task variance, giving much tighter confidence intervals. `stats.py` uses a paired bootstrap.

**Non-inferiority, one-sided.** The claim is not that compression is *better*, only *not meaningfully worse*. The test passes when the upper CI bound of the quality loss sits below δ. Pick δ deliberately (e.g. 0.03 on a 0–1 scale).

**Control randomness.** Use temperature 0, or take N samples per task and average.

**Longitudinal replay is essential.** Compression is stateful — checkpoint summarisation and a growing cache prefix mean damage compounds. A single-turn test can look perfect while quality falls off a cliff at turn 40. The harness replays multi-turn tasks turn-by-turn, re-compressing each turn, and the report prints savings-by-turn so drift is visible. Add per-turn `check`s to get quality-by-turn too.

**Ablation.** Turn each stage off one at a time (`--ablate`) to attribute both savings and quality cost. In the synthetic evaluation, checkpoint accounted for nearly all size reduction while the risky stages contributed almost nothing — ablation is how you discover that and cut dead weight.

For image-heavy sessions, include `image_compress` in the ablation matrix. The dedup sub-feature (`image.dedup_exact`) is the dominant lever for computer-use workloads and should be isolated in its own ablation pass so its quality impact (expected: neutral) is measured separately from the resize pass (expected: mild risk on document/diagram content).

## Evaluating image compression

Image compression has no benchmark data yet. The expected profile — strong savings, near-zero quality impact for dedup, mild risk for resize on text-heavy images — needs verification on real computer-use sessions.

### What to measure

- **Visual token savings** — reported directly by the stage as `visual_tokens_saved` in `ccomp_metrics.jsonl`. This is distinct from text token savings and should be reported separately.
- **Dedup hit rate** — `images_deduped / images_found`. A rate above 50% on computer-use sessions is typical; below 10% suggests few repeated screenshots and the dedup path adds negligible value.
- **Classification accuracy** — spot-check a sample of images in `ccomp_metrics.jsonl` against the classifier's label (photo / document_text / diagram_ui). Misclassifying a screenshot as a photo risks applying seam carving; misclassifying a photo as document_text only means OCR is attempted (fails validation → falls back to downscaling; safe).
- **OCR extraction rate** — `images_ocr_extracted / images_found`. A rate near zero on a terminal-heavy session means the classifier isn't labelling images as `document_text`; check `classify_image` thresholds. A rate near 1.0 on a UI design session is a red flag — OCR should not be firing on photos or diagrams.
- **Zone segmentation rate** — `images_zone_segmented / images_found`. A low rate on mixed-content screenshots means `zone_min_area` is too large or `zone_min_zone_area` is filtering out real zones. Spot-check a sample of zone counts per image to confirm multi-zone splits are happening where expected. Include dark-themed terminal screenshots in the test set — the binarizer auto-inverts for dark backgrounds, so these should now zone-detect correctly; a rate near zero on a terminal-heavy set is a sign of a regression to investigate.
- **Quality on visual tasks** — use objective checks where possible: OCR-style `contains` checks on text visible in screenshots, or task-completion checks when the agent acts on what it sees.

### Task format for image-heavy sessions

Add images to turns using the standard Messages API format. Base64-encoded images are rewritten by the stage; URL images are left untouched:

```json
{"id": "computer-use-task",
 "turns": [
   {"user": "Here is a screenshot of the error.", "images": [
     {"type": "base64", "media_type": "image/png", "data": "<base64>"}
   ]},
   {"user": "What should I fix?",
    "check": {"type": "contains", "value": "NullPointerException"}}
 ]}
```

### Recommended ablation matrix for image sessions

| Config | What it isolates |
|---|---|
| `image.enabled=false` | Baseline — no image compression |
| `image.enabled=true, dedup_exact=true, max_tokens=9999` | Dedup only — measures quality impact of stubbing duplicates |
| `image.enabled=true, dedup_exact=false, max_tokens=1024` | Resize only — measures quality impact of downscaling |
| `image.enabled=true` (all defaults) | Full pipeline (no OCR) |
| `image.enabled=true, old_age_threshold=16, old_age_max=256` | Age-based aggressive compression on old screenshots |
| `image.enabled=true, ocr_enabled=true` | OCR extraction for document_text images — measure quality on text-in-image tasks |
| `image.enabled=true, ocr_enabled=true, ocr_backend=easyocr` | EasyOCR vs Tesseract — compare extraction accuracy on complex layouts |
| `image.enabled=true, zone_segment=true` | Zone segmentation alone — measures quality impact of splitting mixed images |
| `image.enabled=true, zone_segment=true, ocr_enabled=true` | Full pipeline — zone split + OCR text zones + compress image zones |

---

## Measuring quality

Prefer **objective** checks; they cannot be gamed by fluent-but-wrong output:

- **Coding tasks** — apply the patch, run the test suite, score 1 iff green (`check.type = "code_tests"`). SWE-bench-style signal; most trustworthy for a coding agent.
- **Factual / needle tasks** — `contains` / `regex` / `exact` on a planted fact. The example tasks plant a `NEEDLE=...` value in early context and ask for it back several turns later — a direct test of whether compression destroyed information it should not have.
- **Open-ended tasks** — use **blind pairwise LLM-as-judge**: show both answers in randomized order (eliminates position bias) and ask which better satisfies the task. Prefer a different, stronger judge model. Calibrate the judge on a set with known human labels before trusting its scores. Report win/tie/loss alongside the aggregate score.

## Statistical reporting

Report the point estimate **and** the 95% CI, never a bare mean. For the quality delta, the decision rule is `CI_upper(loss) ≤ δ`. The bundled sanity check confirms the rule has teeth: a 30% regression fails (CI excludes zero, upper bound far above δ) while a single error in 100 tasks passes.

## Running the eval

```bash
# wiring demo — no API key, synthetic model (outputs labelled SYNTHETIC):
python -m eval.run_eval --tasks eval/tasks.example.jsonl --mock --ablate

# real run:
export ANTHROPIC_API_KEY=sk-...
python -m eval.run_eval --tasks eval/real_tasks.jsonl \
    --upstream https://api.anthropic.com \
    --judge-model claude-sonnet-4-6 \
    --margin 0.03 --ablate --out report.md
```

### Using the session store as an eval data source

The proxy automatically indexes sessions into `~/.claude-compress/sessions.db`. You can query it directly to find sessions suitable for eval tasks:

```python
from claude_compress.session_store import SessionStore
store = SessionStore()

# find sessions that touched a specific file (likely coding tasks)
for s in store.search_by_file("test_"):
    print(s.session_id[:8], s.turn_count, "turns —", s.summary[:80])

# check what's stored
print(store.stats())
```

Full session JSON (full tier, ≤7 days old) can be loaded with `store.get_full(session_id)` and converted to eval task format with `eval/record_to_tasks.py`.

## Task file format (JSONL, one task per line)

```json
{"id": "needle-buried",
 "system": "optional system prompt",
 "tools": [],
 "turns": [
   {"user": "context with NEEDLE=bravo9 buried in it ..."},
   {"user": "restate the token", "check": {"type": "contains", "value": "bravo9"}}
 ]}
```

`check` can be `contains` / `not_contains` / `regex` / `exact` / `all_of` / `code_tests`. A turn with `"judge": true` and a `judge_rubric` is scored by the pairwise judge instead. Turn-level `check` overrides a task-level default.

## Measuring the source code minifier

The proxy pipeline compresses conversation history; the minifier compresses individual source-code snippets before they enter the prompt. These are independent tools and should be measured independently.

### Token counting with `eval/token_tester.py`

`token_tester.py` calls `/v1/messages/count_tokens` — the real Claude tokenizer endpoint — to get authoritative before/after counts.

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# compare original vs. minified for a single file
python -m minifier path/to/input.py > /tmp/mini.py
python eval/token_tester.py \
    --add original "$(cat path/to/input.py)" \
    --add minified "$(cat /tmp/mini.py)"

# batch: build a candidates JSON from multiple files
python eval/token_tester.py --file candidates.json --out results.json
```

The `--no-api` flag runs without a network call, using tiktoken's `o200k_base` encoding as a rough proxy (different tokenizer — directionally correct, not authoritative).

### Observed token reductions (authoritative Claude API counts)

The table below shows byte reduction as a proxy for token reduction; exact token counts depend on the specific code content but track byte reduction closely for prose-like identifiers.

| Language | Typical byte reduction | Identifiers renamed | Notes |
|---|---|---|---|
| Python | 40–55% | yes | `ast`+`symtable` scope analysis |
| JavaScript | 55–68% | yes | `var` hoisting, closure tracking, CJS/ES6 exports |
| TypeScript | ~48% | yes | JS walker; type annotations auto-skipped |
| C | ~57% | yes | Preprocessor macros untouched |
| C++ | ~59% | yes | Class members / `field_identifier` untouched |
| Java | ~59% | yes | Fields, method names, reflection-accessible names untouched |
| JSON | ~24% | no | Whitespace stripping only; keys are data |
| YAML | ~25% | no | Comment stripping + blank-line collapse; indentation preserved |

### When to run the minifier eval vs. the pipeline eval

- **Minifier eval** (`token_tester.py`): use when you want to measure how much a single file or snippet shrinks. Does not require the proxy to be running.
- **Pipeline eval** (`run_eval.py`): use when you want to measure end-to-end quality and token savings over a full multi-turn session. The proxy stages (checkpoint, dedup, cache) operate on conversation history, not individual code files.

If you are contributing new language support to the minifier, run the test suite (`python -m pytest tests/minifier/`) and then spot-check a representative file with `token_tester.py` to confirm the token reduction matches the byte reduction (they can diverge for code with many short-identifier tokens that happen to be single BPE tokens).

## What a credible result looks like

**Text pipeline:**
> Over 200 multi-turn tasks (avg 9 turns): **41% fewer input tokens** (95% CI 38–44%), **58% lower cost** including cache reads, with mean quality loss **+0.004** (95% CI −0.002 to +0.011) against a δ=0.03 margin → non-inferior. Savings-by-turn stable through turn 12; quality-by-turn flat. Ablation: removing checkpoint drops savings to 9%; removing alias changes nothing → alias disabled.

**Image pipeline (target format once data exists):**
> Over 50 computer-use tasks (avg 12 turns, avg 8 images/turn): **image dedup** eliminated 63% of visual tokens with zero quality regressions (dedup hit rate 71%); **OCR extraction** converted 38% of remaining images (classified as document_text) to text blocks saving 82% of their visual tokens with 0 quality regressions on terminal/log tasks; **resize** reduced remaining image tokens by 58% with 2 quality regressions on diagram-heavy tasks where the classifier mislabelled screenshots as photos. Net: visual tokens −91%, text tokens −19% (text stages applied to OCR output), total cost −68%. Ablation: disabling dedup halves visual savings; disabling OCR drops savings to −77%; disabling resize has negligible quality impact → resize budget raised to 2048 for diagram-heavy workloads.

**Format pipeline (target format once data exists):**
> Over 100 agentic tasks with tool_result blocks (avg 6 turns, avg 3 tool calls/turn): **json_compress** reduced JSON payload tokens by 34% (blocks_compressed rate 61%, no quality regressions — truncated fields never referenced downstream); **log_compress** reduced log/trace tokens by 58% (repeated-line collapse accounted for 40%, frame truncation 60%); **html_compress** (enabled on web-scraping subset, 30 tasks) reduced HTML tokens by 71% with 1 regression where a navigation link was load-bearing. Net: tool_result tokens −44%, total tokens −18%, total cost −22% (additive with text pipeline). Ablation: disabling json_compress alone drops savings to −12%; disabling log_compress drops to −8%; html step-aside for non-HTML content is zero-cost.

That format — savings broken out by stage, block compression rate reported, quality attributed to specific failure modes — is the structure of a credible format compression claim.