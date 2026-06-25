# Evaluating claude-compress

Token savings alone prove nothing. A layer that cuts tokens 80% while degrading answers is a net loss. The claim to support is: **"saves X% with quality non-inferior to baseline within margin δ."** Two axes, measured together, on the same inputs. Please note, due to the fact that tokens are somewhat expensive to buy for sonnet 4.6 and the usage limit of the claude pro plan, we do not have significant amounts of data for testing yet. If possible, please kindly contribute what you can (either that be testing with your own api credits or generating test cases for others to test as well).

## Current results (7 tasks, 44 turns, 10 judged turns — Sonnet 4.6 judge)

> **Note:** FAILs on the non-inferiority test below are driven by wide confidence intervals from small sample size (10 scored turns), not by losses dominating. Compressed wins or ties on 7 of 10 judged turns in every configuration. ~50 scored turns are needed to tighten the CIs enough for a conclusive verdict.

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

## What a credible result looks like

> Over 200 multi-turn tasks (avg 9 turns): **41% fewer input tokens** (95% CI 38–44%), **58% lower cost** including cache reads, with mean quality loss **+0.004** (95% CI −0.002 to +0.011) against a δ=0.03 margin → non-inferior. Savings-by-turn stable through turn 12; quality-by-turn flat. Ablation: removing checkpoint drops savings to 9%; removing alias changes nothing → alias disabled.

That format — savings with bounded quality loss, longitudinally stable, attributed by ablation — is the structure of a credible claim. A bare "80% fewer tokens" is not.