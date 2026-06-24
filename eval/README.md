# Evaluating claude-compress: how to prove it actually works

Token savings alone proves nothing. A layer that cuts tokens 80% while degrading
answers is a net loss. **The claim you must support is: "saves X% with quality
non-inferior to baseline within margin δ."** Two axes, measured together, on the
same inputs.

## The evidence hierarchy (weakest → strongest)

1. **Estimated token reduction** (the proxy's heuristic counter). Useful for
   tuning a stage; not proof — it's an estimate and says nothing about quality.
2. **Ground-truth usage.** Capture the API's own `usage` block (`input_tokens`,
   `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`).
   Now savings is real, and you can compute real cost. The proxy logs this as
   `kind: "ground_truth"` rows.
3. **Paired A/B with quality scoring.** Run each task twice — baseline vs
   compressed — on identical input, score both, and test non-inferiority. This
   is the level that supports a real claim. `eval/run_eval.py` does this.
4. **Shadow / canary on real traffic.** Mirror production requests; run
   compressed in shadow and compare, or canary a small % of sessions with a kill
   switch, monitoring objective outcomes (CI pass rate, task completion,
   thumbs-down). Real distribution beats any benchmark.

## Why size and cost are different numbers

The cache (delta) stage is **lossless on size** — Claude still sees every token —
but moves the stable prefix into the discounted cache-read bucket. So you will
see, correctly, "0% smaller, 30% cheaper." Report both. `total_input` measures
what the model processed; `cost()` weights the three input buckets (1.0× /
1.25× write / 0.10× read). Verify the per-million prices in `usage.py` against
current pricing before quoting dollars.

## Experimental design that matters

- **Paired, not two independent groups.** Same task, only the transform varies.
  Pairing cancels task-to-task variance → much tighter CIs (`stats.py` uses a
  paired bootstrap).
- **Non-inferiority, one-sided.** You're not claiming compression is *better*,
  only *not meaningfully worse*. Test passes when the **upper** CI bound of the
  quality loss sits below δ. Pick δ deliberately (e.g. 0.03 on a 0–1 scale).
- **Control randomness.** Temperature 0, or N samples per task and average.
- **Longitudinal replay is essential.** Compression is *stateful* — checkpoint
  summarisation and a growing cache prefix mean damage **compounds**. A
  single-turn test can look perfect while quality falls off a cliff at turn 40.
  The harness replays multi-turn tasks turn-by-turn, re-compressing each turn,
  and the report prints **savings-by-turn** so you can watch for drift. Add
  per-turn `check`s to get quality-by-turn too.
- **Ablation.** Turn each stage off one at a time (`--ablate`) to attribute both
  savings and quality cost. (In the earlier synthetic run, checkpoint did ~all
  the size reduction and the risky stages earned almost nothing — ablation is
  how you discover that and cut dead weight.)

## Measuring quality

Prefer **objective** checks; they can't be gamed by fluent-but-wrong output:
- coding tasks → apply the patch, run the test suite, score 1 iff green
  (`check.type = "code_tests"`; this is the SWE-bench-style signal and the most
  trustworthy one for a coding agent);
- factual/needle tasks → `contains` / `regex` / `exact` on a planted fact;
- the example tasks plant a `NEEDLE=...` fact in early context and ask for it
  back several turns later — a direct test of whether compression destroyed
  information it shouldn't have.

For open-ended tasks with no checkable answer, use **blind pairwise
LLM-as-judge**: show both answers in randomized order (kills position bias), ask
which better satisfies the task, prefer a *different, stronger* judge model.
A judge is a measurement instrument — **calibrate it**: have it grade a set with
known human labels and confirm agreement before trusting it. Report win/tie/loss
alongside the score.

## Statistical reporting

Report the point estimate **and** the 95% CI, never a bare mean. For the quality
delta, the decision rule is `CI_upper(loss) ≤ δ`. The bundled sanity check shows
the rule has teeth: a 30% regression fails (CI excludes zero, upper bound far
above δ) while a single error in 100 tasks passes.

## Running it

```bash
# wiring demo, no key, synthetic model (numbers marked SYNTHETIC):
python -m eval.run_eval --tasks eval/tasks.example.jsonl --mock --ablate

# real run:
export ANTHROPIC_API_KEY=sk-...
python -m eval.run_eval --tasks eval/tasks.example.jsonl \
    --upstream https://api.anthropic.com --margin 0.03 --ablate --out report.md
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

`check` can be `contains` / `not_contains` / `regex` / `exact` / `all_of` /
`code_tests`. A turn with `"judge": true` and a `judge_rubric` is scored by the
pairwise judge instead. Turn-level `check` overrides task-level.

## What a credible result looks like

> Over 200 multi-turn tasks (avg 9 turns): **41% fewer input tokens**
> (95% CI 38–44%), **58% lower cost** including cache reads, with mean quality
> loss **+0.004** (95% CI −0.002 to +0.011) against a δ=0.03 margin → non-inferior.
> Savings-by-turn stable through turn 12; quality-by-turn flat. Ablation: removing
> checkpoint drops savings to 9%; removing alias changes nothing → alias disabled.

That sentence — savings with bounded quality loss, longitudinally stable,
attributed by ablation — is the proof. A bare "80% fewer tokens" is not.
