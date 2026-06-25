"""Eval runner: produce the evidence.

Runs a task set through baseline vs compressed (and optional per-stage
ablations), aggregates with paired statistics, and writes a markdown report you
can actually cite.

Usage:
    python -m eval.run_eval --tasks eval/tasks.example.jsonl --mock
    python -m eval.run_eval --tasks mytasks.jsonl --upstream https://api.anthropic.com \
        --api-key $ANTHROPIC_API_KEY --margin 0.03 --ablate
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from claude_compress.config import Config

from .harness import (TaskResult, make_mock_model_call, make_real_model_call,
                      run_task)
from .stats import quality_report, savings_report


def load_tasks(path: str) -> List[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _aggregate(results: List[TaskResult]):
    b_in, c_in, b_cost, c_cost = [], [], [], []
    b_q, c_q = [], []
    for r in results:
        for t in r.turns:
            b_in.append(t.baseline_usage.total_input)
            c_in.append(t.compressed_usage.total_input)
            b_cost.append(t.baseline_usage.cost())
            c_cost.append(t.compressed_usage.cost())
        for t in r.turns:
            if t.baseline_score is not None and t.compressed_score is not None:
                b_q.append(t.baseline_score)
                c_q.append(t.compressed_score)
    return b_in, c_in, b_cost, c_cost, b_q, c_q


def _by_turn_savings(results: List[TaskResult]) -> Dict[int, float]:
    acc: Dict[int, List[float]] = {}
    for r in results:
        for t in r.turns:
            b, c = t.baseline_usage.total_input, t.compressed_usage.total_input
            if b:
                acc.setdefault(t.turn_index, []).append(100.0 * (b - c) / b)
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}


def run_suite(tasks, cfg, model_call, judge_call=None, headers=None):
    return [run_task(t, cfg, model_call, judge_call, headers) for t in tasks]


def build_report(name: str, results: List[TaskResult], margin: float) -> str:
    b_in, c_in, b_cost, c_cost, b_q, c_q = _aggregate(results)
    sav = savings_report(b_in, c_in, b_cost, c_cost)
    lines = [f"### {name}", ""]
    lines.append(f"- tasks: {len(results)}  |  scored turns: {len(b_q)}  |  "
                 f"total turns: {len(b_in)}")
    lines.append(f"- **input saved: {sav.mean_saved_pct:.1f}%** "
                 f"(95% CI {sav.ci_lo:.1f}–{sav.ci_hi:.1f}%)")
    lines.append(f"- cost/turn: ${sav.mean_cost_baseline:.5f} → "
                 f"${sav.mean_cost_compressed:.5f}  "
                 f"(**{sav.mean_cost_reduction_pct:.1f}% cheaper**)")
    if b_q:
        q = quality_report(b_q, c_q, margin=margin)
        verdict = "PASS ✅ non-inferior" if q.non_inferior else "FAIL ❌ quality regressed"
        lines.append(f"- quality (0–1): baseline {q.mean_baseline:.3f}, "
                     f"compressed {q.mean_compressed:.3f}")
        lines.append(f"- mean quality loss {q.mean_delta:+.3f} "
                     f"(95% CI {q.delta_ci_lo:+.3f}–{q.delta_ci_hi:+.3f}); "
                     f"margin {margin:.3f} → **{verdict}**")
        lines.append(f"- win/tie/loss for compressed: "
                     f"{q.wins}/{q.ties}/{q.losses}")
    else:
        lines.append("- quality: no scored turns (add `check` or `judge` to tasks)")
    bt = _by_turn_savings(results)
    if bt:
        trend = "  ".join(f"t{k}:{v:.0f}%" for k, v in bt.items())
        lines.append(f"- savings by turn (watch for drift): {trend}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--upstream", default="https://api.anthropic.com")
    ap.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    ap.add_argument("--mock", action="store_true", help="use synthetic model (no network)")
    ap.add_argument("--ablate", action="store_true", help="also run per-stage ablations")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out", default="eval_report.md")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6",help="model to use as pairwise judge")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    headers = {}
    if args.mock:
        model_call = make_mock_model_call()
    else:
        if not args.api_key:
            raise SystemExit("need --api-key or ANTHROPIC_API_KEY (or use --mock)")
        model_call = make_real_model_call(args.upstream)
        headers = {"x-api-key": args.api_key, "anthropic-version": "2023-06-01"}

    sections = ["# claude-compress evaluation",
                "",
                ("> Mock model: numbers are SYNTHETIC, for harness wiring only."
                 if args.mock else
                 "> Live run against the real API. Numbers are ground truth."),
                ""]

    def make_judge_call(judge_model, headers):
        import httpx
        def judge(prompt: str) -> str:
            body = {
                "model": judge_model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            }
            h = dict(headers)
            h["content-type"] = "application/json"
            with httpx.Client(timeout=60) as client:
                r = client.post(
                    args.upstream.rstrip("/") + "/v1/messages",
                    headers=h, json=body
                )
                r.raise_for_status()
                data = r.json()
            return "".join(
                b.get("text", "") for b in data.get("content", [])
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return judge

    judge_call = make_judge_call(args.judge_model, headers) if not args.mock else None

    full_cfg = Config()
    results = run_suite(tasks, full_cfg, model_call, judge_call=judge_call, headers=headers)
    sections.append(build_report("Full pipeline (default config)", results, args.margin))

    if args.ablate:
        # turn each major stage OFF one at a time to attribute its contribution
        ablations = {
            "− checkpoint": ("checkpoint", False),
            "− dedup": ("dedup", False),
            "− delta (cache)": ("delta", False),
        }
        for label, (attr, val) in ablations.items():
            cfg = Config()
            setattr(getattr(cfg, attr), "enabled", val)
            res = run_suite(tasks, cfg, model_call, judge_call=judge_call, headers=headers)
            sections.append(build_report(f"Ablation: {label}", res, args.margin))

    report = "\n".join(sections)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
