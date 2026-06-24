"""Quality scoring for eval outputs.

Two families, in order of trustworthiness:

  1. OBJECTIVE (preferred). The task carries a checkable success condition:
       - "contains" / "regex" / "exact": cheap string checks
       - "code_tests": apply the model's patch and run a test command; score = 1
         if the suite goes green. This is the gold standard for a coding agent --
         it cannot be gamed by fluent-but-wrong output.
     Use objective scoring whenever you can construct it. It needs no judge model
     and has no judge bias.

  2. LLM-AS-JUDGE (for open-ended tasks with no checkable answer). We do a BLIND
     PAIRWISE comparison: show the judge both answers (order randomized to kill
     position bias), ask which better satisfies the task, map win/tie/loss to a
     score. Prefer a *different, strong* judge model from the one under test.

A judge is a measurement instrument: validate it before trusting it (see
eval/README) by checking agreement with human labels on a calibration set.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
from typing import Callable, Optional


# ---- objective scorers ----------------------------------------------------

def score_objective(task: dict, output_text: str) -> Optional[float]:
    """Return 1.0/0.0 if the task defines an objective check, else None."""
    check = task.get("check")
    if not check:
        return None
    kind = check.get("type")
    if kind == "contains":
        return 1.0 if check["value"] in output_text else 0.0
    if kind == "not_contains":
        return 1.0 if check["value"] not in output_text else 0.0
    if kind == "regex":
        return 1.0 if re.search(check["pattern"], output_text) else 0.0
    if kind == "exact":
        return 1.0 if output_text.strip() == check["value"].strip() else 0.0
    if kind == "all_of":
        return 1.0 if all(v in output_text for v in check["values"]) else 0.0
    if kind == "code_tests":
        return _score_code_tests(check, output_text)
    return None


def _score_code_tests(check: dict, output_text: str) -> float:
    """Apply a patch from the output and run a test command in a sandbox dir.

    check = {
      "type": "code_tests",
      "repo_setup": "shell to lay down the repo (or a path to copy)",
      "extract": "fenced",                 # how to pull the patch from output
      "apply": "git apply -",              # how to apply it
      "test_cmd": "pytest -q",             # what to run
    }
    Score 1.0 iff test_cmd exits 0. This is intentionally strict.
    """
    patch = _extract_fenced(output_text) if check.get("extract") == "fenced" else output_text
    with tempfile.TemporaryDirectory() as d:
        try:
            if check.get("repo_setup"):
                subprocess.run(check["repo_setup"], cwd=d, shell=True, check=True,
                               capture_output=True, timeout=120)
            apply_cmd = check.get("apply", "git apply -")
            subprocess.run(apply_cmd, cwd=d, shell=True, input=patch.encode(),
                           check=True, capture_output=True, timeout=60)
            r = subprocess.run(check["test_cmd"], cwd=d, shell=True,
                               capture_output=True, timeout=300)
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0


def _extract_fenced(text: str) -> str:
    m = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


# ---- LLM-as-judge ---------------------------------------------------------

# judge_call(prompt) -> raw judge text. Inject a real API caller in production.
JudgeCall = Callable[[str], str]


def score_pairwise(
    task: dict,
    baseline_out: str,
    compressed_out: str,
    judge_call: JudgeCall,
    seed: int = 0,
) -> dict:
    """Blind pairwise judgement. Returns scores for both arms in [0,1].

    Randomizes which answer is labelled 'A' vs 'B' to defeat position bias, then
    un-shuffles the verdict.
    """
    rng = random.Random(seed)
    swap = rng.random() < 0.5
    a, b = (compressed_out, baseline_out) if swap else (baseline_out, compressed_out)

    prompt = (
        "You are grading two assistant answers to the same task. Judge only how "
        "well each satisfies the task; ignore length and style. Respond with "
        'strict JSON: {"winner":"A"|"B"|"tie","reason":"..."}.\n\n'
        f"TASK:\n{task.get('judge_rubric') or task.get('prompt','')}\n\n"
        f"ANSWER A:\n{a}\n\nANSWER B:\n{b}"
    )
    raw = judge_call(prompt)
    verdict = _parse_verdict(raw)
    # map back to (baseline, compressed)
    if verdict == "tie":
        return {"baseline": 0.5, "compressed": 0.5, "verdict": "tie"}
    a_is_compressed = swap
    a_won = verdict == "A"
    compressed_won = (a_won and a_is_compressed) or ((not a_won) and (not a_is_compressed))
    return {
        "baseline": 0.0 if compressed_won else 1.0,
        "compressed": 1.0 if compressed_won else 0.0,
        "verdict": "compressed" if compressed_won else "baseline",
    }


def _parse_verdict(raw: str) -> str:
    try:
        obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        w = obj.get("winner", "tie").strip()
        return w if w in ("A", "B", "tie") else "tie"
    except Exception:
        low = raw.lower()
        if "winner" in low and '"a"' in low:
            return "A"
        if "winner" in low and '"b"' in low:
            return "B"
        return "tie"
