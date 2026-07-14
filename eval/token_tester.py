#!/usr/bin/env python3
"""
Claude Token Tester
====================
Compares candidate code snippets for token efficiency.

IMPORTANT — read before trusting the numbers:

1. EXACT counts come from Anthropic's real /v1/messages/count_tokens
   endpoint. This is the ground truth for how many tokens a snippet
   costs Claude. It requires an ANTHROPIC_API_KEY and a network call
   per snippet.

2. Anthropic does NOT publish Claude's tokenizer vocabulary, and the
   API does not return individual token strings/boundaries — only a
   total count. So there is no way to show you "here is exactly how
   Claude split this string" locally.

3. To still give you a VISUAL sense of where token boundaries likely
   fall, this script also runs each snippet through tiktoken's
   o200k_base encoding (used by recent OpenAI models). This is a
   DIFFERENT tokenizer than Claude's. Treat the "approx breakdown" as
   an illustrative guess about split points, not a fact about Claude.
   Only the "exact_tokens" column from the API is authoritative.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 token_tester.py                      # runs built-in demo set
    python3 token_tester.py --file candidates.json
    python3 token_tester.py --add "for_loop" "for i in range(10): pass"

candidates.json format:
    {
      "python_for": "for i in range(10):\n    print(i)",
      "c_for":      "for(i=0;i<10;i++){print(i);}"
    }
"""

import argparse
import json
import os
import sys

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


DEFAULT_MODEL = "claude-sonnet-5"

DEMO_SNIPPETS = {
    "python_for": "for i in range(10):\n    print(i)",
    "c_style_for": "for(i=0;i<10;i++){print(i);}",
    "python_func": "def add(a, b):\n    return a + b",
    "arrow_func": "const add = (a, b) => a + b;",
    "terse_lambda": "add=\\a b->a+b",
    "if_else_py": "if x > 0:\n    y = 1\nelse:\n    y = -1",
    "if_else_ternary": "y = 1 if x > 0 else -1",
    "if_else_symbolic": "y=x>0?1:-1",
}


def get_client():
    if anthropic is None:
        sys.exit(
            "The 'anthropic' package is not installed.\n"
            "Install it with: pip install anthropic"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "No API key found.\n"
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Get a key at: https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic(api_key=api_key)


def count_tokens_exact(client, text, model=DEFAULT_MODEL):
    """Real token count from Claude's actual tokenizer via the API."""
    resp = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def approx_breakdown(text):
    """
    Illustrative-only token split using tiktoken's o200k_base encoding.
    NOT Claude's tokenizer. Returns None if tiktoken isn't installed.
    """
    if tiktoken is None:
        return None
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        # First call downloads the encoding file from OpenAI's blob
        # storage. If that's unreachable (offline, blocked network,
        # firewall), skip the approximation gracefully.
        return None
    ids = enc.encode(text)
    pieces = [enc.decode([i]) for i in ids]
    return pieces


def run_comparison(snippets, model=DEFAULT_MODEL, use_api=True):
    client = get_client() if use_api else None
    results = []
    for name, code in snippets.items():
        row = {
            "name": name,
            "code": code,
            "chars": len(code),
            "exact_tokens": None,
            "approx_tokens": None,
            "approx_pieces": None,
            "error": None,
        }
        if use_api:
            try:
                row["exact_tokens"] = count_tokens_exact(client, code, model)
            except Exception as e:
                row["error"] = str(e)

        pieces = approx_breakdown(code)
        if pieces is not None:
            row["approx_pieces"] = pieces
            row["approx_tokens"] = len(pieces)

        results.append(row)
    return results


def print_report(results):
    print("\n" + "=" * 72)
    print("TOKEN COMPARISON REPORT")
    print("=" * 72)

    # Sort by exact tokens when available, else approx
    def sort_key(r):
        if r["exact_tokens"] is not None:
            return (0, r["exact_tokens"])
        if r["approx_tokens"] is not None:
            return (1, r["approx_tokens"])
        return (2, 0)

    ordered = sorted(results, key=sort_key)

    header = f"{'name':<20} {'chars':>6} {'exact(API)':>11} {'approx(o200k)':>14}"
    print(header)
    print("-" * len(header))
    for r in ordered:
        exact = r["exact_tokens"] if r["exact_tokens"] is not None else (
            f"ERR" if r["error"] else "n/a"
        )
        approx = r["approx_tokens"] if r["approx_tokens"] is not None else "n/a"
        print(f"{r['name']:<20} {r['chars']:>6} {str(exact):>11} {str(approx):>14}")

    print("\n" + "-" * 72)
    print("DETAIL")
    print("-" * 72)
    for r in ordered:
        print(f"\n[{r['name']}]")
        print(f"  code: {r['code']!r}")
        if r["error"]:
            print(f"  API error: {r['error']}")
        else:
            print(f"  exact tokens (Claude, authoritative): {r['exact_tokens']}")
        if r["approx_pieces"] is not None:
            print(f"  approx pieces (o200k_base, illustrative only): {r['approx_pieces']}")

    print("\nReminder: only the 'exact(API)' column reflects Claude's real "
          "tokenizer. The o200k_base column is a different tokenizer shown "
          "only to give a rough visual sense of likely split points.\n")


def save_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare token counts across candidate syntax snippets.")
    parser.add_argument("--file", help="JSON file of {name: code} snippets to test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to count against")
    parser.add_argument("--add", nargs=2, metavar=("NAME", "CODE"),
                         action="append", help="Add a single snippet: --add name 'code here'")
    parser.add_argument("--no-api", action="store_true",
                         help="Skip the real Claude API call, use only the local approximation")
    parser.add_argument("--out", help="Save full JSON results to this path")
    args = parser.parse_args()

    snippets = {}
    if args.file:
        with open(args.file) as f:
            snippets.update(json.load(f))
    if args.add:
        for name, code in args.add:
            snippets[name] = code
    if not snippets:
        snippets = DEMO_SNIPPETS
        print("No snippets provided — running built-in demo set.\n"
              "Use --file candidates.json or --add name 'code' to test your own.")

    results = run_comparison(snippets, model=args.model, use_api=not args.no_api)
    print_report(results)

    if args.out:
        save_json(results, args.out)


if __name__ == "__main__":
    main()