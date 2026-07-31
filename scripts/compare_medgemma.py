"""Compare the fine-tuned arms against MedGemma-4b-it on the judged benchmarks.

MedGemma-4b-it is the ceiling this study approximates: Google's own medical
post-train of the same Gemma 3 4b base we fine-tune. The size-matched triple
(gemma-3-4b-it base -> our 4b-it SFT -> medgemma-4b-it) is the only strictly
apples-to-apples row set here; 270m and 1b have no MedGemma counterpart and are
included to show how the gap scales, not to be compared to a 4b model directly.

Reads `judged_summary.json` (gemma_med.judge), never a regex score. All three
statistics are reported because acc_raw alone is not interpretable once SFT
pulls a model off its instruction format:

  acc_raw       -- no_answer counted wrong (what a leaderboard would report)
  acc_answered  -- accuracy among responses that chose an option
  answer_rate   -- fraction that chose an option at all

Every model here must be judged by the SAME judge for the numbers to mean
anything (CLAUDE.md: the judge is an instrument -- freeze it). This script
refuses to build a table out of two judges rather than printing a footnote
nobody reads.

    python scripts/compare_medgemma.py
    python scripts/compare_medgemma.py --out eval/medgemma_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHMARKS = ("medqa", "medmcqa", "pubmedqa")

# (label, dir, group). Order is the reading order of the table.
MODELS = [
    ("gemma-3-270m-it (base)", "eval/gemma-3-270m-it-baseline", "270m"),
    ("  + medical SFT", "eval/traj/270m_it_full/step-4882", "270m"),
    ("gemma-3-1b-it (base)", "eval/gemma-3-1b-it-baseline", "1b"),
    ("  + medical SFT", "eval/traj/1b_it_full_v3/step-4882", "1b"),
    ("gemma-3-4b-it (base)", "eval/gemma-3-4b-it-baseline", "4b"),
    ("  + medical SFT", "eval/traj/4b_it_full/step-4882", "4b"),
    ("MedGemma-4b-it", "eval/medgemma-4b-it", "ref"),
]
REFERENCE = "eval/medgemma-4b-it"


def load(root: Path, d: str) -> dict | None:
    p = root / d / "judged_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def judge_id(s: dict) -> str:
    """provider/model, as `judged_summary.json` records it.

    Both halves matter: the same served model id can front different weights on
    different endpoints, and the same provider serves many models. Comparing on
    the model name alone would let a Kimi-judged dir sit in an Anthropic table.
    """
    p, m = s.get("judge_provider"), s.get("judge_model")
    if p or m:
        return f"{p or '?'}/{m or '?'}"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root")
    ap.add_argument("--out", default=None, help="Write the comparison as JSON")
    ap.add_argument(
        "--allow-mixed-judge",
        action="store_true",
        help="Print anyway when the models were not judged by one judge. Off by "
        "default: two judges' scores are not comparable.",
    )
    args = ap.parse_args()
    root = Path(args.root)

    loaded, missing = [], []
    for label, d, group in MODELS:
        s = load(root, d)
        (missing if s is None else loaded).append((label, d, group, s))
    if missing:
        print("not yet judged (run gemma_med.judge):", file=sys.stderr)
        for label, d, _, _ in missing:
            print(f"  {d}", file=sys.stderr)
    if not loaded:
        sys.exit("nothing judged yet")

    judges = {judge_id(s) for _, _, _, s in loaded}
    if len(judges) > 1:
        msg = f"models were judged by {len(judges)} different judges: {sorted(judges)}"
        if not args.allow_mixed_judge:
            sys.exit(f"refusing to build the table -- {msg}\nre-judge on one, or pass --allow-mixed-judge")
        print(f"WARNING: {msg}", file=sys.stderr)

    ref = next((s for _, d, _, s in loaded if d == REFERENCE), None)

    hdr = f"{'model':<26}"
    for b in BENCHMARKS:
        hdr += f" | {b:>8} {'answrd':>7} {'ans%':>6}"
    print(hdr)
    print("-" * len(hdr))

    rows_out = []
    prev_group = None
    for label, d, group, s in loaded:
        if prev_group is not None and group != prev_group:
            print()
        prev_group = group
        line = f"{label:<26}"
        rec = {"label": label.strip(), "dir": d, "group": group, "benchmarks": {}}
        for b in BENCHMARKS:
            v = s["results"].get(b)
            if not v:
                line += f" | {'--':>8} {'--':>7} {'--':>6}"
                continue
            line += f" | {v['acc_raw']:>8.2f} {v['acc_answered']:>7.2f} {v['answer_rate']:>6.1f}"
            rec["benchmarks"][b] = v
        print(line)
        rows_out.append(rec)

    # The headline the study actually asks: how far is each arm from the ceiling?
    if ref:
        print(f"\ngap to MedGemma-4b-it (acc_raw, negative = below the ceiling)")
        for label, d, group, s in loaded:
            if d == REFERENCE:
                continue
            deltas = []
            for b in BENCHMARKS:
                v, rv = s["results"].get(b), ref["results"].get(b)
                deltas.append(f"{b}={v['acc_raw'] - rv['acc_raw']:+7.2f}" if v and rv else f"{b}=     --")
            print(f"  {label.strip():<24} " + "  ".join(deltas))

    # A judge that contradicts its own extracted choice is a broken instrument.
    worst = max(
        ((s["results"][b].get("judge_self_disagreement_pct") or 0.0, f"{label.strip()}/{b}")
         for _, _, _, s in loaded for b in BENCHMARKS if s["results"].get(b)),
        default=(0.0, "-"),
    )
    print(f"\njudge: {sorted(judges)[0]}   worst self-disagreement: {worst[0]:.2f}% ({worst[1]})")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"judge": sorted(judges), "reference": REFERENCE, "models": rows_out}, indent=2)
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
