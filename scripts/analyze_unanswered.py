"""Why did the judge return answered=False? Bucket the unanswered rows by cause.

`answer_rate` says how often the model failed to commit to an option; it does not
say why, and the two candidate causes call for opposite fixes. A response that was
still reasoning when it hit --max-new-tokens wants a bigger token budget; a
response stuck repeating one sentence wants a different decode. This separates
them (docs/BEHAVIORAL_EVAL.md §8).

    .venv-probe/bin/python scripts/analyze_unanswered.py eval/traj/4b_it_full/step-*
    .venv-probe/bin/python scripts/analyze_unanswered.py --examples eval/medgemma-4b-it

Runs in .venv-probe (needs `tokenizers` + `datasets`), on olab1, over files
already on disk. Predictions written before evaluate.py saved `finish_reason` are
re-tokenised to recover it; newer ones are read straight off the row.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_med.evaluate import BENCHMARKS

DEFAULT_MAX_NEW = 1024
BENCHES = ["medqa", "medmcqa", "pubmedqa"]

# Buckets, in report order. The first two are the ones that matter: both end at
# the token cap, but only B is fixable with a bigger cap.
BUCKETS = [
    ("loop", "at cap, verbatim repetition loop"),
    ("long_cot", "at cap, still reasoning -- real truncation"),
    ("placeholder", "emitted the prompt's literal '<letter>'"),
    ("no_choice", "finished cleanly without choosing"),
    ("judge_miss", "named a letter the judge did not take"),
    ("empty", "empty generation"),
]

PLACEHOLDER_RE = re.compile(r"answer\s*[:\-]\s*<?\s*(letter|yes/no/maybe)\s*>?", re.I)
ANSWER_RE = re.compile(r"answer\s*[:\-]\s*\**\s*<?\s*([A-Da-d]|yes|no|maybe)\b", re.I)


def find_tokenizer(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    roots = [os.environ.get("HF_HOME"), str(Path.home() / ".cache/huggingface")]
    for r in filter(None, roots):
        hits = glob.glob(f"{r}/hub/models--google--gemma-3-*/snapshots/*/tokenizer.json")
        if hits:
            return hits[0]
    return None


def uniq_shingle_ratio(text: str, n: int = 8, tail_words: int = 300) -> float:
    """Unique-8-gram share over the tail. Digits normalised, because the loops
    often come out as an enumeration whose counter keeps incrementing
    ('1. Leukemia 2. Leukemia ... 182. Leukemia')."""
    w = re.sub(r"\d+", "#", text).split()[-tail_words:]
    if len(w) < 4 * n:
        return 1.0
    sh = [" ".join(w[i : i + n]) for i in range(len(w) - n + 1)]
    return len(set(sh)) / len(sh)


def loop_onset_words(text: str, n: int = 8) -> int | None:
    """Word index where the first repeated 8-gram was first seen -- i.e. where
    the model derailed. If that is far below the cap, no token budget saves it."""
    w = re.sub(r"\d+", "#", text).split()
    seen: dict[str, int] = {}
    for i in range(len(w) - n + 1):
        s = " ".join(w[i : i + n])
        if s in seen:
            return seen[s]
        seen[s] = i
    return None


def classify(out: str, at_cap: bool, loop_thresh: float) -> str:
    if at_cap:
        return "loop" if uniq_shingle_ratio(out) < loop_thresh else "long_cot"
    if not out.strip():
        return "empty"
    if PLACEHOLDER_RE.search(out[-300:]):
        return "placeholder"
    if ANSWER_RE.search(out[-400:]):
        return "judge_miss"
    return "no_choice"


_BENCH_CACHE: dict[str, list] = {}


def load_dir(d: Path, bench: str, tok, max_new: int):
    preds = [json.loads(l) for l in open(d / f"{bench}_predictions.jsonl")]
    for i, r in enumerate(preds):
        r.setdefault("idx", i)
    by_idx = {r["idx"]: r for r in preds}
    for line in open(d / f"{bench}_judgments.jsonl"):
        j = json.loads(line)
        p = by_idx[j["idx"]]
        if "finish_reason" in p:
            at_cap = p["finish_reason"] == "length"
        elif tok is not None:
            at_cap = len(tok.encode(p["output"], add_special_tokens=False).ids) >= max_new - 1
        else:
            sys.exit(
                f"{d}: predictions predate finish_reason and no tokenizer was found. "
                "Pass --tokenizer /path/to/tokenizer.json (any Gemma 3 one)."
            )
        yield j, p["output"], at_cap


def analyse(d: Path, bench: str, tok, args):
    n = capped = capped_answered = 0
    un = Counter()
    onsets: list[int] = []
    ex: dict[str, tuple] = {}
    for j, out, at_cap in load_dir(d, bench, tok, args.max_new_tokens):
        n += 1
        capped += at_cap
        if j["answered"]:
            capped_answered += at_cap
            continue
        k = classify(out, at_cap, args.loop_threshold)
        un[k] += 1
        ex.setdefault(k, (j["idx"], out))
        if k == "loop":
            o = loop_onset_words(out)
            if o is not None:
                onsets.append(o)
    return dict(n=n, capped=capped, capped_answered=capped_answered, un=un, onsets=onsets, ex=ex)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="eval dirs holding *_predictions.jsonl + *_judgments.jsonl")
    ap.add_argument("--benchmarks", nargs="+", default=BENCHES, choices=BENCHES)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW, help="cap the generations used")
    ap.add_argument("--loop-threshold", type=float, default=0.6, help="unique-8-gram share below which a tail is a loop")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json, for runs predating finish_reason")
    ap.add_argument("--examples", action="store_true", help="print one generation per bucket")
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    args = ap.parse_args()

    tok = None
    tpath = find_tokenizer(args.tokenizer)
    if tpath:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(tpath)

    keys = [k for k, _ in BUCKETS]
    print(f"{'run':<30}{'bench':<9}{'n':>6}{'unans':>7}{'@cap':>6}{'@cap+ans':>9}  " + "".join(f"{k:>13}" for k in keys))
    rows, allex = [], {}
    for d in args.dirs:
        d = Path(d)
        for b in args.benchmarks:
            if not (d / f"{b}_judgments.jsonl").exists():
                continue
            r = analyse(d, b, tok, args)
            tag = "/".join(d.parts[-2:]) if d.name.startswith(("step-", "t0", "t1")) else d.name
            print(
                f"{tag:<30}{b:<9}{r['n']:>6}{sum(r['un'].values()):>7}{r['capped']:>6}{r['capped_answered']:>9}  "
                + "".join(f"{r['un'][k]:>13}" for k in keys)
            )
            rows.append(
                {
                    "dir": str(d),
                    "bench": b,
                    "n": r["n"],
                    "unanswered": sum(r["un"].values()),
                    "at_cap": r["capped"],
                    "at_cap_but_answered": r["capped_answered"],
                    "buckets": {k: r["un"][k] for k in keys},
                    "loop_onset_words_p50": pct(r["onsets"], 0.5),
                    "loop_onset_words_p90": pct(r["onsets"], 0.9),
                }
            )
            for k, v in r["ex"].items():
                allex.setdefault((k, b), (tag,) + v)

    print("\nloop onset (word index where the repeated span was first seen):")
    for r in rows:
        if r["buckets"]["loop"]:
            print(f"  {r['dir']:<45}{r['bench']:<9} p50={r['loop_onset_words_p50']:>5}  p90={r['loop_onset_words_p90']:>5}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")

    if args.examples:
        for (k, b), (tag, idx, out) in sorted(allex.items()):
            print("\n" + "=" * 100)
            print(f"[{k}] {tag} {b} idx={idx}  ({dict(BUCKETS)[k]})")
            head, tail = out[:350].replace("\n", " "), out[-250:].replace("\n", " ")
            print(head + (f"   ...   {tail}" if len(out) > 600 else ""))


if __name__ == "__main__":
    main()
