"""Materialize the frozen jlens fit corpus using jlens's own WikiText loader.

The jacobian-lens README specifies the released lenses were fit on "1000
sequences of 128 tokens from a pretraining-like corpus", and the repo ships
the loader that produces them: jlens.examples.load_wikitext_prompts takes the
first n records of >= min_chars from wikitext-103-raw-v1 and leaves the
truncation to fit(max_seq_len=128). We call that function rather than
reimplementing it, so our corpus cannot drift from the reference.

We only materialize it to a file because this is a *trajectory* study: the same
corpus must be fed to every checkpoint, size and arm, so it has to be frozen and
version-controlled rather than re-streamed per run.

Sequence length is the thing not to fiddle with -- jlens.fitting skips the first
SKIP_FIRST_N_POSITIONS=16 positions of every sequence, so a 128-token record
contributes ~111 valid source positions where a single sentence contributes ~14.

Run in .venv-jlens (needs jlens + datasets). Output is FROZEN: regenerating with
a different n/min_chars invalidates comparison with existing metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jlens.examples import load_wikitext_prompts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-prompts", type=int, default=1000)
    p.add_argument("--min-chars", type=int, default=600,
                   help="jlens default; ~600 chars clears 128 tokens.")
    p.add_argument("--tokenizer", default=None,
                   help="If set, report the token-length distribution.")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = load_wikitext_prompts(args.n_prompts, min_chars=args.min_chars)
    if len(prompts) != args.n_prompts:
        raise SystemExit(f"got {len(prompts)} prompts, wanted {args.n_prompts}")

    # The corpus file is one sequence per line; wikitext records are single-line
    # but normalize defensively so a stray newline cannot split one in two.
    seqs, rewrapped = [], 0
    for text in prompts:
        flat = " ".join(text.split())
        if len(text.strip().splitlines()) > 1:
            rewrapped += 1
        seqs.append(flat)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(seqs) + "\n")

    manifest = {
        "n_prompts": len(seqs),
        "min_chars": args.min_chars,
        "rewrapped_multiline": rewrapped,
        "source": "jlens.examples.load_wikitext_prompts (Salesforce/wikitext, wikitext-103-raw-v1, train)",
        "spec": "jacobian-lens README: 1000 sequences of 128 tokens, pretraining-like",
        "truncation": "left to jlens.fit(max_seq_len=128)",
    }
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        ns = [len(tok(s).input_ids) for s in seqs]
        valid = [max(0, min(n, 128) - 1 - 16) for n in ns]
        manifest["tokens"] = {"min": min(ns), "max": max(ns),
                              "under_128": sum(1 for n in ns if n < 128)}
        manifest["valid_source_positions"] = sum(valid)

    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
