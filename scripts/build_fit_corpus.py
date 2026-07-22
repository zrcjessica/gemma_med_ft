"""Build the frozen jlens fit corpus to the paper's spec.

The jacobian-lens README specifies the released lenses were fit on "1000
sequences of 128 tokens from a pretraining-like corpus". This packs wikitext
into fixed-length token windows and writes them one-per-line, the format
probe_jlens.py expects.

Sequence length matters more than it looks: jlens.fitting skips the first
SKIP_FIRST_N_POSITIONS=16 positions of every sequence, so a 128-token window
contributes ~111 valid source positions while a single sentence contributes
~14. Do not shorten seq_tokens without re-reading that.

The output is FROZEN -- every checkpoint, size and arm must be fit on the same
file for the trajectory to mean anything. Regenerating with a different seed or
spec invalidates comparison with existing metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

HEADER_RE = re.compile(r"^=+ .* =+$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", required=True,
                   help="Dir of wikitext train-*.parquet shards.")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-seqs", type=int, default=1000)
    p.add_argument("--seq-tokens", type=int, default=128,
                   help="Target length after the probe re-tokenizes.")
    p.add_argument("--headroom", type=int, default=16,
                   help="Extra tokens per window to absorb decode/re-encode drift.")
    p.add_argument("--candidate-mult", type=int, default=20,
                   help="Chunk this many times n_seqs before sampling, for diversity.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def iter_lines(parquet_dir: Path):
    for shard in sorted(parquet_dir.glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(columns=["text"]):
            for text in batch.column("text").to_pylist():
                line = (text or "").strip()
                if line and not HEADER_RE.match(line):
                    yield line


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    window = args.seq_tokens + args.headroom
    n_candidates = args.n_seqs * args.candidate_mult

    ids: list[int] = []
    chunks: list[list[int]] = []
    for line in iter_lines(Path(args.parquet_dir)):
        ids.extend(tok(line, add_special_tokens=False).input_ids)
        while len(ids) >= window:
            chunks.append(ids[:window])
            ids = ids[window:]
        if len(chunks) >= n_candidates:
            break

    if len(chunks) < args.n_seqs:
        raise SystemExit(f"only {len(chunks)} chunks available, need {args.n_seqs}")

    rng = random.Random(args.seed)
    picked = rng.sample(chunks, args.n_seqs)

    seqs, lens = [], []
    for chunk in picked:
        text = " ".join(tok.decode(chunk).split())
        n = len(tok(text).input_ids)
        if n < args.seq_tokens:
            continue
        seqs.append(text)
        lens.append(n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(seqs) + "\n")

    manifest = {
        "n_seqs": len(seqs),
        "seq_tokens_target": args.seq_tokens,
        "window_tokens": window,
        "retokenized_len": {"min": min(lens), "max": max(lens)},
        "source": str(args.parquet_dir),
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "spec": "jacobian-lens README: 1000 sequences of 128 tokens, pretraining-like",
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"dropped {args.n_seqs - len(seqs)} short sequences")


if __name__ == "__main__":
    main()
