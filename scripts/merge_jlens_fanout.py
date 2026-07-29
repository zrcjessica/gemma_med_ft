#!/usr/bin/env python3
"""Merge per-checkpoint metrics.jsonl from a fanned-out jlens probe into one
trajectory file, deduping the step-0 row that every worker carries as its seed."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fanout-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows: dict[int, dict] = {}
    for mp in sorted(Path(args.fanout_dir).glob("ck*/metrics.jsonl")):
        for line in mp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["step"] in rows and r["step"] != 0:
                raise SystemExit(f"duplicate non-base step {r['step']} in {mp}")
            rows[r["step"]] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for step in sorted(rows):
            f.write(json.dumps(rows[step]) + "\n")
    print(f"{len(rows)} checkpoints -> {out}")
    print("steps:", sorted(rows))


if __name__ == "__main__":
    main()
