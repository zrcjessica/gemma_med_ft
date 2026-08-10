#!/usr/bin/env python3
"""Merge per-checkpoint band.jsonl from a fanned-out band probe into one
trajectory file, deduping the step-0 row that every worker carries as its seed.

Same contract as merge_jlens_fanout.py, plus a completeness report: the band
figures compare arms against each other and against t=0, so a silently missing
checkpoint would read as "nothing happened there" rather than "not measured".
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fanout-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect-run-dir", default=None,
                   help="Training run dir; warns about checkpoint-* with no band row.")
    args = p.parse_args()

    rows: dict[int, dict] = {}
    for bp in sorted(Path(args.fanout_dir).glob("ck*/band.jsonl")):
        for line in bp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["step"] in rows and r["step"] != 0:
                raise SystemExit(f"duplicate non-base step {r['step']} in {bp}")
            rows[r["step"]] = r

    if not rows:
        raise SystemExit(f"no band.jsonl rows under {args.fanout_dir}")
    if 0 not in rows:
        print("WARNING: no t=0 row -- the ribbon/delta figures need the untuned "
              "base and will be blank for this arm")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for step in sorted(rows):
            f.write(json.dumps(rows[step]) + "\n")
    print(f"{len(rows)} checkpoints -> {out}")
    print("steps:", sorted(rows))

    if args.expect_run_dir:
        want = {int(d.name.split("-")[1])
                for d in Path(args.expect_run_dir).glob("checkpoint-*") if d.is_dir()}
        missing = sorted(want - set(rows))
        if missing:
            print(f"WARNING: {len(missing)} checkpoint(s) with no band row: {missing}")
        else:
            print(f"complete: all {len(want)} checkpoints present, plus t=0")


if __name__ == "__main__":
    main()
