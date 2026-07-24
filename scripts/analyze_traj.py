"""Join the behavioral trajectory to the jlens trajectory.

Raw accuracy conflates two things once SFT pulls an -it model off its
instruction format: whether the model knows the answer, and whether it still
emits a parseable one. Unparsed responses score as wrong, so a collapsing
format looks identical to collapsing knowledge. This splits them:

  acc_raw      -- unparsed counted wrong (what evaluate.py reports)
  acc_parsed   -- accuracy among responses that parsed (medical signal)
  parse_rate   -- fraction that parsed (format compliance)

Both belong in any figure; acc_raw alone is not interpretable mid-trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BENCHMARKS = ("medqa", "medmcqa", "pubmedqa")


def load_behavioral(traj_dir: Path, baseline: Path | None):
    steps = {}
    dirs = [(int(re.search(r"step-(\d+)", str(d)).group(1)), d) for d in traj_dir.glob("step-*")]
    if baseline:
        dirs.append((0, baseline))
    for step, d in sorted(dirs):
        row = {}
        for b in BENCHMARKS:
            pred = d / f"{b}_predictions.jsonl"
            if not pred.exists():
                continue
            n = n_parsed = n_correct = 0
            for line in open(pred):
                r = json.loads(line)
                n += 1
                if r["pred"] is not None:
                    n_parsed += 1
                    n_correct += r["correct"]
            row[b] = {
                "n": n,
                "acc_raw": 100.0 * n_correct / n,
                "acc_parsed": 100.0 * n_correct / n_parsed if n_parsed else float("nan"),
                "parse_rate": 100.0 * n_parsed / n,
            }
        if row:
            steps[step] = row
    return steps


def load_lens(metrics: Path):
    out = {}
    for line in open(metrics):
        r = json.loads(line)
        c, d = r["concordance"], r["drift"]
        out[r["step"]] = {
            "drift_relfro_max": max(d["relfro_curve"]),
            "drift_cos_min": min(d["cos_curve"]),
            "kl_general_last": c["general"]["kl_curve"][-1],
            "kl_medical_last": c["medical"]["kl_curve"][-1],
            "top1_general_last": c["general"]["agree_curve"][-1],
            "top1_medical_last": c["medical"]["agree_curve"][-1],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True, help="eval/traj/<tag>/ with step-*/ subdirs")
    ap.add_argument("--lens-metrics", required=True, help="eval/jlens/<tag>/metrics.jsonl")
    ap.add_argument("--baseline", default=None, help="t=0 eval dir (the untuned base)")
    ap.add_argument("--out", default=None, help="write joined rows as JSON")
    args = ap.parse_args()

    beh = load_behavioral(Path(args.traj_dir), Path(args.baseline) if args.baseline else None)
    lens = load_lens(Path(args.lens_metrics))

    hdr = f"{'step':>6} {'drift':>7} {'KLgen':>7} {'KLmed':>7} |"
    for b in BENCHMARKS:
        hdr += f" {b[:7]:>7} {'parsed':>7} {'parse%':>7} |"
    print(hdr)
    print("-" * len(hdr))

    joined = []
    for step in sorted(beh):
        L = lens.get(step, {})
        line = (
            f"{step:>6} {L.get('drift_relfro_max', float('nan')):>7.3f} "
            f"{L.get('kl_general_last', float('nan')):>7.3f} {L.get('kl_medical_last', float('nan')):>7.3f} |"
        )
        for b in BENCHMARKS:
            v = beh[step].get(b)
            line += (
                f" {v['acc_raw']:>7.2f} {v['acc_parsed']:>7.2f} {v['parse_rate']:>7.1f} |"
                if v else " " * 25 + "|"
            )
        print(line)
        joined.append({"step": step, "lens": L, "behavioral": beh[step]})

    # The whole point of the join: does the lens signal track behavior, and
    # which behavior -- medical accuracy or format compliance?
    common = [j for j in joined if j["lens"] and "medmcqa" in j["behavioral"]]
    if len(common) >= 3:
        def corr(xs, ys):
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            dx = sum((x - mx) ** 2 for x in xs) ** 0.5
            dy = sum((y - my) ** 2 for y in ys) ** 0.5
            return num / (dx * dy) if dx and dy else float("nan")

        drift = [j["lens"]["drift_relfro_max"] for j in common]
        print()
        for b in BENCHMARKS:
            rows = [j for j in common if b in j["behavioral"]]
            if len(rows) < 3:
                continue
            d = [j["lens"]["drift_relfro_max"] for j in rows]
            print(
                f"pearson(drift, {b:<9}) acc_raw={corr(d, [j['behavioral'][b]['acc_raw'] for j in rows]):+.3f}"
                f"  acc_parsed={corr(d, [j['behavioral'][b]['acc_parsed'] for j in rows]):+.3f}"
                f"  parse_rate={corr(d, [j['behavioral'][b]['parse_rate'] for j in rows]):+.3f}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(joined, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
