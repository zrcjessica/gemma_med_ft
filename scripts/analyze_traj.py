"""Join the behavioral trajectory to the jlens trajectory.

Raw accuracy conflates two things once SFT pulls an -it model off its
instruction format: whether the model knows the answer, and whether it still
gives one. Unanswered responses score as wrong, so a collapsing format looks
identical to collapsing knowledge. This splits them:

  acc_raw       -- no_answer counted wrong
  acc_answered  -- accuracy among responses that chose an option (medical signal)
  answer_rate   -- fraction that chose an option (format compliance)

Both belong in any figure; acc_raw alone is not interpretable mid-trajectory.

Scores come from `*_judgments.jsonl` (gemma_med.judge), not from a regex over
the generations. Judge a trajectory before analysing it:

    python -m gemma_med.judge --root eval/traj/<tag>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BENCHMARKS = ("medqa", "medmcqa", "pubmedqa")


def load_behavioral(traj_dir: Path, baseline: Path | None):
    steps = {}
    dirs = [(int(re.search(r"step-(\d+)", str(d)).group(1)), d) for d in traj_dir.glob("step-*")]
    if baseline:
        dirs.append((0, baseline))
    missing = []
    for step, d in sorted(dirs):
        row = {}
        for b in BENCHMARKS:
            jpath = d / f"{b}_judgments.jsonl"
            if not jpath.exists():
                if (d / f"{b}_predictions.jsonl").exists():
                    missing.append(f"{d.name}/{b}")
                continue
            # Deduplicate by idx: a resumed judge run appends, so a retried item
            # can appear twice. Last write wins, matching gemma_med.judge.
            recs = {r["idx"]: r for r in (json.loads(l) for l in open(jpath))}
            n = len(recs)
            n_ans = sum(1 for r in recs.values() if r["answered"])
            n_correct = sum(1 for r in recs.values() if r["correct"])
            row[b] = {
                "n": n,
                "acc_raw": 100.0 * n_correct / n,
                "acc_answered": 100.0 * n_correct / n_ans if n_ans else float("nan"),
                "answer_rate": 100.0 * n_ans / n,
            }
        if row:
            steps[step] = row
    if missing:
        print(f"warning: unjudged (run gemma_med.judge): {', '.join(missing)}", file=sys.stderr)
    return steps


def load_lens(metrics: Path):
    """Both faithfulness read-outs, named for what they are.

    `*_layermean` is the probe's `kl_final`/`top1_final`: the mean over the last
    n//5 source layers, and the statistic every claim in the deck uses.
    `*_last` is the single final layer. The two disagree in sign on 270m medical
    (1.861 -> 2.137 vs 5.388 -> 5.340), so a figure must say which it plots.
    """
    out = {}
    for line in open(metrics):
        r = json.loads(line)
        c, d = r["concordance"], r["drift"]
        out[r["step"]] = {
            "drift_relfro_max": max(d["relfro_curve"]),
            "drift_cos_min": min(d["cos_curve"]),
            "kl_general_layermean": c["general"]["kl_final"],
            "kl_medical_layermean": c["medical"]["kl_final"],
            "top1_general_layermean": c["general"]["top1_final"],
            "top1_medical_layermean": c["medical"]["top1_final"],
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

    hdr = f"{'step':>6} {'drift':>7} {'KLgen':>7} {'KLmed':>7} |"  # KL = layer mean
    for b in BENCHMARKS:
        hdr += f" {b[:7]:>7} {'answrd':>7} {'ans%':>7} |"
    print(hdr)
    print("-" * len(hdr))

    joined = []
    for step in sorted(beh):
        L = lens.get(step, {})
        line = (
            f"{step:>6} {L.get('drift_relfro_max', float('nan')):>7.3f} "
            f"{L.get('kl_general_layermean', float('nan')):>7.3f} "
            f"{L.get('kl_medical_layermean', float('nan')):>7.3f} |"
        )
        for b in BENCHMARKS:
            v = beh[step].get(b)
            line += (
                f" {v['acc_raw']:>7.2f} {v['acc_answered']:>7.2f} {v['answer_rate']:>7.1f} |"
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
                f"  acc_answered={corr(d, [j['behavioral'][b]['acc_answered'] for j in rows]):+.3f}"
                f"  answer_rate={corr(d, [j['behavioral'][b]['answer_rate'] for j in rows]):+.3f}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(joined, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
