"""A/B the judge's decode config: provider default vs. the frozen temperature=0.

Four passes over the same rows with the same judge model:

    A1, A2   no sampling params sent  -- what every judgment before 2026-08-05 used
    B1, B2   the frozen config        -- temperature=0 (+ seed=0 on the self-hosted server)

A1 vs A2 and B1 vs B2 measure each config's *self*-consistency (the thing
`temperature` is supposed to buy). A1 vs B1 measures whether adopting the new
config moves the reported numbers, i.e. whether already-judged dirs are still
comparable to ones judged from here on.

Runs sync on both providers (batch would be half price on Claude, but ~11 min per
pass and this is 4 of them over a few hundred rows). Writes raw judgments per
pass so a disagreement can be inspected rather than just counted.

    # Claude, from olab1
    PYTHONPATH=. .venv-judge/bin/python scripts/ab_judge_decode.py --n 300

    # Self-hosted Kimi -- BigPurple compute node only, so submit it
    sbatch scripts/ab_judge_decode.sbatch

`--provider local` is the judge of record, so that is the arm that decides
whether existing judgments stay comparable; the Claude arm is the fallback
provider and is cheap enough to run as a cross-check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gemma_med import judge as J

PRED_DIR = Path("eval/traj/4b_it_full/step-512")

# (name, decode-to-send). The empty dict reproduces the pre-2026-08-05 wire: no
# sampling params at all, so the provider samples at its own default.
VARIANTS = {
    "anthropic": [("A1", {}), ("A2", {}), ("B1", {"temperature": 0.0}), ("B2", {"temperature": 0.0})],
    "local": [
        ("A1", {}),
        ("A2", {}),
        ("B1", {"temperature": 0.0, "seed": J.JUDGE_SEED}),
        ("B2", {"temperature": 0.0, "seed": J.JUDGE_SEED}),
    ],
}


def make_judge(provider: str, decode: dict, model: str | None = None):
    """A judge pinned to `decode`, whatever the module defaults say.

    The Anthropic path builds its wire params from the module-level
    `anthropic_decode`, not from the instance, so that is what has to be patched
    -- overriding the method alone would change the recorded config and not the
    request. The OpenAI-compatible path reads `self.decode_config()` in
    `params()`, so the instance override is enough there.
    """
    cls, default_model, _ = J.PROVIDERS[provider]
    model = model or default_model
    if cls is J.AnthropicJudge:
        J.anthropic_decode = lambda _m, _d=dict(decode): dict(_d)
        return cls(model)
    judge = cls(model, os.environ.get("JUDGE_BASE_URL"))
    judge.decode_config = lambda _d=dict(decode): dict(_d)
    return judge


def run_pass(name: str, provider: str, rows, decode: dict, out_dir: Path, concurrency: int, model=None) -> dict:
    orig = J.anthropic_decode
    try:
        judge = make_judge(provider, decode, model)
        out = out_dir / f"{name}.jsonl"
        out.unlink(missing_ok=True)
        n_err = J.judge_sync(judge, rows, out, concurrency)
        if n_err:
            print(f"  {name}: {n_err} errored", flush=True)
    finally:
        J.anthropic_decode = orig
    return {r["idx"]: r for r in (json.loads(l) for l in open(out))}


def compare(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"pair": f"{label_a} vs {label_b}", "n": 0}
    chosen = [i for i in shared if a[i]["chosen"] != b[i]["chosen"]]
    verdict = [i for i in shared if a[i]["verdict"] != b[i]["verdict"]]
    acc = lambda d: 100.0 * sum(d[i]["correct"] for i in shared) / len(shared)
    ans = lambda d: 100.0 * sum(d[i]["answered"] for i in shared) / len(shared)
    return {
        "pair": f"{label_a} vs {label_b}",
        "n": len(shared),
        # `chosen` is the scored field -- `correct`/`answered` are both derived
        # from it, so a chosen-flip is the only kind that can move a number.
        "chosen_flips": len(chosen),
        "chosen_flip_pct": round(100.0 * len(chosen) / len(shared), 2),
        "verdict_flips": len(verdict),
        "acc_raw": (round(acc(a), 2), round(acc(b), 2)),
        "acc_raw_delta_pp": round(acc(b) - acc(a), 2),
        "answer_rate": (round(ans(a), 2), round(ans(b), 2)),
        "flipped_idxs": chosen[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=list(VARIANTS))
    ap.add_argument("--pred-dir", default=str(PRED_DIR))
    ap.add_argument("--benchmarks", nargs="+", default=["medqa", "medmcqa"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    # The self-hosted server answers to its --served-model-name, not the HF name,
    # so the caller (scripts/_judge_provider.sh) discovers it and passes it in.
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL") or None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    J.load_dotenv()
    out_dir = Path(args.out or f"eval/decode_probe/judge_ab_{args.provider}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = args.model or J.PROVIDERS[args.provider][1]
    report = {
        "provider": args.provider,
        "judge_model": model,
        "pred_dir": args.pred_dir,
        "n_requested": args.n,
        "benchmarks": {},
    }

    for bench in args.benchmarks:
        pred = Path(args.pred_dir) / f"{bench}_predictions.jsonl"
        rows = J.load_rows(pred, bench, args.n)
        print(f"\n== {args.provider}/{bench}: {len(rows)} rows x 4 passes ==", flush=True)

        passes = {}
        for name, decode in VARIANTS[args.provider]:
            print(f"  {bench} {name} decode={decode or 'provider default'}", flush=True)
            passes[name] = run_pass(
                f"{bench}_{name}", args.provider, rows, decode, out_dir, args.concurrency, model
            )

        report["benchmarks"][bench] = {
            "n_rows": len(rows),
            "self_consistency_old": compare(passes["A1"], passes["A2"], "A1", "A2"),
            "self_consistency_new": compare(passes["B1"], passes["B2"], "B1", "B2"),
            "old_vs_new": compare(passes["A1"], passes["B1"], "A1", "B1"),
            "old_vs_new_alt": compare(passes["A2"], passes["B2"], "A2", "B2"),
        }
        print(json.dumps(report["benchmarks"][bench], indent=2), flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
