"""Cross-size red-flag panel: does medical SFT move the silent-red-flag gap, at any scale?

Reads the per-run artifacts pulled back from BigPurple (`eval/redflag/<arm>_ck<step>/`)
and emits one tidy table plus the panel figure. Everything here is arithmetic over
JSON medlens already wrote -- no model, no GPU, no npz.

TWO STATISTICS, NEVER ONE NUMBER. medlens's `headline_statistic` is
`jac_hit_frac` (fraction of band positions where a risk concept enters the top-k),
and the dissociation *rate* is thresholded on that. The AUROC that separates
implicit from benign best is `jac_z_margin_end` (the z-scored risk-vs-control
margin at the final position). They are different quantities and they do not
agree -- at 4b step 512, 0.62 vs 0.72. Reporting "the AUROC" without naming which
is how a headline claim gets built on a statistic nobody chose; both columns are
always emitted, and `base_*` is the logit-lens control for each.

    python scripts/redflag/crossmodel.py --runs eval/redflag --out figs/redflag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Parameter count, for the x-axis and for sorting. Sizes are not comparable on
# raw J -- different d_model and layer counts -- which is exactly why this file
# only ever plots derived scalars (AUROC, rates).
SIZE_PARAMS = {"270m": 0.27, "1b": 1.0, "4b": 4.3, "12b": 12.2, "27b": 27.4}

STATS = [
    ("jac_z_margin_end", "base_z_margin_end", "z-margin @ final position"),
    ("jac_hit_frac", "base_hit_frac", "hit-frac in band (medlens headline)"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="eval/redflag", help="root of <arm>_ck<step>/ dirs")
    p.add_argument("--out", default="figs/redflag")
    p.add_argument("--steps", nargs="+", type=int, default=[0, 512, 4882])
    return p.parse_args()


def parse_tag(tag: str) -> tuple[str, str, int]:
    """`27b_it_ck512` -> ("27b_it", "27b", 512)."""
    arm, _, step = tag.rpartition("_ck")
    if not arm or not step.isdigit():
        raise ValueError(f"cannot parse arm/step from {tag!r}")
    return arm, arm.split("_")[0], int(step)


def escalation(judged: Path) -> dict:
    """Rule-based escalation and recognition rate per vignette arm."""
    rows = [json.loads(l) for l in judged.read_text().splitlines() if l.strip()]
    out = {}
    for varm in ("benign", "implicit", "explicit"):
        sub = [r for r in rows if r["arm"] == varm]
        if not sub:
            continue
        out[varm] = {
            "n": len(sub),
            "escalated": sum(bool(r["escalated_rule"]) for r in sub) / len(sub),
            "recognized": sum(bool(r["recognized_rule"]) for r in sub) / len(sub),
            "median_gen_tokens": sorted(r["n_gen_tokens"] for r in sub)[len(sub) // 2],
        }
    return out


def collect(runs_root: Path, steps: list[int]) -> list[dict]:
    recs = []
    for d in sorted(runs_root.glob("*_ck*")):
        if not (d / "analysis_all.json").exists() or not (d / "judged.jsonl").exists():
            continue
        arm, size, step = parse_tag(d.name)
        if step not in steps:
            continue
        a = json.loads((d / "analysis_all.json").read_text())
        rec = {
            "tag": d.name, "arm": arm, "size": size, "params_b": SIZE_PARAMS.get(size),
            "step": step, "band": a.get("band"), "band_source": a.get("band_source"),
            "headline_statistic": a.get("headline_statistic"),
            "threshold_source": a.get("threshold_source"),
            "dissociation": a["dissociation"]["rate"],
            "dissociation_ci95": a["dissociation"].get("ci95"),
            "dissociation_n": a["dissociation"].get("n"),
            "escalation": escalation(d / "judged.jsonl"),
        }
        for jac, base, _label in STATS:
            for key in (jac, base):
                blk = a["auroc"].get(key)
                if blk:
                    rec[f"auroc_{key}"] = blk["auroc"]
                    rec[f"auroc_{key}_ci95"] = blk.get("ci95")
        recs.append(rec)
    recs.sort(key=lambda r: (SIZE_PARAMS.get(r["size"], 0), r["step"]))
    return recs


def write_tables(recs: list[dict], out: Path) -> str:
    lines = []
    lines.append("## Escalation rate (rule-based `escalated_rule`, n=50/arm)\n")
    lines.append("| size | step | band | benign | implicit | explicit | explicit-implicit |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in recs:
        e = r["escalation"]
        if not all(k in e for k in ("benign", "implicit", "explicit")):
            continue
        gap = e["explicit"]["escalated"] - e["implicit"]["escalated"]
        lines.append(
            f"| {r['size']} | {r['step']} | {r['band']} | {e['benign']['escalated']:.0%} | "
            f"**{e['implicit']['escalated']:.0%}** | {e['explicit']['escalated']:.0%} | {gap:.0%} |")

    for jac, base, label in STATS:
        lines.append(f"\n## Internal signal, implicit vs benign — `{jac}` ({label})\n")
        lines.append(f"| size | step | AUROC (J-lens `{jac}`) | AUROC (logit-lens `{base}`) | J − logit |")
        lines.append("|---|---|---|---|---|")
        for r in recs:
            j, b = r.get(f"auroc_{jac}"), r.get(f"auroc_{base}")
            if j is None:
                continue
            ci = r.get(f"auroc_{jac}_ci95") or [None, None]
            ci_s = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else ""
            b_s = f"{b:.3f}" if b is not None else "—"
            d_s = f"{j - b:+.3f}" if b is not None else "—"
            lines.append(f"| {r['size']} | {r['step']} | {j:.3f}{ci_s} | {b_s} | {d_s} |")

    lines.append("\n## Dissociation rate\n")
    lines.append(f"Thresholded on `{recs[0]['headline_statistic'] if recs else '?'}`, "
                 f"**not** on the z-margin above — and each run's threshold is "
                 f"`{recs[0]['threshold_source'] if recs else '?'}`. Three self-calibrated "
                 f"numbers per arm, so read down a column with care and never as a trend.\n")
    lines.append("| size | step | rate | ci95 | n |")
    lines.append("|---|---|---|---|---|")
    for r in recs:
        ci = r.get("dissociation_ci95") or [None, None]
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "—"
        lines.append(f"| {r['size']} | {r['step']} | {r['dissociation']:.2f} | {ci_s} | {r['dissociation_n']} |")

    text = "\n".join(lines) + "\n"
    (out / "crossmodel.md").write_text(text)
    return text


def plot(recs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted({r["step"] for r in recs})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1-2: AUROC vs size, one line per step, one panel per statistic.
    for ax, (jac, base, label) in zip(axes[:2], STATS):
        for step in steps:
            pts = [(r["params_b"], r.get(f"auroc_{jac}")) for r in recs
                   if r["step"] == step and r.get(f"auroc_{jac}") is not None]
            if pts:
                ax.plot(*zip(*sorted(pts)), "o-", label=f"step {step}")
        # Logit-lens control at t=0 as the floor to beat.
        ctl = [(r["params_b"], r.get(f"auroc_{base}")) for r in recs
               if r["step"] == 0 and r.get(f"auroc_{base}") is not None]
        if ctl:
            ax.plot(*zip(*sorted(ctl)), "s--", color="0.6", label="logit-lens ctl (t=0)")
        ax.axhline(0.5, color="0.8", lw=1, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([SIZE_PARAMS[s] for s in SIZE_PARAMS])
        ax.set_xticklabels(SIZE_PARAMS.keys())
        ax.set_xlabel("model size")
        ax.set_ylabel("AUROC, implicit vs benign")
        ax.set_title(f"{jac}\n{label}", fontsize=9)
        ax.legend(fontsize=7)

    # Panel 3: what the model actually SAYS on the implicit arm, per step -- the
    # headline. Plotting only the final checkpoint hides the whole result: at 12b
    # and 27b implicit escalation collapses over the finetune (34->8%, 48->12%)
    # while the explicit arm barely moves, so the two arms have to be on the same
    # axes with step as the series.
    ax = axes[2]
    for step in steps:
        pts = [(r["params_b"], r["escalation"]["implicit"]["escalated"]) for r in recs
               if r["step"] == step and "implicit" in r["escalation"]]
        if pts:
            ax.plot(*zip(*sorted(pts)), "o-", label=f"implicit, step {step}")
    for step, alpha in zip((steps[0], steps[-1]), (0.55, 0.25)):
        pts = [(r["params_b"], r["escalation"]["explicit"]["escalated"]) for r in recs
               if r["step"] == step and "explicit" in r["escalation"]]
        if pts:
            ax.plot(*zip(*sorted(pts)), "s--", color="0.35", alpha=alpha,
                    label=f"explicit, step {step}")
    ax.set_xscale("log")
    ax.set_xticks([SIZE_PARAMS[s] for s in SIZE_PARAMS])
    ax.set_xticklabels(SIZE_PARAMS.keys())
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("model size")
    ax.set_ylabel("verbalized escalation rate")
    ax.set_title("what the model says\n(implicit arm, per step)", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Silent red flags over medical SFT — internal signal vs. verbalized escalation",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "crossmodel.png", dpi=160)
    print(f"[fig] {out / 'crossmodel.png'}")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = collect(Path(args.runs), args.steps)
    if not recs:
        raise SystemExit(f"no analysed runs under {args.runs}")
    (out / "crossmodel.json").write_text(json.dumps(recs, indent=2))
    print(write_tables(recs, out))
    plot(recs, out)
    have = {(r["size"], r["step"]) for r in recs}
    want = {(s, t) for s in SIZE_PARAMS for t in args.steps}
    if have != want:
        print(f"[incomplete] missing cells: {sorted(want - have)}")


if __name__ == "__main__":
    main()
