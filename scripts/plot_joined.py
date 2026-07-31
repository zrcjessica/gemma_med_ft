"""Figure for the behavioral trajectory joined to the j-lens trajectory.

Reads eval/traj/<tag>/joined.json (scripts/analyze_traj.py --out) and renders the
one thing the raw benchmark numbers hide: accuracy among *answered* items rises
over medical SFT while the answer rate collapses, so raw accuracy falls. Raw,
answered, and answer-rate are all percentages and share one y-axis -- no second
scale is introduced anywhere in this figure.

Scores are LLM-judge verdicts (gemma_med.judge); "answered" means the judge found
a committed choice, the successor to the old regex parse rate.

Palette/style are imported from plot_trajectory so the deck stays consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from plot_trajectory import (  # noqa: E402
    INK, MUTED, GENERAL, MEDICAL, DRIFT, infer_label, style_x,
)

BENCHMARKS = [("medqa", "MedQA"), ("medmcqa", "MedMCQA"), ("pubmedqa", "PubMedQA")]

# Roles, assigned once and never cycled: the answer rate is the explanation, the
# answered accuracy is the real signal, the raw accuracy is the misleading number.
C_PARSE = GENERAL    # slot 1 blue
C_PARSED = MEDICAL   # slot 2 orange
C_RAW = DRIFT        # slot 3 aqua


def _series(ax, x, y, color, marker, label, ls="-"):
    ax.plot(x, y, marker=marker, color=color, lw=2, ms=6, mfc="white",
            mew=1.6, mec=color, label=label, linestyle=ls, zorder=3)


def _baseline(ax, y0, color):
    """t=0 reference: every finetuned point is read against the untuned base."""
    ax.axhline(y0, color=color, lw=1.1, ls=(0, (5, 3)), alpha=0.55, zorder=1)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def saturation_step(steps, ys):
    """First step reaching 95% of the final value -- the plateau onset."""
    target = 0.95 * ys[-1]
    for s, y in zip(steps, ys):
        if y >= target:
            return s
    return steps[-1]


def plot(rows, out_pdf, out_png, suptitle):
    steps = [r["step"] for r in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4))

    # --- top row: one panel per benchmark, all three series in percent ---
    for ax, (key, title) in zip(axes[0], BENCHMARKS):
        beh = [r["behavioral"][key] for r in rows]
        parse = [b["answer_rate"] for b in beh]
        accp = [b["acc_answered"] for b in beh]
        accr = [b["acc_raw"] for b in beh]
        _baseline(ax, accp[0], C_PARSED)
        _series(ax, x, parse, C_PARSE, "o", "answer rate")
        _series(ax, x, accp, C_PARSED, "s", "accuracy (answered only)")
        _series(ax, x, accr, C_RAW, "^", "accuracy (raw)")
        ax.set_ylim(0, 104)
        ax.set_title(title, loc="left", color=INK)
        style_x(ax, steps, x)
        # direct-label the endpoints -- the whole story is in the final gap
        ax.annotate(f"{accp[-1]:.0f}", (x[-1], accp[-1]), textcoords="offset points",
                    xytext=(5, 3), fontsize=8, color=MUTED)
        ax.annotate(f"{accr[-1]:.0f}", (x[-1], accr[-1]), textcoords="offset points",
                    xytext=(5, -9), fontsize=8, color=MUTED)
    axes[0][0].set_ylabel("percent")
    axes[0][2].legend(loc="lower left", fontsize=8.5)
    axes[0][0].text(0, 4, "dashed = base (t=0) answered accuracy",
                    fontsize=7.5, color=MEDICAL, alpha=0.85)

    # --- bottom row: the lens side, one measure per panel (never co-plotted) ---
    a_drift, a_kl, a_sc = axes[1]

    drift = [r["lens"]["drift_relfro_max"] for r in rows]
    _series(a_drift, x, drift, DRIFT, "o", "drift")
    a_drift.set_ylabel("rel. Frobenius  ‖J−J₀‖/‖J₀‖")
    a_drift.set_title(f"Jacobian drift — saturates by step {saturation_step(steps, drift)}",
                      loc="left", color=INK)
    style_x(a_drift, steps, x)

    # Layer mean, not the single final layer: the two disagree in sign on 270m
    # medical, and every claim in the deck is made under the layer mean. Plotting
    # the other one here would show medical *rising* beside text saying it is flat.
    klg = [r["lens"]["kl_general_layermean"] for r in rows]
    klm = [r["lens"]["kl_medical_layermean"] for r in rows]
    _baseline(a_kl, klg[0], GENERAL)
    _baseline(a_kl, klm[0], MEDICAL)
    _series(a_kl, x, klg, GENERAL, "o", "general")
    _series(a_kl, x, klm, MEDICAL, "s", "medical")
    a_kl.set_ylabel("KL(model ‖ lens)  (nats, mean of last n/5 layers)")
    a_kl.set_title("Lens faithfulness — still moving after drift stops", loc="left", color=INK)
    a_kl.legend(loc="best", fontsize=8.5)
    style_x(a_kl, steps, x)

    # scatter: the relationship the trajectory panels only imply
    pr = [r["behavioral"]["medmcqa"]["answer_rate"] for r in rows]
    a_sc.scatter(klg, pr, s=46, facecolor="white", edgecolor=C_PARSE, linewidth=1.8, zorder=3)
    for xi, yi, s in zip(klg, pr, steps):
        if s in (0, 32, 512, 4882):
            a_sc.annotate("t=0" if s == 0 else str(s), (xi, yi), textcoords="offset points",
                          xytext=(6, 4), fontsize=7.5, color=MUTED)
    a_sc.set_xlabel("KL(model ‖ lens), general cue (nats, layer mean)")
    a_sc.set_ylabel("MedMCQA answer rate (%)")
    a_sc.set_title(f"Faithfulness vs. format compliance  (r = {pearson(klg, pr):+.2f}, n={len(rows)})",
                   loc="left", color=INK)
    a_sc.grid(alpha=0.7)
    a_sc.set_axisbelow(True)

    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13.5, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True, help="joined.json from analyze_traj.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default=None,
                    help="title label; default derived from the tag dir (e.g. 270m_it_full)")
    # The headline claim is arm-specific and must not be inherited: at 1b parsed
    # accuracy rises on all three benchmarks, at 270m it is flat-to-negative and
    # only format compliance moves. Default to a neutral title; assert only what
    # the arm's own numbers show.
    ap.add_argument("--title", default=None, help="full suptitle; overrides the neutral default")
    args = ap.parse_args()
    rows = json.loads(Path(args.joined).read_text())
    rows = [r for r in rows if r["lens"]]
    # one label rule for the whole deck -- a local copy drifts (a bare split("_")
    # turns '270m_verify' into "Gemma-3-270m-verify")
    label = args.model or infer_label(args.joined)
    suptitle = args.title or f"{label} · medical SFT · behavioral trajectory vs. Jacobian lens"
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot(rows, out / "fig_joined.pdf", out / "fig_joined.png", suptitle)
    print("wrote:", out / "fig_joined.pdf", out / "fig_joined.png")


if __name__ == "__main__":
    main()
