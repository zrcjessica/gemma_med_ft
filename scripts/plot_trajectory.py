"""Render the jlens trajectory metrics into slide-ready figures.

Reads eval/jlens/<run>/metrics.jsonl (one row per checkpoint) and writes vector
PDFs (for the beamer deck) plus a composite PNG (quick-look). Log-spaced
checkpoints are plotted on an even x-index with the real step as the tick label,
which avoids log(0) for the t=0 base and gives every probe point equal width.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# --- palette (dataviz reference, slots 1-3, validated all-pairs both modes) ---
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e6e2"
SURFACE = "#fcfcfb"
GENERAL = "#2a78d6"   # slot 1 blue
MEDICAL = "#eb6834"   # slot 2 orange
DRIFT = "#1baf7a"     # slot 3 aqua

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 140,
})


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    steps = [r["step"] for r in rows]
    x = list(range(len(rows)))
    return rows, steps, x


def style_x(ax, steps, x):
    ax.set_xticks(x)
    ax.set_xticklabels([("t=0" if s == 0 else str(s)) for s in steps],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("training step (log-spaced checkpoints)")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)


def plot_drift(rows, steps, x, out, label):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                                 gridspec_kw={"hspace": 0.12})
    relf = [r["drift"]["relfro_mean"] for r in rows]
    cos = [r["drift"]["cos_mean"] for r in rows]
    a1.plot(x, relf, "-o", color=DRIFT, lw=2, ms=6, mfc="white", mew=1.6, mec=DRIFT)
    a1.set_ylabel("rel. Frobenius\n‖J−J₀‖ / ‖J₀‖")
    a1.grid(axis="y", alpha=0.7); a1.set_axisbelow(True)
    a2.plot(x, cos, "-s", color=MEDICAL, lw=2, ms=6, mfc="white", mew=1.6, mec=MEDICAL)
    a2.set_ylabel("cosine(J, J₀)")
    # The arm belongs in the image: the deck shows this figure for several arms,
    # and an arm-agnostic title is how a mislabeled slide goes unnoticed.
    a1.set_title(f"{label} — Jacobian drift from the base lens", loc="left", color=INK)
    style_x(a2, steps, x)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _series(ax, x, y, color, marker, label):
    ax.plot(x, y, marker=marker, color=color, lw=2, ms=6, mfc="white",
            mew=1.6, mec=color, label=label, linestyle="-")


def _baseline(ax, x, y0, color):
    """Dashed reference at the base-model (t=0) value: every finetuned point is
    read against the untuned baseline, not just against the other arm."""
    ax.axhline(y0, color=color, lw=1.1, ls=(0, (5, 3)), alpha=0.55, zorder=1)


def plot_conc(rows, steps, x, key, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    g = [r["concordance"]["general"][key] for r in rows]
    m = [r["concordance"]["medical"][key] for r in rows]
    _baseline(ax, x, g[0], GENERAL)
    _baseline(ax, x, m[0], MEDICAL)
    _series(ax, x, g, GENERAL, "o", "general (control)")
    _series(ax, x, m, MEDICAL, "s", "medical")
    # annotate the untuned-base reference so the dashes are unambiguous
    ax.text(x[-1], m[0], "  base (t=0)", va="center", ha="left",
            fontsize=7.5, color=MEDICAL, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(loc="best", fontsize=9)
    style_x(ax, steps, x)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_base_vs_final(rows, out, label, ylim_kl=None):
    """Direct base(t=0) vs finetuned comparison — the change medical SFT
    actually produced, per metric and per arm.

    KL and top-1 get separate panels: on one shared axis a +0.12 top-1 move is a
    sliver next to bars 3-5 nats tall, which buries the very number the deck
    calls out. `ylim_kl` pins the KL axis so two arms shown side by side really
    are on the same scale -- autoscaled, 270m's smaller deltas render *taller*
    than 1b's. The step goes in the title because rows[-1] is only the final
    checkpoint for a complete run; on a spike it is step 8.
    """
    b, f = rows[0], rows[-1]
    panels = [
        ("kl_final", "KL(model ‖ lens)  (nats)", ylim_kl),
        ("top1_final", "top-1 agreement  (fraction)", (0, 1)),
    ]
    cues = (("general", GENERAL), ("medical", MEDICAL))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"wspace": 0.3})
    w = 0.36
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        for i, (cue, c) in enumerate(cues):
            bv = b["concordance"][cue][key]
            fv = f["concordance"][cue][key]
            ax.bar(i - w / 2, bv, w, color=c, alpha=0.35, edgecolor=c, lw=1.2)
            ax.bar(i + w / 2, fv, w, color=c, alpha=0.95, edgecolor=c, lw=1.2)
            ax.annotate(f"{fv - bv:+.2f}", (i, max(bv, fv)), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(list(range(len(cues))))
        ax.set_xticklabels([c for c, _ in cues], fontsize=9)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.7); ax.set_axisbelow(True)
    # neutral shading legend: the base/final split is the fill alpha, not the cue color
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MUTED, alpha=0.35, edgecolor=MUTED, label="base (t=0)"),
               Patch(facecolor=MUTED, alpha=0.95, edgecolor=MUTED, label=f"step {f['step']}")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"{label} — base (t=0) vs. step {f['step']}  ·  mean of last n/5 layers",
                 x=0.01, ha="left", fontsize=12, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def print_base_vs_final(rows):
    b, f = rows[0], rows[-1]
    print(f"\nbase (t=0) -> finetuned (step {f['step']}) delta:")
    for arm in ("general", "medical"):
        for key in ("kl_final", "top1_final"):
            bv = b["concordance"][arm][key]
            fv = f["concordance"][arm][key]
            print(f"  {arm:8s} {key:11s} {bv:7.3f} -> {fv:7.3f}  ({fv - bv:+.3f})")
    print(f"  drift    relfro_mean {b['drift']['relfro_mean']:7.3f} -> "
          f"{f['drift']['relfro_mean']:7.3f}  "
          f"({f['drift']['relfro_mean'] - b['drift']['relfro_mean']:+.3f})")


def plot_overview(rows, steps, x, out, label):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    relf = [r["drift"]["relfro_mean"] for r in rows]
    axes[0].plot(x, relf, "-o", color=DRIFT, lw=2, ms=5, mfc="white", mew=1.4, mec=DRIFT)
    axes[0].set_title("Jacobian drift", loc="left", color=INK)
    axes[0].set_ylabel("rel. Frobenius")
    for key, ax, ttl, yl in [
        ("kl_final", axes[1], "Faithfulness KL(model‖lens)", "KL (nats)"),
        ("top1_final", axes[2], "Top-1 agreement", "fraction"),
    ]:
        g = [r["concordance"]["general"][key] for r in rows]
        m = [r["concordance"]["medical"][key] for r in rows]
        _baseline(ax, x, g[0], GENERAL)
        _baseline(ax, x, m[0], MEDICAL)
        _series(ax, x, g, GENERAL, "o", "general")
        _series(ax, x, m, MEDICAL, "s", "medical")
        ax.set_title(ttl, loc="left", color=INK); ax.set_ylabel(yl)
    axes[2].legend(loc="best", fontsize=8)
    for ax in axes:
        style_x(ax, steps, x)
    fig.suptitle(f"{label} · medical SFT · j-lens trajectory "
                 f"(base + {len(rows) - 1} checkpoints)",
                 x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def infer_label(metrics_path):
    """'eval/jlens/1b_it_full_v3/metrics.jsonl' -> 'Gemma-3-1b-it'."""
    # resolve() first: a bare '--metrics metrics.jsonl' run from inside the run
    # dir has no parent component, and an empty label leaves a title starting " · ".
    run_dir = Path(metrics_path).resolve().parent.name
    # not \b after the kind: '_' is a word char, so '270m_it_full' would not match
    m = re.match(r"(270m|\d+b)_(pt|it)(?:_|$)", run_dir)
    return f"Gemma-3-{m.group(1)}-{m.group(2)}" if m else run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default=None,
                    help="Arm name for figure titles (default: inferred from the run dir).")
    ap.add_argument("--ylim-kl", type=float, default=None,
                    help="Pin the base-vs-final KL axis (nats), so arms rendered "
                         "separately can be placed side by side on one scale.")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows, steps, x = load(args.metrics)
    label = args.label or infer_label(args.metrics)

    plot_drift(rows, steps, x, out / "fig_drift.pdf", label)
    # Titles stay descriptive, not claim-laden: the same script renders every arm
    # and the arms do not all move the same way. Interpretation belongs on the slide.
    plot_conc(rows, steps, x, "kl_final",
              "KL(model ‖ lens)  (nats, mean of last n/5 layers)",
              f"{label} — faithfulness vs. the untuned base",
              out / "fig_kl.pdf")
    plot_conc(rows, steps, x, "top1_final",
              "top-1 agreement (mean of last n/5 layers)",
              f"{label} — top-1 agreement vs. the untuned base",
              out / "fig_top1.pdf")
    plot_overview(rows, steps, x, out / "fig_overview.png", label)
    plot_base_vs_final(rows, out / "fig_base_vs_final.pdf", label,
                       ylim_kl=(0, args.ylim_kl) if args.ylim_kl else None)
    print("wrote:", *(p.name for p in sorted(out.glob("fig_*"))))
    print_base_vs_final(rows)


if __name__ == "__main__":
    main()
