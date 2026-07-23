"""Render the jlens trajectory metrics into slide-ready figures.

Reads eval/jlens/<run>/metrics.jsonl (one row per checkpoint) and writes vector
PDFs (for the beamer deck) plus a composite PNG (quick-look). Log-spaced
checkpoints are plotted on an even x-index with the real step as the tick label,
which avoids log(0) for the t=0 base and gives every probe point equal width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
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


def plot_drift(rows, steps, x, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                                 gridspec_kw={"hspace": 0.12})
    relf = [r["drift"]["relfro_mean"] for r in rows]
    cos = [r["drift"]["cos_mean"] for r in rows]
    a1.plot(x, relf, "-o", color=DRIFT, lw=2, ms=6, mfc="white", mew=1.6, mec=DRIFT)
    a1.set_ylabel("rel. Frobenius\n‖J−J₀‖ / ‖J₀‖")
    a1.grid(axis="y", alpha=0.7); a1.set_axisbelow(True)
    a2.plot(x, cos, "-s", color=MEDICAL, lw=2, ms=6, mfc="white", mew=1.6, mec=MEDICAL)
    a2.set_ylabel("cosine(J, J₀)")
    a1.set_title("Jacobian drift from the base lens — fast reorganization, then plateau",
                 loc="left", color=INK)
    style_x(a2, steps, x)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _series(ax, x, y, color, marker, label):
    ax.plot(x, y, marker=marker, color=color, lw=2, ms=6, mfc="white",
            mew=1.6, mec=color, label=label, linestyle="-")


def plot_conc(rows, steps, x, key, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    g = [r["concordance"]["general"][key] for r in rows]
    m = [r["concordance"]["medical"][key] for r in rows]
    _series(ax, x, g, GENERAL, "o", "general (control)")
    _series(ax, x, m, MEDICAL, "s", "medical")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(loc="best", fontsize=9)
    style_x(ax, steps, x)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_overview(rows, steps, x, out):
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
        _series(ax, x, g, GENERAL, "o", "general")
        _series(ax, x, m, MEDICAL, "s", "medical")
        ax.set_title(ttl, loc="left", color=INK); ax.set_ylabel(yl)
    axes[2].legend(loc="best", fontsize=8)
    for ax in axes:
        style_x(ax, steps, x)
    fig.suptitle("Gemma-3-1b-it · medical SFT · j-lens trajectory (base + 14 checkpoints)",
                 x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows, steps, x = load(args.metrics)

    plot_drift(rows, steps, x, out / "fig_drift.pdf")
    plot_conc(rows, steps, x, "kl_final",
              "KL(model ‖ lens)  (nats, final layers)",
              "Faithfulness: medical lens trails the general control by ~1 nat",
              out / "fig_kl.pdf")
    plot_conc(rows, steps, x, "top1_final",
              "top-1 agreement (final layers)",
              "Top-1 agreement is noise-dominated at 15 prompts / arm",
              out / "fig_top1.pdf")
    plot_overview(rows, steps, x, out / "fig_overview.png")
    print("wrote:", *(p.name for p in sorted(out.glob("fig_*"))))


if __name__ == "__main__":
    main()
