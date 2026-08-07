"""Overlay the j-lens trajectories of several size arms on one figure.

plot_trajectory.py renders one arm; this renders the size comparison. Raw J_l is
not comparable across sizes (different hidden dims and layer counts), so every
panel here plots a derived scalar only -- never a Jacobian entry.

Encoding choices that are not free:

* Size is *ordinal* (270m < 1b < 4b < 12b), so the arms take a one-hue blue
  ramp, not categorical hues. Categorical slots 1-3 already mean
  general/medical/drift everywhere else in the deck; reusing them for arms
  would make blue mean two things across slides.
  Validated with `validate_palette.py --ordinal --mode light`: monotone L,
  all adjacent gaps >= 0.06, single hue (4 deg spread), light end 2.06:1.
  The 4th step was added at the *dark* end, not the light one: #86b6ef is
  already at 2.06:1 against the surface, so a paler step fails the 2.0 ordinal
  floor (a #b3d0f5 lead-in measures 1.54:1). Re-step the dark half, never the
  light end, if a 5th arm (27b) is added.
* That light end clears the 2.0 ordinal floor but not the 3.0 categorical one,
  which triggers the relief rule -- so each arm is direct-labeled at its last
  point and carries its own marker shape. Identity is never color alone.
* x is the real training step on a log axis, not the even index plot_trajectory
  uses. The arms have different step grids (1b probed step 1; 270m and 4b start
  at 2), and an even index would silently misalign them. t=0 has no place on a
  log axis, so it appears as each arm's dashed baseline in the absolute panels
  and as the zero line in the delta panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from plot_trajectory import INK, MUTED, GRID  # noqa: E402  (also applies rcParams)

# Ordinal blue ramp, light->dark = small->large. See module docstring for the
# validator run; do not swap these for categorical slots.
ARM_COLOR = ["#86b6ef", "#3f8ae0", "#1d5aa8", "#0b3060"]
ARM_MARKER = ["o", "s", "^", "D"]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step"])
    return rows


def _line(ax, rows, value, color, marker, label, base_ref=False):
    """Plot one arm. t=0 is never a point on a log x-axis -- when base_ref is
    set it becomes a dashed horizontal reference at the base value instead."""
    pts = [(r["step"], value(r)) for r in rows if r["step"] > 0]
    ax.plot([s for s, _ in pts], [v for _, v in pts], marker=marker, color=color,
            lw=2, ms=6, mfc="white", mew=1.6, mec=color, label=label, zorder=3)
    if base_ref:
        b = next(r for r in rows if r["step"] == 0)
        ax.axhline(value(b), color=color, lw=1.1, ls=(0, (5, 3)), alpha=0.55, zorder=1)
    return pts


def _label_last(ax, pts, text, dy=4):
    """Direct label in muted ink -- the colored marker beside it carries identity
    (text never wears the series color)."""
    s, v = pts[-1]
    ax.annotate(f" {text}", (s, v), textcoords="offset points", xytext=(6, dy),
                fontsize=8, color=MUTED, va="center", zorder=4)


def _style_x(ax, last_step):
    ax.set_xscale("log")
    ax.set_xlabel("training step (log)")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(right=last_step * 2.6)  # room for the direct labels


def plot(arms, out_pdf, out_png, suptitle):
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0))
    last = max(r["step"] for _, rows in arms for r in rows)

    def panel(ax, value, ylabel, title, base_ref=False, label_fmt="{:.2f}"):
        for i, (name, rows) in enumerate(arms):
            pts = _line(ax, rows, value, ARM_COLOR[i], ARM_MARKER[i], name,
                        base_ref=base_ref)
            _label_last(ax, pts, name)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK)
        _style_x(ax, last)

    relfro = lambda r: r["drift"]["relfro_mean"]
    cos = lambda r: r["drift"]["cos_mean"]

    def kl(cue):
        return lambda r: r["concordance"][cue]["kl_final"]

    def dkl(cue, rows0):
        b = next(r for r in rows0 if r["step"] == 0)["concordance"][cue]["kl_final"]
        return lambda r: r["concordance"][cue]["kl_final"] - b

    # --- left column: the Jacobian geometry ---
    panel(axes[0][0], relfro, "rel. Frobenius  ‖J−J₀‖/‖J₀‖", "Jacobian drift from t=0")
    panel(axes[1][0], cos, "cosine(J, J₀)", "Jacobian alignment with t=0")
    # cosine cannot exceed 1; the ceiling line makes the fp32 dot-product
    # overshoot at near-zero drift legible as noise rather than signal.
    axes[1][0].axhline(1.0, color=GRID, lw=1.0, zorder=0)

    # --- middle/right columns: faithfulness, per cue ---
    for col, cue in ((1, "general"), (2, "medical")):
        panel(axes[0][col], kl(cue),
              "KL(model ‖ lens)  (nats)",
              f"Faithfulness — {cue} cue", base_ref=True)
        # Park the caption in whichever corner no baseline occupies. Hardcoding
        # a corner does not survive a new arm: top-left was correct for three
        # arms (4b's general baseline sits low) and 12b's medical baseline then
        # landed on top of it.
        ax = axes[0][col]
        lo, hi = ax.get_ylim()
        bases = [(next(r for r in rows if r["step"] == 0)["concordance"][cue]["kl_final"] - lo)
                 / (hi - lo) for _, rows in arms]
        top_clear = not any(f > 0.86 for f in bases)
        ax.text(0.02, 0.97 if top_clear else 0.03,
                "dashed = that arm's base (t=0)",
                transform=ax.transAxes, fontsize=7.5,
                color=MUTED, va="top" if top_clear else "bottom")

    for col, cue in ((1, "general"), (2, "medical")):
        ax = axes[1][col]
        for i, (name, rows) in enumerate(arms):
            pts = _line(ax, rows, dkl(cue, rows), ARM_COLOR[i], ARM_MARKER[i], name)
            _label_last(ax, pts, name)
        ax.axhline(0.0, color=MUTED, lw=1.0, alpha=0.5, zorder=1)
        ax.set_ylabel("Δ KL from base  (nats)")
        ax.set_title(f"Change in faithfulness — {cue} cue", loc="left", color=INK)
        _style_x(ax, last)

    axes[0][0].legend(loc="best", fontsize=9)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13.5, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def summarize(arms):
    print(f"\n{'arm':6s} {'ΔKL gen':>9s} {'ΔKL med':>9s} {'relfro':>8s} "
          f"{'KL gen t=0':>11s} {'top1 gen t=0':>13s}")
    for name, rows in arms:
        b, f = rows[0], rows[-1]
        g = f["concordance"]["general"]["kl_final"] - b["concordance"]["general"]["kl_final"]
        m = f["concordance"]["medical"]["kl_final"] - b["concordance"]["medical"]["kl_final"]
        print(f"{name:6s} {g:+9.3f} {m:+9.3f} {f['drift']['relfro_mean']:8.3f} "
              f"{b['concordance']['general']['kl_final']:11.3f} "
              f"{b['concordance']['general']['top1_final']:13.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable, in ascending size order: --arm 270m=eval/.../metrics.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="Gemma-3 -it · medical SFT · j-lens trajectory by model size")
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms.append((name, load(path)))
    if len(arms) > len(ARM_COLOR):
        raise SystemExit(f"ramp has {len(ARM_COLOR)} steps; got {len(arms)} arms "
                         "-- extend the ramp and re-run the ordinal validator")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot(arms, out / "fig_arms.pdf", out / "fig_arms.png",
         args.title + "  ·  KL = mean over the last n/5 source layers")
    print("wrote:", out / "fig_arms.pdf", out / "fig_arms.png")
    summarize(arms)


if __name__ == "__main__":
    main()
