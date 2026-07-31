"""Locate the "most consequential" j-space layer, under four definitions at once.

plot_arms.py collapses each checkpoint to a scalar and plots it against training
step. This asks the orthogonal question -- *where in depth* the action is -- so
the x axis here is normalized depth (L / n_layers) and each curve is one arm's
final checkpoint read out layer by layer.

"Most consequential" is not one quantity. Four defensible definitions are drawn
side by side; each panel annotates its own argmax per arm, and they do not agree
with each other. That disagreement is the finding, not a defect of the figure.

Encoding choices that are not free:

* x is L / n_layers, never the raw layer index. Raw J_l is not comparable across
  sizes (different d_model, different layer counts: 18 / 26 / 34), so the only
  honest cross-size axis is fractional depth. The last layer is the lens *target*
  and is never a source, so every curve stops at (n_layers-1)/n_layers, not 1.0.
* Size is ordinal, so the arms reuse the one-hue blue ramp from plot_arms
  (re-validated here with `validate_palette.py --ordinal --mode light`: monotone
  L, adjacent gaps >= 0.06, single hue 3 deg, light end 2.06:1). The light end
  clears the ordinal floor but not the categorical one, so identity is never
  color alone -- each arm also carries its own marker shape and a direct label.
* **Dashed always means t=0**, in every panel that has a base to show. Panels 1
  and 3 are already differences from the base, so their base is the zero line,
  drawn and labelled as such. No other meaning is ever attached to a dash here.
* Markers are subsampled (markevery) -- at 33 layers a marker per point is a
  smear, but the shape still has to be present often enough to identify the arm.

Usage:
    python scripts/plot_layer_defs.py \
        --arm 270m=eval/jlens/270m_it_full/metrics.jsonl \
        --arm 1b=eval/jlens/1b_it_full_v3/metrics.jsonl \
        --arm 4b=eval/jlens/4b_it_full/metrics.jsonl \
        --outdir figs/layer_defs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from plot_trajectory import INK, MUTED, GRID  # noqa: E402  (also applies rcParams)
from plot_arms import ARM_COLOR, ARM_MARKER, load  # noqa: E402

AGREE_THRESHOLD = 0.5


def depth(row, layers):
    return [L / row["n_layers"] for L in layers]


def _curve(ax, x, y, i, name, solid=True, label=None):
    """One arm's layer curve. Marker shape + direct label carry identity so the
    ordinal ramp never has to do it alone."""
    every = max(1, len(x) // 6)
    ax.plot(x, y, color=ARM_COLOR[i], lw=2 if solid else 1.2,
            ls="-" if solid else (0, (5, 3)), alpha=1.0 if solid else 0.55,
            marker=ARM_MARKER[i] if solid else None, markevery=every, ms=6,
            mfc="white", mew=1.6, mec=ARM_COLOR[i], label=label, zorder=3 if solid else 2)


def _mark_peak(ax, x, y, i, key=abs):
    """Ring the argmax -- this is the panel's answer for that arm."""
    j = max(range(len(y)), key=lambda k: key(y[k]))
    ax.plot([x[j]], [y[j]], marker="o", ms=11, mfc="none", mew=2.0,
            mec=ARM_COLOR[i], zorder=5)
    ax.annotate(f"{x[j]:.2f}", (x[j], y[j]), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=8, color=MUTED, zorder=5)
    return j


def _place_end_labels(ax, entries, min_gap=0.062):
    """Direct-label every arm in the right margin. The curves converge at the
    deep end, so the labels are pushed apart to a minimum gap rather than left
    to land on top of each other. Call after the y-limits are final."""
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    placed = []
    for frac, name in sorted(((y - lo) / span, name) for y, name in entries):
        if placed and frac - placed[-1][0] < min_gap:
            frac = placed[-1][0] + min_gap
        placed.append((frac, name))
    for frac, name in placed:
        ax.annotate(name, xy=(1.015, min(max(frac, 0.0), 1.0)), xycoords="axes fraction",
                    fontsize=8.5, color=MUTED, va="center", ha="left", zorder=4)


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("normalized depth  L / n_layers")
    ax.set_xlim(-0.02, 1.0)           # arm labels sit outside, in the right margin
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)


def _note(ax, text, y=0.97, va="top"):
    ax.text(0.02, y, text, transform=ax.transAxes, fontsize=7.5,
            color=MUTED, va=va, zorder=6)


def plot(arms, out_pdf, out_png, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    peaks = {}

    # --- 1. where fine-tuning moved the Jacobian most -------------------------
    ax = axes[0][0]
    ends = []
    for i, (name, rows) in enumerate(arms):
        f = rows[-1]
        L = f["drift"]["layers"]
        x = depth(f, L)
        y = [1.0 - c for c in f["drift"]["cos_curve"]]
        _curve(ax, x, y, i, name, label=name)
        j = _mark_peak(ax, x, y, i)
        ends.append((y[-1], name))
        peaks.setdefault(name, {})["drift"] = (L[j], x[j])
    ax.axhline(0.0, color=MUTED, lw=1.0, alpha=0.5, zorder=1)
    _style(ax, "1 · Jacobian drift — where FT changed the lens", "1 − cos(J_l, J_l⁰)")
    _place_end_labels(ax, ends)

    # --- 2. where the medical cue departs from the general control -----------
    ax = axes[0][1]
    ends = []
    for i, (name, rows) in enumerate(arms):
        b, f = rows[0], rows[-1]
        L = f["concordance"]["medical"]["layers"]
        x = depth(f, L)
        gap = [m - g for m, g in zip(f["concordance"]["medical"]["kl_curve"],
                                     f["concordance"]["general"]["kl_curve"])]
        gap0 = [m - g for m, g in zip(b["concordance"]["medical"]["kl_curve"],
                                      b["concordance"]["general"]["kl_curve"])]
        _curve(ax, x, gap0, i, name, solid=False)
        _curve(ax, x, gap, i, name, label=name)
        j = _mark_peak(ax, x, gap, i)
        ends.append((gap[-1], name))
        peaks[name]["med_gap"] = (L[j], x[j])
    ax.axhline(0.0, color=MUTED, lw=1.0, alpha=0.5, zorder=1)
    _style(ax, "2 · Medical − general KL gap   (>0 = lens less faithful on medical)",
           "ΔKL  medical − general  (nats)")
    _place_end_labels(ax, ends)

    # --- 3. where faithfulness moved most over training ----------------------
    ax = axes[1][0]
    ends = []
    for i, (name, rows) in enumerate(arms):
        b, f = rows[0], rows[-1]
        L = f["concordance"]["medical"]["layers"]
        x = depth(f, L)
        d = [t - z for t, z in zip(f["concordance"]["medical"]["kl_curve"],
                                   b["concordance"]["medical"]["kl_curve"])]
        _curve(ax, x, d, i, name, label=name)
        j = _mark_peak(ax, x, d, i)          # argmax |ΔKL| -- sign differs by arm
        ends.append((d[-1], name))
        peaks[name]["dkl_med"] = (L[j], x[j], d[j])
    ax.axhline(0.0, color=MUTED, lw=1.0, alpha=0.5, zorder=1)
    _style(ax, "3 · Change in faithfulness over FT — medical cue",
           "KL(t=T) − KL(t=0)  (nats)")
    # top-right is the only empty corner here; 270m dips below zero at the left.
    ax.text(0.98, 0.97, "ring = argmax |ΔKL| — sign is not constant across arms",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top", ha="right")
    _place_end_labels(ax, ends)

    # --- 4. where the lens becomes faithful at all ---------------------------
    ax = axes[1][1]
    ends = []
    for i, (name, rows) in enumerate(arms):
        b, f = rows[0], rows[-1]
        L = f["concordance"]["medical"]["layers"]
        x = depth(f, L)
        a0 = b["concordance"]["medical"]["agree_curve"]
        a = f["concordance"]["medical"]["agree_curve"]
        _curve(ax, x, a0, i, name, solid=False)
        _curve(ax, x, a, i, name, label=name)
        ends.append((a[-1], name))
        k = next((n for n, v in enumerate(a) if v >= AGREE_THRESHOLD), None)
        if k is None:
            peaks[name]["crossover"] = None
        else:
            ax.plot([x[k]], [a[k]], marker="o", ms=11, mfc="none", mew=2.0,
                    mec=ARM_COLOR[i], zorder=5)
            ax.annotate(f"{x[k]:.2f}", (x[k], a[k]), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=8, color=MUTED, zorder=5)
            peaks[name]["crossover"] = (L[k], x[k])
    ax.axhline(AGREE_THRESHOLD, color=GRID, lw=1.2, zorder=0)
    _style(ax, "4 · Lens↔model top-1 agreement — medical cue", "top-1 agreement")
    ax.set_ylim(-0.04, 1.04)
    missing = [n for n, p in peaks.items() if p["crossover"] is None]
    # upper-left is empty here -- every arm sits near zero until ~0.5 depth.
    _note(ax, "ring = first layer reaching 0.5"
              + (f"\n{', '.join(missing)} never reaches 0.5 — no ring" if missing else ""))
    _place_end_labels(ax, ends)

    axes[0][0].legend(loc="upper right", fontsize=9)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13.5, color=INK, weight="bold")
    fig.text(0.01, 0.012,
             "Solid = final checkpoint (step 4882);  dashed = that arm's untuned base (t=0);  "
             "ring = the panel's answer for that arm.  Curves stop at (n_layers−1)/n_layers — "
             "the last layer is the lens target, never a fitted source.",
             ha="left", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.028, 1, 0.955))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return peaks


def summarize(arms, peaks):
    def cell(entry):
        if entry is None:
            return "none"
        return f"L{entry[0]} ({entry[1]:.2f})"

    print(f"\n{'arm':6s} {'n_lay':>5s} {'1 drift':>13s} {'2 med gap':>13s} "
          f"{'3 |dKL| med':>13s} {'4 crossover':>13s}")
    for name, rows in arms:
        p = peaks[name]
        print(f"{name:6s} {rows[-1]['n_layers']:5d} "
              f"{cell(p['drift']):>13s} {cell(p['med_gap']):>13s} "
              f"{cell(p['dkl_med']):>13s} {cell(p['crossover']):>13s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable, in ascending size order")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="Gemma-3 -it · medical SFT · where in depth is the "
                                       "j-space layer that matters?")
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
    peaks = plot(arms, out / "fig_layer_defs.pdf", out / "fig_layer_defs.png",
                 args.title + "  ·  final checkpoint, read out layer by layer")
    print("wrote:", out / "fig_layer_defs.pdf", out / "fig_layer_defs.png")
    summarize(arms, peaks)


if __name__ == "__main__":
    main()
