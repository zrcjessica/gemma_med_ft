"""Plot the workspace band across arms x size x training step.

Reads the `band.jsonl` files written by `scripts/probe_band.py` (one row per
checkpoint, carrying the per-layer Fig. 28 curves and the suggested band) and
renders four figures:

  band_ribbon.png    the band itself: relative depth vs step, one panel per arm
  band_edges.png     onset / offset / width, all arms overlaid on one axis
  band_profiles.png  the depth profiles behind the band, metric x arm,
                     one curve per step
  band_delta.png     the same profiles as change from t=0 -- the view that
                     answers "did medical SFT move the band, and where"

Encoding choices that are not free:

* **Depth, never layer index.** Arms differ in layer count (18 at 270m, 62 at
  27b), so a raw index means a different depth in every panel and the arms are
  not comparable. Every y-axis here is `layer / n_layers`. This is the same
  cross-size rule the j-lens work already follows: compare derived scalars,
  never raw `J_l`.
* **Size is ordinal**, so the arms take the same validated one-hue blue ramp
  (`plot_arms.ARM_COLOR`) they wear everywhere else in the deck, not
  categorical hues -- slots 1-3 already mean general/medical/drift. Reusing
  them for arms would make blue mean two things across slides.
* **t=0 is always drawn.** A finetuned-only panel cannot show that something
  moved. In the log-x panel it is each arm's dashed baseline (t=0 has no place
  on a log axis); everywhere else it is the first column / the orange curve /
  the zero of the delta.
* **Step within an arm is a second sequential context**, so per the one-hue
  rule it takes the *next* slot's ramp rather than a second blue: t=0 in slot-2
  orange, later steps on a light->dark grey ramp. Blue keeps meaning size.
* **The delta panels are diverging** (blue <-> red, neutral grey midpoint,
  symmetric limits shared across a row) because their zero is meaningful.
  Absolute-magnitude panels stay one-hue sequential.
* `band_ribbon` / `band_profiles` / `band_delta` put steps on an *index* axis
  labelled with the real step, so t=0 has a column. Only `band_edges` overlays
  arms on one axis, and it uses a true log-x for exactly the reason
  plot_arms.py documents: the arms have different step grids, and an index axis
  would silently misalign them.

Example:
  python scripts/plot_band.py --arm 270m eval/band/270m_it/band.jsonl \
                              --arm 1b   eval/band/1b_it/band.jsonl \
                              --out figures/band
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_trajectory import INK, MUTED, GRID, SURFACE  # noqa: E402 (also applies rcParams)
from plot_arms import ARM_COLOR, ARM_MARKER  # noqa: E402

BASE_C = "#eb6834"   # slot 2 orange -- the t=0 reference, never a step
# Sequential grey ramp for step-within-an-arm (light->dark = early->late).
STEP_RAMP = LinearSegmentedColormap.from_list(
    "step_grey", ["#c9c8c3", "#8d8c87", "#52514e", "#0b0b0b"])
# Diverging blue<->red with a neutral grey midpoint (the documented pair;
# blue<->aqua was rejected -- both cool, the midpoint reads as "something").
DIVERGE = LinearSegmentedColormap.from_list(
    "band_div", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                 "#f3a3a3", "#d03b3b", "#8c1f1f"])
SEQ = LinearSegmentedColormap.from_list(
    "band_seq", ["#f0efec", "#cde2fb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

# The Fig. 28 quartet. Row labels are kept short on purpose -- a 4x5 grid has
# no room for the full sentence, and a label that runs into the tick numbers is
# worse than a terse one. The long form lives in the row annotation.
METRICS = [
    ("acc_topk", "next-token\naccuracy", "any top-10 = model top-1"),
    ("kurtosis", "excess\nkurtosis", "of the readout over the vocabulary"),
    ("autocorr_dlog", "top-1\nautocorr.", "Δ log p vs position-shuffled null"),
    ("dim_frac", "J-space\ndimensionality", "frac. of dims for 90% of variance"),
]


def load_arm(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step"])
    for r in rows:
        c = r["curves"]
        # band.py reports the autocorrelation as a delta log-prob against the
        # position-shuffled null; the raw rate alone is not interpretable
        # (it rises with any peaky distribution).
        c["autocorr_dlog"] = (
            np.log(np.maximum(np.array(c["autocorr"]), 1e-9))
            - np.log(np.maximum(np.array(c["autocorr_null"]), 1e-9))
        ).tolist()
    return rows


def _steps(rows):
    return [r["step"] for r in rows]


def _tick_labels(steps):
    return ["0\n(base)" if s == 0 else str(s) for s in steps]


def _thin_ticks(ax, steps, every=2):
    """Label every `every`-th step, always the first and last.

    Drops the second-to-last tick when the stride would place it adjacent to
    the last one -- log-spaced checkpoints end with an uneven final step (…,
    2048, 4096, 4882) and those two labels otherwise print on top of each other.
    """
    idx = sorted({0, len(steps) - 1} | set(range(0, len(steps), every)))
    if len(idx) > 2 and idx[-1] - idx[-2] < every:
        idx.pop(-2)
    ax.set_xticks(idx)
    ax.set_xticklabels([_tick_labels(steps)[i] for i in idx], fontsize=7.5)


def _declutter(ax, points, min_gap_frac=0.05):
    """Nudge direct labels apart along y so converged arms stay readable.

    Returns [(x, y_label, text)] spread by at least `min_gap_frac` of the axis
    range, then shifted back inside the axis if the spread pushed the stack off
    the top. The marker still sits at the true y -- only the text moves, and it
    stays in the same order, so nothing is misrepresented.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    placed = []
    for x, y, text in sorted(points, key=lambda p: p[1]):
        placed.append((x, y if not placed else max(y, placed[-1][1] + gap), text))
    if placed:
        over = placed[-1][1] - (hi - gap / 2)
        under = (lo + gap / 2) - placed[0][1]
        shift = -over if over > 0 else (under if under > 0 else 0.0)
        if shift:
            placed = [(x, y + shift, t) for x, y, t in placed]
    return placed


# --------------------------------------------------------------------------- #
# Figure 1: the band itself


def fig_ribbon(arms, out_path, suptitle):
    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.6, 3.9), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for i, (label, rows) in enumerate(arms):
        ax, color = axes[i], ARM_COLOR[i % len(ARM_COLOR)]
        steps = _steps(rows)
        x = np.arange(len(rows))
        lo = np.array([r["band_depth"][0] for r in rows])
        hi = np.array([r["band_depth"][1] for r in rows])
        ax.fill_between(x, lo, hi, color=color, alpha=0.30, lw=0, zorder=2)
        ax.plot(x, lo, color=color, lw=2, zorder=3)
        ax.plot(x, hi, color=color, lw=2, zorder=3)
        if rows and rows[0]["step"] == 0:
            # The untuned reference, carried across the panel so any movement
            # away from it is visible rather than inferred.
            for y in rows[0]["band_depth"]:
                ax.axhline(y, color=MUTED, lw=1.0, ls=(0, (5, 3)), alpha=0.6, zorder=1)
        ax.set_title(f"{label}  ({rows[0]['n_layers']} layers)", loc="left",
                     color=INK, fontsize=10)
        ax.set_xlabel("training step")
        _thin_ticks(ax, steps)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("relative depth  (layer / n_layers)")
    handles = [Patch(facecolor=ARM_COLOR[2], alpha=0.30, edgecolor=ARM_COLOR[2],
                     label="suggested workspace band"),
               Line2D([], [], color=MUTED, lw=1.0, ls=(0, (5, 3)),
                      label="band at t=0 (untuned base)")]
    axes[-1].legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
# Figure 2: band edges across arms, one axis


def fig_edges(arms, out_path, suptitle):
    panels = [("onset (band start)", lambda r: r["band_depth"][0]),
              ("offset (band end)", lambda r: r["band_depth"][1]),
              ("band width", lambda r: r["band_depth"][1] - r["band_depth"][0])]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    ends_by_ax = {}
    for ax, (title, value) in zip(axes, panels):
        ends = []
        for i, (label, rows) in enumerate(arms):
            color, marker = ARM_COLOR[i % len(ARM_COLOR)], ARM_MARKER[i % len(ARM_MARKER)]
            pts = [(r["step"], value(r)) for r in rows if r["step"] > 0]
            if not pts:
                continue
            ax.plot([s for s, _ in pts], [v for _, v in pts], marker=marker,
                    color=color, lw=2, ms=6, mfc="white", mew=1.6, mec=color,
                    label=label, zorder=3)
            base = next((r for r in rows if r["step"] == 0), None)
            if base is not None:
                # t=0 has no place on a log axis, so the base becomes this
                # arm's dashed reference line.
                ax.axhline(value(base), color=color, lw=1.1, ls=(0, (5, 3)),
                           alpha=0.55, zorder=1)
            ends.append((pts[-1][0], pts[-1][1], label))
        ax.set_xscale("log")
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel("training step (log)")
        ax.grid(True, alpha=0.6)
        ax.set_axisbelow(True)
        ends_by_ax[ax] = ends

    # Direct labels in a second pass: the light end of the ordinal ramp clears
    # the 2:1 ordinal floor but not the 3:1 categorical one, so identity is
    # never carried by color alone. Arms converge here, so the text is spread
    # apart (the markers stay at their true y) -- and the spread needs the
    # FINAL shared ylim, which only exists once every panel has been drawn.
    for ax, ends in ends_by_ax.items():
        for x, y, label in _declutter(ax, ends):
            ax.annotate(f" {label}", (x, y), textcoords="offset points",
                        xytext=(7, 0), fontsize=8, color=MUTED, va="center",
                        zorder=4, annotation_clip=False)
    axes[0].set_ylabel("relative depth  (layer / n_layers)")
    axes[0].legend(fontsize=8, frameon=False, loc="best", title="arm",
                   title_fontsize=8)
    handles = [Line2D([], [], color=MUTED, lw=1.1, ls=(0, (5, 3)),
                      label="each arm at t=0 (untuned base)")]
    axes[-1].legend(handles=handles, fontsize=8, frameon=False, loc="best")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
# Figures 3 & 4: depth profiles, metric x arm


def fig_profiles(arms, out_path, suptitle):
    # sharey per row: the arms are only comparable if a curve's height means
    # the same thing across the row. Per-panel autoscaling would make a flat
    # 27b profile look as tall as a real 270m one.
    fig, axes = plt.subplots(len(METRICS), len(arms),
                             figsize=(2.9 * len(arms) + 1.4, 2.6 * len(METRICS)),
                             squeeze=False, sharey="row", sharex=True)
    for r_i, (key, short, long) in enumerate(METRICS):
        row_axes = axes[r_i]
        for c_i, (label, rows) in enumerate(arms):
            ax = row_axes[c_i]
            n_ft = max(1, sum(1 for r in rows if r["step"] > 0) - 1)
            k = 0
            for r in rows:
                depth = np.array(r["layer_depth"])
                y = np.array(r["curves"][key])
                if r["step"] == 0:
                    ax.plot(depth, y, color=BASE_C, lw=2.2, zorder=4,
                            label="t=0 (base)")
                else:
                    ax.plot(depth, y, color=STEP_RAMP(k / n_ft), lw=1.3,
                            alpha=0.9, zorder=3)
                    k += 1
            if key == "autocorr_dlog":
                ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)
            if r_i == 0:
                ax.set_title(label, loc="left", color=INK, fontsize=10)
            if r_i == len(METRICS) - 1:
                ax.set_xlabel("relative depth")
            if c_i == 0:
                ax.set_ylabel(short, fontsize=9, labelpad=8)
            if c_i == len(arms) - 1:
                # Long form on the right rail, where nothing collides with it.
                ax.annotate(long, xy=(1.02, 0.5), xycoords="axes fraction",
                            rotation=270, va="center", ha="left",
                            fontsize=7.5, color=MUTED, annotation_clip=False)
            ax.grid(True, alpha=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.5)
    handles = [Line2D([], [], color=BASE_C, lw=2.2, label="t=0 (untuned base)"),
               Line2D([], [], color=STEP_RAMP(0.15), lw=1.3, label="early steps"),
               Line2D([], [], color=STEP_RAMP(1.0), lw=1.3, label="late steps")]
    axes[0][0].legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 0.985, 0.955))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


def fig_delta(arms, out_path, suptitle):
    """depth x step heatmap of each metric's change from that arm's own t=0.

    Limits are symmetric and shared across a row, so a cell means the same
    change in every arm; the colorbar sits per row for the same reason.
    """
    fig, axes = plt.subplots(len(METRICS), len(arms),
                             figsize=(2.9 * len(arms) + 2.0, 2.6 * len(METRICS)),
                             squeeze=False)
    for r_i, (key, short, long) in enumerate(METRICS):
        # Pass 1: the shared symmetric scale for this metric.
        deltas = {}
        for label, rows in arms:
            base = next((r for r in rows if r["step"] == 0), None)
            if base is None:
                continue
            b = np.array(base["curves"][key])
            ft = [r for r in rows if r["step"] > 0]
            if not ft:
                continue
            deltas[label] = (np.array([np.array(r["curves"][key]) - b for r in ft]).T,
                             [r["step"] for r in ft])
        if not deltas:
            continue
        vmax = max(np.abs(d).max() for d, _ in deltas.values()) or 1e-9
        im = None
        for c_i, (label, rows) in enumerate(arms):
            ax = axes[r_i][c_i]
            if label not in deltas:
                ax.set_axis_off()
                continue
            d, steps = deltas[label]
            im = ax.imshow(d, origin="lower", aspect="auto", cmap=DIVERGE,
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                           extent=[-0.5, len(steps) - 0.5, 0.0, 1.0])
            # The band edges, so the change is read against the structure it
            # is supposed to be moving.
            ft = [r for r in rows if r["step"] > 0]
            xs = np.arange(len(ft))
            ax.plot(xs, [r["band_depth"][0] for r in ft], color=INK, lw=1.2, zorder=3)
            ax.plot(xs, [r["band_depth"][1] for r in ft], color=INK, lw=1.2, zorder=3)
            _thin_ticks(ax, steps, every=3)
            if r_i == 0:
                ax.set_title(label, loc="left", color=INK, fontsize=10)
            if r_i == len(METRICS) - 1:
                ax.set_xlabel("training step")
            if c_i == 0:
                ax.set_ylabel("relative depth", fontsize=8.5, labelpad=6)
                ax.annotate(f"{short.replace(chr(10), ' ')}\n{long}",
                            xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center",
                            fontsize=8.5, color=INK, annotation_clip=False)
            ax.tick_params(labelsize=7.5)
        if im is not None:
            cb = fig.colorbar(im, ax=list(axes[r_i]), fraction=0.02, pad=0.01)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("Δ vs t=0", fontsize=7.5, color=MUTED)
    handles = [Line2D([], [], color=INK, lw=1.2, label="suggested band edges")]
    axes[0][0].legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL",
                    help="Label for the next positional band.jsonl (repeatable).")
    ap.add_argument("paths", nargs="+", help="band.jsonl files, in arm order.")
    ap.add_argument("--out", default="figures/band", help="Output directory.")
    ap.add_argument("--suptitle", default="Workspace band across medical SFT")
    args = ap.parse_args()

    if args.arm and len(args.arm) != len(args.paths):
        ap.error(f"{len(args.arm)} --arm labels for {len(args.paths)} paths")
    labels = args.arm or [Path(p).parent.name for p in args.paths]
    arms = [(lbl, load_arm(p)) for lbl, p in zip(labels, args.paths)]
    arms = [(lbl, rows) for lbl, rows in arms if rows]
    if not arms:
        ap.error("no rows in any band.jsonl")
    for lbl, rows in arms:
        if rows[0]["step"] != 0:
            print(f"WARNING: arm {lbl} has no t=0 row -- its delta/reference "
                  f"panels will be blank")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_ribbon(arms, out / "band_ribbon.png",
               f"{args.suptitle} — where the band sits")
    fig_edges(arms, out / "band_edges.png",
              f"{args.suptitle} — band edges by size")
    fig_profiles(arms, out / "band_profiles.png",
                 f"{args.suptitle} — depth profiles behind the band")
    fig_delta(arms, out / "band_delta.png",
              f"{args.suptitle} — change from the untuned base")


if __name__ == "__main__":
    main()
