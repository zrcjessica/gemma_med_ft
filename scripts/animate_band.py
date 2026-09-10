"""Animate one arm's band diagnostic across training steps, oldest to newest.

Frames are rendered by `band.band_diagnostic_plot` -- the vendored medlens
function, same as the static panels -- and stitched into a GIF with PIL.

The one thing this adds is a **shared scale across frames**. Left alone,
`band_diagnostic_plot` autoscales each curve panel to its own frame and
contrast-stretches the CKA colormap to that frame's off-diagonal minimum. That
is right for reading one checkpoint and wrong for an animation: every frame
renormalizes, so a curve that doubles looks identical from frame to frame and
only the axis labels move. Here the y-limits (and the CKA color limits) are
computed once over every step in the arm and pinned, so motion on screen is
motion in the metric.

Pinning happens in a `Figure.savefig` hook rather than a fork of
`band_diagnostic_plot`: the panels are found by their titles and their limits
set just before the frame is written. Nothing about the plotted data changes.

    python scripts/animate_band.py --arm 4b \
        --npz-root <band_out> --out figs/band

Writes figs/band/<arm>_it/<arm>_it_band.gif plus the pinned-scale frames.
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402

import band as band_mod  # noqa: E402
from plot_band_diagnostics import STAT_KEYS, find_npz  # noqa: E402

# Panel title -> which shared limit it takes. Titles come from band.py; matched
# by prefix so a wording tweak upstream degrades to "unpinned", not a crash.
KURT_TITLE = "excess kurtosis"
AUTO_TITLE = "top-1 autocorrelation"
CKA_TITLE = "CKA among J-lens vectors"


def _pad(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    return lo - frac * span, hi + frac * span


@contextmanager
def pinned_axes(ylims: dict[str, tuple[float, float]], cka_vmin: float):
    """Force shared limits onto the figure at savefig time."""
    orig = matplotlib.figure.Figure.savefig

    def hooked(fig, *a, **kw):
        for ax in fig.axes:
            title = ax.get_title()
            for key, lim in ylims.items():
                if title.startswith(key):
                    ax.set_ylim(*lim)
            if title.startswith(CKA_TITLE):
                for im in ax.get_images():
                    im.set_clim(cka_vmin, 1.0)
        return orig(fig, *a, **kw)

    matplotlib.figure.Figure.savefig = hooked
    try:
        yield
    finally:
        matplotlib.figure.Figure.savefig = orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--band-root", type=Path, default=Path("eval/band"))
    ap.add_argument("--npz-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figs/band"))
    ap.add_argument("--ms-per-frame", type=int, default=700)
    ap.add_argument("--hold-ms", type=int, default=2200,
                    help="extra dwell on the first and last frame")
    ap.add_argument("--width", type=int, default=1900,
                    help="GIF width in px; frames are rendered at full res first")
    args = ap.parse_args()

    from PIL import Image

    for arm in args.arm:
        rows_path = args.band_root / f"{arm}_it" / "band.jsonl"
        rows = {json.loads(l)["step"]: json.loads(l) for l in rows_path.open()}
        npz_by_step = find_npz(args.npz_root, arm)
        steps = sorted(rows)

        loaded = {s: np.load(npz_by_step[s]) for s in steps}

        # One pass for the shared scale, over every step in the arm.
        kurt = np.concatenate([loaded[s]["kurtosis"] for s in steps])
        delta_log = np.concatenate([
            np.log(np.maximum(loaded[s]["autocorr"], 1e-9))
            - np.log(np.maximum(loaded[s]["autocorr_null"], 1e-9)) for s in steps])
        ylims = {
            KURT_TITLE: _pad(float(kurt.min()), float(kurt.max())),
            AUTO_TITLE: _pad(float(delta_log.min()), float(delta_log.max())),
        }
        cka_vmin = min(
            float(loaded[s]["cka"][np.triu_indices_from(loaded[s]["cka"], 1)].min())
            for s in steps)

        frame_dir = args.out / f"{arm}_it" / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for step in steps:
            d = loaded[step]
            stats = {"layers": d["layers"], **{k: d[k] for k in STAT_KEYS}}
            lo, hi = rows[step]["band"]
            d0, d1 = rows[step]["band_depth"]
            out = frame_dir / f"step{step:05d}.png"
            with pinned_axes(ylims, cka_vmin):
                band_mod.band_diagnostic_plot(
                    d["cka"], list(d["layers"]), stats, (lo, hi), str(out),
                    dim_frac=d["dim_frac"],
                    title=f"{arm}_it   step {step}   —   workspace band: layers "
                          f"{lo}-{hi} ({d0:.0%}-{d1:.0%} of depth)")
            paths.append(out)

        frames = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            if args.width and im.width > args.width:
                h = round(im.height * args.width / im.width)
                im = im.resize((args.width, h), Image.LANCZOS)
            frames.append(im)

        durations = [args.ms_per_frame] * len(frames)
        durations[0] += args.hold_ms
        durations[-1] += args.hold_ms

        gif = args.out / f"{arm}_it" / f"{arm}_it_band.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True)
        print(f"{arm}: {len(frames)} frames (steps {steps[0]}..{steps[-1]}) -> {gif} "
              f"[{gif.stat().st_size / 1e6:.1f} MB]")
        print(f"   pinned kurtosis {ylims[KURT_TITLE][0]:.2f}..{ylims[KURT_TITLE][1]:.2f}, "
              f"autocorr {ylims[AUTO_TITLE][0]:.2f}..{ylims[AUTO_TITLE][1]:.2f}, "
              f"CKA vmin {cka_vmin:.2f}")


if __name__ == "__main__":
    main()
