"""Replot the medlens Fig. 27-28 diagnostic panel for every probed checkpoint.

`probe_band.py` already writes one `band_diagnostic.png` next to each
`band_stats.npz`, but they are scattered across the fanout tree and the t=0 dir.
This regenerates them from the saved npz into one directory per arm, so a whole
trajectory can be flipped through in step order.

Plotting is `band.band_diagnostic_plot` -- the vendored medlens function, not a
local redraw. Nothing here recomputes a metric; the npz is the result.

    python scripts/plot_band_diagnostics.py --arm 270m --arm 1b --arm 4b \
        --band-root eval/band --npz-root <band_out> --out figs/band
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import band as band_mod

STAT_KEYS = ("kurtosis", "autocorr", "autocorr_null", "acc_top1", "acc_topk", "acc_k")


def find_npz(npz_root: Path, arm: str) -> dict[int, Path]:
    """step -> band_stats.npz, over both the t=0 dir and the fanout tree."""
    found: dict[int, Path] = {}
    for base in (npz_root / f"{arm}_it", npz_root / f"{arm}_it_fanout"):
        for p in base.rglob("band_stats.npz"):
            m = re.fullmatch(r"ck(\d+)", p.parent.name)
            if not m:
                continue
            step = int(m.group(1))
            # The fanout nests ck<N>/ck<N>/; either depth resolves to the same
            # step, so first-wins is fine but assert we never disagree.
            if step in found and found[step] != p:
                raise SystemExit(f"{arm}: two npz for step {step}: {found[step]} and {p}")
            found[step] = p
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--band-root", type=Path, default=Path("eval/band"))
    ap.add_argument("--npz-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figs/band"))
    args = ap.parse_args()

    for arm in args.arm:
        rows_path = args.band_root / f"{arm}_it" / "band.jsonl"
        if not rows_path.exists():
            raise SystemExit(f"no merged rows: {rows_path}")
        rows = {json.loads(l)["step"]: json.loads(l) for l in rows_path.open()}
        npz_by_step = find_npz(args.npz_root, arm)

        missing = sorted(set(rows) - set(npz_by_step))
        if missing:
            raise SystemExit(f"{arm}: rows without an npz: {missing}")

        out_dir = args.out / f"{arm}_it" / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)

        for step in sorted(rows):
            row = rows[step]
            d = np.load(npz_by_step[step])
            stats = {"layers": d["layers"], **{k: d[k] for k in STAT_KEYS}}
            lo, hi = row["band"]
            d0, d1 = row["band_depth"]
            # Zero-pad the step so the files sort the way the trajectory runs.
            out = out_dir / f"step{step:05d}.png"
            band_mod.band_diagnostic_plot(
                d["cka"], list(d["layers"]), stats, (lo, hi), str(out),
                dim_frac=d["dim_frac"],
                title=f"{arm}_it step {step} — workspace band: layers {lo}-{hi} "
                      f"({d0:.0%}-{d1:.0%} of depth)")
            print(f"{arm} step {step:>5}: band {lo}-{hi} -> {out}")


if __name__ == "__main__":
    main()
