"""Exemplar heatmap + transcript for each red-flag run over OUR checkpoints.

A fork of yeb04's `medlens/redflag_exemplars.py` with three things changed: our
run/results roots instead of theirs, our configs, and the band read from our own
band probe (`eval/band/<arm>/band.jsonl`, matched per step) instead of their
CONFIRMED table -- which only covers 27b models and says nothing about a
finetuned checkpoint. The rendering itself is theirs, called not copied
(`medlens.make_figures.exemplar_figure` / `exemplar_transcript`), so our panels
are the same object as the lab's.

Each run is registered as its own one-off "model" in make_figures' registries,
which is how the upstream script gets arbitrary runs through a code path whose
tables only know three headline models.

    python scripts/redflag/exemplars.py --runs /gpfs/.../redflag_out \
        --out figs/redflag --stem rf-001 --concept suicide
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(os.environ.get("REPO_ROOT", "/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft"))
CFG_DIR = Path(__file__).parent / "configs"
# medlens is read-only on yeb04's tree and not installed in .venv-jlens; put it
# on the path here so this runs without a PYTHONPATH incantation.
MEDLENS = os.environ.get("MEDLENS", "/gpfs/data/oermannlab/users/yeb04/jlens/code/medlens")
if MEDLENS not in sys.path:
    sys.path.insert(0, MEDLENS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="root holding <tag>/ run dirs")
    p.add_argument("--out", required=True, help="figure root; one subdir per run")
    p.add_argument("--stem", default="rf-001", help="vignette pair stem (<stem>-i / <stem>-b)")
    p.add_argument("--concept", default="suicide", help="probe concept whose ranks are colored")
    p.add_argument("--band-root", default=str(REPO / "eval" / "band"),
                   help="dir of <arm>/band.jsonl from probe_band.py")
    return p.parse_args()


def parse_tag(tag: str) -> tuple[str, int]:
    """`4b_it_ck512` -> ("4b_it", 512)."""
    arm, _, step = tag.rpartition("_ck")
    if not arm or not step.isdigit():
        raise ValueError(f"cannot parse arm/step from run tag {tag!r}")
    return arm, int(step)


def band_for(arm: str, step: int, run_dir: Path, band_root: Path) -> tuple[int, int]:
    """Our own band pass for this exact checkpoint, else the paper-relative
    38-92% of the run's layer axis (upstream's last-resort fallback).

    Never falls back to another step's band: the band MOVES over the finetune
    (4b: [13,31] at step 512 -> [9,30] at 4882), so borrowing one would draw the
    wrong lines on the heatmap.
    """
    # A gap-filled step's row lives in its own dir (fill_band_gaps.sh writes
    # eval/band/<arm>_ck<step>/band.jsonl rather than appending to the arm's
    # file), so check both -- same lookup analyze_all.sh does.
    for bj in (band_root / arm / "band.jsonl", band_root / f"{arm}_ck{step}" / "band.jsonl"):
        if not bj.exists():
            continue
        for line in bj.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("step") == step and row.get("band"):
                return int(row["band"][0]), int(row["band"][1])
    layers = np.load(next(iter(sorted((run_dir / "npz").glob("*.npz")))))["layers"]
    top = int(layers.max())
    print(f"[band] {arm} step {step}: no band row, using paper-relative 38-92%")
    return round(0.38 * top), round(0.92 * top)


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from medlens import make_figures as mf

    runs_root, out_root = Path(args.runs), Path(args.out)
    # make_figures resolves runs and transcripts off its own module-level roots.
    mf.RUNS = runs_root
    mf.RES = out_root
    mf._setup_mpl()

    done, skipped = [], []
    for run_dir in sorted(runs_root.glob("*")):
        if not run_dir.is_dir():
            continue
        run = run_dir.name
        cfg_path = CFG_DIR / f"{run}.yaml"
        ready = ((run_dir / "npz" / f"{args.stem}-i.npz").exists()
                 and (run_dir / "npz" / f"{args.stem}-b.npz").exists()
                 and (run_dir / "judged.jsonl").exists() and cfg_path.exists())
        if not ready:
            skipped.append(run)
            continue
        arm, step = parse_tag(run)
        cfg = yaml.safe_load(cfg_path.read_text())
        band = band_for(arm, step, run_dir, Path(args.band_root))

        key = run  # this run is its own exemplar "model"
        mf.EXEMPLAR_MODELS[key] = {"tokenizer": cfg["hf_model_name"], "run": run, "band": band}
        mf.MODEL_LABEL[key] = f"gemma-3-{arm.replace('_', '-')} @ step {step}"
        mf.TRANSCRIPT_RUNS[key] = [(None, run)]

        out = out_root / run
        out.mkdir(parents=True, exist_ok=True)
        try:
            mf.exemplar_figure(out, args.stem, args.concept, model=key)
            mf.exemplar_transcript(out, args.stem, args.concept, model=key)
            print(f"[exemplar] {run}  band={band}  -> {out}")
            done.append(run)
        except Exception as e:  # keep sweeping; report at the end
            print(f"[exemplar] {run} FAILED: {type(e).__name__}: {e}")
            skipped.append(run)

    print(f"\nswept: {len(done)} done, {len(skipped)} skipped/not-ready: {skipped}")


if __name__ == "__main__":
    main()
