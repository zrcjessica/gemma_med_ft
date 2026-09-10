"""Emit medlens ModelConfig YAMLs pairing one of our checkpoints with its banked lens.

yeb04's `medlens` (the silent-red-flag study) takes one YAML per model: an HF
path plus a *pre-fit* J-lens `.pt`. Our trajectory already has both -- the band
grid ran with SAVE_LENS=1 -- so a red-flag read-out over the finetune costs a
forward pass per vignette, not a refit (0.35 h at 270m, 12.8 h at 12b, 35 h at 27b).

**The band probe's own record is the authority.** `eval/band/<arm>/band.jsonl`
has one row per step carrying the exact `path` (model) and `lens_reused` (lens)
that produced that step's band. Reading the pairing back out of it means the
red-flag run and the band it is analysed in are the same object, by construction
rather than by two independent path guesses agreeing. Globbing is the fallback,
for the rows where the lens was fit fresh (`lens_reused: null`).

Refuses to guess. A missing checkpoint or a missing lens is an error, because
the silent alternative (medlens refitting, or worse, pairing a checkpoint with
another step's lens) invalidates the run. Lens paths are variously nested
(`ck<N>/lens.pt`, `ck<N>/ck<N>/lens.pt`) under variously suffixed arm roots
(`<arm>`, `<arm>_fanout`, `<arm>_array`); every layout is tried.

Also reports each lens's stored dtype. The 270m/1b t=0 lenses were banked in
fp16 and everything else in fp32 -- see `--require-fp32`.

Run on BigPurple -- the paths it validates only exist there.

    python scripts/redflag/make_configs.py --arm 4b_it \
        --run-dir outputs/4b/full_lr1e-5_25911616 --steps 0 512 4882
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT", "/gpfs/data/oermannlab/users/zhouj14/gemma_med_ft"))
BAND_OUT = Path(os.environ.get("BAND_OUT", "/gpfs/data/oermannlab/users/zhouj14/band_out"))
JLENS_OUT = Path(os.environ.get("JLENS_OUT", "/gpfs/data/oermannlab/users/zhouj14/jlens_out"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, help="e.g. 4b_it")
    p.add_argument("--run-dir", help="training run dir holding checkpoint-*/ (not needed for --steps 0)")
    p.add_argument("--steps", nargs="+", type=int, required=True, help="0 = the untuned base (t=0)")
    p.add_argument("--out-dir", default=str(Path(__file__).parent / "configs"))
    p.add_argument("--band-root", default=str(REPO / "eval" / "band"))
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="cuda",
                   help="'cuda' pins one GPU; 'auto' shards (27b on 2x40GB A100 needs this, "
                        "and MAX_MEM_HEADROOM_GIB=8 with it -- see medlens/loading.py).")
    p.add_argument("--require-fp32", action="store_true",
                   help="Error on an fp16-banked lens instead of warning. The 270m/1b t=0 "
                        "lenses are fp16 while every other lens in the study is fp32; that "
                        "is a precision split sitting exactly at the trajectory origin.")
    return p.parse_args()


def band_row(arm: str, step: int, band_root: Path) -> dict | None:
    """The band probe's own record for this (arm, step), if it ran."""
    bj = band_root / arm / "band.jsonl"
    if not bj.exists():
        return None
    for line in bj.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("step") == step:
                return row
    return None


def model_path(arm: str, step: int, run_dir: str | None, row: dict | None) -> Path:
    if step == 0:
        # Whatever the band probe called t=0 -- a converted text base for 4b+,
        # the stock HF snapshot for 270m/1b (which need no conversion).
        if row and row.get("path"):
            return Path(row["path"])
        return REPO / "data" / "text_bases" / arm
    if not run_dir:
        raise SystemExit("--run-dir is required for any step > 0")
    return (REPO / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)) / f"checkpoint-{step}"


def lens_path(arm: str, step: int, row: dict | None) -> Path:
    """The lens the band probe used at this step, else the banked one it wrote."""
    # An fp32 t=0 refit OUTRANKS the band row, which still names the fp16 lens it
    # was computed with. Checked before the row, not after, or the shortcut below
    # returns the fp16 one and the refit is silently ignored.
    fp32_base = REPO / "eval" / "band" / f"{arm}_fp32base" / "ck0" / "lens.pt"
    if step == 0 and fp32_base.exists():
        return fp32_base
    if row and row.get("lens_reused"):
        p = Path(row["lens_reused"])
        if p.exists():
            return p

    roots = [BAND_OUT / f"{arm}_fanout", BAND_OUT / f"{arm}_array", BAND_OUT / arm]
    cands: list[Path] = []
    if step == 0:
        # fp32 t=0 refit first (fill_band_gaps.sh) -- 270m/1b banked their t=0 in
        # fp16 while every finetuned step is fp32, and the origin is the arm's
        # own reference point. Measured: the refit reproduces the fp16 band
        # exactly, so this buys uniformity, not a different answer.
        cands += [REPO / "eval" / "band" / f"{arm}_fp32base" / "ck0" / "lens.pt",
                  JLENS_OUT / arm / ".base_lens.pt"]
    for r in roots:
        cands += [r / f"ck{step}" / f"ck{step}" / "lens.pt", r / f"ck{step}" / "lens.pt"]
    for c in cands:
        if c.exists():
            return c
    # Last resort: any lens.pt under a matching ck<N>/ anywhere below an arm root.
    for r in roots:
        hits = sorted(r.glob(f"**/ck{step}/lens.pt")) if r.exists() else []
        if hits:
            return hits[0]
    raise SystemExit(
        f"no banked lens for {arm} step {step}; tried:\n  " + "\n  ".join(str(c) for c in cands)
        + "\nFitting one is 0.35 h at 270m and 35 h at 27b -- do that deliberately, "
          "not as a side effect of a red-flag run."
    )


def lens_meta(path: Path) -> dict:
    """Stored dtype / corpus size, without materializing the lens.

    `mmap=True` is not an optimization here. A lens is 7.0 GB at 27b and 2.8 GB
    at 12b, and a plain `torch.load` of one gets the process OOM-killed on the
    login node -- silently, with no traceback and exit 0, which reads exactly
    like "this arm has no configs to write". Only dtype/shape are touched, so
    nothing is ever paged in.
    """
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    j = d["J"][sorted(d["J"])[0]]
    return {"dtype": str(j.dtype).replace("torch.", ""), "d_model": d["d_model"],
            "n_src": len(d["source_layers"]), "n_prompts": d["n_prompts"]}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    band_root = Path(args.band_root)

    written, warnings = [], []
    for step in args.steps:
        row = band_row(args.arm, step, band_root)
        model = model_path(args.arm, step, args.run_dir, row)
        if not (model / "config.json").exists():
            raise SystemExit(f"no model at {model}")
        lens = lens_path(args.arm, step, row)
        meta = lens_meta(lens)
        tag = f"{args.arm}_ck{step}"

        band = row.get("band") if row else None
        if band is None:
            warnings.append(f"{tag}: NO band row -- analysis will fall back to 38-92% of the layer axis")
        if meta["dtype"] != "float32":
            msg = f"{tag}: lens banked in {meta['dtype']}, not fp32 ({lens})"
            if args.require_fp32:
                raise SystemExit(msg)
            warnings.append(msg)

        text = (
            f"# gemma_med_ft red-flag probe: {args.arm} @ step {step}\n"
            f"# model + lens pairing taken from eval/band/{args.arm}/band.jsonl"
            f"{'' if row else ' (NO ROW -- paths inferred)'}\n"
            f"# lens: {meta['dtype']}, d_model={meta['d_model']}, {meta['n_src']} source layers, "
            f"n_prompts={meta['n_prompts']} (fit_corpus_v2.txt)\n"
            f"# workspace band at this step: {band}\n"
            f"np_model_id: {tag}\n"
            f"hf_model_name: {model}\n"
            f"lens_path: {lens}\n"
            f"dtype: {args.dtype}\n"
            f"max_seq_len: {args.max_seq_len}\n"
            f"device_map: {args.device_map}\n"
        )
        (out_dir / f"{tag}.yaml").write_text(text)
        written.append(out_dir / f"{tag}.yaml")
        print(f"[cfg] {tag}  band={band}  lens={meta['dtype']}\n"
              f"      model={model}\n      lens ={lens}")

    for w in warnings:
        print(f"[WARN] {w}")
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
