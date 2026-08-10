"""Workspace-band trajectory probe -- medlens `band.py` over our SFT checkpoints.

Runs in the standalone `.venv-jlens` (transformers>=5.5 + jlens @ 581d398);
imports nothing from `gemma_med`. Sibling of `probe_jlens.py`, and it reuses that
script's checkpoint discovery and (load-bearing) text-only loader rather than
re-deriving them -- the two probes must agree on what "the model at step t" is.

For each checkpoint on a training trajectory it:

  1. obtains a Jacobian lens -- refit on the FROZEN generic corpus, or loaded
     from `--lens` / `--lens-dir` when a previous run already paid for it;
  2. computes the paper's Fig. 27/28 band metrics (`band.py`): per-layer CKA
     block structure, next-token prediction accuracy, excess kurtosis, top-1
     autocorrelation vs a shuffled null, and J-space dimensionality;
  3. calls `locate_band` for a suggested contiguous workspace band, recorded in
     BOTH raw layer indices and relative depth -- only the latter is comparable
     across sizes.

t=0 is the untouched base model, prepended to the checkpoint list, so every
trajectory carries its own untuned reference. Per checkpoint it writes
`<out>/ck<step>/band_stats.npz` (full curves + CKA matrix) and
`<out>/ck<step>/band_diagnostic.png`, and appends the scalars to
`<out>/band.jsonl`. `scripts/plot_band.py` reads those across arms.

THE COST IS THE LENS FIT, not the band metrics. Fitting is ~(d_model x params)
-- measured 3.5 h/ckpt at 4b on an H100, ~15 h at 12b, ~47 h at 27b -- while the
band pass on an existing lens is minutes to tens of minutes. So: pass `--lens`
whenever a lens for that exact checkpoint already exists, and `--save-lens`
whenever you are paying for a fit, so the next analysis never pays again.

Example (one checkpoint, reusing a lens that is already on disk):
  python scripts/probe_band.py \
    --base-model data/text_bases/4b_it \
    --checkpoints outputs/4b/full_lr1e-5_25911616/checkpoint-512 \
    --lens /gpfs/data/oermannlab/users/zhouj14/jlens_out/4b_it_fanout/ck512/lens.pt \
    --fit-corpus data/jlens/fit_corpus_v2.txt \
    --out eval/band/4b_it
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import jlens

import band as band_mod
from probe_jlens import discover_checkpoints, is_full_model, load_hf

log = logging.getLogger("probe_band")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="t=0 model + tokenizer source.")
    p.add_argument("--run-dir", default=None, help="Dir containing checkpoint-*/.")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="Explicit checkpoint dirs; overrides --run-dir.")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default=None,
                   help="Arm label recorded in every row (default: basename of --out).")

    # Lens: reuse or refit.
    p.add_argument("--lens", default=None,
                   help="Pre-fit lens .pt for the SINGLE checkpoint being probed.")
    p.add_argument("--lens-dir", default=None,
                   help="Dir of lens_step<N>.pt / ck<N>/lens.pt, matched per step.")
    p.add_argument("--save-lens", action="store_true",
                   help="Write the fitted lens to <out>/ck<step>/lens.pt (fp16).")
    p.add_argument("--fit-corpus", required=True,
                   help="Frozen generic corpus: fits the lens AND feeds the text stats.")
    p.add_argument("--fit-prompts", type=int, default=None, help="Cap corpus size for the fit.")
    p.add_argument("--dim-batch", type=int, default=16)
    p.add_argument("--fit-max-seq", type=int, default=128)
    p.add_argument("--fit-ckpt-every", type=int, default=50)

    # Band metrics.
    p.add_argument("--cka-tokens", type=int, default=8192,
                   help="Vocabulary subsample for the CKA/dimensionality features.")
    p.add_argument("--var-share", type=float, default=0.90)
    p.add_argument("--text-prompts", type=int, default=50,
                   help="Prompts for the kurtosis/autocorrelation/accuracy pass.")
    p.add_argument("--text-from-tail", action="store_true",
                   help="Draw the text-stats prompts from the END of the fit corpus "
                        "(held out from the fit when --fit-prompts truncates it).")
    p.add_argument("--text-max-seq", type=int, default=128)
    p.add_argument("--skip-first", type=int, default=16)
    p.add_argument("--acc-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--fresh", action="store_true", help="Ignore existing band.jsonl.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve_lens_path(args, step: int) -> str | None:
    """Pre-fit lens for `step`, or None if this step has to be refit."""
    if args.lens:
        return args.lens
    if not args.lens_dir:
        return None
    d = Path(args.lens_dir)
    for cand in (d / f"lens_step{step}.pt", d / f"ck{step}" / "lens.pt",
                 d / f"lens_{step}.pt"):
        if cand.exists():
            return str(cand)
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = args.arm or out.name
    rows_path = out / "band.jsonl"

    # Resume: skip checkpoints already written, so a requeue continues rather
    # than re-paying for fits it has already done.
    done_steps: set[int] = set()
    if args.fresh or not rows_path.exists():
        rows_path.write_text("")
    else:
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done_steps.add(json.loads(line)["step"])
        if done_steps:
            log.info("resuming: %d checkpoints already in %s", len(done_steps), rows_path)

    tok = transformers.AutoTokenizer.from_pretrained(args.base_model)
    corpus = [l for l in Path(args.fit_corpus).read_text().splitlines() if l.strip()]
    fit_corpus = corpus[: args.fit_prompts] if args.fit_prompts else corpus
    # The paper reads the text statistics off the distribution the lens was
    # fitted on. --text-from-tail takes them from the far end instead, which is
    # genuinely held out whenever --fit-prompts truncated the corpus.
    text_prompts = (corpus[-args.text_prompts:] if args.text_from_tail
                    else corpus[: args.text_prompts])
    ckpts = discover_checkpoints(args.base_model, args.run_dir, args.checkpoints)
    log.info("arm=%s fit_corpus=%d text_prompts=%d checkpoints=%d",
             arm, len(fit_corpus), len(text_prompts), len(ckpts))

    for step, path in ckpts:
        if step in done_steps:
            log.info("skip step %d (already in band.jsonl)", step)
            continue
        if not is_full_model(path):
            log.warning("skip step %d: LoRA adapter only (%s)", step, path)
            continue
        log.info("=== step %d :: %s ===", step, path)
        ck_dir = out / f"ck{step}"
        ck_dir.mkdir(exist_ok=True)

        hf = load_hf(path, dtype).to(args.device)
        model = jlens.from_hf(hf, tok)

        lens_path = resolve_lens_path(args, step)
        if lens_path:
            log.info("  lens: loading %s", lens_path)
            lens = jlens.JacobianLens.from_pretrained(lens_path)
            fit_s = 0.0
        else:
            log.info("  lens: fitting (this is the expensive part)")
            t0 = time.time()
            lens = jlens.fit(model, fit_corpus, dim_batch=args.dim_batch,
                             max_seq_len=args.fit_max_seq,
                             checkpoint_path=str(ck_dir / ".fit_ckpt.pt"),
                             checkpoint_every=(args.fit_ckpt_every or None))
            fit_s = time.time() - t0
            log.info("  lens: fitted in %.0f s", fit_s)
            if args.save_lens:
                lens.save(str(ck_dir / "lens.pt"))
                log.info("  lens: saved -> %s", ck_dir / "lens.pt")

        # band.readout_text_stats assumes the jacobians are already on the model
        # device (medlens preloads them for this reason). jlens.lens.readout
        # would otherwise .to() each J per layer per prompt -- at 27b that is
        # 115 MB x 62 layers x every prompt over PCIe, which turns a minutes-long
        # text pass into an hours-long one. Fitted lenses are usually already
        # there; loaded ones are not.
        try:
            for L in lens.source_layers:
                lens.jacobians[L] = lens.jacobians[L].to(model.input_device)
        except torch.cuda.OutOfMemoryError:
            log.warning("  jacobians do not fit on device; falling back to "
                        "per-call transfer (slower, same numbers)")
            torch.cuda.empty_cache()

        # One vocabulary draw shared by CKA and dimensionality: building the
        # meaningful-token mask decodes all 262k Gemma 3 ids one at a time.
        t0 = time.time()
        w_sub = band_mod.unembed_token_sample(
            model, n_tokens=args.cka_tokens, seed=args.seed, device=model.input_device)
        log.info("  token sample %s (%.0f s)", tuple(w_sub.shape), time.time() - t0)

        t0 = time.time()
        stats = band_mod.readout_text_stats(
            model, lens, text_prompts, max_seq_len=args.text_max_seq,
            skip_first=args.skip_first, seed=args.seed, acc_k=args.acc_k)
        log.info("  text stats over %d prompts (%.0f s)", len(text_prompts), time.time() - t0)

        t0 = time.time()
        cka, layers = band_mod.jlens_vector_cka(model, lens, w_sub=w_sub)
        log.info("  CKA over %d layers (%.0f s)", len(layers), time.time() - t0)

        t0 = time.time()
        dim_frac = band_mod.jspace_dimensionality(
            model, lens, var_share=args.var_share, w_sub=w_sub)
        log.info("  dimensionality over %d layers (%.0f s)", len(layers), time.time() - t0)

        autocorr_excess = stats["autocorr"] - stats["autocorr_null"]
        lo, hi = band_mod.locate_band(cka, layers, stats["kurtosis"], autocorr_excess)
        # Relative depth is the ONLY cross-size-comparable form of the band:
        # sizes differ in layer count, so a raw index means a different depth.
        n_layers = model.n_layers
        depth = [lo / n_layers, hi / n_layers]

        np.savez(
            ck_dir / "band_stats.npz",
            cka=cka, layers=np.array(layers),
            kurtosis=stats["kurtosis"],
            autocorr=stats["autocorr"], autocorr_null=stats["autocorr_null"],
            acc_top1=stats["acc_top1"], acc_topk=stats["acc_topk"],
            acc_k=stats["acc_k"], dim_frac=dim_frac,
        )
        band_mod.band_diagnostic_plot(
            cka, layers, stats, (lo, hi), str(ck_dir / "band_diagnostic.png"),
            dim_frac=dim_frac,
            title=f"{arm} step {step} — suggested workspace band: layers {lo}-{hi} "
                  f"({depth[0]:.0%}-{depth[1]:.0%} of depth)")

        # Row carries the full per-layer curves as well as the scalars: the
        # cross-arm plots need the curves, and re-opening 5 arms x 14 npz files
        # to get them is slower and easier to get out of sync.
        row = {
            "arm": arm, "step": step, "path": path,
            "n_layers": n_layers, "d_model": model.d_model,
            "band": [lo, hi], "band_depth": depth,
            "band_width_depth": depth[1] - depth[0],
            "fit_seconds": round(fit_s, 1),
            "lens_reused": lens_path,
            "layers": [int(x) for x in layers],
            "layer_depth": [x / n_layers for x in layers],
            "curves": {
                "acc_top1": stats["acc_top1"].tolist(),
                "acc_topk": stats["acc_topk"].tolist(),
                "kurtosis": stats["kurtosis"].tolist(),
                "autocorr": stats["autocorr"].tolist(),
                "autocorr_null": stats["autocorr_null"].tolist(),
                "autocorr_excess": autocorr_excess.tolist(),
                "dim_frac": dim_frac.tolist(),
            },
            "cka_mean_offdiag": float(cka[np.triu_indices_from(cka, 1)].mean()),
        }
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        Path(ck_dir / ".fit_ckpt.pt").unlink(missing_ok=True)

        log.info("  band = layers %d-%d (%.0f%%-%.0f%% of depth), width %.0f%%",
                 lo, hi, 100 * depth[0], 100 * depth[1], 100 * (depth[1] - depth[0]))

        del hf, model, lens, w_sub
        torch.cuda.empty_cache()

    log.info("wrote %s", rows_path)


if __name__ == "__main__":
    main()
