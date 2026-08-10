#!/usr/bin/env python3
"""Retro-correct the fp32 drift cosine in an existing jlens metrics.jsonl.

Until 2026-08-10 jac_drift divided an fp32 torch.dot (accurate) by a product of
fp32 torch.linalg.vector_norm (which underestimates |J| -- it accumulates
d_model^2 terms until the smallest are absorbed). The quotient was inflated by
a factor 1/(r_t*r_0), r(x) = |x|_fp32 / |x|_fp64, and since r falls with
d_model the bias was also cross-size: ~0.01pp at 270m, ~1.4pp at 27b.

dot was already accurate, so the recorded numbers can be corrected in place:

    cos_true = cos_measured * r(J_t) * r(J_0) ~= cos_measured * r(J_0)^2

r(J_0)^2 comes from one of two sources, in order of preference:

  identity  -- a trajectory whose first checkpoint is bit-identical to t=0
               (relfro == 0) has cos_true == 1 exactly at that step, so
               1/cos_measured IS r(J_0)^2 per layer. No approximation.
  baselens  -- otherwise compute r per layer from the saved fp32 .base_lens.pt.
               Then r(J_t) ~= r(J_0) is an approximation; validated to <=0.27pp
               residual out to relfro=0.45, against a 1.4pp bias removed.

relfro is left alone: both its terms are vector_norm, so the biases largely
cancel and what survives is ~+0.1% relative at 4b, ~+0.5% at 27b.

Raw values are preserved as *_fp32raw and the original file is backed up.
"""
import argparse
import json
import shutil
from pathlib import Path


def factors_from_identity(rows, max_relfro):
    """r(J0)^2 per layer, read off the least-drifted checkpoint.

    Exact when that checkpoint is bit-identical to t=0 (relfro==0, cos_true==1).
    Otherwise cos_true >= 1 - relfro^2/2, so a near-identity step is usable and
    the shortfall is reported as the residual.
    """
    cands = [r for r in sorted(rows, key=lambda r: r["step"]) if r["step"] != 0]
    if not cands:
        return None, None, None
    best = min(cands, key=lambda r: max(r["drift"]["relfro_curve"]))
    worst = max(best["drift"]["relfro_curve"])
    if worst > max_relfro:
        return None, None, None
    d = best["drift"]
    fac = {L: 1.0 / c for L, c in zip(d["layers"], d["cos_curve"])}
    return fac, best["step"], worst ** 2 / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--base-lens-r", help="JSON {layer: r} from the saved base lens")
    p.add_argument("--max-relfro", type=float, default=1e-2,
                   help="reject an identity source whose relfro exceeds this")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    mp = Path(args.metrics)
    rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
    if any("cos_mean_fp32raw" in r["drift"] for r in rows):
        raise SystemExit(f"{mp} is already corrected -- refusing to double-apply")

    # Prefer the base lens when we have it: it needs no near-identity checkpoint.
    fac = None
    if args.base_lens_r:
        r = {int(k): v for k, v in json.load(open(args.base_lens_r)).items()}
        fac, resid = {L: r[L] ** 2 for L in r}, None   # resid: r(J_t)~=r(J_0), see below
        method = f"baselens({Path(args.base_lens_r).name})"
    if fac is None:
        fac, src_step, resid = factors_from_identity(rows, args.max_relfro)
        method = f"identity(step={src_step})"
    if fac is None:
        raise SystemExit("no near-identity checkpoint and no --base-lens-r; cannot correct")

    vals = sorted(fac.values())
    print(f"{mp}\n  method={method}  factor {vals[0]:.6f}..{vals[-1]:.6f} "
          f"(bias {(1-vals[-1])*100:.4f}-{(1-vals[0])*100:.4f} pp)"
          + (f"  residual<={resid*100:.1e} pp" if resid is not None
             else "  residual: r(J_t)~=r(J_0), <=0.27pp at relfro 0.45"
             if args.base_lens_r else "  residual=0 (bit-identical)"))

    n_fixed = 0
    for row in rows:
        d = row["drift"]
        if row["step"] == 0:                      # hardcoded 1.0, never measured
            continue
        if set(d["layers"]) - set(fac):
            raise SystemExit(f"step {row['step']}: layers missing from factors")
        corr = [c * fac[L] for L, c in zip(d["layers"], d["cos_curve"])]
        n_fixed += sum(1 for c in d["cos_curve"] if c > 1.0)
        d["cos_curve_fp32raw"] = d["cos_curve"]
        d["cos_mean_fp32raw"] = d["cos_mean"]
        d["cos_curve"] = corr
        d["cos_mean"] = sum(corr) / len(corr)
        d["cos_correction"] = {"method": method, "relfro": "uncorrected"}

    # A factor derived from a stored fp32 cosine carries eps~1.2e-7, so a
    # residual excess at that scale is round-off, not unexplained bias.
    excess = [c - 1.0 for r in rows if r["step"] != 0
              for c in r["drift"]["cos_curve"] if c > 1.0]
    print(f"  cos>1 layer-steps: {n_fixed} -> {len(excess)}"
          + (f"  (max excess {max(excess):.1e})" if excess else ""))
    if excess and max(excess) > 1e-5:
        print("  WARNING: violations remain above round-off; bias not fully explained")
    if args.dry_run:
        print("  (dry run, not written)")
        return
    shutil.copy2(mp, mp.with_suffix(".jsonl.fp32raw"))
    mp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"  wrote {mp}  (raw kept at {mp.name}.fp32raw)")


if __name__ == "__main__":
    main()
