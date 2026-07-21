"""Convert a Gemma 3 4b+ multimodal base into a text-only Gemma3ForCausalLM base.

Run in the TRAINING env (.venv, transformers 4.53.2), which correctly remaps the
base's nested `language_model.model.*` keys onto Gemma3ForCausalLM's `model.*`.
The probe env (transformers 5.14.1) does NOT do that remap -- it silently
random-initializes all 444 LM tensors -- so the probe must be handed an
already-converted base rather than the raw multimodal snapshot.

This makes t=0 identical in class and key layout to the checkpoints train.py
writes, so the whole trajectory is measured on one code path.

    python scripts/make_text_base.py --size 4b --kind it --out data/text_bases/4b_it
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", required=True)
    p.add_argument("--kind", default="it")
    p.add_argument("--base-model", default=None, help="Skip resolution; use this snapshot dir.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = args.base_model
    if base is None:
        base = subprocess.check_output(
            ["bash", "-c", f"source scripts/_resolve_model.sh; resolve_gemma {args.size} {args.kind}"]
        ).decode().strip()
    if not base:
        raise SystemExit(f"could not resolve gemma-3-{args.size}-{args.kind}")
    print(f"base: {base}")

    model, info = Gemma3ForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, output_loading_info=True
    )
    missing = info.get("missing_keys", [])
    if missing:
        raise RuntimeError(
            f"{len(missing)} LM weights missing -- this env does not remap the "
            f"multimodal keys; run under .venv (transformers 4.53.2). {missing[:5]}"
        )
    unexpected = info.get("unexpected_keys", [])
    print(f"loaded {type(model).__name__}: missing=0 dropped={len(unexpected)} non-LM weights")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out))
    print(f"saved text-only base -> {out}")


if __name__ == "__main__":
    main()
