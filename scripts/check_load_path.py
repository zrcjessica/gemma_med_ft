"""Verify the train.py load path populates ALL language-model weights.

train.py loads every size with AutoModelForCausalLM, trusting transformers to
resolve gemma3 -> Gemma3ForCausalLM (text-only) even for 4b+, whose checkpoints
declare the multimodal Gemma3ForConditionalGeneration wrapper. If the wrapper's
LM weights are nested under a prefix the bare CausalLM doesn't expect, they land
in `missing_keys` and get silently re-initialized -- poisoning every checkpoint
in the trajectory with no hard error.

This loads exactly as train.py does (CPU, no GPU needed) and asserts nothing the
CausalLM expects is missing. Vision-tower / projector weights showing up in
`unexpected_keys` is fine and expected -- that's the multimodal head being
correctly dropped for text-only use.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def check(model_path: str) -> bool:
    cfg = AutoConfig.from_pretrained(model_path)
    print(f"  model_type={cfg.model_type}  architectures={cfg.architectures}")

    model, info = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    print(f"  resolved class -> {type(model).__name__}")

    missing = info.get("missing_keys", [])
    unexpected = info.get("unexpected_keys", [])
    mismatched = info.get("mismatched_keys", [])
    errors = info.get("error_msgs", [])

    print(f"  missing_keys={len(missing)}  unexpected_keys={len(unexpected)}"
          f"  mismatched={len(mismatched)}  errors={len(errors)}")
    # Unexpected keys are the dropped vision/projector head -- show the distinct
    # top-level prefixes so it's obvious that's all they are.
    if unexpected:
        prefixes = sorted({k.split(".")[0] for k in unexpected})
        print(f"  unexpected top-level prefixes: {prefixes}")

    # Quantify what train.py's freeze_vision_tower will catch (same match rule).
    total = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for n, p in model.named_parameters()
                 if "vision_tower" in n or "multi_modal_projector" in n)
    if vision:
        print(f"  vision-tower/projector params: {vision/1e6:.1f}M "
              f"({100*vision/total:.1f}% of {total/1e9:.2f}B) -> frozen for text-only SFT")

    ok = True
    if missing:
        ok = False
        print(f"  !! {len(missing)} MISSING LM weights (silently re-initialized):", file=sys.stderr)
        for k in missing[:20]:
            print(f"       {k}", file=sys.stderr)
        if len(missing) > 20:
            print(f"       ... +{len(missing) - 20} more", file=sys.stderr)
    if mismatched or errors:
        ok = False
        print(f"  !! mismatched={mismatched} errors={errors}", file=sys.stderr)

    print(f"  => {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_paths", nargs="+", help="One or more snapshot dirs to check.")
    p.add_argument("--labels", nargs="*", default=None, help="Optional labels, parallel to paths.")
    args = p.parse_args()

    labels = args.labels or [""] * len(args.model_paths)
    all_ok = True
    for label, path in zip(labels, args.model_paths):
        print(f"\n=== {label or path} ===")
        try:
            all_ok &= check(path)
        except Exception as e:
            all_ok = False
            print(f"  !! load raised: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\nOVERALL: {'ALL OK' if all_ok else 'FAILURES PRESENT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
