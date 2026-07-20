"""Throwaway spike: does jlens.from_hf / fit / apply work on Gemma 3?

Gates the whole interp study. Success = layout auto-detects, fit runs grads
through Gemma3 blocks, apply returns sane concordant logits. Run under srun with
a GPU; loads the 1b-it from the lab shared cache offline.
"""

import os
import sys

import torch
import transformers

import jlens

MODEL = os.environ.get("SPIKE_MODEL", "google/gemma-3-1b-it")

# ~generic web text, each comfortably > skip_first(16)+1 tokens.
PROMPTS = [
    "The city council met on Tuesday evening to discuss the proposed changes to the downtown parking regulations and the new bike lanes.",
    "Scientists have long debated whether the earliest human settlements in the region were seasonal camps or permanent year-round villages.",
    "After the storm passed, residents returned to find fallen trees blocking the main road and power lines down across several neighborhoods.",
    "The recipe calls for two cups of flour, a pinch of salt, three eggs, and enough warm milk to bring the dough together into a soft ball.",
    "Investors reacted cautiously to the earnings report, and by the closing bell the index had given back most of the gains from the morning.",
    "She opened the old wooden box carefully, half expecting to find letters, but instead there were dozens of faded black and white photographs.",
    "The documentary follows a small team of researchers as they track a population of migratory birds across three continents over two years.",
    "Learning a new language as an adult takes patience, consistent daily practice, and a willingness to make mistakes in front of other people.",
]

APPLY_PROMPT = "Fact: The currency used in the country shaped like a boot is"


def top(tok, logits, k=5):
    return [tok.decode([t]) for t in logits.topk(k).indices.tolist()]


def load_hf(model, dtype=torch.bfloat16):
    """Gemma 3 at 4b+ is Gemma3ForConditionalGeneration, which may not map to the
    CausalLM auto-class in transformers 5.x. Try loaders in order and report which
    one wins so the same pattern can go into probe_jlens.py."""
    errs = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            m = cls.from_pretrained(model, dtype=dtype)
            print(f"load_hf: {name} OK", flush=True)
            return m
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}: {e}")
            print(f"load_hf: {name} FAILED -- {type(e).__name__}", flush=True)
    raise RuntimeError("all loaders failed:\n" + "\n".join(errs))


def main():
    print(f"transformers={transformers.__version__} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = load_hf(MODEL).cuda()
    print(f"loaded {type(hf).__name__}  arch={hf.config.architectures}", flush=True)

    model = jlens.from_hf(hf, tok)
    print(f"layout={model.layout}  n_layers={model.n_layers}  d_model={model.d_model}",
          flush=True)

    lens = jlens.fit(model, PROMPTS, dim_batch=16, max_seq_len=64)
    print(f"fit OK -> source_layers={lens.source_layers[:4]}...{lens.source_layers[-3:]}",
          flush=True)

    lens_logits, model_logits, _ = lens.apply(model, APPLY_PROMPT, positions=[-2])
    print(f"\napply OK  model_logits.shape={tuple(model_logits.shape)}")
    print(f"MODEL top-5 @ -2: {top(tok, model_logits[0])}")

    model_top1 = model_logits[0].argmax().item()
    agree = []
    for L in sorted(lens_logits):
        ll = lens_logits[L]
        assert ll.shape == model_logits.shape, f"shape mismatch layer {L}: {ll.shape}"
        agree.append(ll[0].argmax().item() == model_top1)
    print(f"lens/model top-1 agreement: {sum(agree)}/{len(agree)} layers")
    for L in list(sorted(lens_logits))[::max(1, model.n_layers // 6)]:
        print(f"  L{L:>2} lens top-5: {top(tok, lens_logits[L][0])}")

    ok = (
        model.d_model > 0
        and len(agree) == model.n_layers - 1
        and any(agree[-5:])  # late layers should concord with the model
    )
    print(f"\n{'PASS' if ok else 'FAIL'}: jlens <-> Gemma 3 integration", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
